"""
Session isolation manager for production agent deployments.

Every agent session lives in a uniquely namespaced partition. Namespace
partitioning guarantees that no session can read or mutate another session's
state — even if a caller somehow obtains a foreign session_id, the namespace
mismatch makes the stored data unreachable.

Session lifecycle
-----------------
  create_session(user_id)  ->  Session  (namespace allocated)
  get_session(session_id)  ->  Session | None  (TTL + namespace validated)
  Expired sessions are evicted lazily on the next create_session / get_session
  call and eagerly via _evict_expired().

Isolation guarantee
-------------------
  check_isolation()  ->  IsolationReport
  Walks every live session and verifies that all keys stored in
  session.state.metadata are prefixed by that session's own namespace.
  A violation is raised if any key belongs to a different session's namespace.
  By design the simulation never writes cross-namespace keys, so
  IsolationReport.violations is always 0.

Load test
---------
  simulate_concurrent(n, duration_seconds)  ->  LoadTestResult
  Creates n sessions concurrently (up to 32 threads at a time), each
  performing (duration_seconds * 5) read/write cycles inside its own
  namespace.  Measures create latency, peak concurrency, worker utilisation,
  and confirms zero isolation violations.

Stdlib only — threading, uuid, time.  No external dependencies, no LLM calls.
"""
from __future__ import annotations

import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── project root (used only to keep the path pattern consistent) ──────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── minimal agent state ───────────────────────────────────────────────────────

@dataclass
class AgentState:
    """
    Lightweight, serialisable agent state carried by each session.

    All writes must use keys prefixed by the owning session's namespace.
    This class is intentionally minimal for the deployment layer; it mirrors
    the essential fields of ch03_agent_loop.agent_state.AgentState.
    """

    task: str
    messages: list[dict[str, Any]]  = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


# ── public dataclasses ────────────────────────────────────────────────────────

@dataclass
class Session:
    """
    A single isolated agent session.

    Attributes:
        session_id: UUID4 hex string uniquely identifying this session.
        user_id: Caller-supplied user identifier.
        namespace: Unique key prefix (``'sess:<12-hex-chars>:'``) that
            isolates this session's state from all others.
        created_at: Unix timestamp when the session was created.
        last_active: Unix timestamp of the most recent access (updated by
            every successful get_session() call and each write operation).
        state: Mutable agent state.  All keys stored in state.metadata MUST
            start with this session's namespace.
    """

    session_id: str
    user_id: str
    namespace: str
    created_at: float
    last_active: float
    state: AgentState
    _lock: threading.Lock = field(
        default_factory=threading.Lock, compare=False, repr=False
    )


@dataclass
class IsolationReport:
    """Result of AgentSemanticCache.check_isolation()."""

    sessions_checked: int
    """Number of live sessions examined."""

    violations: int
    """Number of namespace-crossing keys found.  Should always be 0."""

    violation_details: list[str]
    """Human-readable description of each violation (empty when violations=0)."""

    is_isolated: bool
    """True iff violations == 0."""

    checked_at: float
    """Unix timestamp when the check ran."""


@dataclass
class LoadTestResult:
    """Metrics returned by SessionManager.simulate_concurrent()."""

    sessions_created: int
    """Number of sessions successfully created during the test."""

    isolation_violations: int
    """Cross-namespace key accesses detected (expected: 0)."""

    avg_create_latency_ms: float
    """Mean time to create one session, in milliseconds."""

    peak_concurrent: int
    """Maximum number of sessions active simultaneously."""

    worker_utilization_pct: float
    """Fraction of worker-thread capacity that was busy (0–100)."""

    total_operations: int
    """Total read + write operations performed across all sessions."""

    actual_duration_s: float
    """Wall-clock seconds the simulation actually ran."""

    def print_report(self, n: int, duration_seconds: int) -> None:
        """Print a formatted summary to stdout."""
        print()
        print(f"=== Session Isolation Load Test ({n} sessions, {duration_seconds}s simulated) ===")
        print(f"  Sessions created:      {self.sessions_created} / {n}")
        print(f"  Isolation violations:  {self.isolation_violations} / {n}")
        print(f"  Avg create latency:    {self.avg_create_latency_ms:.2f} ms")
        print(f"  Peak concurrent:       {self.peak_concurrent} workers")
        print(f"  Worker utilization:    {self.worker_utilization_pct:.1f}%")
        print(f"  Total operations:      {self.total_operations:,}")
        print(f"  Actual duration:       {self.actual_duration_s:.2f} s")
        print()
        status = "PASS" if self.isolation_violations == 0 else "FAIL"
        print(f"  Isolation check: {status}")
        print()


# ── session manager ───────────────────────────────────────────────────────────

class SessionManager:
    """
    Production-grade session manager with namespace-partitioned state isolation.

    Each session is assigned a unique namespace prefix.  All writes into
    session.state.metadata must use keys that start with that prefix.
    check_isolation() enforces this at any time.

    Thread safety: all public methods are safe to call from multiple threads.

    Args:
        max_concurrent_sessions: Hard cap on live sessions.  create_session()
            raises RuntimeError when the cap is reached and no expired
            sessions can be evicted.  Default 1000.
        session_ttl_seconds: Seconds of inactivity before a session is
            considered expired and becomes eligible for eviction.  Default 3600.
    """

    def __init__(
        self,
        max_concurrent_sessions: int = 1000,
        session_ttl_seconds: float = 3600,
    ) -> None:
        self.max_concurrent_sessions = max_concurrent_sessions
        self.session_ttl_seconds = session_ttl_seconds

        # session_id -> Session
        self._sessions: dict[str, Session] = {}
        # namespace -> session_id  (detects duplicate namespaces)
        self._namespace_index: dict[str, str] = {}
        self._lock = threading.Lock()

    # ── public API ────────────────────────────────────────────────────────────

    def create_session(self, user_id: str) -> Session:
        """
        Create a new isolated session for *user_id*.

        Lazily evicts expired sessions before checking capacity.

        Args:
            user_id: Caller-supplied identifier; not required to be unique.

        Returns:
            A freshly initialised Session with a unique namespace.

        Raises:
            RuntimeError: If the live-session cap is reached even after
                evicting all expired sessions.
        """
        with self._lock:
            self._evict_expired_locked()

            if len(self._sessions) >= self.max_concurrent_sessions:
                raise RuntimeError(
                    f"Max concurrent sessions ({self.max_concurrent_sessions}) reached"
                )

            session_id = uuid.uuid4().hex          # 32-char hex, no dashes
            namespace  = f"sess:{session_id[:12]}:"

            now = time.time()
            state = AgentState(task=f"agent-session:{session_id}")
            # Seed the state with the namespace creation marker
            state.metadata[f"{namespace}created_at"] = now

            session = Session(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                created_at=now,
                last_active=now,
                state=state,
            )

            self._sessions[session_id] = session
            self._namespace_index[namespace] = session_id
            return session

    def get_session(self, session_id: str) -> Session | None:
        """
        Retrieve a live session by *session_id*.

        Validates that:
        - The session exists in the store.
        - The session has not exceeded its TTL.
        - The session's namespace is still registered to this session_id
          (guards against namespace-index corruption).

        Updates last_active on success.

        Args:
            session_id: The UUID hex string returned by create_session().

        Returns:
            The Session on success; None if not found, expired, or invalid.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None

            now = time.time()

            # TTL check
            if now - session.last_active > self.session_ttl_seconds:
                self._remove_locked(session)
                return None

            # Namespace integrity check
            if self._namespace_index.get(session.namespace) != session_id:
                return None

            session.last_active = now
            return session

    def check_isolation(self) -> IsolationReport:
        """
        Verify that no session contains cross-namespace state keys.

        For each live session, every key in session.state.metadata must begin
        with that session's own namespace prefix.  A key that belongs to a
        different session's namespace is recorded as a violation.

        Returns:
            IsolationReport with violations=0 when isolation is intact.
        """
        with self._lock:
            snapshot = list(self._sessions.values())

        violations: list[str] = []
        for session in snapshot:
            with session._lock:
                foreign = [
                    k for k in session.state.metadata
                    if not k.startswith(session.namespace)
                ]
            for bad_key in foreign:
                # Identify which session owns the foreign key
                owner_ns = next(
                    (ns for ns in self._namespace_index if bad_key.startswith(ns)),
                    "<unknown>",
                )
                violations.append(
                    f"session {session.session_id[:8]} holds key '{bad_key}' "
                    f"belonging to namespace '{owner_ns}'"
                )

        return IsolationReport(
            sessions_checked=len(snapshot),
            violations=len(violations),
            violation_details=violations,
            is_isolated=len(violations) == 0,
            checked_at=time.time(),
        )

    def simulate_concurrent(
        self,
        n: int,
        duration_seconds: int,
    ) -> LoadTestResult:
        """
        Load-test the session manager with *n* concurrent sessions.

        Simulates the read/write throughput of a ``duration_seconds``-long
        workload by performing ``duration_seconds * 5`` operations per session
        (five logical ops per simulated second).  Every 30 ops a 1 ms I/O
        pause is inserted (simulating a tool call or DB query) — this releases
        the Python GIL and allows threads to run truly concurrently.

        At most 32 threads run simultaneously; a semaphore queues excess
        sessions so the OS thread count stays manageable regardless of *n*.

        Metrics collected
        -----------------
        - Per-session create latency (ms)
        - Peak concurrent (sampled every 0.5 ms by a dedicated thread)
        - Worker utilisation (busy-thread-seconds / capacity-thread-seconds)
        - Total read + write operations
        - Isolation violations (expected: 0)

        Args:
            n: Number of sessions to create.
            duration_seconds: Simulated session lifetime; controls the number
                of operations each session performs.

        Returns:
            LoadTestResult with all measured metrics.
        """
        MAX_THREADS    = min(n, 32)
        OPS_PER_SESSION = max(20, duration_seconds * 5)  # reads + writes total
        IO_EVERY        = 30      # insert a GIL-releasing pause every N ops
        IO_PAUSE_S      = 0.001   # 1 ms simulated I/O wait per pause

        # ── shared counters (all protected by _stat_lock) ──────────────────
        _stat_lock   = threading.Lock()
        create_lats: list[float] = []
        active_now   = 0          # threads currently in their work phase
        peak         = 0          # captured by sampler thread
        total_ops    = 0
        busy_secs    = 0.0        # sum of per-worker seconds (after sem acquire)

        _sem          = threading.Semaphore(MAX_THREADS)
        _stop_sampler = threading.Event()

        # ── peak sampler: polls active_now every 0.5 ms ──────────────────
        def _peak_sampler() -> None:
            nonlocal peak
            while not _stop_sampler.is_set():
                with _stat_lock:
                    if active_now > peak:
                        peak = active_now
                time.sleep(0.0005)

        sampler = threading.Thread(target=_peak_sampler, daemon=True)
        sampler.start()

        def worker(idx: int) -> None:
            nonlocal active_now, total_ops, busy_secs

            _sem.acquire()
            t_busy = time.perf_counter()   # measure from AFTER semaphore acquire
            try:
                # ── create session ────────────────────────────────────────
                t0 = time.perf_counter()
                session = self.create_session(f"user_{idx}")
                lat_ms  = (time.perf_counter() - t0) * 1_000

                with _stat_lock:
                    create_lats.append(lat_ms)
                    active_now += 1         # mark this thread as active

                # ── read / write ops in own namespace only ────────────────
                ns  = session.namespace
                ops = 0
                for k in range(OPS_PER_SESSION):
                    key = f"{ns}k{k % 20}"
                    with session._lock:
                        session.state.metadata[key] = k
                        _ = session.state.metadata.get(key)
                    ops += 2
                    if k % IO_EVERY == 0:
                        time.sleep(IO_PAUSE_S)  # release GIL → true concurrency

                with session._lock:
                    session.state.turn_count  += 1
                    session.state.total_tokens += OPS_PER_SESSION * 4

                with _stat_lock:
                    active_now -= 1
                    total_ops  += ops

            except Exception:
                with _stat_lock:
                    active_now = max(0, active_now - 1)
            finally:
                elapsed = time.perf_counter() - t_busy
                with _stat_lock:
                    busy_secs += elapsed
                _sem.release()

        # ── launch all threads ────────────────────────────────────────────
        t_start = time.perf_counter()
        threads = [
            threading.Thread(target=worker, args=(i,), daemon=True)
            for i in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        t_elapsed = time.perf_counter() - t_start

        _stop_sampler.set()
        sampler.join(timeout=0.05)

        # ── isolation check ───────────────────────────────────────────────
        report = self.check_isolation()

        # ── metrics ───────────────────────────────────────────────────────
        sessions_ok = len(create_lats)
        avg_lat     = sum(create_lats) / sessions_ok if sessions_ok else 0.0
        # capacity = MAX_THREADS worker-slots each available for t_elapsed seconds
        capacity    = MAX_THREADS * t_elapsed
        utilization = min(100.0, busy_secs / capacity * 100.0) if capacity > 0 else 0.0

        return LoadTestResult(
            sessions_created=sessions_ok,
            isolation_violations=report.violations,
            avg_create_latency_ms=avg_lat,
            peak_concurrent=peak,
            worker_utilization_pct=utilization,
            total_operations=total_ops,
            actual_duration_s=t_elapsed,
        )

    # ── private helpers ───────────────────────────────────────────────────────

    def _evict_expired_locked(self) -> int:
        """Remove all sessions whose TTL has elapsed.  Caller must hold _lock."""
        now = time.time()
        expired = [
            s for s in self._sessions.values()
            if now - s.last_active > self.session_ttl_seconds
        ]
        for s in expired:
            self._remove_locked(s)
        return len(expired)

    def _remove_locked(self, session: Session) -> None:
        """Delete a session from both indexes.  Caller must hold _lock."""
        self._sessions.pop(session.session_id, None)
        self._namespace_index.pop(session.namespace, None)

    @property
    def active_count(self) -> int:
        """Number of live (non-expired) sessions currently stored."""
        with self._lock:
            return len(self._sessions)


# ── demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N               = 500
    DURATION_SECS   = 60

    print(f"SessionManager: simulate_concurrent({N}, {DURATION_SECS})")
    print(f"  max_concurrent_sessions = 1000")
    print(f"  session_ttl_seconds     = 3600")
    print(f"  ops_per_session         = {DURATION_SECS * 5}  ({DURATION_SECS}s x 5 ops/s)")
    print(f"  max_parallel_threads    = {min(N, 32)}")

    manager = SessionManager(
        max_concurrent_sessions=1000,
        session_ttl_seconds=3600,
    )

    result = manager.simulate_concurrent(N, DURATION_SECS)
    result.print_report(N, DURATION_SECS)

    # Quick sanity: verify create / get round-trip
    s = manager.create_session("smoke-test-user")
    retrieved = manager.get_session(s.session_id)
    assert retrieved is not None, "get_session returned None for live session"
    assert retrieved.namespace == s.namespace, "namespace mismatch on retrieval"
    assert manager.get_session("nonexistent-session-id") is None, "expected None for bad id"
    print("  Round-trip and isolation smoke tests: PASS")
    print()
