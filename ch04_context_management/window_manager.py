"""
Central context-window manager for LLM agent loops.

Three eviction strategies keep the window within bounds:

  FIFO       -- Drops the oldest non-pinned message on every overflow cycle.
  IMPORTANCE -- Scores messages by recency, role, and content keywords;
                removes the lowest-scored non-pinned pair first.
  SUMMARY    -- When the window crosses the critical threshold, calls Claude
                Haiku to compress all but the last three turns into a single
                summary message, preserving semantics at low token cost.
                Falls back to extractive summarisation when no client is set.

Each message carries an importance score and an optional pin flag.
Pinned messages are never evicted by any strategy.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


# ── strategy enum ─────────────────────────────────────────────────────────────

class Strategy(str, Enum):
    """Eviction strategy triggered when the context window fills up."""
    FIFO       = "FIFO"
    IMPORTANCE = "IMPORTANCE"
    SUMMARY    = "SUMMARY"


# ── public types ──────────────────────────────────────────────────────────────

@dataclass
class WindowStatus:
    """State returned by WindowManager.add_message() after each insertion.

    Attributes:
        current_tokens:          Token count in the window after this message.
        window_pct:              Percentage of max_tokens consumed (0-100).
        strategy_applied:        Eviction strategy name ("fifo" | "importance"
                                 | "summary"), or None if no eviction ran.
        messages_dropped:        Number of messages removed in this cycle.
        quality_impact_estimate: Fraction of information likely lost:
                                 0.0 = no loss, 1.0 = complete loss.
    """
    current_tokens:          int
    window_pct:              float
    strategy_applied:        str | None
    messages_dropped:        int
    quality_impact_estimate: float


# ── internal message wrapper ──────────────────────────────────────────────────

@dataclass
class _Msg:
    """Internal representation of a stored message with eviction metadata."""
    id:               str
    role:             str
    content:          str
    tokens:           int
    importance_score: float
    pinned:           bool
    turn_index:       int   # 1-based turn in which this message was added


# ── importance scoring ────────────────────────────────────────────────────────

_SIGNAL_WORDS: frozenset[str] = frozenset({
    "error", "fail", "exception", "critical", "important", "result",
    "conclusion", "decision", "final", "answer", "solution", "summary",
    "key", "note", "warning", "problem", "issue", "bug", "fix", "resolved",
    "deadline", "breaking", "urgent", "blocked",
})


def _score_importance(
    content:     str,
    role:        str,
    turn_index:  int,
    total_turns: int,
) -> float:
    """Heuristic importance score in [0.0, 1.0].

    Combines recency (more recent = higher), keyword density (signal words
    from _SIGNAL_WORDS), and a small role bonus for assistant messages which
    carry synthesised answers.

    Args:
        content:     Message text.
        role:        "user" or "assistant".
        turn_index:  1-based turn number when the message was added.
        total_turns: Running turn count at the time of scoring.

    Returns:
        Float in [0.0, 1.0]; higher means more important to keep.
    """
    recency = turn_index / max(total_turns, 1)

    words           = set(content.lower().split())
    keyword_density = min(1.0, len(words & _SIGNAL_WORDS) * 0.10)

    role_bonus = 0.08 if role == "assistant" else 0.0

    return min(1.0, 0.55 * recency + 0.30 * keyword_density + 0.15 + role_bonus)


# ── window manager ────────────────────────────────────────────────────────────

class WindowManager:
    """Context window manager with automatic eviction for LLM agent loops.

    Usage::

        wm = WindowManager(max_tokens=100_000, strategy="SUMMARY", client=client)
        for user_text, assistant_text in dialogue:
            wm.add_message({"role": "user",      "content": user_text})
            status = wm.add_message({"role": "assistant", "content": assistant_text})
            if status.strategy_applied:
                print(f"Eviction: {status.strategy_applied}")
        messages = wm.get_context()   # pass directly to client.messages.create()

    Args:
        max_tokens:         Hard upper bound on context size (tokens).
        strategy:           Eviction strategy (FIFO | IMPORTANCE | SUMMARY).
        warn_threshold:     Fraction at which a WARN is surfaced (default 0.70).
        critical_threshold: Fraction at which eviction triggers (default 0.90).
        client:             Anthropic client -- required only for SUMMARY API mode.
        summary_model:      Model used for summarisation calls (default haiku).
    """

    def __init__(
        self,
        max_tokens:         int             = 100_000,
        strategy:           str | Strategy  = Strategy.FIFO,
        warn_threshold:     float           = 0.70,
        critical_threshold: float           = 0.90,
        client:             Any | None      = None,
        summary_model:      str             = "claude-haiku-4-5",
    ) -> None:
        if not (0 < warn_threshold < critical_threshold <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 < warn < critical <= 1. "
                f"Got warn={warn_threshold}, critical={critical_threshold}."
            )
        self.max_tokens         = max_tokens
        self.strategy           = Strategy(strategy)
        self.warn_threshold     = warn_threshold
        self.critical_threshold = critical_threshold
        self._client            = client
        self._summary_model     = summary_model

        self._messages:     list[_Msg] = []
        self._total_tokens: int        = 0
        self._turn_counter: int        = 0
        self._pinned_ids:   set[str]   = set()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_message(self, message: dict[str, Any]) -> WindowStatus:
        """Append a message and evict when the critical threshold is crossed.

        Eviction, if triggered, runs after appending the new message and is
        reflected immediately in the returned WindowStatus.

        Args:
            message: Dict with at least ``role`` and ``content`` keys.
                     An optional ``id`` key preserves a caller-supplied ID.

        Returns:
            WindowStatus after this addition (and any eviction).
        """
        role    = message.get("role", "user")
        content = message.get("content", "")
        if not isinstance(content, str):
            import json as _json
            content = _json.dumps(content)

        if role == "user":
            self._turn_counter += 1

        msg_id    = message.get("id") or str(uuid.uuid4())
        tokens    = _estimate_tokens(content)
        importance = _score_importance(
            content, role, self._turn_counter, max(self._turn_counter, 1)
        )
        stored = _Msg(
            id=msg_id,
            role=role,
            content=content,
            tokens=tokens,
            importance_score=importance,
            pinned=(msg_id in self._pinned_ids),
            turn_index=self._turn_counter,
        )
        self._messages.append(stored)
        self._total_tokens += tokens

        strategy_applied = None
        dropped          = 0
        quality_impact   = 0.0

        if self._total_tokens > self._critical_limit():
            dropped, quality_impact = self._evict()
            strategy_applied = self.strategy.value.lower()

        return self._build_status(strategy_applied, dropped, quality_impact)

    def pin_message(self, message_id: str) -> bool:
        """Mark a message as permanent -- it will never be evicted.

        Args:
            message_id: The ``id`` field of the message to protect.

        Returns:
            True if the message was found and pinned; False if not found.
        """
        self._pinned_ids.add(message_id)
        for msg in self._messages:
            if msg.id == message_id:
                msg.pinned = True
                return True
        return False

    def get_context(self) -> list[dict[str, Any]]:
        """Return the current window as plain dicts ready for the API.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts, oldest first.
        """
        return [{"role": m.role, "content": m.content} for m in self._messages]

    @property
    def message_count(self) -> int:
        """Number of messages currently stored in the window."""
        return len(self._messages)

    @property
    def total_tokens(self) -> int:
        """Running token estimate for all stored messages."""
        return self._total_tokens

    # ------------------------------------------------------------------
    # Eviction strategies
    # ------------------------------------------------------------------

    def _evict(self) -> tuple[int, float]:
        """Dispatch to the active strategy."""
        if self.strategy == Strategy.FIFO:
            return self._apply_fifo()
        if self.strategy == Strategy.IMPORTANCE:
            return self._apply_importance()
        if self.strategy == Strategy.SUMMARY:
            return self._apply_summary()
        return (0, 0.0)

    def _apply_fifo(self) -> tuple[int, float]:
        """Drop oldest non-pinned messages one at a time until under threshold.

        Quality impact is the importance-weighted fraction of content removed.

        Returns:
            (messages_dropped, quality_impact_estimate)
        """
        total_importance   = sum(m.importance_score for m in self._messages) or 1.0
        dropped_importance = 0.0
        dropped            = 0

        while self._total_tokens > self._critical_limit():
            idx = next(
                (i for i, m in enumerate(self._messages) if not m.pinned),
                None,
            )
            if idx is None:
                break
            removed = self._messages.pop(idx)
            self._total_tokens -= removed.tokens
            dropped_importance += removed.importance_score
            dropped            += 1

        return (dropped, round(dropped_importance / total_importance, 4))

    def _apply_importance(self) -> tuple[int, float]:
        """Drop lowest-scored non-pinned pair, falling back to singles.

        Consecutive pairs (user + assistant) are scored by their mean
        importance and the worst pair is removed first.

        Returns:
            (messages_dropped, quality_impact_estimate)
        """
        total_importance   = sum(m.importance_score for m in self._messages) or 1.0
        dropped_importance = 0.0
        dropped            = 0

        while self._total_tokens > self._critical_limit():
            # Locate the lowest-scored consecutive evictable pair
            pair_idx: int | None = None
            lowest               = float("inf")
            for i in range(len(self._messages) - 1):
                a, b = self._messages[i], self._messages[i + 1]
                if a.pinned or b.pinned:
                    continue
                score = (a.importance_score + b.importance_score) / 2
                if score < lowest:
                    lowest   = score
                    pair_idx = i

            if pair_idx is not None:
                for msg in self._messages[pair_idx : pair_idx + 2]:
                    self._total_tokens -= msg.tokens
                    dropped_importance += msg.importance_score
                    dropped            += 1
                del self._messages[pair_idx : pair_idx + 2]
            else:
                # No full evictable pair -- drop worst single message
                candidates = [m for m in self._messages if not m.pinned]
                if not candidates:
                    break
                worst = min(candidates, key=lambda m: m.importance_score)
                self._messages.remove(worst)
                self._total_tokens -= worst.tokens
                dropped_importance += worst.importance_score
                dropped            += 1

        return (dropped, round(dropped_importance / total_importance, 4))

    def _apply_summary(self) -> tuple[int, float]:
        """Compress the middle turns into a summary user/assistant pair.

        Preserves pinned messages and the last 6 non-pinned messages (3 full
        turns). Everything before that is replaced by a compact summary.
        Falls back to FIFO when fewer than 8 messages are available.

        The summary message is automatically pinned so it is never evicted.

        Returns:
            (messages_dropped, quality_impact_estimate)
        """
        unpinned = [m for m in self._messages if not m.pinned]
        if len(unpinned) < 8:
            return self._apply_fifo()

        KEEP_TAIL   = 6
        pinned_msgs = [m for m in self._messages if m.pinned]
        tail        = unpinned[-KEEP_TAIL:]
        to_compress = unpinned[:-KEEP_TAIL]

        dropped          = len(to_compress)
        total_importance = sum(m.importance_score for m in self._messages) or 1.0
        lost_importance  = sum(m.importance_score for m in to_compress)

        summary_text = self._call_summariser(to_compress)
        s_tokens     = _estimate_tokens(summary_text)
        ack_text     = "Understood, I have the prior context."
        a_tokens     = _estimate_tokens(ack_text)

        summary_msg = _Msg(
            id=str(uuid.uuid4()), role="user",
            content=f"[Context summary]\n{summary_text}",
            tokens=s_tokens, importance_score=0.92, pinned=True, turn_index=0,
        )
        ack_msg = _Msg(
            id=str(uuid.uuid4()), role="assistant",
            content=ack_text,
            tokens=a_tokens, importance_score=0.50, pinned=False, turn_index=0,
        )

        self._messages     = pinned_msgs + [summary_msg, ack_msg] + tail
        self._total_tokens = sum(m.tokens for m in self._messages)

        # Information is compressed, not lost -- cap impact at 0.35
        quality_impact = min(0.35, (lost_importance / total_importance) * 0.55)
        return (dropped, round(quality_impact, 4))

    # ------------------------------------------------------------------
    # Summarisation helpers
    # ------------------------------------------------------------------

    def _call_summariser(self, messages: list[_Msg]) -> str:
        """Route to API or extractive summariser depending on client availability."""
        if self._client is not None:
            try:
                return self._api_summary(messages)
            except Exception:
                pass  # fall through to extractive on any API error
        return self._extractive_summary(messages)

    def _api_summary(self, messages: list[_Msg]) -> str:
        """Call Claude Haiku to produce a concise bullet-point summary.

        Args:
            messages: Messages to compress.

        Returns:
            3-5 bullet points as a plain string.
        """
        transcript = "\n".join(
            f"{m.role.upper()}: {m.content}" for m in messages
        )
        resp = self._client.messages.create(
            model=self._summary_model,
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": (
                    "Summarise this conversation in 3-5 bullet points. "
                    "Focus on decisions, key facts, and open questions. "
                    "Be terse -- this replaces raw transcript in the context window.\n\n"
                    f"{transcript}"
                ),
            }],
        )
        return resp.content[0].text.strip()

    def _extractive_summary(self, messages: list[_Msg]) -> str:
        """Offline fallback: first sentence of each assistant message.

        Args:
            messages: Messages to compress.

        Returns:
            Up to 5 bullet points extracted from assistant content.
        """
        bullets: list[str] = []
        for m in messages:
            if m.role == "assistant":
                first = m.content.split(".")[0][:100].strip()
                if first:
                    bullets.append(f"- {first}.")
        if not bullets:
            bullets.append(
                f"- {len(messages)} messages compressed (no client; no assistant content)."
            )
        return "\n".join(bullets[:5])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _critical_limit(self) -> int:
        return int(self.max_tokens * self.critical_threshold)

    def _build_status(
        self,
        strategy_applied: str | None,
        messages_dropped: int,
        quality_impact:   float,
    ) -> WindowStatus:
        pct = (self._total_tokens / self.max_tokens * 100.0) if self.max_tokens > 0 else 0.0
        return WindowStatus(
            current_tokens=          self._total_tokens,
            window_pct=              round(pct, 2),
            strategy_applied=        strategy_applied,
            messages_dropped=        messages_dropped,
            quality_impact_estimate= round(quality_impact, 4),
        )


# ── module-level helpers ──────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Cheap chars/4 token estimate used throughout the module."""
    return max(1, len(text) // 4) if text else 0


# ── demo conversation content ─────────────────────────────────────────────────
# 20 software-engineering Q&A pairs.
# User ~96 chars (24 tok) + Assistant ~440 chars (110 tok) = ~134 tok/turn.
# With max_tokens=1900 and critical_threshold=0.90 (limit=1710):
#   after turn 11 assistant: ~1650 < 1710  (no trigger)
#   after turn 12 user:      ~1650+25=1675 < 1710  (no trigger)
#   after turn 12 assistant: ~1803 > 1710  -> 1st SUMMARY (18 msgs dropped)
#   after turn 20 assistant: ~1688 < 1710, next msg pushes over  -> 2nd SUMMARY

_DEMO_TURNS: list[tuple[str, str]] = [
    (
        "What is the time complexity of a B-tree lookup?",
        "B-tree lookups run in O(log n) time. The height is log_t((n+1)/2) for minimum "
        "degree t. Each node holds up to 2t-1 keys; binary search within a node costs "
        "O(log t), giving total O(log t * log_t n) = O(log n). Disk-based B-trees align "
        "each node to one disk page, so real cost equals the number of I/Os. A million-"
        "record table with t=100 has at most 3 levels, meaning 2-4 reads. This predictable "
        "I/O behaviour is why PostgreSQL, MySQL, and most file systems use B-trees by default.",
    ),
    (
        "How does the Python GIL limit multi-threaded CPU work?",
        "The Global Interpreter Lock allows only one thread to execute Python bytecode at a "
        "time. The GIL is released every 5ms for switching and during blocking I/O syscalls. "
        "For CPU-bound work -- sorting, encryption, numerical computation -- multiple threads "
        "offer no throughput gain because they contend for the lock. Solutions: multiprocessing "
        "gives each process its own GIL; C extensions like NumPy release the GIL around their "
        "inner loops. For I/O-bound workloads threads remain effective because the GIL is "
        "surrendered while waiting for network or disk.",
    ),
    (
        "Explain the CAP theorem and its practical trade-offs.",
        "CAP states a distributed system can guarantee at most two of: Consistency (all nodes "
        "see the same data simultaneously), Availability (every request gets a response), and "
        "Partition tolerance (the system runs despite network splits). Partitions are unavoidable "
        "in real networks, so the real choice is CP vs AP. CP systems (ZooKeeper, HBase) halt "
        "or reject writes during a partition to stay consistent. AP systems (Cassandra, DynamoDB) "
        "remain writable and reconcile conflicts later via last-write-wins or vector clocks.",
    ),
    (
        "Walk me through the TCP three-way handshake.",
        "TCP connection setup uses three segments. First, the client sends SYN with its initial "
        "sequence number (ISN). Second, the server replies SYN-ACK, acknowledging ISN+1 and "
        "advertising its own ISN. Third, the client sends ACK confirming the server's ISN+1. "
        "Both sides now share agreed sequence numbers and can send data. Setup costs 1.5 RTTs "
        "before the first byte. TLS 1.3 piggybacks its key exchange onto this handshake, "
        "reducing total secure-connection setup to 1-RTT or 0-RTT for session resumption.",
    ),
    (
        "When should I use memory-mapped files instead of read()?",
        "Memory-mapped files map a file directly into the process address space; the OS pages "
        "in only the accessed regions on demand, making startup instant for multi-gigabyte files. "
        "The kernel page cache is shared across processes mapping the same file, so RAM is not "
        "duplicated. Use mmap for: sparse random access on large files, sharing data between "
        "processes, or zero-copy reads. Avoid it for strictly sequential streaming (buffered "
        "read() is simpler), NFS-mounted files (mmap semantics are unreliable), and files that "
        "grow frequently after mapping.",
    ),
    (
        "What are the key differences between Redis and Memcached?",
        "Redis supports rich data structures -- strings, lists, sets, sorted sets, hashes, "
        "streams, HyperLogLog -- while Memcached stores only opaque byte strings. Redis persists "
        "data via RDB snapshots or append-only file logging; Memcached is purely in-memory. "
        "Redis offers Lua scripting, pub/sub messaging, and geospatial queries. Memcached scales "
        "horizontally via client-side consistent hashing; Redis Cluster adds server-side sharding. "
        "Choose Memcached for a pure high-throughput LRU cache; choose Redis when you need "
        "persistence, secondary data structures, or atomic multi-key operations.",
    ),
    (
        "How does consistent hashing reduce resharding overhead?",
        "Consistent hashing maps both cache nodes and keys to positions on a circular ring. A "
        "key is owned by the first node clockwise from its hash position. Adding or removing a "
        "node migrates only the keys between that node and its predecessor -- roughly 1/n of all "
        "keys. Traditional modulo-N hashing remaps nearly all keys on any change, causing a "
        "cache-miss thundering herd. Virtual nodes (vnodes) -- where each physical node occupies "
        "multiple ring slots -- smooth load distribution and allow capacity-weighted assignment "
        "by giving larger nodes more vnodes.",
    ),
    (
        "Explain how LSM trees achieve high write throughput.",
        "Log-Structured Merge trees buffer all writes in a sorted in-memory structure (MemTable). "
        "When the MemTable fills, it is flushed as an immutable sorted file (SSTable) on disk. "
        "All writes are sequential appends, maximising throughput and eliminating random I/O. "
        "Reads must search multiple SSTables; periodic compaction merges them to reduce read "
        "amplification. Bloom filters per SSTable avoid reading files that cannot contain a key. "
        "RocksDB and Cassandra use LSM trees and outperform B-trees for write-heavy workloads "
        "by trading read amplification for near-sequential write performance.",
    ),
    (
        "What strategies prevent database deadlocks?",
        "Deadlock prevention ensures at least one Coffman condition cannot hold. Common approaches: "
        "lock ordering (always acquire A before B, never the reverse); timeout-based detection "
        "(roll back transactions holding locks beyond a threshold); wait-for graph cycle detection "
        "(the DBMS scans periodically and kills a victim); and optimistic concurrency control "
        "(no locks during execution; validate at commit and retry on conflict). PostgreSQL uses "
        "cycle detection and aborts one transaction. Applications should retry on serialisation "
        "failures and access shared resources in a consistent global order.",
    ),
    (
        "When should I choose JWTs over server-side sessions?",
        "JWTs are self-contained: the server encodes claims and signs with a secret, so any "
        "server can verify without a database lookup -- ideal for stateless, horizontally scaled "
        "services and cross-domain auth (OAuth2, OIDC). Server-side sessions store state centrally "
        "(Redis, DB) and send only a session ID to the client. Sessions are trivially revocable; "
        "JWTs remain valid until expiry unless you maintain a revocation list, reintroducing "
        "state. Use JWTs for microservices and third-party API access; use sessions for web apps "
        "that require instant logout or privilege escalation.",
    ),
    (
        "What are the trade-offs of horizontal database sharding?",
        "Sharding splits data across instances by a shard key, enabling linear write scalability "
        "and keeping per-node dataset size manageable. Trade-offs: cross-shard joins require "
        "scatter-gather queries or denormalisation; resharding is operationally complex and may "
        "need double-writes or downtime; distributed transactions spanning shards require two-phase "
        "commit or the saga pattern. Shard-key choice is critical -- a timestamp key creates hot "
        "spots, while a hashed entity ID distributes load evenly. Plan for resharding from day one "
        "by using consistent hashing or a virtual-shard layer.",
    ),
    # --- SUMMARY triggers here (turn 12 assistant message) ---
    (
        "What is event sourcing and when is it the right choice?",
        "Event sourcing stores state as an immutable sequence of domain events rather than the "
        "current snapshot. Current state is derived by replaying events (or from a snapshot "
        "checkpoint). This gives complete audit history, point-in-time queries, and the ability "
        "to derive new projections from stored history. Trade-offs: eventual consistency between "
        "the event store and read projections; query complexity requiring CQRS read models; and "
        "replay time growing with event count without snapshotting. Best suited for financial "
        "ledgers, order management, and domains with strict audit requirements.",
    ),
    (
        "How do circuit breakers improve service resilience?",
        "A circuit breaker wraps calls to a dependency and tracks recent failure rates. When "
        "failures exceed a threshold the breaker trips to OPEN, returning errors immediately "
        "without attempting the failing call -- preventing latency pile-up and thread exhaustion "
        "in the caller. After a configurable timeout the breaker enters HALF-OPEN, allowing one "
        "probe request. Success resets it to CLOSED; failure keeps it OPEN. Libraries like "
        "Resilience4j and Hystrix implement this pattern. Pair with exponential backoff and "
        "fallbacks (cached responses, degraded mode) for full resilience.",
    ),
    (
        "How does Kubernetes schedule pods onto nodes?",
        "The kube-scheduler assigns pods to nodes in two phases: filtering and scoring. Filtering "
        "removes nodes that violate hard constraints -- resource requests, node affinity, taints "
        "and tolerations, port availability. Scoring ranks remaining nodes by soft preferences: "
        "spreading pods across zones (pod anti-affinity), preferring nodes with the image already "
        "cached (ImageLocalityPriority), and balancing CPU/memory utilisation. The highest-scored "
        "node wins. Custom schedulers or scheduler extenders can inject domain-specific logic, "
        "and PriorityClasses determine eviction order under resource pressure.",
    ),
    (
        "Explain how Bloom filters work and their limitations.",
        "A Bloom filter is a probabilistic data structure that tests set membership using k hash "
        "functions mapping each element to k bit positions in a bit array. On insert, all k bits "
        "are set. On query, if any bit is unset the element is definitely absent; if all are set "
        "the element is probably present (false positives are possible, false negatives are not). "
        "Space efficiency is O(n) with low constant. Limitations: cannot delete elements (use "
        "Counting Bloom Filter), false-positive rate grows as the filter fills, and the bit array "
        "size must be chosen at creation time based on expected element count and desired FPR.",
    ),
    (
        "What is zero-copy I/O and when does it matter?",
        "Zero-copy transfers data between kernel and user space (or between file descriptors) "
        "without copying through the application buffer. Linux sendfile() copies data from a "
        "file descriptor directly to a socket in kernel space, eliminating two user/kernel "
        "crossings and two memcpy calls. splice() and vmsplice() extend this to pipe-based "
        "pipelines. Zero-copy matters for high-throughput file serving (Nginx uses sendfile), "
        "Kafka log replication, and network storage. Gains are most significant when CPU copy "
        "cost exceeds network/disk bandwidth; for small payloads the syscall overhead dominates.",
    ),
    (
        "How does a service mesh differ from an API gateway?",
        "An API gateway sits at the cluster edge and handles north-south traffic: authentication, "
        "rate limiting, routing, and protocol translation for external clients. A service mesh "
        "operates as a sidecar proxy (Envoy) injected into every pod, managing east-west traffic "
        "between services: mutual TLS, circuit breaking, retries, load balancing, and distributed "
        "tracing -- without changes to application code. The gateway owns the external contract; "
        "the mesh owns internal reliability. They are complementary: Istio or Linkerd for the mesh, "
        "Kong or AWS API Gateway for the edge.",
    ),
    (
        "What data does distributed tracing capture and why is it useful?",
        "Distributed tracing records the causal chain of work across service boundaries. Each "
        "logical operation is a trace; each unit of work within a service is a span. Spans carry "
        "a shared trace ID propagated via HTTP headers (W3C traceparent). Collected spans are "
        "assembled into a trace tree showing call graph, latencies, and error tags. This reveals "
        "which service introduced a latency spike, where errors originate in a dependency chain, "
        "and how traffic distributes across instances. OpenTelemetry provides vendor-neutral "
        "instrumentation; Jaeger, Zipkin, and Tempo are common backends.",
    ),
    (
        "What are CRDTs and how do they enable conflict-free merges?",
        "Conflict-free Replicated Data Types are data structures with a merge operation that is "
        "commutative, associative, and idempotent. Any two replicas can be merged in any order "
        "and converge to the same result without coordination. G-Counter (grow-only counter per "
        "node), OR-Set (observed-remove set), and LWW-Element-Set are common examples. CRDTs "
        "enable AP systems (Riak, Automerge, Yjs for collaborative editing) to accept concurrent "
        "writes on all replicas and merge lazily. The trade-off is richer data semantics -- not "
        "all application logic can be expressed as a CRDT.",
    ),
    (
        "Explain eventual consistency and how systems achieve it.",
        "Eventual consistency guarantees that, given no new updates, all replicas will converge "
        "to the same value. It does not bound when. Systems achieve it via: gossip protocols "
        "(nodes periodically exchange state and reconcile differences), anti-entropy repair "
        "(periodic background scans compare Merkle trees across replicas and sync divergent "
        "ranges), and read-repair (on a read, the coordinator compares versions from multiple "
        "replicas and fixes stale ones). Cassandra, DynamoDB, and Riak use these techniques. "
        "Monotonic read and read-your-writes are stronger session guarantees layered on top.",
    ),
]


# ── demo runner ───────────────────────────────────────────────────────────────

def run_demo(
    num_turns:  int         = 20,
    strategy:   str         = "SUMMARY",
    client:     Any | None  = None,
    max_tokens: int         = 1900,
) -> None:
    """Simulate num_turns of conversation and print window evolution.

    Runs the specified strategy and a FIFO baseline in parallel on the same
    conversation. Prints a turn-by-turn table and a quality comparison.

    Args:
        num_turns:  Number of user+assistant turns to simulate (max 20).
        strategy:   Strategy under test ("FIFO", "IMPORTANCE", "SUMMARY").
        client:     Anthropic client (enables API summarisation).
        max_tokens: Context window size for both managers in the demo.
    """
    num_turns = min(num_turns, len(_DEMO_TURNS))

    wm_test = WindowManager(
        max_tokens=max_tokens, strategy=strategy,
        warn_threshold=0.70, critical_threshold=0.90, client=client,
    )
    wm_fifo = WindowManager(
        max_tokens=max_tokens, strategy="FIFO",
        warn_threshold=0.70, critical_threshold=0.90,
    )

    sep = "=" * 82
    warn_limit     = int(max_tokens * 0.70)
    critical_limit = int(max_tokens * 0.90)

    print(sep)
    print(
        f"  Window manager demo  |  strategy={strategy}  "
        f"|  window={max_tokens:,}  |  warn={warn_limit:,}  |  critical={critical_limit:,}"
    )
    print(sep)
    print(
        f"  {'Turn':>4}  {'Msgs':>4}  {'Tokens':>7}  {'Window%':>7}  "
        f"{'Alert':>8}  {'Strategy':>10}  {'Dropped':>7}  {'Quality':>7}"
    )
    print(f"  {'-'*78}")

    events: list[str] = []

    for turn_idx, (user_text, asst_text) in enumerate(
        _DEMO_TURNS[:num_turns], start=1
    ):
        wm_fifo.add_message({"role": "user",      "content": user_text})
        wm_fifo.add_message({"role": "assistant", "content": asst_text})

        wm_test.add_message({"role": "user", "content": user_text})
        st = wm_test.add_message({"role": "assistant", "content": asst_text})

        pct   = st.window_pct
        alert = (
            "CRITICAL" if pct >= 90.0 else
            "WARN"     if pct >= 70.0 else
            "OK"
        )
        strat_label = st.strategy_applied or "-"
        quality_str = f"{st.quality_impact_estimate:.3f}" if st.strategy_applied else "-"

        eviction_marker = ""
        if st.strategy_applied:
            marker = (
                f"  [!] Turn {turn_idx}: {st.strategy_applied.upper()} eviction -- "
                f"{st.messages_dropped} msgs dropped, "
                f"quality impact={st.quality_impact_estimate:.3f}"
            )
            eviction_marker = "  <--"
            events.append(marker)

        print(
            f"  {turn_idx:>4}  {wm_test.message_count:>4}  "
            f"{st.current_tokens:>7,}  {pct:>6.1f}%  "
            f"{alert:>8}  {strat_label:>10}  {st.messages_dropped:>7}  "
            f"{quality_str:>7}{eviction_marker}"
        )

    # ── quality comparison ────────────────────────────────────────────────────
    def _mean_importance(wm: WindowManager) -> float:
        msgs = wm._messages  # access internal list for demo analytics only
        if not msgs:
            return 0.0
        return sum(m.importance_score for m in msgs) / len(msgs)

    test_quality = _mean_importance(wm_test)
    fifo_quality = _mean_importance(wm_fifo)
    delta        = test_quality - fifo_quality

    print(f"\n  Eviction events:")
    if events:
        for ev in events:
            print(ev)
    else:
        print("  (none -- window never filled)")

    print(f"\n{sep}")
    print("  Quality comparison (mean importance score of retained messages)")
    print(f"  {'-'*60}")
    print(f"  {strategy:<12} strategy  msgs={wm_test.message_count:>3}  "
          f"tokens={wm_test.total_tokens:>6,}  mean_importance={test_quality:.4f}")
    print(f"  {'FIFO':<12} baseline   msgs={wm_fifo.message_count:>3}  "
          f"tokens={wm_fifo.total_tokens:>6,}  mean_importance={fifo_quality:.4f}")
    print(f"\n  Delta (test - FIFO): {delta:+.4f}  "
          f"({'test retains higher-importance messages' if delta > 0 else 'similar or lower quality'})")
    print(sep)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    _ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_ROOT))

    from pathlib import Path as _Path

    # .env loader (mirrors ch03_agent_loop pattern)
    def _load_env(path: Path) -> None:
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    _load_env(_ROOT / ".env")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    client  = None
    if api_key:
        try:
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=api_key)
            print("Anthropic client ready -- SUMMARY will use Claude Haiku API.\n")
        except ImportError:
            print("anthropic package not installed -- using extractive summary.\n")
    else:
        print("ANTHROPIC_API_KEY not set -- using extractive summary (no API calls).\n")

    run_demo(num_turns=20, strategy="SUMMARY", client=client, max_tokens=1900)
