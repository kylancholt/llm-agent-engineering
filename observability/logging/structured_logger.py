"""
Structured logging for LLM agent pipelines.

Provides a typed event schema for all agent activity (turn boundaries,
tool calls, reasoning steps, cost tracking, loop detection, errors) and
writes events asynchronously to JSONL files with daily rotation.

Key properties:
  - Non-blocking: log() enqueues immediately; a background thread writes.
  - Typed schema: EventType enum + AgentEvent dataclass.
  - Configurable sampling (default 100%).
  - Daily JSONL rotation:  {log_dir}/agent_events_{YYYY-MM-DD}.jsonl
  - Searchable by task_id, event_type, and time window.
  - Overhead target: < 1.5 ms per log() call (queue enqueue only).

Stdlib only -- no external dependencies.
"""
from __future__ import annotations

import datetime
import json
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# EventType
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """Typed event categories for all agent pipeline activity."""

    TURN_START        = "TURN_START"
    TOOL_CALL         = "TOOL_CALL"
    TOOL_RESULT       = "TOOL_RESULT"
    REASONING         = "REASONING"
    COST_UPDATE       = "COST_UPDATE"
    LOOP_DETECTED     = "LOOP_DETECTED"
    CONTEXT_TRUNCATED = "CONTEXT_TRUNCATED"
    FINAL_ANSWER      = "FINAL_ANSWER"
    ERROR             = "ERROR"


# ---------------------------------------------------------------------------
# AgentEvent
# ---------------------------------------------------------------------------

@dataclass
class AgentEvent:
    """
    One structured log record for an agent pipeline event.

    Attributes
    ----------
    event_type:
        Category; one of the :class:`EventType` string values.
    task_id:
        Identifier for the top-level task being executed.
    agent_id:
        Identifier for the agent that generated this event.
    turn_id:
        Sequential turn number within the task (1-based by convention).
    timestamp:
        Epoch seconds when the event occurred; defaults to ``time.time()``.
    tool_name:
        Name of the tool involved; ``None`` for non-tool events.
    token_count:
        Token count associated with this event (0 if not applicable).
    latency_ms:
        Wall-clock latency in milliseconds (0.0 if not applicable).
    cost_usd:
        Incremental USD cost for this event (0.0 if not applicable).
    metadata:
        Arbitrary key/value pairs for event-specific detail.
    """

    event_type: str
    task_id: str
    agent_id: str
    turn_id: int
    timestamp: float = field(default_factory=time.time)
    tool_name: Optional[str] = None
    token_count: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal flush sentinel
# ---------------------------------------------------------------------------

class _FlushSentinel:
    """
    Placed on the writer queue to signal a flush checkpoint.

    The writer sets ``done`` after processing all prior items, allowing
    the calling thread to block until those items have been written.
    """

    __slots__ = ("done",)

    def __init__(self) -> None:
        self.done = threading.Event()


# ---------------------------------------------------------------------------
# AgentStructuredLogger
# ---------------------------------------------------------------------------

class AgentStructuredLogger:
    """
    Asynchronous structured logger for agent pipeline events.

    Events are enqueued in the calling thread and written to a dated
    JSONL file by a dedicated background thread.  This design keeps
    ``log()`` non-blocking and its overhead well below 1.5 ms per call
    regardless of file I/O speed.

    Log files are rotated daily:
        {log_dir}/agent_events_{YYYY-MM-DD}.jsonl

    Parameters
    ----------
    log_dir:
        Directory where JSONL files are written.  Created if absent.
    buffer_size:
        Maximum events held in the in-memory queue before new ``log()``
        calls drop events (counted via :attr:`dropped_count`).
    """

    def __init__(
        self,
        log_dir: str = "logs/",
        buffer_size: int = 1_000,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._sampling_rate: float = 1.0
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._running: bool = True

        # overhead counters -- only modified by calling thread(s)
        self._total_overhead_ns: int = 0
        self._call_count: int = 0
        self._dropped_count: int = 0

        # written counter -- only modified by the writer thread
        self._events_written: int = 0

        self._thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="AgentStructuredLogger-writer",
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Core logging API
    # ------------------------------------------------------------------

    def log(self, event: AgentEvent) -> None:
        """
        Enqueue *event* for asynchronous writing to the JSONL log file.

        The call is non-blocking: sampling is checked first, then the
        event is placed on the internal queue with ``put_nowait``.  If
        the queue is full, the event is dropped and the drop counter is
        incremented.

        The time spent inside this method (sampling check + enqueue) is
        accumulated and exposed via :meth:`get_overhead`.

        Parameters
        ----------
        event:
            The :class:`AgentEvent` to record.
        """
        t0 = time.perf_counter_ns()
        try:
            if self._sampling_rate < 1.0 and random.random() > self._sampling_rate:
                return
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self._dropped_count += 1
        finally:
            self._total_overhead_ns += time.perf_counter_ns() - t0
            self._call_count += 1

    def set_sampling_rate(self, rate: float) -> None:
        """
        Configure what fraction of ``log()`` calls are written to disk.

        Parameters
        ----------
        rate:
            Float in ``[0.0, 1.0]``.  ``1.0`` records every event
            (default); ``0.5`` records roughly half; ``0.0`` silences
            all output.

        Raises
        ------
        ValueError
            If *rate* is outside ``[0.0, 1.0]``.
        """
        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"sampling rate must be in [0.0, 1.0], got {rate!r}"
            )
        self._sampling_rate = rate

    def get_overhead(self) -> float:
        """
        Return the average overhead of a single ``log()`` call in milliseconds.

        Measures the time spent inside ``log()`` (sampling check +
        ``queue.put_nowait``), not the actual file write.

        Returns
        -------
        float
            Average overhead per call in ms.  Target: < 1.5 ms.
        """
        if self._call_count == 0:
            return 0.0
        return (self._total_overhead_ns / self._call_count) / 1_000_000

    def get_stats(self) -> dict:
        """
        Return a snapshot of logger activity since construction.

        Returns
        -------
        dict
            Keys: ``call_count``, ``events_written``, ``dropped``,
            ``sampling_rate``, ``overhead_ms_avg``.
        """
        return {
            "call_count": self._call_count,
            "events_written": self._events_written,
            "dropped": self._dropped_count,
            "sampling_rate": self._sampling_rate,
            "overhead_ms_avg": self.get_overhead(),
        }

    # ------------------------------------------------------------------
    # Flow control
    # ------------------------------------------------------------------

    def flush(self, timeout: float = 5.0) -> bool:
        """
        Block until all queued events have been written to disk.

        Inserts a sentinel into the queue; the writer sets a flag once
        it processes the sentinel (meaning all prior events are on disk).

        Parameters
        ----------
        timeout:
            Maximum seconds to wait.

        Returns
        -------
        bool
            ``True`` if the flush completed within *timeout*.
        """
        sentinel = _FlushSentinel()
        self._queue.put(sentinel, timeout=timeout)
        return sentinel.done.wait(timeout=timeout)

    def close(self, timeout: float = 5.0) -> None:
        """
        Flush pending events and stop the background writer thread.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait for flush + thread join.
        """
        self.flush(timeout=timeout)
        self._running = False
        self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        task_id: Optional[str] = None,
        event_type: Optional[str] = None,
        hours: int = 24,
    ) -> list[AgentEvent]:
        """
        Search log files for events matching the specified criteria.

        Reads all JSONL files that fall within the *hours* time window,
        filtering by timestamp, *task_id*, and *event_type*.

        Parameters
        ----------
        task_id:
            If given, return only events whose ``task_id`` matches.
        event_type:
            If given, return only events whose ``event_type`` matches.
            Accepts plain strings (``"TOOL_CALL"``) or
            :class:`EventType` members.
        hours:
            How far back to search (default 24 hours).

        Returns
        -------
        list[AgentEvent]
            Matching events in file-order (approximately chronological).
        """
        cutoff_ts = time.time() - hours * 3600
        et_str: Optional[str]
        if event_type is None:
            et_str = None
        elif isinstance(event_type, Enum):
            et_str = event_type.value
        else:
            et_str = event_type
        results: list[AgentEvent] = []

        for date_str in _search_dates(cutoff_ts):
            log_path = self._log_dir / f"agent_events_{date_str}.jsonl"
            if not log_path.exists():
                continue
            try:
                with open(log_path, encoding="utf-8") as fh:
                    for raw in fh:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if data.get("timestamp", 0.0) < cutoff_ts:
                            continue
                        if task_id and data.get("task_id") != task_id:
                            continue
                        if et_str and data.get("event_type") != et_str:
                            continue
                        results.append(_dict_to_event(data))
            except OSError:
                continue

        return results

    # ------------------------------------------------------------------
    # Background writer thread
    # ------------------------------------------------------------------

    def _writer_loop(self) -> None:
        """Drain the queue and write events to the current day's JSONL file."""
        current_date: Optional[str] = None
        fh = None

        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.05)
                except queue.Empty:
                    if not self._running:
                        break
                    if fh is not None:
                        fh.flush()
                    continue

                if isinstance(item, _FlushSentinel):
                    if fh is not None:
                        fh.flush()
                    item.done.set()
                    continue

                # daily rotation
                today = datetime.date.today().isoformat()
                if today != current_date:
                    if fh is not None:
                        fh.close()
                    log_path = self._log_dir / f"agent_events_{today}.jsonl"
                    fh = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
                    current_date = today

                fh.write(json.dumps(_event_to_dict(item)) + "\n")  # type: ignore[union-attr]
                self._events_written += 1
        finally:
            if fh is not None:
                fh.close()


# ---------------------------------------------------------------------------
# Serialisation helpers (module-level for use in both class and search)
# ---------------------------------------------------------------------------

def _event_to_dict(event: AgentEvent) -> dict:
    """Serialise an :class:`AgentEvent` to a JSON-compatible dict."""
    return {
        "timestamp": event.timestamp,
        "task_id": event.task_id,
        "agent_id": event.agent_id,
        "turn_id": event.turn_id,
        "event_type": (
            event.event_type.value
            if isinstance(event.event_type, Enum)
            else event.event_type
        ),
        "tool_name": event.tool_name,
        "token_count": event.token_count,
        "latency_ms": event.latency_ms,
        "cost_usd": event.cost_usd,
        "metadata": event.metadata,
    }


def _dict_to_event(data: dict) -> AgentEvent:
    """Deserialise a JSON dict to an :class:`AgentEvent`."""
    return AgentEvent(
        timestamp=data["timestamp"],
        task_id=data["task_id"],
        agent_id=data["agent_id"],
        turn_id=data["turn_id"],
        event_type=data["event_type"],
        tool_name=data.get("tool_name"),
        token_count=data.get("token_count", 0),
        latency_ms=data.get("latency_ms", 0.0),
        cost_usd=data.get("cost_usd", 0.0),
        metadata=data.get("metadata", {}),
    )


def _search_dates(cutoff_ts: float) -> list[str]:
    """
    Return sorted ISO date strings for every calendar day in
    ``[cutoff_ts, now]``, always including today.
    """
    today = datetime.date.today().isoformat()
    dates: set[str] = set()
    t = cutoff_ts
    now = time.time()
    while t <= now:
        dates.add(datetime.datetime.fromtimestamp(t).date().isoformat())
        t += 86_400
    dates.add(today)
    return sorted(d for d in dates if d <= today)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _run_demo() -> None:
    """
    Log 30 simulated events across 2 agents and 2 tasks, covering all 9
    event types.  Measures per-call overhead, then searches by event_type
    and task_id.

    Expected output:
      - overhead < 1.5 ms per event
      - 6 TOOL_CALL events found (3 per task)
      - 15 events found for task_web_research
    """
    SEP = "=" * 68
    MID = "-" * 68

    print(SEP)
    print("  AgentStructuredLogger -- Structured Logging Demo")
    print("  30 events  |  2 agents  |  2 tasks  |  9 event types")
    print(SEP)
    print()

    # Use a dedicated demo directory; clear any stale JSONL from prior runs
    # so search results are reproducible.
    demo_log_dir = Path("logs") / "_demo"
    demo_log_dir.mkdir(parents=True, exist_ok=True)
    for stale in demo_log_dir.glob("*.jsonl"):
        stale.unlink()

    logger = AgentStructuredLogger(log_dir=str(demo_log_dir), buffer_size=500)
    T0 = time.time()

    def mk(et: str, task_id: str, agent_id: str, turn_id: int, **kw) -> AgentEvent:
        return AgentEvent(
            event_type=et,
            task_id=task_id,
            agent_id=agent_id,
            turn_id=turn_id,
            timestamp=T0,
            **kw,
        )

    # ── Task 1: agent_A  web-research pipeline (15 events) ───────────
    T1, A1 = "task_web_research", "agent_A"

    # turn 1
    logger.log(mk(EventType.TURN_START,  T1, A1, 1, metadata={"goal": "fetch AAPL price"}))
    logger.log(mk(EventType.TOOL_CALL,   T1, A1, 1, tool_name="web_search",
                  latency_ms=231.0, metadata={"query": "AAPL stock price Q3 2025"}))
    logger.log(mk(EventType.TOOL_RESULT, T1, A1, 1, tool_name="web_search",
                  token_count=110, latency_ms=0.5))
    logger.log(mk(EventType.REASONING,   T1, A1, 1, token_count=380, latency_ms=178.0))
    logger.log(mk(EventType.COST_UPDATE, T1, A1, 1, cost_usd=0.00126, token_count=630,
                  metadata={"running_total_usd": 0.00126}))

    # turn 2
    logger.log(mk(EventType.TURN_START,  T1, A1, 2, metadata={"goal": "fetch market cap"}))
    logger.log(mk(EventType.TOOL_CALL,   T1, A1, 2, tool_name="web_search",
                  latency_ms=215.0, metadata={"query": "AAPL market cap Q3 2025"}))
    logger.log(mk(EventType.TOOL_RESULT, T1, A1, 2, tool_name="web_search",
                  token_count=95, latency_ms=0.4))
    logger.log(mk(EventType.REASONING,   T1, A1, 2, token_count=290, latency_ms=140.0))
    logger.log(mk(EventType.COST_UPDATE, T1, A1, 2, cost_usd=0.00115, token_count=575,
                  metadata={"running_total_usd": 0.00241}))

    # turn 3
    logger.log(mk(EventType.TURN_START,  T1, A1, 3, metadata={"goal": "summarise findings"}))
    logger.log(mk(EventType.TOOL_CALL,   T1, A1, 3, tool_name="summarize",
                  latency_ms=254.0))
    logger.log(mk(EventType.TOOL_RESULT, T1, A1, 3, tool_name="summarize",
                  token_count=210, latency_ms=0.3))
    logger.log(mk(EventType.FINAL_ANSWER, T1, A1, 3,
                  metadata={"answer": "AAPL $189.30 +1.2%. 42-share value: $7,950.60."}))
    logger.log(mk(EventType.COST_UPDATE, T1, A1, 3, cost_usd=0.00161, token_count=850,
                  metadata={"running_total_usd": 0.00402}))

    # ── Task 2: agent_B  data-analysis pipeline (15 events) ──────────
    T2, A2 = "task_data_analysis", "agent_B"

    # turn 1
    logger.log(mk(EventType.TURN_START,  T2, A2, 1, metadata={"goal": "load dataset"}))
    logger.log(mk(EventType.TOOL_CALL,   T2, A2, 1, tool_name="fetch_data",
                  latency_ms=310.0, metadata={"source": "s3://datasets/aapl_q3"}))
    logger.log(mk(EventType.TOOL_RESULT, T2, A2, 1, tool_name="fetch_data",
                  token_count=180, latency_ms=0.6))
    logger.log(mk(EventType.REASONING,   T2, A2, 1, token_count=420, latency_ms=195.0))

    # turn 2 -- loop detected + context truncation
    logger.log(mk(EventType.TURN_START,  T2, A2, 2, metadata={"goal": "run regression"}))
    logger.log(mk(EventType.TOOL_CALL,   T2, A2, 2, tool_name="analyze",
                  latency_ms=520.0, metadata={"algorithm": "linear_regression"}))
    logger.log(mk(EventType.TOOL_RESULT, T2, A2, 2, tool_name="analyze",
                  token_count=350, latency_ms=0.7))
    logger.log(mk(EventType.LOOP_DETECTED, T2, A2, 2,
                  metadata={"repeated_call": "analyze", "count": 3}))
    logger.log(mk(EventType.CONTEXT_TRUNCATED, T2, A2, 2,
                  token_count=8192,
                  metadata={"strategy": "FIFO", "dropped_turns": 2}))

    # turn 3 -- error on first attempt, success on retry
    logger.log(mk(EventType.TURN_START,  T2, A2, 3, metadata={"goal": "write report"}))
    logger.log(mk(EventType.ERROR,       T2, A2, 3,
                  metadata={"exc": "TimeoutError", "tool": "write_report", "retry": True}))
    logger.log(mk(EventType.TOOL_CALL,   T2, A2, 3, tool_name="write_report",
                  latency_ms=780.0, metadata={"attempt": 2}))
    logger.log(mk(EventType.TOOL_RESULT, T2, A2, 3, tool_name="write_report",
                  token_count=280, latency_ms=0.4))
    logger.log(mk(EventType.FINAL_ANSWER, T2, A2, 3,
                  metadata={"answer": "Report written to /output/aapl_q3.pdf"}))
    logger.log(mk(EventType.COST_UPDATE, T2, A2, 3, cost_usd=0.00183, token_count=1230,
                  metadata={"running_total_usd": 0.00183}))

    # ── flush before reading back ─────────────────────────────────────
    logger.flush(timeout=5.0)

    # ── overhead report ───────────────────────────────────────────────
    overhead_ms = logger.get_overhead()
    target_ok = overhead_ms < 1.5
    print(
        f"  overhead : {overhead_ms:.4f} ms/event"
        f"  (target < 1.5ms)  {'OK' if target_ok else 'WARN'}"
    )
    print()

    # ── stats ─────────────────────────────────────────────────────────
    stats = logger.get_stats()
    print("  stats:")
    print(f"    events logged   : {stats['call_count']}")
    print(f"    events written  : {stats['events_written']}")
    print(f"    dropped         : {stats['dropped']}")
    print(f"    sampling_rate   : {stats['sampling_rate']:.0%}")
    print()

    # ── search by event_type ──────────────────────────────────────────
    tool_calls = logger.search(event_type=EventType.TOOL_CALL, hours=1)
    print(f"  search event_type=TOOL_CALL -> {len(tool_calls)} events")
    print(MID)
    for ev in tool_calls:
        print(
            f"    task={ev.task_id:<22}  agent={ev.agent_id}  "
            f"turn={ev.turn_id}  tool={ev.tool_name}  "
            f"latency={ev.latency_ms:.0f}ms"
        )
    print()

    # ── search by task_id ─────────────────────────────────────────────
    task1_events = logger.search(task_id=T1, hours=1)
    print(f"  search task_id={T1!r} -> {len(task1_events)} events")
    print(MID)
    counts: dict[str, int] = {}
    for ev in task1_events:
        counts[ev.event_type] = counts.get(ev.event_type, 0) + 1
    for et, cnt in sorted(counts.items()):
        print(f"    {et:<20}  x{cnt}")
    print()

    # ── sample JSONL record ───────────────────────────────────────────
    if tool_calls:
        print("  sample JSONL record (first TOOL_CALL):")
        print(MID)
        print(json.dumps(_event_to_dict(tool_calls[0]), indent=2))
        print(MID)
        print()

    logger.close()

    # ── final verdict ─────────────────────────────────────────────────
    assert target_ok, f"overhead {overhead_ms:.4f}ms exceeds 1.5ms target"
    print(f"  overhead check: {overhead_ms:.4f}ms < 1.5ms  OK")


if __name__ == "__main__":
    _run_demo()
