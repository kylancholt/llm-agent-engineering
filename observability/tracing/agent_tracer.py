"""
Turn-level agent tracer with an OpenTelemetry-compatible data model.

OpenTelemetry concepts mapped to this module:
  Trace  <->  one agent run (task)            -> TraceFile
  Span   <->  one agent turn                  -> TurnEvent
  Event  <->  tool call or reasoning step

JSONL file format (one JSON object per line):
  Line 1     : {"record_type": "trace_header", ...}
  Lines 2..N : {"record_type": "turn", ...}
  Last line  : {"record_type": "trace_footer", ...}

Trace files are written to:
    {output_dir}/trace_{task_id}_{ts_ms:013d}.jsonl

Stdlib only -- no external dependencies, no LLM calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TraceContext:
    """Returned by start_trace(); holds the active-trace identifiers."""
    trace_id: str         # 32-char hex (128-bit, OTel-compatible)
    task_id: str
    agent_id: str
    start_time_ms: float  # epoch-ms
    task: str


@dataclass
class TurnEvent:
    """One agent turn (OTel span)."""
    turn_id: int
    span_id: str           # 16-char hex (64-bit, OTel-compatible)
    kind: str              # "tool_call" | "reasoning"
    tool_name: Optional[str]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    result_valid: bool
    cost_usd: float
    timestamp_ms: float    # epoch-ms of turn start
    reasoning_tokens: int = 0  # non-zero for kind="reasoning"


@dataclass
class TraceFile:
    """Complete, closed trace with automatic diagnosis."""
    task_id: str
    trace_id: str
    agent_id: str
    task: str
    start_time_ms: float
    end_time_ms: float
    duration_ms: float
    turns: list[TurnEvent]
    total_cost_usd: float
    total_tokens: int
    status: str            # "complete" | "error" | "partial"
    final_answer: str
    diagnosis: str


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def _diagnose(turns: list[TurnEvent]) -> str:
    """Inspect turn metadata for anti-patterns; return a one-line summary."""
    issues: list[str] = []

    # Pattern: reasoning immediately after a successful tool call,
    # followed by another tool call -> reasoning may have ignored the result.
    for i, turn in enumerate(turns):
        if turn.kind != "reasoning":
            continue
        if i == 0:
            continue
        prev = turns[i - 1]
        if prev.kind == "tool_call" and prev.result_valid:
            for later in turns[i + 1:]:
                if later.kind == "tool_call":
                    issues.append(
                        f"turn {turn.turn_id} reasoning may have ignored the "
                        f"{prev.tool_name} result from turn {prev.turn_id} "
                        f"({later.tool_name} called at turn {later.turn_id} "
                        f"without processing prior output)"
                    )
                    break

    # Invalid tool results
    for turn in turns:
        if turn.kind == "tool_call" and not turn.result_valid:
            issues.append(
                f"turn {turn.turn_id} {turn.tool_name} returned an invalid result"
            )

    # Latency outlier among tool calls (>3x average AND >500 ms)
    tool_lat = [t.latency_ms for t in turns if t.kind == "tool_call"]
    if len(tool_lat) >= 2:
        avg_lat = sum(tool_lat) / len(tool_lat)
        for turn in turns:
            if (
                turn.kind == "tool_call"
                and turn.latency_ms > avg_lat * 3
                and turn.latency_ms > 500
            ):
                issues.append(
                    f"turn {turn.turn_id} {turn.tool_name} latency "
                    f"{turn.latency_ms:.0f}ms is "
                    f"{turn.latency_ms / avg_lat:.1f}x the mean "
                    f"({avg_lat:.0f}ms)"
                )

    return "; ".join(issues) if issues else "no anomalies detected"


# ---------------------------------------------------------------------------
# AgentTracer
# ---------------------------------------------------------------------------

class AgentTracer:
    """
    Record and persist a turn-level trace for a single agent run.

    Usage:
        tracer = AgentTracer("my_task", "agent_v1")
        ctx    = tracer.start_trace("summarise this document")
        tracer.record_turn(1, "web_search", 400, 80, 230.0, True, 0.001)
        tracer.record_reasoning(2, 200, 110.0)
        tf = tracer.finish_trace("Final answer here", "complete")
    """

    def __init__(
        self,
        task_id: str,
        agent_id: str,
        output_dir: str = "traces/",
    ) -> None:
        self.task_id = task_id
        self.agent_id = agent_id
        self.output_dir = Path(output_dir)
        self._ctx: Optional[TraceContext] = None
        self._turns: list[TurnEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_trace(self, task: str) -> TraceContext:
        """Begin a new trace; resets any prior state."""
        self._turns = []
        self._ctx = TraceContext(
            trace_id=uuid.uuid4().hex,          # 32-char hex
            task_id=self.task_id,
            agent_id=self.agent_id,
            start_time_ms=time.time() * 1_000,
            task=task,
        )
        return self._ctx

    def record_turn(
        self,
        turn_id: int,
        tool_name: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        result_valid: bool,
        cost_usd: float,
    ) -> None:
        """Record a tool-call turn (span)."""
        self._require_active()
        self._turns.append(TurnEvent(
            turn_id=turn_id,
            span_id=uuid.uuid4().hex[:16],
            kind="tool_call",
            tool_name=tool_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            result_valid=result_valid,
            cost_usd=cost_usd,
            timestamp_ms=time.time() * 1_000,
            reasoning_tokens=0,
        ))

    def record_reasoning(
        self,
        turn_id: int,
        tokens: int,
        latency_ms: float,
    ) -> None:
        """Record a reasoning turn (no tool call)."""
        self._require_active()
        self._turns.append(TurnEvent(
            turn_id=turn_id,
            span_id=uuid.uuid4().hex[:16],
            kind="reasoning",
            tool_name=None,
            input_tokens=tokens,
            output_tokens=0,
            latency_ms=latency_ms,
            result_valid=True,
            cost_usd=0.0,
            timestamp_ms=time.time() * 1_000,
            reasoning_tokens=tokens,
        ))

    def finish_trace(self, final_answer: str, status: str) -> TraceFile:
        """
        Close the trace, run diagnosis, write JSONL, and return TraceFile.
        """
        self._require_active()
        ctx = self._ctx  # type: ignore[assignment]

        end_ms = time.time() * 1_000
        duration_ms = end_ms - ctx.start_time_ms
        total_cost = sum(t.cost_usd for t in self._turns)
        total_tokens = sum(
            (t.input_tokens + t.output_tokens) for t in self._turns
        )
        diag = _diagnose(self._turns)

        tf = TraceFile(
            task_id=ctx.task_id,
            trace_id=ctx.trace_id,
            agent_id=ctx.agent_id,
            task=ctx.task,
            start_time_ms=ctx.start_time_ms,
            end_time_ms=end_ms,
            duration_ms=duration_ms,
            turns=list(self._turns),
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            status=status,
            final_answer=final_answer,
            diagnosis=diag,
        )
        self._write_jsonl(tf)
        # reset for next trace
        self._ctx = None
        return tf

    # ------------------------------------------------------------------
    # Replay (static method -- reads any saved trace)
    # ------------------------------------------------------------------

    @staticmethod
    def replay(trace_path: str) -> None:
        """
        Read a JSONL trace file and print a human-readable reconstruction.
        """
        path = Path(trace_path)
        if not path.exists():
            print(f"[replay] file not found: {trace_path}")
            return

        records = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            print("[replay] empty trace file")
            return

        header = next((r for r in records if r.get("record_type") == "trace_header"), {})
        footer = next((r for r in records if r.get("record_type") == "trace_footer"), {})
        turn_records = [r for r in records if r.get("record_type") == "turn"]

        SEP = "=" * 68
        MID = "-" * 68

        print(SEP)
        print(f"  TRACE REPLAY")
        print(f"  task_id  : {header.get('task_id', 'n/a')}")
        print(f"  agent_id : {header.get('agent_id', 'n/a')}")
        print(f"  trace_id : {header.get('trace_id', 'n/a')[:16]}...")
        print(f"  task     : {header.get('task', 'n/a')}")
        print(SEP)
        print()

        for rec in turn_records:
            tid = rec.get("turn_id", "?")
            kind = rec.get("kind", "?")
            tool = rec.get("tool_name") or ""
            lat = rec.get("latency_ms", 0.0)
            inp = rec.get("input_tokens", 0)
            out = rec.get("output_tokens", 0)
            cost = rec.get("cost_usd", 0.0)
            valid = rec.get("result_valid", True)

            if kind == "tool_call":
                valid_str = "yes" if valid else "NO"
                print(
                    f"  turn {tid:>2}  [tool_call]  {tool}"
                )
                print(
                    f"          latency={lat:.0f}ms"
                    f"   in={inp} out={out} tok"
                    f"   cost=${cost:.5f}"
                    f"   valid={valid_str}"
                )
            else:
                rtok = rec.get("reasoning_tokens", inp)
                print(f"  turn {tid:>2}  [reasoning]")
                print(
                    f"          latency={lat:.0f}ms"
                    f"   tokens={rtok}"
                )
            print()

        print(MID)
        print("  DIAGNOSIS")
        print(MID)
        diag = footer.get("diagnosis", "n/a")
        # wrap long diagnosis lines
        for part in diag.split("; "):
            print(f"  {part}")
        print(MID)

        dur = footer.get("duration_ms", 0.0)
        cost_total = footer.get("total_cost_usd", 0.0)
        tok_total = footer.get("total_tokens", 0)
        status = footer.get("status", "n/a")
        answer = footer.get("final_answer", "")
        print(f"  task          : {header.get('task', 'n/a')}")
        print(f"  final_answer  : {answer}")
        print(f"  total_tokens  : {tok_total}")
        print(f"  total_cost    : ${cost_total:.5f}")
        print(f"  duration      : {dur:.0f}ms")
        print(f"  status        : {status}")
        print(SEP)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_active(self) -> None:
        if self._ctx is None:
            raise RuntimeError(
                "No active trace -- call start_trace() first"
            )

    def _write_jsonl(self, tf: TraceFile) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts_ms = int(tf.start_time_ms)
        fname = f"trace_{tf.task_id}_{ts_ms:013d}.jsonl"
        fpath = self.output_dir / fname

        header = {
            "record_type": "trace_header",
            "trace_id": tf.trace_id,
            "task_id": tf.task_id,
            "agent_id": tf.agent_id,
            "task": tf.task,
            "start_time_ms": tf.start_time_ms,
        }
        footer = {
            "record_type": "trace_footer",
            "status": tf.status,
            "final_answer": tf.final_answer,
            "total_cost_usd": tf.total_cost_usd,
            "total_tokens": tf.total_tokens,
            "duration_ms": tf.duration_ms,
            "diagnosis": tf.diagnosis,
        }

        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(header) + "\n")
            for turn in tf.turns:
                rec = {
                    "record_type": "turn",
                    "turn_id": turn.turn_id,
                    "span_id": turn.span_id,
                    "kind": turn.kind,
                    "tool_name": turn.tool_name,
                    "input_tokens": turn.input_tokens,
                    "output_tokens": turn.output_tokens,
                    "latency_ms": turn.latency_ms,
                    "result_valid": turn.result_valid,
                    "cost_usd": turn.cost_usd,
                    "timestamp_ms": turn.timestamp_ms,
                    "reasoning_tokens": turn.reasoning_tokens,
                }
                fh.write(json.dumps(rec) + "\n")
            fh.write(json.dumps(footer) + "\n")

        print(f"[tracer] trace saved -> {fpath}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _run_demo() -> None:
    """
    Simulate a 6-turn agent trace for an AAPL stock analysis task.

    Turn breakdown:
      1  tool_call   web_search  (AAPL price lookup)
      2  tool_call   web_search  (market cap data) <- result ignored by turn 3
      3  reasoning               <- problematic: ignores turn 2 result
      4  tool_call   web_search  (repeated search, confirms turn 3 ignored data)
      5  tool_call   summarize   (synthesis)
      6  reasoning               (final answer preparation)

    The diagnosis should flag: turn 3 reasoning ignored web_search at turn 2.
    """
    print("=" * 68)
    print("  AgentTracer -- Chapter 10 Demo")
    print("  Task: Analyze AAPL stock for Q3 and compute 42-share value")
    print("=" * 68)
    print()

    tracer = AgentTracer(
        task_id="aapl_analysis_q3",
        agent_id="research_agent_v1",
        output_dir=str(_ROOT / "traces"),
    )

    ctx = tracer.start_trace(
        "Analyze AAPL stock for Q3 and compute 42-share value"
    )
    print(f"[tracer] trace started  trace_id={ctx.trace_id[:16]}...")
    print()

    # turn 1 -- first web_search: AAPL current price
    tracer.record_turn(1, "web_search", 520, 110, 231.0, True, 0.00126)
    print("[turn 1] tool_call  web_search  (AAPL price)  231ms  ok")

    # turn 2 -- second web_search: market data (its result will be "ignored")
    tracer.record_turn(2, "web_search", 480, 95, 215.0, True, 0.00115)
    print("[turn 2] tool_call  web_search  (market cap)  215ms  ok")

    # turn 3 -- reasoning that does NOT act on turn 2's result
    tracer.record_reasoning(3, 380, 178.0)
    print("[turn 3] reasoning  (no tool)  178ms")

    # turn 4 -- another web_search: repeats the query from turn 2
    tracer.record_turn(4, "web_search", 510, 108, 220.0, True, 0.00123)
    print("[turn 4] tool_call  web_search  (repeated)  220ms  ok")

    # turn 5 -- summarize: synthesis step
    tracer.record_turn(5, "summarize", 640, 210, 254.0, True, 0.00161)
    print("[turn 5] tool_call  summarize  254ms  ok")

    # turn 6 -- final reasoning: prepare the answer
    tracer.record_reasoning(6, 290, 135.0)
    print("[turn 6] reasoning  (final answer prep)  135ms")
    print()

    tf = tracer.finish_trace(
        final_answer="AAPL: $189.30, +1.2%. 42-share portfolio value: $7,950.60.",
        status="complete",
    )

    print()
    print(f"[tracer] diagnosis  : {tf.diagnosis}")
    print(f"[tracer] total_cost : ${tf.total_cost_usd:.5f}")
    print(f"[tracer] total_tok  : {tf.total_tokens}")
    print(f"[tracer] duration   : {tf.duration_ms:.0f}ms")
    print()

    # --- replay ----------------------------------------------------------
    trace_files = sorted(
        (Path(_ROOT / "traces")).glob(f"trace_{tracer.task_id}_*.jsonl")
    )
    if trace_files:
        latest = str(trace_files[-1])
        print("=" * 68)
        print("  REPLAY")
        print("=" * 68)
        print()
        AgentTracer.replay(latest)


if __name__ == "__main__":
    _run_demo()
