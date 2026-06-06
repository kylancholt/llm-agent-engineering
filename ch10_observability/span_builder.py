"""
Span hierarchy builder for multi-agent pipeline tracing.

Builds a parent/child span graph that is compatible with the OpenTelemetry
data model:

  Trace  --  one pipeline run identified by a shared trace_id
  Span   --  one unit of work (tool call, reasoning step, agent root)
             with a unique span_id and an optional parent_span_id

Multiple SpanBuilder instances that share the same trace_id contribute
to the same logical trace.  Calling correlate_agents() records which
agent IDs participate, making the shared correlation ID explicit.

OTLP-JSON export produces one dict per span matching the OpenTelemetry
proto Span message fields (traceId, spanId, parentSpanId, name, kind,
startTimeUnixNano, endTimeUnixNano, attributes, status).

Stdlib only -- no external dependencies.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------

@dataclass
class Span:
    """
    One unit of traced work, compatible with the OTel Span proto.

    Attributes
    ----------
    span_id:
        16-char hex (64-bit), unique within a trace.
    parent_span_id:
        span_id of the enclosing span; ``None`` for root spans.
    trace_id:
        32-char hex (128-bit), shared across every span in the trace.
    name:
        Human-readable operation name (e.g. ``"web_search"``).
    start_time:
        Epoch seconds (float) when the span was opened.
    end_time:
        Epoch seconds when the span was closed; ``None`` while open.
    duration_ms:
        Wall-clock milliseconds; computed by ``end_span()``.
    attributes:
        Arbitrary string key/value metadata.
    status:
        ``"OK"`` or ``"ERROR"``.
    error:
        Error message when status is ``"ERROR"``; otherwise ``None``.
    agent_id:
        Identifier of the agent that created this span.
    """

    span_id: str
    parent_span_id: Optional[str]
    trace_id: str
    name: str
    start_time: float
    end_time: Optional[float]
    duration_ms: Optional[float]
    attributes: dict[str, str]
    status: str
    error: Optional[str]
    agent_id: str


# ---------------------------------------------------------------------------
# SpanTree
# ---------------------------------------------------------------------------

class SpanTree:
    """
    Hierarchical view of all spans in a single trace.

    Instances are produced by :meth:`SpanBuilder.get_trace_tree`.
    """

    def __init__(self, trace_id: str, spans: list[Span]) -> None:
        self.trace_id = trace_id
        self.spans = spans
        # index spans by their parent; root spans key on None
        self._children: dict[Optional[str], list[Span]] = {}
        self._by_id: dict[str, Span] = {}
        for span in spans:
            self._children.setdefault(span.parent_span_id, []).append(span)
            self._by_id[span.span_id] = span

    @property
    def root_spans(self) -> list[Span]:
        """Spans with no parent within this trace."""
        return self._children.get(None, [])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str:
        """
        Return an ASCII tree representation of the span hierarchy.

        Single-root trace::

            Task "Analyze report" [8.4s] OK
              +-- turn_1: web_search [340ms] OK
              +-- turn_2: web_search [290ms] OK
              +-- turn_3: reasoning [1240ms] OK

        Multi-root trace (multiple root spans under the same trace_id)::

            Trace abc12345...
              +-- span_A [1.5s] OK
              +-- span_B [1.6s] OK
        """
        roots = self.root_spans
        if not roots:
            return f"(empty trace {self.trace_id[:8]}...)"

        lines: list[str] = []
        if len(roots) == 1:
            root = roots[0]
            lines.append(
                f'Task "{root.name}" [{self._fmt_dur(root.duration_ms)}] {root.status}'
            )
            for child in self._children.get(root.span_id, []):
                self._render_node(child, indent=1, lines=lines)
        else:
            lines.append(f"Trace {self.trace_id[:8]}...")
            for root in roots:
                self._render_node(root, indent=1, lines=lines)

        return "\n".join(lines)

    def _render_node(self, span: Span, indent: int, lines: list[str]) -> None:
        """Recursively append *span* and its descendants to *lines*."""
        prefix = "  " * indent + "+-- "
        lines.append(
            f"{prefix}{span.name} [{self._fmt_dur(span.duration_ms)}] {span.status}"
        )
        for child in self._children.get(span.span_id, []):
            self._render_node(child, indent + 1, lines=lines)

    @staticmethod
    def _fmt_dur(duration_ms: Optional[float]) -> str:
        """Format a duration for display: seconds if >= 1 s, otherwise ms."""
        if duration_ms is None:
            return "open"
        if duration_ms >= 1_000:
            return f"{duration_ms / 1_000:.1f}s"
        return f"{duration_ms:.0f}ms"

    # ------------------------------------------------------------------
    # OTLP export
    # ------------------------------------------------------------------

    def to_otel_format(self) -> list[dict]:
        """
        Export all spans as OTLP JSON span objects.

        Each dict matches the OpenTelemetry proto ``Span`` message.
        Returns one object per span; callers may wrap the list in a
        ``resourceSpans`` envelope for a full OTLP export request.

        Status codes follow the OTLP spec:
            ``STATUS_CODE_OK``    -- span finished without error
            ``STATUS_CODE_ERROR`` -- span finished with an error
        """
        result: list[dict] = []
        for span in self.spans:
            end_ns = int(
                (span.end_time if span.end_time is not None else span.start_time)
                * 1_000_000_000
            )
            result.append(
                {
                    "traceId": span.trace_id,
                    "spanId": span.span_id,
                    "parentSpanId": span.parent_span_id or "",
                    "name": span.name,
                    "kind": 1,  # SPAN_KIND_INTERNAL
                    "startTimeUnixNano": int(span.start_time * 1_000_000_000),
                    "endTimeUnixNano": end_ns,
                    "attributes": [
                        {"key": k, "value": {"stringValue": str(v)}}
                        for k, v in span.attributes.items()
                    ],
                    "status": {
                        "code": (
                            "STATUS_CODE_OK"
                            if span.status == "OK"
                            else "STATUS_CODE_ERROR"
                        ),
                        "message": span.error or "",
                    },
                    "resource": {
                        "attributes": [
                            {
                                "key": "agent.id",
                                "value": {"stringValue": span.agent_id},
                            }
                        ]
                    },
                }
            )
        return result


# ---------------------------------------------------------------------------
# SpanBuilder
# ---------------------------------------------------------------------------

class SpanBuilder:
    """
    Creates and tracks spans for one agent within a distributed trace.

    Multiple ``SpanBuilder`` instances that share the same *trace_id*
    contribute spans to the same logical trace.  A class-level registry
    makes spans visible across builder instances without explicit
    state passing, enabling cross-agent :meth:`get_trace_tree` calls.

    Parameters
    ----------
    agent_id:
        Identifier for the agent using this builder (e.g. ``"agent_A"``).
    trace_id:
        Shared trace correlation ID.  Generated as a UUID hex string
        if omitted.  Pass the same value to all builders in a pipeline
        to place their spans under one logical trace.

    Notes
    -----
    Call :meth:`reset` at the start of each test or demo run to clear
    accumulated class-level state from prior invocations.
    """

    # Class-level shared storage (visible across all instances)
    _all_spans: dict[str, Span] = {}            # span_id -> Span
    _trace_index: dict[str, list[str]] = {}     # trace_id -> [span_id] (insertion order)
    _correlations: dict[str, list[str]] = {}    # trace_id -> [agent_id]

    def __init__(
        self,
        agent_id: str,
        trace_id: Optional[str] = None,
    ) -> None:
        self.agent_id = agent_id
        self.trace_id = trace_id or uuid.uuid4().hex

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def start_span(
        self,
        name: str,
        parent_span_id: Optional[str] = None,
        _start_time: Optional[float] = None,
        **attributes: str,
    ) -> Span:
        """
        Open a new span and register it in the class-level trace registry.

        Parameters
        ----------
        name:
            Human-readable operation name (e.g. ``"web_search"``).
        parent_span_id:
            ``span_id`` of the enclosing span; ``None`` for root spans.
        _start_time:
            Override the start timestamp (epoch seconds).  Used by tests
            and demos to inject deterministic timing without sleeping.
        **attributes:
            Arbitrary string key/value metadata attached to the span.

        Returns
        -------
        Span
            The newly created, open span (``end_time`` and
            ``duration_ms`` are ``None`` until :meth:`end_span` is called).
        """
        span = Span(
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent_span_id,
            trace_id=self.trace_id,
            name=name,
            start_time=(
                _start_time if _start_time is not None else time.time()
            ),
            end_time=None,
            duration_ms=None,
            attributes=dict(attributes),
            status="OK",
            error=None,
            agent_id=self.agent_id,
        )
        SpanBuilder._all_spans[span.span_id] = span
        SpanBuilder._trace_index.setdefault(self.trace_id, []).append(
            span.span_id
        )
        return span

    def end_span(
        self,
        span_id: str,
        status: str = "OK",
        error: Optional[str] = None,
        _end_time: Optional[float] = None,
    ) -> Span:
        """
        Close a span, compute its duration, and set its final status.

        Parameters
        ----------
        span_id:
            Identifier of the span to close.
        status:
            ``"OK"`` or ``"ERROR"``.  Automatically promoted to
            ``"ERROR"`` when *error* is provided and *status* is ``"OK"``.
        error:
            Optional error message attached to the span.
        _end_time:
            Override the end timestamp (epoch seconds).

        Returns
        -------
        Span
            The closed span, mutated in place.

        Raises
        ------
        KeyError
            If *span_id* is not found in the registry.
        """
        span = SpanBuilder._all_spans[span_id]
        end = _end_time if _end_time is not None else time.time()
        span.end_time = end
        span.duration_ms = (end - span.start_time) * 1_000
        # promote status to ERROR if an error message was supplied
        span.status = (
            "ERROR" if (error is not None and status == "OK") else status
        )
        span.error = error
        return span

    # ------------------------------------------------------------------
    # Trace tree
    # ------------------------------------------------------------------

    def get_trace_tree(self, trace_id: Optional[str] = None) -> SpanTree:
        """
        Build a :class:`SpanTree` for all spans registered under *trace_id*.

        Parameters
        ----------
        trace_id:
            Trace to retrieve; defaults to this builder's ``trace_id``.

        Returns
        -------
        SpanTree
        """
        tid = trace_id if trace_id is not None else self.trace_id
        span_ids = SpanBuilder._trace_index.get(tid, [])
        spans = [
            SpanBuilder._all_spans[sid]
            for sid in span_ids
            if sid in SpanBuilder._all_spans
        ]
        return SpanTree(trace_id=tid, spans=spans)

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    def correlate_agents(
        self, agent_ids: list[str], trace_id: str
    ) -> None:
        """
        Register *agent_ids* as contributors to *trace_id*.

        This is a metadata operation: it does **not** re-tag existing
        spans.  Use it to make the agent membership queryable and to
        document which agents ran in parallel under a shared trace.

        Parameters
        ----------
        agent_ids:
            Identifiers of all agents whose spans appear in this trace.
        trace_id:
            The shared trace correlation ID.
        """
        existing = SpanBuilder._correlations.get(trace_id, [])
        # dedup while preserving insertion order
        seen: set[str] = set(existing)
        merged = list(existing)
        for aid in agent_ids:
            if aid not in seen:
                merged.append(aid)
                seen.add(aid)
        SpanBuilder._correlations[trace_id] = merged

    def get_correlated_agents(
        self, trace_id: Optional[str] = None
    ) -> list[str]:
        """
        Return the agent IDs registered under *trace_id*.

        Parameters
        ----------
        trace_id:
            Defaults to this builder's ``trace_id``.

        Returns
        -------
        list[str]
            Agent IDs in registration order; empty if none registered.
        """
        tid = trace_id if trace_id is not None else self.trace_id
        return list(SpanBuilder._correlations.get(tid, []))

    # ------------------------------------------------------------------
    # OTLP export (delegate to SpanTree)
    # ------------------------------------------------------------------

    def to_otel_format(self, trace_id: Optional[str] = None) -> list[dict]:
        """
        Export all spans for *trace_id* in OTLP JSON span format.

        Delegates to :meth:`SpanTree.to_otel_format`.

        Parameters
        ----------
        trace_id:
            Defaults to this builder's ``trace_id``.

        Returns
        -------
        list[dict]
            One OTLP span object per span in the trace.
        """
        return self.get_trace_tree(trace_id).to_otel_format()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Clear all class-level state (useful for test isolation)."""
        cls._all_spans.clear()
        cls._trace_index.clear()
        cls._correlations.clear()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _run_demo() -> None:
    """
    Simulate two parallel agents (3 spans each) under a shared trace_id.

    Pipeline layout:
        AAPL multi-agent analysis          <- root span (pipeline)
          +-- agent_A: research            <- agent span
              +-- web_search [340ms]       <- AAPL stock price
              +-- web_search [290ms]       <- AAPL market cap
              +-- summarize [890ms]
          +-- agent_B: validation          <- agent span (parallel)
              +-- web_search [310ms]       <- AAPL Q3 earnings
              +-- reasoning [520ms]
              +-- write_report [780ms]

    Demonstrates:
        - shared trace_id as the cross-agent correlation ID
        - ASCII tree rendering via SpanTree.render()
        - OTLP JSON export via SpanTree.to_otel_format()
        - agent membership via SpanBuilder.correlate_agents()
    """
    SpanBuilder.reset()  # clean slate for this run

    SEP = "=" * 68
    MID = "-" * 68

    print(SEP)
    print("  SpanBuilder -- Multi-Agent Pipeline Trace Demo")
    print("  2 parallel agents, 3 spans each, shared trace_id")
    print(SEP)
    print()

    # Shared trace_id is the correlation ID that ties both agents together.
    shared_trace_id = uuid.uuid4().hex
    print(f"  correlation ID (trace_id) : {shared_trace_id}")
    print()

    # Fixed base epoch for deterministic demo timings (no real sleeping).
    T0 = 1_748_816_000.0  # arbitrary but reproducible

    # One builder per logical role; all share the same trace_id.
    b_pipeline = SpanBuilder("pipeline", trace_id=shared_trace_id)
    b_a        = SpanBuilder("agent_A",  trace_id=shared_trace_id)
    b_b        = SpanBuilder("agent_B",  trace_id=shared_trace_id)

    # ── Root pipeline span ────────────────────────────────────────────
    s_root = b_pipeline.start_span(
        "AAPL multi-agent analysis",
        _start_time=T0,
        pipeline_version="1.0",
    )

    # ── Agent A: research (3 spans, starts at +10ms) ──────────────────
    s_a = b_a.start_span(
        "agent_A: research",
        parent_span_id=s_root.span_id,
        _start_time=T0 + 0.010,
        role="researcher",
    )
    s_a1 = b_a.start_span(
        "web_search", parent_span_id=s_a.span_id,
        _start_time=T0 + 0.015, query="AAPL stock price",
    )
    b_a.end_span(s_a1.span_id, _end_time=T0 + 0.355)   # 340 ms

    s_a2 = b_a.start_span(
        "web_search", parent_span_id=s_a.span_id,
        _start_time=T0 + 0.360, query="AAPL market cap",
    )
    b_a.end_span(s_a2.span_id, _end_time=T0 + 0.650)   # 290 ms

    s_a3 = b_a.start_span(
        "summarize", parent_span_id=s_a.span_id,
        _start_time=T0 + 0.655, model="claude-sonnet-4-6",
    )
    b_a.end_span(s_a3.span_id, _end_time=T0 + 1.545)   # 890 ms

    b_a.end_span(s_a.span_id, _end_time=T0 + 1.550)    # agent_A total: 1540 ms

    # ── Agent B: validation (3 spans, starts at +20ms in parallel) ────
    s_b = b_b.start_span(
        "agent_B: validation",
        parent_span_id=s_root.span_id,
        _start_time=T0 + 0.020,
        role="validator",
    )
    s_b1 = b_b.start_span(
        "web_search", parent_span_id=s_b.span_id,
        _start_time=T0 + 0.025, query="AAPL Q3 earnings",
    )
    b_b.end_span(s_b1.span_id, _end_time=T0 + 0.335)   # 310 ms

    s_b2 = b_b.start_span(
        "reasoning", parent_span_id=s_b.span_id,
        _start_time=T0 + 0.340,
    )
    b_b.end_span(s_b2.span_id, _end_time=T0 + 0.860)   # 520 ms

    s_b3 = b_b.start_span(
        "write_report", parent_span_id=s_b.span_id,
        _start_time=T0 + 0.865,
    )
    b_b.end_span(s_b3.span_id, _end_time=T0 + 1.645)   # 780 ms

    b_b.end_span(s_b.span_id, _end_time=T0 + 1.650)    # agent_B total: 1630 ms

    # ── Close root ────────────────────────────────────────────────────
    b_pipeline.end_span(s_root.span_id, _end_time=T0 + 1.660)

    # ── Register correlation metadata ─────────────────────────────────
    b_pipeline.correlate_agents(["agent_A", "agent_B"], shared_trace_id)

    # ── ASCII tree ────────────────────────────────────────────────────
    tree = b_pipeline.get_trace_tree(shared_trace_id)
    print(tree.render())
    print()

    # ── Correlation summary ───────────────────────────────────────────
    agents = b_pipeline.get_correlated_agents(shared_trace_id)
    print(f"  correlated agents : {agents}")
    print(f"  total spans       : {len(tree.spans)}")
    print()

    # ── OTLP export overview ──────────────────────────────────────────
    otel_spans = tree.to_otel_format()
    preview_n = min(4, len(otel_spans))
    print(f"  OTLP export ({len(otel_spans)} spans, showing first {preview_n}):")
    print(MID)
    for s in otel_spans[:preview_n]:
        sid = s["spanId"]
        code = s["status"]["code"]
        name = s["name"]
        agent = next(
            (a["value"]["stringValue"]
             for a in s["resource"]["attributes"]
             if a["key"] == "agent.id"),
            "?",
        )
        print(f"  spanId={sid}  agent={agent:<12}  {name!r:<34} {code}")
    if len(otel_spans) > preview_n:
        print(f"  ... and {len(otel_spans) - preview_n} more spans")
    print(MID)
    print()

    # ── Full JSON for the root span ───────────────────────────────────
    root_otel = otel_spans[0]
    print("  Full OTLP JSON -- root span:")
    print(MID)
    print(json.dumps(root_otel, indent=4))
    print(MID)


if __name__ == "__main__":
    _run_demo()
