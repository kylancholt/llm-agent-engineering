"""
Classifies agent failure modes from JSONL execution trace logs.

Each line in the trace file is a JSON object representing one agent turn:
  type        : "message" | "tool_call" | "tool_result" | "final_answer"
  tool_name   : str   (tool_call / tool_result only)
  tool_args   : dict  (tool_call only)
  tool_schema : dict[str, str]  expected Python type per param (tool_call only)
  tokens      : int   cumulative context-window tokens at this turn
  cost_usd    : float cumulative spend in USD at this turn
"""
from __future__ import annotations

import json
import tempfile
import textwrap
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


CONTEXT_WINDOW_TOKENS: int = 200_000
DEFAULT_BUDGET_USD: float = 10.0

_TYPE_MAP: dict[str, type] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
}


class AgentFailureType(Enum):
    """Taxonomy of observable failure modes in production LLM agents."""

    LOOP_DETECTED = "loop_detected"
    CONTEXT_OVERFLOW = "context_overflow"
    TOOL_HALLUCINATION = "tool_hallucination"
    MEMORY_DRIFT = "memory_drift"
    SILENT_WRONG_ANSWER = "silent_wrong_answer"
    COST_EXPLOSION = "cost_explosion"
    OBSERVABILITY_BLINDNESS = "observability_blindness"


@dataclass
class FailureReport:
    """Structured result of a failure-mode classification.

    Attributes:
        failure_type: The classified AgentFailureType.
        turns_before_detection: 1-based turn index where the failure was caught.
        cost_usd_burned: Cumulative USD spend at detection time.
        budget_remaining_usd: Remaining budget after the burn.
        confidence: Classifier confidence in [0.0, 1.0].
        recommendation: Actionable remediation advice.
    """

    failure_type: AgentFailureType
    turns_before_detection: int
    cost_usd_burned: float
    budget_remaining_usd: float
    confidence: float
    recommendation: str

    def __str__(self) -> str:
        sep = "=" * 64
        wrapped = textwrap.fill(
            self.recommendation, width=60, initial_indent="    ", subsequent_indent="    "
        )
        return (
            f"\n{sep}\n"
            f"  FAILURE REPORT\n"
            f"{sep}\n"
            f"  Type             : {self.failure_type.value.upper()}\n"
            f"  Detected at turn : {self.turns_before_detection}\n"
            f"  Cost burned      : ${self.cost_usd_burned:.4f}\n"
            f"  Budget remaining : ${self.budget_remaining_usd:.4f}\n"
            f"  Confidence       : {self.confidence:.0%}\n"
            f"\n  Recommendation:\n{wrapped}\n"
            f"{sep}\n"
        )


_RECOMMENDATIONS: dict[AgentFailureType, str] = {
    AgentFailureType.LOOP_DETECTED: (
        "Inject a loop-break guard: track (tool_name, frozen_args) in a seen-set "
        "and abort after 3 identical calls. Reflect the constraint in the system prompt."
    ),
    AgentFailureType.CONTEXT_OVERFLOW: (
        "Enable sliding-window context compression or summarise older turns before "
        "they exceed 80% of the context window. Tune chunk granularity accordingly."
    ),
    AgentFailureType.TOOL_HALLUCINATION: (
        "Add a pre-call Pydantic validation layer that coerces or rejects malformed "
        "tool arguments before they reach the tool executor."
    ),
    AgentFailureType.MEMORY_DRIFT: (
        "Ground every long-term memory read against the original task specification "
        "to detect semantic drift before it compounds across turns."
    ),
    AgentFailureType.SILENT_WRONG_ANSWER: (
        "Require at least one verification tool call (e.g. self_check, re_read, "
        "assert_facts) within the last 2 turns before emitting FINAL_ANSWER."
    ),
    AgentFailureType.COST_EXPLOSION: (
        "Enforce a per-turn cost hard-cap and abort with a partial result when "
        "cumulative spend exceeds 90% of the budget before turn 5."
    ),
    AgentFailureType.OBSERVABILITY_BLINDNESS: (
        "Instrument every turn with structured span fields (tokens, cost_usd, latency) "
        "so silent failures become detectable before they reach production."
    ),
}


class FailureModeClassifier:
    """Classifies agent failure modes from JSONL execution trace files.

    Detectors run in priority order:
      LOOP_DETECTED → CONTEXT_OVERFLOW → TOOL_HALLUCINATION →
      COST_EXPLOSION → SILENT_WRONG_ANSWER → OBSERVABILITY_BLINDNESS (fallback)

    Args:
        budget_usd: Total allowed spend for the agent run (default $10).
        context_window: Model context window in tokens (default 200,000).
    """

    def __init__(
        self,
        budget_usd: float = DEFAULT_BUDGET_USD,
        context_window: int = CONTEXT_WINDOW_TOKENS,
    ) -> None:
        self.budget_usd = budget_usd
        self.context_window = context_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, trace_path: str) -> FailureReport:
        """Read a JSONL trace file and return the first failure detected.

        Args:
            trace_path: Path to a JSONL file; each non-empty line is one turn.

        Returns:
            FailureReport for the first matching failure pattern.

        Raises:
            FileNotFoundError: If trace_path does not exist.
            ValueError: If the file contains no parseable turns.
        """
        turns = self._load_trace(trace_path)
        if not turns:
            raise ValueError(f"No turns found in trace: {trace_path}")

        for detector in (
            self._detect_loop,
            self._detect_context_overflow,
            self._detect_tool_hallucination,
            self._detect_cost_explosion,
            self._detect_silent_wrong_answer,
        ):
            result = detector(turns)
            if result is not None:
                return result

        return self._detect_observability_blindness(turns)

    @staticmethod
    def simulate_trace(failure_type: AgentFailureType) -> list[dict[str, Any]]:
        """Generate a minimal synthetic trace that exhibits the given failure.

        Args:
            failure_type: The AgentFailureType scenario to simulate.

        Returns:
            A list of turn dicts ready to be serialised as JSONL.
        """
        if failure_type == AgentFailureType.LOOP_DETECTED:
            _repeated: dict[str, Any] = {
                "type": "tool_call",
                "tool_name": "web_search",
                "tool_args": {"query": "current stock price AAPL"},
                "tool_schema": {"query": "str"},
                "tokens": 4_200,
                "cost_usd": 0.012,
            }
            return [
                {"type": "message", "content": "Find the latest AAPL price.", "tokens": 800, "cost_usd": 0.002},
                {**_repeated, "turn": 2},
                {"type": "tool_result", "tool_name": "web_search", "result": "timeout", "tokens": 4_400, "cost_usd": 0.014},
                {**_repeated, "turn": 4},
                {"type": "tool_result", "tool_name": "web_search", "result": "timeout", "tokens": 4_600, "cost_usd": 0.016},
                {**_repeated, "turn": 6},
                {"type": "tool_result", "tool_name": "web_search", "result": "timeout", "tokens": 4_800, "cost_usd": 0.018},
            ]

        if failure_type == AgentFailureType.TOOL_HALLUCINATION:
            return [
                {"type": "message", "content": "Calculate compound interest for 5 years.", "tokens": 900, "cost_usd": 0.003},
                {
                    "type": "tool_call",
                    "turn": 2,
                    "tool_name": "calculator",
                    # 'years' is declared int in schema but passed as str — hallucination
                    "tool_args": {"principal": 1000, "rate": 0.05, "years": "five"},
                    "tool_schema": {"principal": "int", "rate": "float", "years": "int"},
                    "tokens": 1_800,
                    "cost_usd": 0.008,
                },
            ]

        if failure_type == AgentFailureType.CONTEXT_OVERFLOW:
            return [
                {"type": "message", "content": "Summarise this 200-page document.", "tokens": 1_000, "cost_usd": 0.003},
                {"type": "tool_call", "tool_name": "read_document", "tool_args": {"path": "doc.pdf"},
                 "tool_schema": {"path": "str"}, "tokens": 50_000, "cost_usd": 0.20},
                {"type": "tool_result", "tool_name": "read_document", "result": "...(full text)...",
                 "tokens": 165_000, "cost_usd": 0.70},
            ]

        if failure_type == AgentFailureType.COST_EXPLOSION:
            return [
                {"type": "message", "content": "Run deep research on all competitors.", "tokens": 2_000, "cost_usd": 0.01},
                {"type": "tool_call", "tool_name": "deep_research", "tool_args": {"depth": 10},
                 "tool_schema": {"depth": "int"}, "tokens": 30_000, "cost_usd": 9.50},
            ]

        if failure_type == AgentFailureType.SILENT_WRONG_ANSWER:
            return [
                {"type": "message", "content": "What is 2 + 2?", "tokens": 400, "cost_usd": 0.001},
                {"type": "message", "content": "Let me think about this carefully.", "tokens": 500, "cost_usd": 0.002},
                # No tool_call in the 2 turns preceding this final_answer
                {"type": "final_answer", "content": "The answer is 5.", "tokens": 600, "cost_usd": 0.003},
            ]

        if failure_type == AgentFailureType.MEMORY_DRIFT:
            return [
                {"type": "message", "content": "Task: book a flight to Paris.", "tokens": 500, "cost_usd": 0.002},
                {"type": "tool_call", "tool_name": "memory_read", "tool_args": {"key": "user_prefs"},
                 "tool_schema": {"key": "str"}, "tokens": 1_000, "cost_usd": 0.005},
                {"type": "tool_result", "tool_name": "memory_read", "result": "User prefers London.",
                 "tokens": 1_500, "cost_usd": 0.008},
                {"type": "final_answer", "content": "Booked a flight to London.", "tokens": 1_800, "cost_usd": 0.010},
            ]

        # OBSERVABILITY_BLINDNESS: turns intentionally lack tokens/cost_usd fields
        return [
            {"type": "message", "content": "Do something complex."},
            {"type": "tool_call", "tool_name": "some_tool", "tool_args": {}},
            {"type": "final_answer", "content": "Done."},
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_trace(self, path: str) -> list[dict[str, Any]]:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return [json.loads(ln) for ln in lines if ln.strip()]

    def _cost_at(self, turns: list[dict[str, Any]], idx: int) -> float:
        return float(turns[idx].get("cost_usd", 0.0))

    def _make_report(
        self,
        failure_type: AgentFailureType,
        turn_idx: int,
        turns: list[dict[str, Any]],
        confidence: float,
    ) -> FailureReport:
        cost = self._cost_at(turns, turn_idx)
        return FailureReport(
            failure_type=failure_type,
            turns_before_detection=turn_idx + 1,
            cost_usd_burned=cost,
            budget_remaining_usd=max(0.0, self.budget_usd - cost),
            confidence=confidence,
            recommendation=_RECOMMENDATIONS[failure_type],
        )

    def _detect_loop(self, turns: list[dict[str, Any]]) -> FailureReport | None:
        """Same (tool_name, serialised_args) pair seen 3+ times."""
        counts: Counter[tuple[str, str]] = Counter()
        for i, turn in enumerate(turns):
            if turn.get("type") != "tool_call":
                continue
            key = (
                turn.get("tool_name", ""),
                json.dumps(turn.get("tool_args", {}), sort_keys=True),
            )
            counts[key] += 1
            if counts[key] >= 3:
                return self._make_report(AgentFailureType.LOOP_DETECTED, i, turns, confidence=0.97)
        return None

    def _detect_context_overflow(self, turns: list[dict[str, Any]]) -> FailureReport | None:
        """Token count exceeds 80% of the context window in a single turn."""
        threshold = 0.80 * self.context_window
        for i, turn in enumerate(turns):
            if int(turn.get("tokens", 0)) > threshold:
                return self._make_report(AgentFailureType.CONTEXT_OVERFLOW, i, turns, confidence=0.99)
        return None

    def _detect_tool_hallucination(self, turns: list[dict[str, Any]]) -> FailureReport | None:
        """A tool argument's runtime type does not match its declared schema type."""
        for i, turn in enumerate(turns):
            if turn.get("type") != "tool_call":
                continue
            schema: dict[str, str] = turn.get("tool_schema", {})
            args: dict[str, Any] = turn.get("tool_args", {})
            for param, expected_name in schema.items():
                expected = _TYPE_MAP.get(expected_name)
                if expected is None:
                    continue
                value = args.get(param)
                if value is not None and not isinstance(value, expected):
                    return self._make_report(AgentFailureType.TOOL_HALLUCINATION, i, turns, confidence=0.92)
        return None

    def _detect_cost_explosion(self, turns: list[dict[str, Any]]) -> FailureReport | None:
        """Cumulative cost exceeds 90% of budget before turn 5."""
        threshold = 0.90 * self.budget_usd
        for i, turn in enumerate(turns[:5]):
            if self._cost_at(turns, i) > threshold:
                return self._make_report(AgentFailureType.COST_EXPLOSION, i, turns, confidence=0.95)
        return None

    def _detect_silent_wrong_answer(self, turns: list[dict[str, Any]]) -> FailureReport | None:
        """FINAL_ANSWER emitted without any tool_call in the preceding 2 turns."""
        for i, turn in enumerate(turns):
            if turn.get("type") != "final_answer":
                continue
            window = turns[max(0, i - 2) : i]
            if not any(t.get("type") == "tool_call" for t in window):
                return self._make_report(AgentFailureType.SILENT_WRONG_ANSWER, i, turns, confidence=0.78)
        return None

    def _detect_observability_blindness(self, turns: list[dict[str, Any]]) -> FailureReport:
        """Fallback: no turn carries both 'tokens' and 'cost_usd' span fields."""
        has_spans = any("tokens" in t and "cost_usd" in t for t in turns)
        confidence = 0.30 if has_spans else 0.60
        return self._make_report(
            AgentFailureType.OBSERVABILITY_BLINDNESS, len(turns) - 1, turns, confidence=confidence
        )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    classifier = FailureModeClassifier(budget_usd=10.0)

    for scenario in (AgentFailureType.LOOP_DETECTED, AgentFailureType.TOOL_HALLUCINATION):
        trace = FailureModeClassifier.simulate_trace(scenario)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp:
            for turn in trace:
                tmp.write(json.dumps(turn) + "\n")
            tmp_path = tmp.name

        print(f"\nSimulating scenario : {scenario.value}")
        print(f"Trace turns         : {len(trace)}")
        print(f"Trace file          : {tmp_path}")

        report = classifier.classify(tmp_path)
        print(report)
