"""
Reproduces all 7 AgentFailureType patterns under controlled conditions.

Each test builds a minimal synthetic JSONL trace, runs FailureModeClassifier,
and asserts that the detected type matches the expected one.

Detector priority in FailureModeClassifier (relevant for trace design):
  LOOP_DETECTED > CONTEXT_OVERFLOW > TOOL_HALLUCINATION >
  COST_EXPLOSION > SILENT_WRONG_ANSWER > OBSERVABILITY_BLINDNESS
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Allow direct execution from any working directory: python ch01_why_agents_break/test_failure_modes.py
sys.path.insert(0, str(Path(__file__).parent))

from failure_modes import AgentFailureType, FailureModeClassifier


# ------------------------------------------------------------------
# Result types
# ------------------------------------------------------------------

@dataclass
class TestResult:
    """Outcome of a single failure-mode test case.

    Attributes:
        name: Descriptive test identifier.
        passed: True when detected_type == expected_type with no exception.
        expected_type: AgentFailureType the classifier should return.
        detected_type: AgentFailureType actually returned; None on exception.
        turns_in_trace: Number of turns in the synthetic trace.
        cost_simulated_usd: cost_usd_burned reported by FailureModeClassifier.
        trace: The turn list used for this test (printed on pass).
        error_message: Exception text on failure; None on success.
    """

    name: str
    passed: bool
    expected_type: AgentFailureType
    detected_type: AgentFailureType | None
    turns_in_trace: int
    cost_simulated_usd: float
    trace: list[dict[str, Any]] = field(default_factory=list)
    error_message: str | None = None


@dataclass
class TestReport:
    """Aggregate result of the full test suite.

    Attributes:
        tests_run: Total tests executed.
        tests_passed: Tests that passed.
        tests_failed: Tests that failed or raised.
        failure_details: Full TestResult list (all tests, not only failures).
    """

    tests_run: int
    tests_passed: int
    tests_failed: int
    failure_details: list[TestResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Percentage of tests that passed."""
        return (self.tests_passed / self.tests_run * 100.0) if self.tests_run else 0.0

    def __str__(self) -> str:
        sep = "=" * 66
        lines = [
            f"\n{sep}",
            "  FINAL TEST REPORT  --  FailureModeTestSuite",
            sep,
            f"  Tests run    : {self.tests_run}",
            f"  Passed       : {self.tests_passed}",
            f"  Failed       : {self.tests_failed}",
            f"  Pass rate    : {self.pass_rate:.1f}%",
            "",
            f"  {'Test':<40} {'Expected':<26} Result",
            f"  {'-'*62}",
        ]
        for r in self.failure_details:
            status = "PASS" if r.passed else "FAIL"
            detected = r.detected_type.value if r.detected_type else "EXCEPTION"
            lines.append(
                f"  {r.name:<40} {r.expected_type.value:<26} [{status}]"
            )
            if not r.passed:
                detail = r.error_message or f"got {detected}"
                lines.append(f"  {'':40} {detail}")
        lines.append(sep + "\n")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Test suite
# ------------------------------------------------------------------

class FailureModeTestSuite:
    """Reproduces each of the 7 AgentFailureType patterns under controlled conditions.

    Call run_all() to execute the full suite. Each passing test prints
    its synthetic trace, the detected type, and the simulated cost.
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_all(self) -> TestReport:
        """Execute all 7 tests and return an aggregate TestReport.

        For passing tests: prints the trace turns, detected type, and cost.
        For failing tests: prints the mismatch or exception detail.

        Returns:
            TestReport with per-test breakdown and overall pass rate.
        """
        tests: list[Callable[[], TestResult]] = [
            self._test_loop_detection,
            self._test_context_overflow,
            self._test_tool_hallucination,
            self._test_cost_explosion,
            self._test_silent_wrong_answer,
            self._test_memory_drift,
            self._test_observability_blindness,
        ]

        results: list[TestResult] = []
        for fn in tests:
            result = fn()
            results.append(result)
            _print_result(result)

        passed = sum(1 for r in results if r.passed)
        return TestReport(
            tests_run=len(results),
            tests_passed=passed,
            tests_failed=len(results) - passed,
            failure_details=results,
        )

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    def _test_loop_detection(self) -> TestResult:
        """12 turns; the same web_search call repeats from turn 8 onward.

        Turns 1-7: varied tool calls.
        Turns 8-12: identical (web_search, {"query": "NVDA real-time price"}).

        The counter-based detector reaches 3 on turn 10 → LOOP_DETECTED.
        Turns 11-12 are unreachable in production once the classifier halts.
        """
        repeated: dict[str, Any] = {
            "type": "tool_call",
            "tool_name": "web_search",
            "tool_args": {"query": "NVDA real-time price"},
            "tool_schema": {"query": "str"},
            "tokens": 1_400,
            "cost_usd": 0.012,
        }
        turns: list[dict[str, Any]] = [
            # Turns 1-7: normal varied activity
            {"type": "message",     "content": "Research task started.",             "tokens":   300, "cost_usd": 0.001},
            {"type": "tool_call",   "tool_name": "web_search",    "tool_args": {"query": "global GDP 2024"},    "tool_schema": {"query": "str"}, "tokens":   600, "cost_usd": 0.003},
            {"type": "tool_result", "tool_name": "web_search",    "result": "GDP up 2.1%",                     "tokens":   800, "cost_usd": 0.005},
            {"type": "tool_call",   "tool_name": "compute",       "tool_args": {"expr": "100 * 1.021"},        "tool_schema": {"expr":  "str"}, "tokens":   900, "cost_usd": 0.006},
            {"type": "tool_result", "tool_name": "compute",       "result": "102.1",                           "tokens":   950, "cost_usd": 0.007},
            {"type": "tool_call",   "tool_name": "web_search",    "tool_args": {"query": "inflation rate 2024"}, "tool_schema": {"query": "str"}, "tokens": 1_100, "cost_usd": 0.009},
            {"type": "tool_result", "tool_name": "web_search",    "result": "3.2%",                            "tokens": 1_200, "cost_usd": 0.010},
            # Turns 8-12: loop — same call, agent retrying a timed-out price feed
            {**repeated, "cost_usd": 0.012},   # count = 1
            {**repeated, "cost_usd": 0.014},   # count = 2
            {**repeated, "cost_usd": 0.016},   # count = 3  ← LOOP_DETECTED here
            {**repeated, "cost_usd": 0.018},   # unreachable in production
            {**repeated, "cost_usd": 0.020},   # unreachable in production
        ]
        return self._run(
            name="test_loop_detection",
            turns=turns,
            expected=AgentFailureType.LOOP_DETECTED,
        )

    def _test_context_overflow(self) -> TestResult:
        """3 turns on a 100 k-token window; turn 3 loads 85 k tokens.

        Threshold = 80% of 100 000 = 80 000 tokens.
        85 000 > 80 000 → CONTEXT_OVERFLOW.

        Classifier initialised with context_window=100_000 (not the default 200 k).
        The tool_result at index 2 carries the large token count.
        """
        turns: list[dict[str, Any]] = [
            {"type": "message",     "content": "Summarise the full document corpus.",  "tokens":  1_000, "cost_usd": 0.003},
            {"type": "tool_call",   "tool_name": "read_corpus", "tool_args": {"path": "corpus.txt"}, "tool_schema": {"path": "str"}, "tokens": 10_000, "cost_usd": 0.030},
            {"type": "tool_result", "tool_name": "read_corpus", "result": "...(full corpus)...",      "tokens": 85_000, "cost_usd": 0.255},
        ]
        return self._run(
            name="test_context_overflow",
            turns=turns,
            expected=AgentFailureType.CONTEXT_OVERFLOW,
            context_window=100_000,
        )

    def _test_tool_hallucination(self) -> TestResult:
        """2 turns; 'error_code' is declared str in schema but the agent passes int 404.

        Schema: {"error_code": "str", "message": "str"}
        Args:   {"error_code": 404,   "message": "Not Found"}

        isinstance(404, str) is False → TOOL_HALLUCINATION.
        """
        turns: list[dict[str, Any]] = [
            {"type": "message", "content": "Format the HTTP error response.", "tokens": 400, "cost_usd": 0.001},
            {
                "type":        "tool_call",
                "tool_name":   "format_error",
                "tool_args":   {"error_code": 404, "message": "Not Found"},
                "tool_schema": {"error_code": "str", "message": "str"},
                "tokens": 800, "cost_usd": 0.004,
            },
        ]
        return self._run(
            name="test_tool_hallucination",
            turns=turns,
            expected=AgentFailureType.TOOL_HALLUCINATION,
        )

    def _test_cost_explosion(self) -> TestResult:
        """3 turns on a $1.00 budget; cumulative cost reaches $0.95 at turn 3.

        The cost_usd field in each trace turn represents the cumulative spend.
        Threshold = 90% of $1.00 = $0.90.

          Turn 1: cumulative $0.20  (20%) — OK
          Turn 2: cumulative $0.55  (55%) — OK
          Turn 3: cumulative $0.95  (95%) → exceeds threshold before turn 5 → COST_EXPLOSION

        Classifier initialised with budget_usd=1.0.
        All tool_args match their schemas to avoid TOOL_HALLUCINATION firing first.
        """
        turns: list[dict[str, Any]] = [
            {"type": "tool_call", "tool_name": "vector_scan",  "tool_args": {"corpus": "all"},  "tool_schema": {"corpus": "str"}, "tokens": 10_000, "cost_usd": 0.20},
            {"type": "tool_call", "tool_name": "embed_batch",  "tool_args": {"size": 5_000},    "tool_schema": {"size":   "int"}, "tokens": 50_000, "cost_usd": 0.55},
            {"type": "tool_call", "tool_name": "cross_rerank", "tool_args": {"top_k": 100},     "tool_schema": {"top_k":  "int"}, "tokens": 80_000, "cost_usd": 0.95},
        ]
        return self._run(
            name="test_cost_explosion",
            turns=turns,
            expected=AgentFailureType.COST_EXPLOSION,
            budget_usd=1.0,
        )

    def _test_silent_wrong_answer(self) -> TestResult:
        """4 turns; FINAL_ANSWER emitted after two consecutive message-only turns.

        Window before final_answer (index 3) = [message, message].
        No tool_call in the window → SILENT_WRONG_ANSWER.
        """
        turns: list[dict[str, Any]] = [
            {"type": "message",      "content": "Solve: 2x + 3 = 11.",         "tokens": 300, "cost_usd": 0.001},
            {"type": "message",      "content": "Let me reason step by step.",  "tokens": 500, "cost_usd": 0.002},
            {"type": "message",      "content": "I believe I have the answer.", "tokens": 600, "cost_usd": 0.003},
            {"type": "final_answer", "content": "x = 3",                        "tokens": 700, "cost_usd": 0.004},
        ]
        return self._run(
            name="test_silent_wrong_answer",
            turns=turns,
            expected=AgentFailureType.SILENT_WRONG_ANSWER,
        )

    def _test_memory_drift(self) -> TestResult:
        """6 turns; stale memory returns 'London' for a task that asks for Paris.

        After reading the wrong memory value the agent produces two reasoning
        messages and then a FINAL_ANSWER with no re-verification tool call in
        the preceding 2-turn window.

        Window before final_answer (index 5) = [message, message].
        No tool_call → SILENT_WRONG_ANSWER (the observable signature of memory drift).

        Note: MEMORY_DRIFT has no dedicated static detector; it surfaces through
        the SILENT_WRONG_ANSWER pattern — an unverified answer grounded on stale
        memory.  A higher-level semantic checker would be needed to distinguish
        the root cause.
        """
        turns: list[dict[str, Any]] = [
            {"type": "message",      "content": "Task: book a flight to Paris.",           "tokens":   400, "cost_usd": 0.001},
            {"type": "tool_call",    "tool_name": "memory_read", "tool_args": {"key": "last_destination"}, "tool_schema": {"key": "str"}, "tokens": 600, "cost_usd": 0.003},
            {"type": "tool_result",  "tool_name": "memory_read", "result": "London",       "tokens":   700, "cost_usd": 0.004},
            # Two non-tool turns — no re-verification before the (wrong) answer
            {"type": "message",      "content": "Memory says last destination: London.",   "tokens":   800, "cost_usd": 0.005},
            {"type": "message",      "content": "Proceeding with that destination.",       "tokens":   900, "cost_usd": 0.006},
            {"type": "final_answer", "content": "Flight booked to London (via memory).",   "tokens": 1_000, "cost_usd": 0.007},
        ]
        return self._run(
            name="test_memory_drift",
            turns=turns,
            expected=AgentFailureType.SILENT_WRONG_ANSWER,
        )

    def _test_observability_blindness(self) -> TestResult:
        """4 turns with no 'tokens' or 'cost_usd' span fields anywhere.

        All active detectors produce None (no loops, no overflow, no type errors,
        no cost data, no unverified final_answer — the tool_call at index 1 is
        within the 2-turn window before final_answer).

        Fallback detector fires: has_spans=False → OBSERVABILITY_BLINDNESS
        with confidence 0.60.
        """
        turns: list[dict[str, Any]] = [
            {"type": "message",      "content": "Execute preprocessing pipeline."},
            {"type": "tool_call",    "tool_name": "pipeline_run", "tool_args": {"stage": "preprocess"}},
            {"type": "tool_result",  "tool_name": "pipeline_run", "result": "done"},
            {"type": "final_answer", "content": "Pipeline complete."},
        ]
        return self._run(
            name="test_observability_blindness",
            turns=turns,
            expected=AgentFailureType.OBSERVABILITY_BLINDNESS,
        )

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    def _run(
        self,
        name: str,
        turns: list[dict[str, Any]],
        expected: AgentFailureType,
        budget_usd: float = 10.0,
        context_window: int = 200_000,
    ) -> TestResult:
        """Write trace to a temp file, classify, and compare against expected.

        Args:
            name: Test identifier used in output.
            turns: List of turn dicts to serialise as JSONL.
            expected: The AgentFailureType the test asserts.
            budget_usd: Passed to FailureModeClassifier (default $10).
            context_window: Passed to FailureModeClassifier (default 200 k).

        Returns:
            TestResult with pass/fail status and supporting metadata.
        """
        try:
            tmp_path = _write_trace(turns)
            classifier = FailureModeClassifier(
                budget_usd=budget_usd,
                context_window=context_window,
            )
            report = classifier.classify(tmp_path)
            passed = report.failure_type == expected
            return TestResult(
                name=name,
                passed=passed,
                expected_type=expected,
                detected_type=report.failure_type,
                turns_in_trace=len(turns),
                cost_simulated_usd=report.cost_usd_burned,
                trace=turns,
                error_message=(
                    f"expected {expected.value!r}, "
                    f"got {report.failure_type.value!r}"
                ) if not passed else None,
            )
        except Exception as exc:
            return TestResult(
                name=name,
                passed=False,
                expected_type=expected,
                detected_type=None,
                turns_in_trace=len(turns),
                cost_simulated_usd=0.0,
                trace=turns,
                error_message=f"{type(exc).__name__}: {exc}",
            )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _write_trace(turns: list[dict[str, Any]]) -> str:
    """Serialise turns as JSONL to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n")
        return f.name


def _print_result(result: TestResult) -> None:
    """Print a single test result to stdout.

    Passing tests show the full trace, detected type, and simulated cost.
    Failing tests show the mismatch or exception detail.
    """
    sep = "-" * 66
    tag = "PASS" if result.passed else "FAIL"
    detected = result.detected_type.value if result.detected_type else "EXCEPTION"

    print(f"\n[{tag}] {result.name}")
    print(f"  Expected  : {result.expected_type.value}")
    print(f"  Detected  : {detected}")
    print(f"  Turns     : {result.turns_in_trace}")
    print(f"  Cost      : ${result.cost_simulated_usd:.6f}")

    if result.passed:
        print(f"  {sep}")
        print("  Trace:")
        for i, turn in enumerate(result.trace):
            t_type = turn.get("type", "?")
            tool   = turn.get("tool_name", "")
            args   = turn.get("tool_args", {})
            tokens = turn.get("tokens", "n/a")
            cost   = turn.get("cost_usd", "n/a")
            if tool:
                print(f"    [{i+1:>2}] {t_type:<12} {tool}({args})"
                      f"  tokens={tokens}  cost_usd={cost}")
            else:
                snippet = str(turn.get("content") or turn.get("result", ""))[:50]
                print(f"    [{i+1:>2}] {t_type:<12} {snippet!r}"
                      f"  tokens={tokens}  cost_usd={cost}")
    else:
        print(f"  Error     : {result.error_message}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    suite = FailureModeTestSuite()
    report = suite.run_all()
    print(report)
