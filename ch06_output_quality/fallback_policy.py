"""
Fallback policy: a decision tree for handling low-confidence agent outputs.

Given a ConfidenceResult (and, optionally, the ValidationReport that produced
it), the policy chooses exactly one recovery action:

  PASS_THROUGH     Confidence is high enough -- deliver the output as-is.
  RETRY_WITH_HINT  Medium confidence with retries left -- retry, steering the
                   agent with a hint derived from the weakest signal.
  PARTIAL_RESULT   Medium confidence, retries exhausted -- return a best-effort
                   partial result with an explicit incompleteness caveat.
  ESCALATE_HUMAN   Confidence too low -- enqueue the case for human review.
  ABORT            Schema validation failed -- the output is structurally
                   invalid; stop immediately, no retry can fix bad structure.

Decision tree (evaluated top-down):
  1. schema FAIL (validation_report)          -> ABORT
  2. confidence >= confidence_threshold        -> PASS_THROUGH
  3. medium AND attempt_count <  max_retries   -> RETRY_WITH_HINT
  4. medium AND attempt_count >= max_retries   -> PARTIAL_RESULT
  5. confidence <  _MEDIUM_FLOOR (0.5)         -> ESCALATE_HUMAN

"confidence" means the calibrated_score from ConfidenceResult.

Escalations are appended as JSON lines to escalation_queue_path so a human or a
separate worker can drain the queue. Standard library only; the report types
come from the sibling self_eval modules.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# ── import previous modules' report types ────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ch06_output_quality.confidence_scorer import ConfidenceResult
from ch06_output_quality.output_validator import ValidationReport


# ── constants ─────────────────────────────────────────────────────────────────

# Lower edge of the "medium confidence" band. Below this -> escalate.
_MEDIUM_FLOOR = 0.50

# Human-readable hint per weakest contributing factor
_FACTOR_HINTS: dict[str, str] = {
    "tool_success_rate": (
        "Several tool calls failed. Retry the failed calls, fix their arguments, "
        "or switch to an alternative tool before answering."
    ),
    "evidence_coverage": (
        "Key claims are not backed by tool results. Gather more evidence and cite "
        "concrete tool outputs for each claim before answering."
    ),
    "reasoning_consistency": (
        "The tool calls look disconnected from the task. Focus each action on the "
        "task's core requirements and ensure every tool call is resolved."
    ),
    "budget_pressure": (
        "Budget is nearly exhausted. Prioritise the single most critical sub-task "
        "and produce a focused answer instead of broad exploration."
    ),
}

# budget_pressure is "bad when high"; all other factors are "bad when low"
_INVERTED_FACTORS: frozenset[str] = frozenset({"budget_pressure"})


# ── action enum ───────────────────────────────────────────────────────────────

class FallbackAction(str, Enum):
    """The five mutually-exclusive recovery actions a FallbackPolicy can take."""
    PASS_THROUGH    = "PASS_THROUGH"
    RETRY_WITH_HINT = "RETRY_WITH_HINT"
    PARTIAL_RESULT  = "PARTIAL_RESULT"
    ESCALATE_HUMAN  = "ESCALATE_HUMAN"
    ABORT           = "ABORT"


# ── decision record ───────────────────────────────────────────────────────────

@dataclass
class FallbackDecision:
    """The outcome of FallbackPolicy.decide().

    Attributes:
        action:             The chosen FallbackAction.
        reasoning:          Human-readable justification for the action.
        retry_hint:         Steering hint for the next attempt (RETRY_WITH_HINT only).
        partial_result:     Best-effort partial payload (PARTIAL_RESULT only).
        escalation_payload: Case record enqueued for humans (ESCALATE_HUMAN only).
    """
    action:             FallbackAction
    reasoning:          str
    retry_hint:         str | None             = None
    partial_result:     str | None             = None
    escalation_payload: dict[str, Any] | None  = None


# ── fallback policy ───────────────────────────────────────────────────────────

class FallbackPolicy:
    """Decision tree that maps low-confidence outputs to recovery actions.

    Usage::

        policy = FallbackPolicy(confidence_threshold=0.75, max_retries=1)
        decision = policy.decide(confidence_result, attempt_count=0, validation_report=vr)
        if decision.action is FallbackAction.RETRY_WITH_HINT:
            next_prompt = base_prompt + "\\n\\nHint: " + decision.retry_hint
        ...
        print(policy.get_policy_stats())

    Args:
        confidence_threshold:  Calibrated score at/above which output passes through.
        max_retries:           Maximum RETRY_WITH_HINT attempts before falling back
                               to PARTIAL_RESULT.
        escalation_queue_path: JSONL file to which escalations are appended. When
                               None, escalations are still returned but not persisted.
    """

    def __init__(
        self,
        confidence_threshold:  float           = 0.75,
        max_retries:           int             = 1,
        escalation_queue_path: str | Path | None = None,
    ) -> None:
        self.confidence_threshold  = confidence_threshold
        self.max_retries           = max_retries
        self.escalation_queue_path = (
            Path(escalation_queue_path) if escalation_queue_path else None
        )

        self._action_counts: Counter[str] = Counter()
        self._escalations_written         = 0
        self._decision_count              = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        confidence_result: ConfidenceResult,
        attempt_count:     int,
        validation_report: ValidationReport | None = None,
    ) -> FallbackDecision:
        """Choose a recovery action for one agent output.

        Args:
            confidence_result: Score from ConfidenceScorer.score().
            attempt_count:     How many attempts have already been made for this
                               task (0 on the first try).
            validation_report: Optional report from OutputValidator.validate();
                               only its schema_result is consulted (for ABORT).

        Returns:
            A FallbackDecision; the matching detail field (retry_hint /
            partial_result / escalation_payload) is populated for that action.
        """
        confidence = confidence_result.calibrated_score
        self._decision_count += 1

        # ── 1. Schema failure → ABORT (no retry can fix bad structure) ───────
        if self._schema_failed(validation_report):
            errs = "; ".join((validation_report.schema_result.errors or [])[:2]) \
                if validation_report else "unknown schema error"
            decision = FallbackDecision(
                action=FallbackAction.ABORT,
                reasoning=(
                    f"Schema validation FAILED ({errs}). Output is structurally "
                    "invalid; aborting because a retry cannot repair bad structure."
                ),
            )
            return self._record(decision)

        # ── 2. High confidence → PASS_THROUGH ────────────────────────────────
        if confidence >= self.confidence_threshold:
            decision = FallbackDecision(
                action=FallbackAction.PASS_THROUGH,
                reasoning=(
                    f"Confidence {confidence:.2f} >= threshold "
                    f"{self.confidence_threshold:.2f}; delivering output as-is."
                ),
            )
            return self._record(decision)

        # ── 3 & 4. Medium confidence band ─────────────────────────────────────
        if confidence >= _MEDIUM_FLOOR:
            if attempt_count < self.max_retries:
                weakest, hint = self._build_hint(confidence_result.contributing_factors)
                decision = FallbackDecision(
                    action=FallbackAction.RETRY_WITH_HINT,
                    reasoning=(
                        f"Confidence {confidence:.2f} in medium band "
                        f"[{_MEDIUM_FLOOR:.2f}, {self.confidence_threshold:.2f}); "
                        f"attempt {attempt_count} < max_retries {self.max_retries}. "
                        f"Retrying with a hint targeting the weakest signal "
                        f"('{weakest}')."
                    ),
                    retry_hint=hint,
                )
                return self._record(decision)

            # retries exhausted → partial result
            decision = FallbackDecision(
                action=FallbackAction.PARTIAL_RESULT,
                reasoning=(
                    f"Confidence {confidence:.2f} in medium band but attempt "
                    f"{attempt_count} >= max_retries {self.max_retries}. "
                    "Returning a best-effort partial result with a caveat."
                ),
                partial_result=self._build_partial(confidence_result),
            )
            return self._record(decision)

        # ── 5. Low confidence → ESCALATE_HUMAN ───────────────────────────────
        payload  = self._build_escalation(confidence_result, attempt_count)
        decision = FallbackDecision(
            action=FallbackAction.ESCALATE_HUMAN,
            reasoning=(
                f"Confidence {confidence:.2f} < {_MEDIUM_FLOOR:.2f}; too low to "
                "deliver autonomously. Escalating for human review."
            ),
            escalation_payload=payload,
        )
        self._enqueue_escalation(payload)
        return self._record(decision)

    def get_policy_stats(self) -> dict[str, Any]:
        """Return the distribution of actions taken so far.

        Returns:
            Dict with:
              total_decisions     -- decisions made since construction.
              by_action           -- count per FallbackAction value.
              by_action_pct       -- percentage per action (0-100, rounded).
              escalations_written -- escalations persisted to the queue file.
        """
        total = self._decision_count
        by_action = {a.value: self._action_counts.get(a.value, 0) for a in FallbackAction}
        by_pct = {
            k: (round(v / total * 100, 1) if total else 0.0)
            for k, v in by_action.items()
        }
        return {
            "total_decisions":     total,
            "by_action":           by_action,
            "by_action_pct":       by_pct,
            "escalations_written": self._escalations_written,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _schema_failed(report: ValidationReport | None) -> bool:
        """True when a schema was applied and it did not pass."""
        if report is None:
            return False
        sr = report.schema_result
        return (not sr.skipped) and (not sr.passed)

    def _build_hint(self, factors: dict[str, float]) -> tuple[str, str]:
        """Pick the weakest contributing factor and return (factor_name, hint).

        For normal factors "weakest" means lowest value; for inverted factors
        (budget_pressure) it means highest value. The factor whose deviation from
        ideal is largest wins.
        """
        if not factors:
            return ("unknown", "Re-examine the task requirements and try again.")

        def severity(name: str, value: float) -> float:
            # Severity = how far from the ideal (1.0 normal, 0.0 inverted)
            return value if name in _INVERTED_FACTORS else (1.0 - value)

        weakest = max(factors, key=lambda n: severity(n, factors[n]))
        hint    = _FACTOR_HINTS.get(
            weakest, "Address the weakest signal and try again."
        )
        return (weakest, hint)

    def _build_partial(self, result: ConfidenceResult) -> str:
        """Compose a best-effort partial result caveat from the strong signals."""
        factors  = result.contributing_factors
        strong   = [
            name for name, val in factors.items()
            if (val < 0.5 if name in _INVERTED_FACTORS else val >= 0.5)
        ]
        strong_str = ", ".join(strong) if strong else "no signal"
        return (
            "[PARTIAL RESULT -- confidence "
            f"{result.calibrated_score:.2f}, band {result.confidence_band}] "
            "The agent could not reach full confidence after exhausting retries. "
            f"Trustworthy aspects (signals above threshold): {strong_str}. "
            "Treat remaining claims as unverified and confirm before acting."
        )

    def _build_escalation(
        self,
        result:        ConfidenceResult,
        attempt_count: int,
    ) -> dict[str, Any]:
        """Build the JSON-serialisable escalation case record."""
        return {
            "timestamp":            _utcnow_iso(),
            "calibrated_score":     result.calibrated_score,
            "raw_score":            result.raw_score,
            "confidence_band":      result.confidence_band,
            "contributing_factors": result.contributing_factors,
            "recommendation":       result.recommendation,
            "attempt_count":        attempt_count,
            "reason":               "confidence below medium floor",
        }

    def _enqueue_escalation(self, payload: dict[str, Any]) -> None:
        """Append one escalation record as a JSON line to the queue file."""
        if self.escalation_queue_path is None:
            return
        self.escalation_queue_path.parent.mkdir(parents=True, exist_ok=True)
        with self.escalation_queue_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._escalations_written += 1

    def _record(self, decision: FallbackDecision) -> FallbackDecision:
        """Tally the action and return the decision unchanged."""
        self._action_counts[decision.action.value] += 1
        return decision


# ── module helpers ────────────────────────────────────────────────────────────

def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(tzinfo=None).isoformat()


def _make_confidence(
    calibrated: float,
    factors: dict[str, float] | None = None,
    band: str | None = None,
    recommendation: str = "",
) -> ConfidenceResult:
    """Build a ConfidenceResult for the demo without running the scorer."""
    if factors is None:
        factors = {
            "tool_success_rate":     calibrated,
            "evidence_coverage":     calibrated,
            "reasoning_consistency": 1.0,
            "budget_pressure":       0.0,
        }
    if band is None:
        band = "HIGH" if calibrated >= 0.75 else "MEDIUM" if calibrated >= 0.50 else "LOW"
    return ConfidenceResult(
        raw_score=            calibrated,
        calibrated_score=     calibrated,
        confidence_band=      band,
        contributing_factors= factors,
        recommendation=       recommendation,
    )


def _make_report(schema_passed: bool, skipped: bool = False) -> ValidationReport:
    """Build a minimal ValidationReport carrying only a schema verdict."""
    from ch06_output_quality.output_validator import SchemaResult
    errors = [] if schema_passed else ["root: missing required field 'summary'"]
    return ValidationReport(
        schema_result=      SchemaResult(passed=schema_passed, errors=errors, skipped=skipped),
        semantic_score=     0.0,
        missing_topics=     [],
        grounding_score=    0.0,
        grounded_claims=    0,
        total_claims=       0,
        overall_confidence= 0.0,
        decision=           "PASS" if schema_passed else "ESCALATE",
        reasons=            [],
    )


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    sep = "=" * 80
    queue_path = Path(tempfile.gettempdir()) / "fallback_escalations.jsonl"
    if queue_path.exists():
        queue_path.unlink()

    policy = FallbackPolicy(
        confidence_threshold=0.75,
        max_retries=1,
        escalation_queue_path=queue_path,
    )

    print(sep)
    print(f"  FallbackPolicy demo  |  threshold=0.75  |  max_retries=1")
    print(f"  Escalation queue: {queue_path}")
    print(sep)

    # ── 15 scenarios: (label, calibrated_conf, attempt, schema_ok|None, factors) ─
    # schema_ok=None -> no schema applied (skipped); False -> schema FAIL -> ABORT
    F_LOW_TOOL   = {"tool_success_rate": 0.30, "evidence_coverage": 0.70,
                    "reasoning_consistency": 0.90, "budget_pressure": 0.10}
    F_LOW_EVID   = {"tool_success_rate": 0.90, "evidence_coverage": 0.25,
                    "reasoning_consistency": 0.90, "budget_pressure": 0.10}
    F_LOW_CONS   = {"tool_success_rate": 0.90, "evidence_coverage": 0.70,
                    "reasoning_consistency": 0.40, "budget_pressure": 0.10}
    F_HIGH_BGT   = {"tool_success_rate": 0.80, "evidence_coverage": 0.70,
                    "reasoning_consistency": 0.90, "budget_pressure": 0.95}

    scenarios: list[tuple[str, float, int, bool | None, dict[str, float] | None]] = [
        ("high-conf-clean",          0.92, 0, None,  None),
        ("high-conf-borderline",     0.78, 0, None,  None),
        ("medium-first-try-lowtool", 0.62, 0, None,  F_LOW_TOOL),
        ("medium-first-try-lowevid", 0.60, 0, None,  F_LOW_EVID),
        ("medium-first-try-lowcons", 0.65, 0, None,  F_LOW_CONS),
        ("medium-first-try-budget",  0.58, 0, None,  F_HIGH_BGT),
        ("medium-retry-exhausted",   0.62, 1, None,  F_LOW_EVID),
        ("medium-retry-exhausted-2", 0.55, 2, None,  F_LOW_TOOL),
        ("low-conf-escalate",        0.42, 0, None,  None),
        ("low-conf-escalate-retry",  0.38, 1, None,  None),
        ("very-low-conf",            0.18, 0, None,  None),
        ("schema-fail-abort",        0.90, 0, False, None),
        ("schema-fail-low-conf",     0.30, 0, False, None),
        ("exactly-threshold",        0.75, 0, None,  None),
        ("exactly-medium-floor",     0.50, 0, None,  F_LOW_CONS),
    ]

    print(
        f"\n  {'#':<3} {'Scenario':<26} {'Conf':>5} {'Att':>3} {'Schema':>7}  "
        f"{'Action':<16} Detail"
    )
    print(f"  {'-'*116}")

    for i, (label, conf, attempt, schema_ok, factors) in enumerate(scenarios, 1):
        cr = _make_confidence(conf, factors=factors)
        vr = None if schema_ok is None else _make_report(schema_passed=schema_ok)

        decision = policy.decide(cr, attempt_count=attempt, validation_report=vr)

        schema_str = "n/a" if schema_ok is None else ("PASS" if schema_ok else "FAIL")

        # Compact detail per action
        if decision.action is FallbackAction.RETRY_WITH_HINT:
            detail = f"hint: {decision.retry_hint[:46]}..."
        elif decision.action is FallbackAction.PARTIAL_RESULT:
            detail = "partial result returned with caveat"
        elif decision.action is FallbackAction.ESCALATE_HUMAN:
            detail = f"queued (score={decision.escalation_payload['calibrated_score']})"
        elif decision.action is FallbackAction.ABORT:
            detail = "structurally invalid -- stopped"
        else:
            detail = "delivered as-is"

        print(
            f"  {i:<3} {label:<26} {conf:>5.2f} {attempt:>3} {schema_str:>7}  "
            f"{decision.action.value:<16} {detail}"
        )

    # ── full reasoning for 3 representative decisions ──────────────────────────
    print(f"\n{sep}")
    print("  DETAILED REASONING (3 representative cases)")
    print(sep)

    samples = [
        ("RETRY example",    _make_confidence(0.60, factors=F_LOW_EVID), 0,   None),
        ("PARTIAL example",  _make_confidence(0.62, factors=F_LOW_TOOL), 1,   None),
        ("ABORT example",    _make_confidence(0.90),                     0,   _make_report(False)),
    ]
    for title, cr, att, vr in samples:
        d = policy.decide(cr, attempt_count=att, validation_report=vr)
        print(f"\n  [{title}] -> {d.action.value}")
        print(f"    reasoning: {d.reasoning}")
        if d.retry_hint:
            print(f"    retry_hint: {d.retry_hint}")
        if d.partial_result:
            print(f"    partial_result: {d.partial_result}")
        if d.escalation_payload:
            print(f"    escalation_payload: {json.dumps(d.escalation_payload, indent=6)[:200]}...")

    # ── policy stats ───────────────────────────────────────────────────────────
    stats = policy.get_policy_stats()

    print(f"\n{sep}")
    print("  POLICY STATS (action distribution)")
    print(f"  {'-'*50}")
    print(f"  Total decisions     : {stats['total_decisions']}")
    print(f"  Escalations written : {stats['escalations_written']}")
    print(f"\n  {'Action':<18} {'Count':>5} {'Pct':>6}  Distribution")
    print(f"  {'-'*60}")
    for action in FallbackAction:
        name = action.value
        cnt  = stats["by_action"][name]
        pct  = stats["by_action_pct"][name]
        bar  = "#" * int(pct / 100 * 30)
        print(f"  {name:<18} {cnt:>5} {pct:>5.1f}%  {bar}")

    # ── verify the escalation queue file ───────────────────────────────────────
    if queue_path.exists():
        lines = queue_path.read_text(encoding="utf-8").strip().splitlines()
        print(f"\n  Escalation queue file written: {len(lines)} JSON line(s)")
        if lines:
            first = json.loads(lines[0])
            print(f"  First record: score={first['calibrated_score']}, "
                  f"band={first['confidence_band']}, attempt={first['attempt_count']}")
    print(sep)
