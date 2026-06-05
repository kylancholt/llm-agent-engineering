"""
Confidence scorer: estimate output reliability from trace signals, no LLM calls.

The scorer combines four cheap, deterministic signals extracted from the agent's
conversation trace and the OutputValidator report:

  tool_success_rate     -- Fraction of tool calls that completed without error.
                           Failed/errored tool results lower confidence.
  evidence_coverage     -- grounding_score from the ValidationReport: how many
                           output claims are backed by tool-result evidence.
  reasoning_consistency -- Whether tool calls are logically connected to the task:
                           every tool_use should resolve to a tool_result, and the
                           tool inputs should share vocabulary with the task.
  budget_pressure       -- Agents running low on budget cut corners; high pressure
                           applies a confidence penalty.

raw_score is a weighted sum of these signals. calibrated_score applies an
adjustment factor learned from historical (score, error) pairs so that a stated
confidence of c predicts a success rate of c (i.e. confidence 0.6 -> ~40% errors).

Standard library only; the sole external dependency is the ValidationReport
dataclass imported from the sibling output_validator module.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── import the previous module's report type ─────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ch06_output_quality.output_validator import ValidationReport


# ── constants ─────────────────────────────────────────────────────────────────

# Signal weights for the raw confidence score (must sum to 1.0)
_W_TOOL_SUCCESS = 0.30
_W_EVIDENCE     = 0.30
_W_CONSISTENCY  = 0.25
_W_BUDGET       = 0.15

# Confidence-band thresholds on the calibrated score
_BAND_HIGH   = 0.75
_BAND_MEDIUM = 0.50

# Grounding gate: evidence_coverage is the signal most predictive of factual
# correctness, so an ungrounded output cannot earn a high band no matter how
# cleanly its tools ran. Below _EVIDENCE_GATE_MED the band is capped at MEDIUM;
# below _EVIDENCE_GATE_LOW it is forced to LOW. Mirrors OutputValidator's
# escalate-on-low-grounding philosophy.
_EVIDENCE_GATE_MED = 0.30
_EVIDENCE_GATE_LOW = 0.15

# Band ordering for cap operations (higher rank = more confident)
_BAND_RANK: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_RANK_BAND: dict[int, str] = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

# Budget pressure begins to bite at 50% spend, maxes out at 100%
_BUDGET_PRESSURE_FLOOR = 0.50

# Tokens shorter than this are ignored when measuring task/tool keyword overlap
_MIN_KEYWORD_LEN = 4

# Substrings that mark a tool result as a failure
_ERROR_MARKERS: tuple[str, ...] = (
    "error", "exception", "failed", "failure", "traceback",
    "not found", "no such", "permission denied", "timeout",
    "refused", "invalid", "cannot", "could not", "unable to",
)

# Calibration: a confidence bin is "well calibrated" if observed success rate is
# within this absolute tolerance of the bin's predicted confidence.
_CALIBRATION_TOLERANCE = 0.10


# ── public types ──────────────────────────────────────────────────────────────

@dataclass
class ConfidenceResult:
    """Output of ConfidenceScorer.score().

    Attributes:
        raw_score:           Weighted signal sum before calibration (0-1).
        calibrated_score:    raw_score after applying the learned adjustment (0-1).
        confidence_band:     "HIGH" | "MEDIUM" | "LOW" based on calibrated_score.
        contributing_factors: Per-signal values that fed the raw score.
        recommendation:      Suggested action for the agent loop.
    """
    raw_score:            float
    calibrated_score:     float
    confidence_band:      str
    contributing_factors: dict[str, float]
    recommendation:       str


@dataclass
class CalibrationBin:
    """One reliability-diagram bin produced by ConfidenceScorer.calibrate().

    Attributes:
        lower:            Bin lower edge (inclusive).
        upper:            Bin upper edge (exclusive, except the last bin).
        count:            Number of historical samples in this bin.
        predicted_conf:   Mean predicted confidence of samples in the bin.
        observed_success: Observed success rate (1 - error rate) in the bin.
        gap:              predicted_conf - observed_success (positive = overconfident).
    """
    lower:            float
    upper:            float
    count:            int
    predicted_conf:   float
    observed_success: float
    gap:              float


@dataclass
class CalibrationResult:
    """Output of ConfidenceScorer.calibrate().

    Attributes:
        expected_calibration_error: Sample-weighted mean |predicted - observed| (ECE).
        is_well_calibrated:         True when ECE <= _CALIBRATION_TOLERANCE.
        adjustment_factor:          Multiplier applied to future raw scores.
                                    < 1 shrinks overconfident scores, > 1 lifts
                                    underconfident ones.
        bins:                       Per-bin reliability detail.
        sample_count:               Total historical samples evaluated.
    """
    expected_calibration_error: float
    is_well_calibrated:         bool
    adjustment_factor:          float
    bins:                       list[CalibrationBin]
    sample_count:               int


# ── confidence scorer ─────────────────────────────────────────────────────────

class ConfidenceScorer:
    """Estimate output confidence from trace signals without extra LLM calls.

    Usage::

        scorer = ConfidenceScorer()
        result = scorer.score(trace, validation_report, budget_used_pct=0.3)
        print(result.confidence_band, result.calibrated_score)

        # Optionally learn a calibration adjustment from history:
        cal = scorer.calibrate(historical_scores, actual_errors)
        # Subsequent score() calls now apply cal.adjustment_factor.

    Args:
        adjustment_factor: Initial calibration multiplier (default 1.0 = identity).
    """

    def __init__(self, adjustment_factor: float = 1.0) -> None:
        self._adjustment_factor = adjustment_factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        trace:             list[dict[str, Any]],
        validation_report: ValidationReport,
        task:              str   = "",
        budget_used_pct:   float = 0.0,
    ) -> ConfidenceResult:
        """Compute a calibrated confidence score for an agent output.

        Args:
            trace:             Conversation trace (list of message dicts with
                               tool_use / tool_result blocks).
            validation_report: Report from OutputValidator.validate(); supplies
                               evidence_coverage (its grounding_score).
            task:              Original task text, used for reasoning_consistency
                               keyword overlap. Optional but improves accuracy.
            budget_used_pct:   Fraction of budget consumed (0-1) at output time.

        Returns:
            ConfidenceResult with raw and calibrated scores, band, factors, and
            a recommended action.
        """
        tool_success = self._tool_success_rate(trace)
        evidence     = max(0.0, min(1.0, validation_report.grounding_score))
        consistency  = self._reasoning_consistency(trace, task)
        pressure     = self._budget_pressure(budget_used_pct)

        raw = (
            _W_TOOL_SUCCESS * tool_success
            + _W_EVIDENCE   * evidence
            + _W_CONSISTENCY* consistency
            + _W_BUDGET     * (1.0 - pressure)
        )
        raw = round(max(0.0, min(1.0, raw)), 4)

        calibrated = round(max(0.0, min(1.0, raw * self._adjustment_factor)), 4)
        band       = self._band(calibrated)
        band       = self._apply_grounding_gate(band, evidence)

        factors = {
            "tool_success_rate":     round(tool_success, 4),
            "evidence_coverage":     round(evidence, 4),
            "reasoning_consistency": round(consistency, 4),
            "budget_pressure":       round(pressure, 4),
        }

        return ConfidenceResult(
            raw_score=            raw,
            calibrated_score=     calibrated,
            confidence_band=      band,
            contributing_factors= factors,
            recommendation=       self._recommend(band, validation_report),
        )

    def calibrate(
        self,
        historical_scores: list[float],
        actual_errors:     list[bool],
        n_bins:            int = 5,
    ) -> CalibrationResult:
        """Check calibration against history and learn an adjustment factor.

        A perfectly calibrated scorer satisfies: among outputs scored with
        confidence c, the success rate is c (so the error rate is 1 - c).
        We bin the historical scores, compare each bin's mean predicted
        confidence to its observed success rate, and compute the expected
        calibration error (ECE).

        The adjustment_factor is the ratio of overall observed success to overall
        mean predicted confidence; it rescales future scores toward reality and
        is stored on the scorer for subsequent score() calls.

        Args:
            historical_scores: Past calibrated confidence scores (0-1).
            actual_errors:     Parallel list; True where that output was wrong.
            n_bins:            Number of equal-width bins across [0, 1].

        Returns:
            CalibrationResult with ECE, calibration verdict, adjustment factor,
            and per-bin reliability detail.

        Raises:
            ValueError: If the two lists differ in length or are empty.
        """
        if len(historical_scores) != len(actual_errors):
            raise ValueError(
                f"historical_scores ({len(historical_scores)}) and actual_errors "
                f"({len(actual_errors)}) must have equal length."
            )
        if not historical_scores:
            raise ValueError("Cannot calibrate on empty history.")

        n = len(historical_scores)
        successes = [not err for err in actual_errors]

        # ── build bins ────────────────────────────────────────────────────────
        bins: list[CalibrationBin] = []
        ece  = 0.0
        for b in range(n_bins):
            lower = b / n_bins
            upper = (b + 1) / n_bins
            # Last bin is inclusive of 1.0
            in_bin = [
                i for i, s in enumerate(historical_scores)
                if (lower <= s < upper) or (b == n_bins - 1 and s == 1.0)
            ]
            if not in_bin:
                bins.append(CalibrationBin(
                    lower=lower, upper=upper, count=0,
                    predicted_conf=0.0, observed_success=0.0, gap=0.0,
                ))
                continue

            pred = sum(historical_scores[i] for i in in_bin) / len(in_bin)
            obs  = sum(1 for i in in_bin if successes[i]) / len(in_bin)
            gap  = pred - obs
            ece += (len(in_bin) / n) * abs(gap)
            bins.append(CalibrationBin(
                lower=lower, upper=upper, count=len(in_bin),
                predicted_conf=round(pred, 4),
                observed_success=round(obs, 4),
                gap=round(gap, 4),
            ))

        # ── adjustment factor: overall observed / overall predicted ───────────
        mean_pred    = sum(historical_scores) / n
        mean_success = sum(1 for s in successes if s) / n
        adjustment   = (mean_success / mean_pred) if mean_pred > 1e-9 else 1.0
        adjustment   = round(max(0.1, min(2.0, adjustment)), 4)

        self._adjustment_factor = adjustment

        return CalibrationResult(
            expected_calibration_error= round(ece, 4),
            is_well_calibrated=         ece <= _CALIBRATION_TOLERANCE,
            adjustment_factor=          adjustment,
            bins=                       bins,
            sample_count=               n,
        )

    @property
    def adjustment_factor(self) -> float:
        """The current calibration multiplier applied to raw scores."""
        return self._adjustment_factor

    # ------------------------------------------------------------------
    # Signal extractors
    # ------------------------------------------------------------------

    def _tool_success_rate(self, trace: list[dict[str, Any]]) -> float:
        """Fraction of tool_result blocks that did not signal an error.

        A tool result counts as a failure when its ``is_error`` flag is set or
        when its text contains any marker in _ERROR_MARKERS. Returns 1.0 when the
        trace contains no tool results (nothing could have failed).
        """
        total    = 0
        failures = 0
        for msg in trace:
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                total += 1
                if block.get("is_error"):
                    failures += 1
                    continue
                text = _block_text(block).lower()
                if any(marker in text for marker in _ERROR_MARKERS):
                    failures += 1
        if total == 0:
            return 1.0
        return (total - failures) / total

    def _reasoning_consistency(self, trace: list[dict[str, Any]], task: str) -> float:
        """Score how logically the tool calls connect to the task.

        Two equally-weighted components:
          completion_rate -- fraction of tool_use blocks that have a matching
                             tool_result (a dangling call is incoherent).
          keyword_overlap -- fraction of tool calls whose serialised input shares
                             at least one significant token with the task. When no
                             task is provided this component defaults to 1.0.

        Returns 1.0 when the trace issues no tool calls (a pure-reasoning answer
        is trivially self-consistent).
        """
        tool_use_ids: list[str] = []
        result_ids:   set[str]  = set()
        tool_inputs:  list[str] = []

        for msg in trace:
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "tool_use":
                    tool_use_ids.append(block.get("id", ""))
                    tool_inputs.append(_json_text(block.get("input", {})))
                elif btype == "tool_result":
                    rid = block.get("tool_use_id", "")
                    if rid:
                        result_ids.add(rid)

        if not tool_use_ids:
            return 1.0

        # Component 1: completion rate
        matched          = sum(1 for tid in tool_use_ids if tid in result_ids)
        completion_rate  = matched / len(tool_use_ids)

        # Component 2: keyword overlap with the task
        task_keywords = _keywords(task)
        if not task_keywords:
            keyword_overlap = 1.0
        else:
            relevant = sum(
                1 for inp in tool_inputs
                if _keywords(inp) & task_keywords
            )
            keyword_overlap = relevant / len(tool_inputs)

        return 0.5 * completion_rate + 0.5 * keyword_overlap

    def _budget_pressure(self, budget_used_pct: float) -> float:
        """Map budget consumption to a pressure score in [0, 1].

        Pressure is 0 below _BUDGET_PRESSURE_FLOOR (50% spend) and rises linearly
        to 1.0 at 100% spend. High pressure means the agent may have cut corners.
        """
        used = max(0.0, min(1.0, budget_used_pct))
        if used <= _BUDGET_PRESSURE_FLOOR:
            return 0.0
        return (used - _BUDGET_PRESSURE_FLOOR) / (1.0 - _BUDGET_PRESSURE_FLOOR)

    # ------------------------------------------------------------------
    # Band and recommendation
    # ------------------------------------------------------------------

    @staticmethod
    def _band(score: float) -> str:
        if score >= _BAND_HIGH:
            return "HIGH"
        if score >= _BAND_MEDIUM:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _apply_grounding_gate(band: str, evidence: float) -> str:
        """Cap the band when evidence coverage is too low to trust the output.

        Ungrounded outputs cannot be HIGH confidence: grounding is the dominant
        correctness signal, so clean tool execution alone must not certify a
        fabricated answer.
        """
        if evidence < _EVIDENCE_GATE_LOW:
            return "LOW"
        if evidence < _EVIDENCE_GATE_MED:
            return _RANK_BAND[min(_BAND_RANK[band], _BAND_RANK["MEDIUM"])]
        return band

    @staticmethod
    def _recommend(band: str, report: ValidationReport) -> str:
        """Suggest an action given the confidence band and validator decision."""
        if band == "HIGH":
            return "Return output to the user; confidence is sufficient."
        if band == "MEDIUM":
            if report.decision == "ESCALATE":
                return "Escalate: medium confidence but validator flagged fabrication risk."
            return "Return with a caveat, or run one verification pass before delivering."
        # LOW
        if report.decision == "RETRY":
            return "Retry: low confidence and output is incomplete."
        return "Escalate to a human; confidence is too low to deliver autonomously."


# ── module helpers ────────────────────────────────────────────────────────────

def _block_text(block: dict[str, Any]) -> str:
    """Extract textual content from a tool_result block (str or list form)."""
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for sub in content:
            if isinstance(sub, dict) and sub.get("type") == "text":
                parts.append(sub.get("text", ""))
        return " ".join(parts)
    return str(content)


def _json_text(obj: Any) -> str:
    """Flatten a tool input dict into a single lowercase string."""
    if isinstance(obj, dict):
        return " ".join(f"{k} {_json_text(v)}" for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return " ".join(_json_text(v) for v in obj)
    return str(obj)


def _keywords(text: str) -> set[str]:
    """Return the set of significant lowercase tokens in text."""
    tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return {t for t in tokens if len(t) >= _MIN_KEYWORD_LEN}


def _make_report(grounding_score: float, decision: str = "PASS") -> ValidationReport:
    """Build a minimal ValidationReport for the demo (no embedding model needed)."""
    from ch06_output_quality.output_validator import SchemaResult
    return ValidationReport(
        schema_result=      SchemaResult(passed=True, errors=[], skipped=True),
        semantic_score=     grounding_score,   # demo proxy
        missing_topics=     [],
        grounding_score=    grounding_score,
        grounded_claims=    int(grounding_score * 10),
        total_claims=       10,
        overall_confidence= grounding_score,
        decision=           decision,
        reasons=            [],
    )


# ── demo trace builders ───────────────────────────────────────────────────────

def _make_trace(
    task:           str,
    n_tools:        int,
    n_failures:     int,
    dangling:       int = 0,
    irrelevant:     int = 0,
) -> list[dict[str, Any]]:
    """Construct a synthetic trace with controllable success/coherence patterns.

    Args:
        task:       Task text (tool inputs reuse its keywords for relevance).
        n_tools:    Number of tool_use / tool_result pairs to emit.
        n_failures: How many tool results carry an error marker.
        dangling:   How many tool_use blocks get NO matching result (incoherence).
        irrelevant: How many tool inputs use off-task keywords (low overlap).

    Returns:
        A conversation trace as a list of message dicts.
    """
    task_kw   = list(_keywords(task)) or ["data"]
    trace: list[dict[str, Any]] = [{"role": "user", "content": task}]

    for i in range(n_tools):
        tid     = f"tu_{i:03d}"
        is_irrel = i < irrelevant
        kw       = "miscellaneous offtopic placeholder" if is_irrel else " ".join(task_kw[:3])
        trace.append({
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": tid, "name": "read_file",
                 "input": {"query": kw, "path": f"file_{i}.py"}},
            ],
        })

        # Dangling tool calls (first `dangling`) get no result message
        if i < dangling:
            continue

        if i < n_failures:
            result_content = f"Error: file_{i}.py not found (exception raised)."
            is_error       = True
        else:
            result_content = (
                f"Contents of file_{i}.py: function processes {kw} successfully "
                f"and returns the expected result for the task."
            )
            is_error = False

        trace.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tid,
                 "is_error": is_error, "content": result_content},
            ],
        })

    return trace


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    random.seed(42)
    sep = "=" * 78
    TASK = "Analyze the authentication module and report security vulnerabilities."

    print(sep)
    print("  ConfidenceScorer demo  |  20 synthetic traces  |  no LLM calls")
    print(sep)

    scorer = ConfidenceScorer()

    # ── 20 trace scenarios: (label, n_tools, n_failures, dangling, irrelevant,
    #                         grounding_score, decision, budget_used_pct) ───────
    scenarios: list[tuple[str, int, int, int, int, float, str, float]] = [
        ("clean-5tools-grounded",      5, 0, 0, 0, 0.85, "PASS",     0.20),
        ("clean-3tools-grounded",      3, 0, 0, 0, 0.80, "PASS",     0.15),
        ("clean-high-budget",          4, 0, 0, 0, 0.75, "PASS",     0.90),
        ("one-failure",                5, 1, 0, 0, 0.70, "PASS",     0.30),
        ("two-failures",               5, 2, 0, 0, 0.60, "PASS",     0.35),
        ("half-failures",              6, 3, 0, 0, 0.50, "RETRY",    0.40),
        ("mostly-failures",            5, 4, 0, 0, 0.30, "RETRY",    0.50),
        ("all-failures",               4, 4, 0, 0, 0.10, "ESCALATE", 0.60),
        ("dangling-calls",             5, 0, 2, 0, 0.65, "PASS",     0.25),
        ("irrelevant-tools",           5, 0, 0, 3, 0.55, "RETRY",    0.30),
        ("low-grounding-clean",        4, 0, 0, 0, 0.25, "ESCALATE", 0.20),
        ("high-budget-pressure",       4, 1, 0, 0, 0.60, "PASS",     0.95),
        ("max-budget-pressure",        3, 0, 0, 0, 0.70, "PASS",     1.00),
        ("no-tools-grounded",          0, 0, 0, 0, 0.80, "PASS",     0.10),
        ("no-tools-ungrounded",        0, 0, 0, 0, 0.20, "ESCALATE", 0.15),
        ("mixed-fail-irrelevant",      6, 2, 1, 2, 0.45, "RETRY",    0.55),
        ("perfect-output",             6, 0, 0, 0, 0.95, "PASS",     0.10),
        ("borderline",                 5, 1, 1, 1, 0.60, "PASS",     0.45),
        ("budget-and-failures",        5, 3, 0, 0, 0.40, "RETRY",    0.85),
        ("everything-wrong",           5, 4, 2, 3, 0.15, "ESCALATE", 0.95),
    ]

    print(
        f"\n  {'#':<3} {'Scenario':<24} {'Tool%':>6} {'Evid':>5} {'Cons':>5} "
        f"{'Bgt':>5} {'Raw':>6} {'Band':<7} Recommendation"
    )
    print(f"  {'-'*120}")

    band_counts: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    all_results: list[ConfidenceResult] = []

    for i, (label, nt, nf, dang, irr, grnd, dec, budget) in enumerate(scenarios, 1):
        trace  = _make_trace(TASK, nt, nf, dang, irr)
        report = _make_report(grnd, dec)
        result = scorer.score(trace, report, task=TASK, budget_used_pct=budget)
        all_results.append(result)
        band_counts[result.confidence_band] += 1

        f = result.contributing_factors
        print(
            f"  {i:<3} {label:<24} "
            f"{f['tool_success_rate']*100:>5.0f}% {f['evidence_coverage']:>5.2f} "
            f"{f['reasoning_consistency']:>5.2f} {f['budget_pressure']:>5.2f} "
            f"{result.raw_score:>6.3f} {result.confidence_band:<7} "
            f"{result.recommendation[:48]}"
        )

    # ── band distribution ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  CONFIDENCE BAND DISTRIBUTION")
    print(f"  {'-'*40}")
    total = len(scenarios)
    for band in ("HIGH", "MEDIUM", "LOW"):
        c   = band_counts[band]
        bar = "#" * int(c / total * 40)
        print(f"  {band:<7} {c:>2}/{total}  {bar}")
    mean_raw = sum(r.raw_score for r in all_results) / len(all_results)
    print(f"\n  Mean raw score: {mean_raw:.3f}")

    # ── calibration test ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  CALIBRATION TEST")
    print(sep)

    # Simulate 200 historical (score, error) pairs from an OVERCONFIDENT scorer:
    # the model reports confidence c, but the true success rate is only ~c - 0.15.
    print("\n  Simulating 200 historical predictions from an overconfident model")
    print("  (true success rate ~ confidence - 0.15)...\n")

    hist_scores: list[float] = []
    hist_errors: list[bool]  = []
    for _ in range(200):
        conf         = round(random.uniform(0.3, 0.99), 3)
        true_success = conf - 0.15           # systematic overconfidence
        is_error     = random.random() > true_success
        hist_scores.append(conf)
        hist_errors.append(is_error)

    cal = scorer.calibrate(hist_scores, hist_errors, n_bins=5)

    print(f"  {'Bin':<14} {'N':>4} {'Predicted':>10} {'Observed':>9} {'Gap':>7}")
    print(f"  {'-'*50}")
    for b in cal.bins:
        if b.count == 0:
            continue
        flag = "  <-- overconfident" if b.gap > _CALIBRATION_TOLERANCE else ""
        print(
            f"  [{b.lower:.1f}, {b.upper:.1f})    {b.count:>4} "
            f"{b.predicted_conf:>10.3f} {b.observed_success:>9.3f} {b.gap:>+7.3f}{flag}"
        )

    print(f"\n  Expected Calibration Error (ECE): {cal.expected_calibration_error:.4f}")
    print(f"  Well calibrated (ECE <= {_CALIBRATION_TOLERANCE}): {cal.is_well_calibrated}")
    print(f"  Learned adjustment factor       : {cal.adjustment_factor:.4f}")
    print(f"  (scores will be multiplied by {cal.adjustment_factor:.3f} to correct overconfidence)")

    # ── verify the adjustment corrects calibration ─────────────────────────────
    print(f"\n  Re-scoring scenario #1 with the learned adjustment:")
    trace1  = _make_trace(TASK, 5, 0, 0, 0)
    report1 = _make_report(0.85, "PASS")
    result1 = scorer.score(trace1, report1, task=TASK, budget_used_pct=0.20)
    print(f"    raw_score={result1.raw_score:.3f}  "
          f"calibrated_score={result1.calibrated_score:.3f}  "
          f"band={result1.confidence_band}")
    print(f"    (calibration pulled confidence down to reflect true reliability)")
    print(sep)
