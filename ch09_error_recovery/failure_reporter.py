"""
Failure reporter for honest partial results and fabrication detection.

When an agent cannot complete a task — or when it might be about to return a
hallucinated answer — this module provides three things:

1. **FabricationRisk** — three lightweight signals that flag an answer the
   agent produced without adequate tool evidence:

   * No tool calls in the last two assistant turns (weight 0.40).
   * Numbers or quoted phrases in the answer that are absent from every tool
     result (weight up to 0.40).
   * Key topic words from the task that are absent from the answer (weight 0.20).

2. **PartialResult** — honest accounting of what work was done and what is
   missing, with a ``downstream_safe`` flag so callers know whether to pass
   the partial output forward.

3. **FailureResponse** — a fully JSON-serializable structure that downstream
   systems can consume without raising exceptions, regardless of how much work
   the agent managed to complete.

Counters for audit are exposed via ``get_stats()``.  Failed responses can be
written to disk for audit trail via ``log_failure()``.

Stdlib only -- no external dependencies, no LLM calls.
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# -- project root: ch09_error_recovery/../ = root -----------------------------
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ch03_agent_loop.agent_state import AgentState  # noqa: E402


# -- NLP helpers --------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "that", "this", "these", "those",
    "what", "which", "who", "how", "when", "where", "why", "it", "its",
    "as", "if", "not", "no", "than", "then", "also", "about", "up", "out",
    "into", "through", "during", "before", "after", "above", "below",
    "get", "give", "make", "take", "use", "can", "just", "only", "very",
})

# Matches numbers with >= 2 digits (optionally decimal/percentage)
_NUMBER_RE = re.compile(r"\b\d{2,}(?:[.,]\d+)?%?\b")
# Matches double-quoted phrases of at least 4 characters
_QUOTED_RE = re.compile(r'"([^"]{4,})"')
# Slug: keep only alphanumeric, replace runs of non-alnum with underscore
_SLUG_RE   = re.compile(r"[^a-z0-9]+")


# -- result types -------------------------------------------------------------

@dataclass
class FabricationRisk:
    """Fabrication risk assessment for a proposed final answer.

    Attributes:
        is_risky:       True when risk_score is at or above 0.40.
        risk_score:     Composite score from 0.0 (none) to 1.0 (certain).
        signals:        Human-readable list of signals that raised the score.
        recommendation: Action recommendation based on the risk tier.
    """

    is_risky:       bool
    risk_score:     float
    signals:        list[str]
    recommendation: str


@dataclass
class PartialResult:
    """Honest accounting of incomplete work for downstream consumption.

    Attributes:
        completed_work:      Mapping of step label -> result content.
        missing_data:        Labels of steps that failed or were skipped.
        confidence_estimate: Fraction of required steps that completed (0-1).
        honest_summary:      One-paragraph plain-English description.
        downstream_safe:     True when the partial output is safe to forward
                             (majority completed AND no blocking step failed).
    """

    completed_work:      dict[str, Any]
    missing_data:        list[str]
    confidence_estimate: float
    honest_summary:      str
    downstream_safe:     bool


@dataclass
class FailureResponse:
    """Structured failure/partial response safe for downstream system consumption.

    Downstream services branch on ``status`` ("partial" | "failed"), consume
    whatever is in ``completed_sections``, and surface ``missing_sections`` to
    the user or a retry queue -- without ever raising a KeyError or AttributeError.

    Attributes:
        task_id:             Short URL-safe slug derived from the task string.
        task:                Original task description.
        status:              "partial" (some work done) or "failed" (nothing done).
        completed_sections:  Mapping of step label -> result.
        missing_sections:    Labels of data that could not be obtained.
        reason:              Why the task did not fully complete.
        retry_possible:      True when the failure looks transient.
        confidence:          Fraction of required work that completed.
        downstream_safe:     Whether the partial result is safe to forward.
        timestamp:           Unix timestamp when this response was built.
    """

    task_id:            str
    task:               str
    status:             str
    completed_sections: dict[str, Any]
    missing_sections:   list[str]
    reason:             str
    retry_possible:     bool
    confidence:         float
    downstream_safe:    bool
    timestamp:          float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "task_id":            self.task_id,
            "task":               self.task,
            "status":             self.status,
            "completed_sections": self.completed_sections,
            "missing_sections":   self.missing_sections,
            "reason":             self.reason,
            "retry_possible":     self.retry_possible,
            "confidence":         round(self.confidence, 4),
            "downstream_safe":    self.downstream_safe,
            "timestamp":          self.timestamp,
        }


# -- reporter -----------------------------------------------------------------

class FailureReporter:
    """Detects fabricated answers and builds honest partial/failure responses.

    All methods are pure (no side effects) except ``log_failure`` and the
    internal counters incremented for ``get_stats()``.

    Signal weights for ``detect_fabrication``::

        no tool calls in last 2 turns : 0.40
        ungrounded facts              : up to 0.40
        scope mismatch                : 0.20

    Risk tiers: score >= 0.70 -> HIGH, >= 0.40 -> MEDIUM, else LOW.
    ``is_risky`` is True when score >= 0.40.
    """

    _RISKY_THRESHOLD:    float = 0.40
    _HIGH_RISK:          float = 0.70
    _W_NO_TOOL_CALLS:    float = 0.40
    _W_UNGROUNDED_FACTS: float = 0.40
    _W_SCOPE_MISMATCH:   float = 0.20

    _RETRY_KEYWORDS: tuple[str, ...] = (
        "timeout", "timed out", "rate limit", "429", "503", "502",
        "temporarily", "transient", "retry", "overloaded", "connection",
        "unavailable", "try again",
    )

    def __init__(self) -> None:
        self._fabrication_risks_caught: int = 0
        self._partial_results_emitted:  int = 0
        self._full_failures:            int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_fabrication(
        self,
        state:           AgentState,
        proposed_answer: str,
    ) -> FabricationRisk:
        """Assess the fabrication risk of a proposed final answer.

        Evaluates three signals without calling any LLM:

        1. **No tool calls in last 2 turns** (weight 0.40) -- an assistant
           that produces a data-claim without issuing tool calls in its recent
           turns is likely hallucinating.
        2. **Ungrounded facts** (weight up to 0.40) -- numbers (>=2 digits)
           and quoted phrases in the proposed answer that do not appear
           anywhere in the agent's accumulated tool results.
        3. **Scope mismatch** (weight 0.20) -- fewer than 30 % of significant
           topic keywords from the task appear in the proposed answer,
           suggesting the answer addresses a different question.

        Args:
            state:           Current agent state.
            proposed_answer: The answer the agent is about to return.

        Returns:
            FabricationRisk with risk_score, signals, and recommendation.
        """
        score:   float      = 0.0
        signals: list[str]  = []

        # -- signal 1: no tool calls in last 2 assistant turns ---------------
        last_turns = _last_assistant_turns(state.messages, n=2)
        if not any(_has_tool_use(m) for m in last_turns):
            score += self._W_NO_TOOL_CALLS
            if last_turns:
                signals.append(
                    "no tool calls in the last 2 assistant turns before final answer"
                )
            else:
                signals.append(
                    "no assistant turns at all -- answer produced without any agent interaction"
                )

        # -- signal 2: claimed facts not grounded in tool results -------------
        facts       = _extract_facts(proposed_answer)
        results_txt = _tool_results_text(state)
        if facts:
            ungrounded = sorted(f for f in facts if f not in results_txt)
            if ungrounded:
                frac   = len(ungrounded) / len(facts)
                weight = min(self._W_UNGROUNDED_FACTS, frac * self._W_UNGROUNDED_FACTS)
                score += weight
                preview = ", ".join(repr(u) for u in ungrounded[:5])
                signals.append(
                    f"{len(ungrounded)}/{len(facts)} claimed facts not found in "
                    f"tool results: {preview}"
                )

        # -- signal 3: task scope vs answer scope ----------------------------
        task_kws = _task_keywords(state.task)
        if task_kws:
            ans_lower = proposed_answer.lower()
            covered   = sum(1 for k in task_kws if k in ans_lower)
            if covered / len(task_kws) < 0.30:
                score += self._W_SCOPE_MISMATCH
                missing_kws = [k for k in task_kws if k not in ans_lower][:5]
                signals.append(
                    f"answer covers {covered}/{len(task_kws)} task keywords; "
                    f"missing: {missing_kws}"
                )

        score    = min(1.0, round(score, 4))
        is_risky = score >= self._RISKY_THRESHOLD
        if is_risky:
            self._fabrication_risks_caught += 1

        if score >= self._HIGH_RISK:
            rec = "HIGH RISK: suppress answer and request a tool-grounded re-run"
        elif score >= self._RISKY_THRESHOLD:
            rec = "MEDIUM RISK: request source citations before publishing"
        else:
            rec = "LOW RISK: answer appears grounded in tool evidence"

        return FabricationRisk(
            is_risky=is_risky,
            risk_score=score,
            signals=signals,
            recommendation=rec,
        )

    def build_partial_result(
        self,
        state:           AgentState,
        completed_steps: list[dict[str, Any]],
        failed_steps:    list[dict[str, Any]],
    ) -> PartialResult:
        """Construct an honest partial result from an incomplete run.

        Each step dict is expected to have at least ``"id"`` and ``"tool"``
        keys.  Completed steps may carry a ``"result"`` key.  Failed steps may
        carry ``"reason"`` and ``"blocking"`` (defaults True) keys.

        Additional tool results already accumulated in ``state.tool_results``
        are merged in for any step not already covered by ``completed_steps``.

        Args:
            state:           Current agent state at the point of failure.
            completed_steps: Steps that finished successfully.
            failed_steps:    Steps that failed or were skipped.

        Returns:
            PartialResult with completed_work, missing_data, confidence, etc.
        """
        completed_work: dict[str, Any] = {}

        for step in completed_steps:
            label = _step_label(step)
            completed_work[label] = step.get("result", "(completed -- no result captured)")

        # Merge in tool results from state for steps not explicitly listed.
        for tr in state.tool_results:
            if not isinstance(tr, dict):
                continue
            sid  = tr.get("step")
            tool = tr.get("tool", "unknown")
            lbl  = f"step_{sid}:{tool}"
            if lbl not in completed_work and sid is not None:
                completed_work[lbl] = tr.get("content", "(no content)")

        missing_data: list[str] = []
        for s in failed_steps:
            entry = _step_label(s)
            if "reason" in s:
                entry += f" -- {s['reason']}"
            missing_data.append(entry)

        total = len(completed_steps) + len(failed_steps)
        conf  = round(len(completed_steps) / total, 4) if total > 0 else 0.0

        blocking_failures = sum(
            1 for s in failed_steps if s.get("blocking", True)
        )
        ds_safe = conf >= 0.5 and blocking_failures == 0

        n_done = len(completed_steps)
        n_fail = len(failed_steps)
        n_tr   = len(state.tool_results)
        parts  = [
            f"{n_done}/{total} steps completed ({n_tr} tool results preserved)."
        ]
        if n_fail:
            labels = ", ".join(_step_label(s) for s in failed_steps)
            parts.append(f"Failed steps: {labels}.")
        parts.append(f"Confidence: {conf:.0%}.")
        parts.append(
            "Downstream safe: yes."
            if ds_safe
            else "Downstream safe: no -- blocking steps are missing."
        )
        summary = " ".join(parts)

        self._partial_results_emitted += 1
        return PartialResult(
            completed_work=completed_work,
            missing_data=missing_data,
            confidence_estimate=conf,
            honest_summary=summary,
            downstream_safe=ds_safe,
        )

    def build_failure_response(
        self,
        task:           str,
        failure_reason: str,
        partial:        PartialResult,
    ) -> FailureResponse:
        """Build a structured, JSON-safe failure response for downstream systems.

        ``status`` is "partial" when any work was completed, "failed" when
        nothing was completed.  ``retry_possible`` is inferred from transient
        keywords in ``failure_reason``.

        Args:
            task:           Original task description.
            failure_reason: Human-readable explanation of the failure.
            partial:        PartialResult from ``build_partial_result``.

        Returns:
            FailureResponse ready for serialisation or forwarding.
        """
        status = "partial" if partial.completed_work else "failed"
        if status == "failed":
            self._full_failures += 1

        reason_lower   = failure_reason.lower()
        retry_possible = any(kw in reason_lower for kw in self._RETRY_KEYWORDS)

        return FailureResponse(
            task_id=_make_task_id(task),
            task=task,
            status=status,
            completed_sections=partial.completed_work,
            missing_sections=partial.missing_data,
            reason=failure_reason,
            retry_possible=retry_possible,
            confidence=partial.confidence_estimate,
            downstream_safe=partial.downstream_safe,
            timestamp=time.time(),
        )

    def log_failure(
        self,
        failure_response: FailureResponse,
        path:             str = "failures/",
    ) -> None:
        """Persist the failure response to disk for audit trail.

        File is written to ``{path}/{task_id}_{ts_ms}.json``.  The directory
        is created if it does not exist.

        Args:
            failure_response: Response to persist.
            path:             Directory to write into.
        """
        out_dir  = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_ms    = int(failure_response.timestamp * 1000)
        filename = f"{failure_response.task_id}_{ts_ms}.json"
        target   = out_dir / filename
        target.write_text(
            json.dumps(failure_response.to_dict(), indent=2),
            encoding="utf-8",
        )

    def get_stats(self) -> dict[str, int]:
        """Return cumulative counters for this reporter instance.

        Returns:
            Dict with keys: ``fabrication_risks_caught``,
            ``partial_results_emitted``, ``full_failures``.
        """
        return {
            "fabrication_risks_caught": self._fabrication_risks_caught,
            "partial_results_emitted":  self._partial_results_emitted,
            "full_failures":            self._full_failures,
        }


# -- module helpers -----------------------------------------------------------

def _last_assistant_turns(
    messages: list[dict[str, Any]],
    n:        int = 2,
) -> list[dict[str, Any]]:
    """Return the last ``n`` assistant messages from the history."""
    turns = [m for m in messages if m.get("role") == "assistant"]
    return turns[-n:] if turns else []


def _has_tool_use(message: dict[str, Any]) -> bool:
    """Return True if the message contains at least one tool_use content block."""
    content = message.get("content", [])
    if isinstance(content, str):
        return False
    return any(
        isinstance(c, dict) and c.get("type") == "tool_use"
        for c in content
    )


def _extract_facts(text: str) -> list[str]:
    """Extract numbers (>=2 digits) and double-quoted phrases from ``text``."""
    numbers = _NUMBER_RE.findall(text)
    quoted  = _QUOTED_RE.findall(text)
    # deduplicate while preserving order
    seen: set[str] = set()
    facts: list[str] = []
    for f in numbers + quoted:
        if f not in seen:
            seen.add(f)
            facts.append(f)
    return facts


def _tool_results_text(state: AgentState) -> str:
    """Concatenate all string values from tool_results into a single blob."""
    parts: list[str] = []
    for r in state.tool_results:
        if isinstance(r, dict):
            parts.extend(str(v) for v in r.values())
        else:
            parts.append(str(r))
    return " ".join(parts)


def _task_keywords(task: str) -> list[str]:
    """Extract significant non-stopword words (length > 3) from ``task``."""
    words = re.findall(r"\b[a-zA-Z]+\b", task.lower())
    return [w for w in words if len(w) > 3 and w not in _STOPWORDS]


def _step_label(step: dict[str, Any]) -> str:
    return f"step_{step.get('id', '?')}:{step.get('tool', 'unknown')}"


def _make_task_id(task: str) -> str:
    """Return a short URL-safe slug for the task string."""
    slug = _SLUG_RE.sub("_", task.lower())
    return slug[:40].strip("_")


# -- entry point --------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    SEP  = "=" * 68
    SEP2 = "-" * 40

    reporter = FailureReporter()

    print(f"\n{SEP}")
    print("  FAILURE REPORTER  |  3 scenarios")
    print(SEP)

    # =========================================================================
    # Scenario 1: agent attempts to fabricate a data-heavy answer without tools
    # =========================================================================
    print("\n[SCENARIO 1] FABRICATION DETECTION")
    print(SEP2)

    fab_state = AgentState(
        task=(
            "Fetch AAPL live stock price, 30-day trend, "
            "and P/E ratio from the market API"
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    "Fetch AAPL live stock price, 30-day trend, "
                    "and P/E ratio from the market API"
                ),
            },
            # assistant turn 1: no tool_use
            {
                "role": "assistant",
                "content": "I will research this for you right away.",
            },
            {"role": "user", "content": "Please proceed."},
            # assistant turn 2: no tool_use -- about to fabricate
            {
                "role": "assistant",
                "content": (
                    "Based on my knowledge, AAPL is at $195.30 with a "
                    "30-day gain of 3.2% and a P/E ratio of 29.4."
                ),
            },
        ],
        tool_results=[],   # no tool was actually called
    )

    proposed_fab = (
        "AAPL is currently trading at $195.30, up 3.2% over the past "
        "30 days. The trailing P/E ratio stands at 29.4, "
        "slightly above the sector average of 27.1."
    )

    risk = reporter.detect_fabrication(fab_state, proposed_fab)

    print(f"  task      : {fab_state.task}")
    print(f"  proposed  : {proposed_fab[:70]!r}...")
    print(f"  signals   :")
    for sig in risk.signals:
        print(f"    * {sig}")
    print(f"  risk_score     : {risk.risk_score:.2f}")
    print(f"  is_risky       : {risk.is_risky}")
    print(f"  recommendation : {risk.recommendation}")

    # =========================================================================
    # Scenario 2: task 2/4 steps completed, honest partial result + log
    # =========================================================================
    print(f"\n[SCENARIO 2] PARTIAL COMPLETION (2/4 steps)")
    print(SEP2)

    partial_state = AgentState(
        task="Collect market data, parse it, run statistical analysis, and generate report",
        messages=[
            {
                "role": "user",
                "content": (
                    "Collect market data, parse it, "
                    "run statistical analysis, and generate report"
                ),
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "web_search",
                        "input": {"query": "AAPL market data"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "12 data points collected from 3 APIs",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t2",
                        "name": "parse_data",
                        "input": {"raw": "..."},
                    }
                ],
            },
        ],
        tool_results=[
            {
                "step": 1,
                "tool": "web_search",
                "content": "12 market data points collected from 3 APIs",
            },
            {
                "step": 2,
                "tool": "parse_data",
                "content": "parsed: 12 records, 2 outliers removed, schema OK",
            },
        ],
        turn_count=2,
    )

    completed_2 = [
        {
            "id": 1,
            "tool": "web_search",
            "result": "12 market data points collected from 3 APIs",
        },
        {
            "id": 2,
            "tool": "parse_data",
            "result": "parsed: 12 records, 2 outliers removed, schema OK",
        },
    ]
    failed_2 = [
        {
            "id": 3,
            "tool": "run_analysis",
            "blocking": True,
            "reason": "TimeoutError after 30 s",
        },
        {
            "id": 4,
            "tool": "generate_report",
            "blocking": True,
            "reason": "blocked -- depends on step 3",
        },
    ]

    partial_2 = reporter.build_partial_result(partial_state, completed_2, failed_2)

    print(f"  confidence_estimate : {partial_2.confidence_estimate:.0%}")
    print(f"  downstream_safe     : {partial_2.downstream_safe}")
    print(f"  completed_work keys : {list(partial_2.completed_work.keys())}")
    print(f"  missing_data        : {partial_2.missing_data}")
    print(f"  honest_summary      : {partial_2.honest_summary}")

    resp_2 = reporter.build_failure_response(
        task=partial_state.task,
        failure_reason=(
            "Step 3 (run_analysis) timed out after 30 s; "
            "step 4 (generate_report) blocked on it."
        ),
        partial=partial_2,
    )

    print(f"\n  FAILURE RESPONSE (status={resp_2.status!r}):")
    print(json.dumps(resp_2.to_dict(), indent=4))

    with tempfile.TemporaryDirectory() as tmpdir:
        reporter.log_failure(resp_2, path=tmpdir)
        log_file = next(Path(tmpdir).glob("*.json"))
        print(f"\n  audit log written: {log_file.name}")

    # =========================================================================
    # Scenario 3: full failure -- nothing completed, transient root cause
    # =========================================================================
    print(f"\n[SCENARIO 3] FULL FAILURE (0/4 steps, rate limit)")
    print(SEP2)

    empty_state = AgentState(
        task="Fetch Q3 2024 earnings report from SEC EDGAR for MSFT",
        messages=[
            {
                "role": "user",
                "content": "Fetch Q3 2024 earnings report from SEC EDGAR for MSFT",
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "api_call",
                        "input": {"url": "https://sec.gov/..."},
                    }
                ],
            },
        ],
        tool_results=[],   # api_call never returned
    )

    failed_3 = [
        {
            "id": 1,
            "tool": "api_call",
            "blocking": True,
            "reason": "HTTP 429 -- rate limit exceeded",
        },
        {
            "id": 2,
            "tool": "parse_response",
            "blocking": True,
            "reason": "blocked -- depends on step 1",
        },
        {
            "id": 3,
            "tool": "extract_data",
            "blocking": True,
            "reason": "blocked -- depends on step 1",
        },
        {
            "id": 4,
            "tool": "format_report",
            "blocking": True,
            "reason": "blocked -- depends on step 3",
        },
    ]

    partial_3 = reporter.build_partial_result(empty_state, [], failed_3)
    resp_3    = reporter.build_failure_response(
        task=empty_state.task,
        failure_reason=(
            "API rate limit exceeded (HTTP 429). All 4 pipeline steps blocked. "
            "Retry after the rate-limit window resets (approx. 60 s)."
        ),
        partial=partial_3,
    )

    print(f"  status         : {resp_3.status}")
    print(f"  retry_possible : {resp_3.retry_possible}")
    print(f"  confidence     : {resp_3.confidence:.0%}")

    print(f"\n  FAILURE RESPONSE (status={resp_3.status!r}):")
    print(json.dumps(resp_3.to_dict(), indent=4))

    # =========================================================================
    # Final stats
    # =========================================================================
    print(f"\n{SEP2}")
    stats = reporter.get_stats()
    print(f"  FINAL STATS:")
    for k, v in stats.items():
        print(f"    {k}: {v}")
    print(SEP)
