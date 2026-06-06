"""
Recovery state machine for step-level and task-level failure handling.

When a tool call inside an agent step fails, the RecoveryEngine decides what to
do next. It first classifies the failure, then maps the class to a concrete
recovery action, honouring a per-step retry budget and an optional fallback
tool per tool name:

  FailureClass.TRANSIENT  -> RETRY (with exponential backoff) while the retry
                             budget remains; then FALLBACK if a fallback tool is
                             configured; otherwise ESCALATE (blocking) / SKIP.
  FailureClass.PERMANENT  -> SKIP (non-blocking step) or ESCALATE (blocking).
  FailureClass.AMBIGUOUS  -> one diagnostic RETRY, then ESCALATE.

Step-level recovery (`recover_step`) returns a RecoveryAction. Task-level
recovery (`recover_task`) assembles a TaskRecoveryPlan over a whole run: it
preserves the work of already-completed steps (they are listed as preserved and
are NEVER re-executed) and points the executor at where to resume.

Progress preservation: every non-RETRY RecoveryAction carries the partial
results accumulated so far (from AgentState.tool_results), and TaskRecoveryPlan
lists the completed step ids so the orchestrator can resume without redoing
finished work.

Stdlib only — no external dependencies, no LLM calls, no I/O.

Step metadata contract: `recover_step` reads the run's step plan from
``state.metadata["steps"]`` — a list of dicts shaped like
``{"id": int, "tool": str, "blocking": bool}``. Missing entries default to
``tool="unknown"`` and ``blocking=True`` (fail safe).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ── project root: ch09_error_recovery/../ = root ────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ch03_agent_loop.agent_state import AgentState


# ── failure taxonomy ────────────────────────────────────────────────────────────

class FailureClass(str, Enum):
    """How a failure should be treated by the recovery state machine.

    Attributes:
        TRANSIENT: Temporary fault (timeout, rate limit, connection reset).
            Worth retrying, then falling back.
        PERMANENT: Deterministic fault (bad input, not found, unauthorized).
            Retrying will not help — skip or escalate.
        AMBIGUOUS: Unclassified fault. Worth one diagnostic retry before
            escalating to a human/operator.
    """
    TRANSIENT = "TRANSIENT"
    PERMANENT = "PERMANENT"
    AMBIGUOUS = "AMBIGUOUS"


class RecoveryActionType(str, Enum):
    """The concrete action the executor should take for a failed step.

    Attributes:
        RETRY:    Re-run the same tool after ``backoff_seconds``.
        FALLBACK: Re-run the step with ``fallback_tool`` instead.
        SKIP:     Abandon this (non-blocking) step and continue the task.
        ESCALATE: Hand the step off to a human/operator; the task pauses.
        ABORT:    Stop the whole task; recovery is not possible.
    """
    RETRY    = "RETRY"
    FALLBACK = "FALLBACK"
    SKIP     = "SKIP"
    ESCALATE = "ESCALATE"
    ABORT    = "ABORT"


# ── transient / permanent signals ───────────────────────────────────────────────

_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError)
_PERMANENT_TYPES: tuple[type[BaseException], ...] = (
    ValueError, TypeError, KeyError, PermissionError,
    FileNotFoundError, NotImplementedError,
)
_TRANSIENT_KEYWORDS: tuple[str, ...] = (
    "timeout", "timed out", "temporarily", "rate limit", "ratelimit",
    "throttle", "429", "503", "502", "504", "overloaded", "unavailable",
    "connection reset", "reset by peer", "try again",
)
_PERMANENT_KEYWORDS: tuple[str, ...] = (
    "not found", "invalid", "unauthorized", "forbidden", "permission denied",
    "malformed", "does not exist", "no such", "schema", "bad request",
    "400", "401", "403", "404", "unsupported",
)


# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class RecoveryAction:
    """The recovery decision for a single failed step.

    Attributes:
        action: What the executor should do (RETRY / FALLBACK / SKIP /
            ESCALATE / ABORT).
        retry_hint: Human-readable guidance for a RETRY (None otherwise).
        fallback_tool: Tool to use for a FALLBACK (None otherwise).
        partial_result: Progress accumulated before the failure (list of prior
            tool results); None for RETRY since nothing is finalised yet.
        backoff_seconds: How long to wait before a RETRY (0.0 otherwise).
        failure_class: The classification that produced this action.
        reason: Short explanation of the decision.
    """
    action:          RecoveryActionType
    retry_hint:      str | None        = None
    fallback_tool:   str | None        = None
    partial_result:  Any               = None
    backoff_seconds: float             = 0.0
    failure_class:   FailureClass | None = None
    reason:          str               = ""


@dataclass
class StepRecord:
    """A completed step, passed to recover_task to preserve its work.

    Attributes:
        id: Step identifier.
        tool: Tool that produced the result (the fallback tool if recovered).
        result: The step's output (reused, never recomputed).
        recovered_via: Fallback tool name if the step was recovered, else None.
        partial: True if the result came from a fallback / degraded path.
    """
    id:            int
    tool:          str
    result:        Any        = None
    recovered_via: str | None = None
    partial:       bool       = False


@dataclass
class TaskRecoveryPlan:
    """Task-level plan that preserves completed work and resumes the run.

    Attributes:
        task_id: The task this plan applies to.
        action: One of "COMPLETE", "PARTIAL_COMPLETE", "RESUME", "ESCALATE",
            "ABORT".
        preserved_steps: Ids of completed steps — reused, NOT re-executed.
        resume_from_step: Next step id to execute, or None when the task is
            complete / escalated / aborted.
        skipped_steps: Ids abandoned during recovery.
        partial_result: Collected results of the preserved steps.
        summary: One-line human-readable description of the plan.
    """
    task_id:          str
    action:           str
    preserved_steps:  list[int]
    resume_from_step: int | None
    skipped_steps:    list[int]
    partial_result:   Any
    summary:          str


# ── engine ──────────────────────────────────────────────────────────────────────

class RecoveryEngine:
    """Step- and task-level recovery state machine (stdlib only).

    Args:
        max_step_retries: Maximum RETRY actions issued for a single step before
            falling back or escalating.
        fallback_tools: Mapping of tool_name -> fallback tool_name, used when a
            transient failure exhausts its retry budget.
        base_backoff_seconds: Base delay for exponential backoff; the delay for
            retry ``n`` (0-indexed) is ``base_backoff_seconds * 2**n``.

    The engine is stateful across a single run: it tracks the number of retries
    issued per step and the last action taken per step. Call ``reset`` to reuse
    it for a fresh task.

    Usage::

        engine = RecoveryEngine(max_step_retries=2,
                                fallback_tools={"database_query": "cache_lookup"})
        action = engine.recover_step(step_id=2, error=err, state=state)
        if action.action is RecoveryActionType.FALLBACK:
            ...  # run action.fallback_tool instead
    """

    def __init__(
        self,
        max_step_retries:     int = 2,
        fallback_tools:       dict[str, str] | None = None,
        base_backoff_seconds: float = 0.5,
    ) -> None:
        self.max_step_retries     = max_step_retries
        self.fallback_tools       = dict(fallback_tools or {})
        self.base_backoff_seconds = base_backoff_seconds
        self._retries:     dict[int, int] = {}
        self._last_action: dict[int, RecoveryActionType] = {}

    def reset(self) -> None:
        """Clear per-step retry counters and recorded actions for a new task."""
        self._retries.clear()
        self._last_action.clear()

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_failure(
        self, error: Exception, tool_name: str, attempt: int
    ) -> FailureClass:
        """Classify a failure as TRANSIENT, PERMANENT, or AMBIGUOUS.

        Classification uses the exception type first, then case-insensitive
        keyword matching on the error message. Transient signals are checked
        before permanent ones so that, e.g., a timeout is treated as transient
        even if its message also contains a generic word.

        Args:
            error: The exception raised by the tool call.
            tool_name: Name of the tool that failed (for caller context).
            attempt: Number of retries already issued for this step (0-based).

        Returns:
            The FailureClass for this error.
        """
        message = str(error).lower()

        if isinstance(error, _TRANSIENT_TYPES) or _matches(message, _TRANSIENT_KEYWORDS):
            return FailureClass.TRANSIENT
        if isinstance(error, _PERMANENT_TYPES) or _matches(message, _PERMANENT_KEYWORDS):
            return FailureClass.PERMANENT
        return FailureClass.AMBIGUOUS

    # ------------------------------------------------------------------
    # Step-level recovery
    # ------------------------------------------------------------------

    def recover_step(
        self, step_id: int, error: Exception, state: AgentState
    ) -> RecoveryAction:
        """Decide how to recover a single failed step.

        The step's tool and blocking flag are read from
        ``state.metadata["steps"]`` (see the module docstring). The decision
        follows the state machine:

          TRANSIENT, retries remaining  -> RETRY (exponential backoff)
          TRANSIENT, budget exhausted   -> FALLBACK (if configured) else
                                           ESCALATE (blocking) / SKIP
          PERMANENT                     -> SKIP (non-blocking) / ESCALATE (blocking)
          AMBIGUOUS, first time         -> RETRY once
          AMBIGUOUS, after one retry    -> ESCALATE

        Args:
            step_id: Identifier of the failed step.
            error: The exception raised.
            state: Current agent state (source of the step plan and the
                accumulated tool results used as the partial result).

        Returns:
            A RecoveryAction describing the chosen recovery.
        """
        step      = self._step_meta(state).get(step_id, {})
        tool_name = step.get("tool", "unknown")
        blocking  = bool(step.get("blocking", True))
        attempt   = self._retries.get(step_id, 0)          # retries already issued
        fclass    = self.classify_failure(error, tool_name, attempt)
        progress  = list(state.tool_results)

        if fclass is FailureClass.TRANSIENT:
            if attempt < self.max_step_retries:
                self._retries[step_id] = attempt + 1
                backoff = round(self.base_backoff_seconds * (2 ** attempt), 3)
                return self._record(step_id, RecoveryAction(
                    action=RecoveryActionType.RETRY,
                    retry_hint=f"retry {attempt + 1}/{self.max_step_retries}, "
                               f"backoff {backoff}s",
                    backoff_seconds=backoff,
                    failure_class=fclass,
                    reason=f"transient failure on '{tool_name}' within retry budget",
                ))
            fallback = self.fallback_tools.get(tool_name)
            if fallback:
                return self._record(step_id, RecoveryAction(
                    action=RecoveryActionType.FALLBACK,
                    fallback_tool=fallback,
                    partial_result=progress,
                    failure_class=fclass,
                    reason=f"retries exhausted on '{tool_name}'; "
                           f"falling back to '{fallback}'",
                ))
            return self._record(step_id, self._skip_or_escalate(
                blocking, progress, fclass,
                reason=f"transient retries exhausted, no fallback for '{tool_name}'",
            ))

        if fclass is FailureClass.PERMANENT:
            return self._record(step_id, self._skip_or_escalate(
                blocking, progress, fclass,
                reason=f"permanent failure on '{tool_name}'",
            ))

        # AMBIGUOUS: one diagnostic retry, then escalate.
        if attempt < 1:
            self._retries[step_id] = attempt + 1
            return self._record(step_id, RecoveryAction(
                action=RecoveryActionType.RETRY,
                retry_hint=f"retry 1/1 (diagnostic), backoff "
                           f"{round(self.base_backoff_seconds, 3)}s",
                backoff_seconds=round(self.base_backoff_seconds, 3),
                failure_class=fclass,
                reason=f"ambiguous failure on '{tool_name}'; one diagnostic retry",
            ))
        return self._record(step_id, RecoveryAction(
            action=RecoveryActionType.ESCALATE,
            partial_result=progress,
            failure_class=fclass,
            reason=f"ambiguous failure on '{tool_name}' persisted after retry",
        ))

    # ------------------------------------------------------------------
    # Task-level recovery
    # ------------------------------------------------------------------

    def recover_task(
        self,
        task_id:         str,
        completed_steps: list[StepRecord] | list[dict[str, Any]],
        failed_step:     int,
    ) -> TaskRecoveryPlan:
        """Build a task-level recovery plan that preserves completed work.

        Completed steps are recorded as preserved and are NOT re-executed. The
        plan's action is derived from how ``failed_step`` resolved:

          - failed_step is among the completed steps (recovered, e.g. via
            fallback) -> "PARTIAL_COMPLETE" if any completed step is partial,
            else "COMPLETE".
          - last action for failed_step was SKIP -> "PARTIAL_COMPLETE",
            resuming after it.
          - ESCALATE / ABORT -> the corresponding terminal action.
          - otherwise -> "RESUME" from failed_step.

        Args:
            task_id: The task identifier.
            completed_steps: StepRecords (or equivalent dicts) for finished
                steps, in execution order.
            failed_step: Id of the step that triggered recovery.

        Returns:
            A TaskRecoveryPlan with preserved steps, resume point, and the
            collected partial result.
        """
        preserved:  list[int] = []
        partial_results: list[dict[str, Any]] = []
        any_partial = False
        for record in completed_steps:
            sid, result, is_partial, tool = _read_step(record)
            preserved.append(sid)
            partial_results.append({"step": sid, "tool": tool, "result": result})
            any_partial = any_partial or is_partial

        last = self._last_action.get(failed_step)

        if failed_step in preserved:
            action      = "PARTIAL_COMPLETE" if any_partial else "COMPLETE"
            resume_from = None
            skipped: list[int] = []
            summary = f"{len(preserved)} step(s) completed"
            if any_partial:
                summary += f"; step {failed_step} recovered via fallback (partial data)"
        elif last is RecoveryActionType.SKIP:
            action      = "PARTIAL_COMPLETE"
            resume_from = failed_step + 1
            skipped     = [failed_step]
            summary     = (f"step {failed_step} skipped (non-blocking); "
                           f"resume at step {resume_from}")
        elif last is RecoveryActionType.ESCALATE:
            action, resume_from, skipped = "ESCALATE", failed_step, []
            summary = f"step {failed_step} escalated; task paused for operator review"
        elif last is RecoveryActionType.ABORT:
            action, resume_from, skipped = "ABORT", None, []
            summary = f"task aborted at step {failed_step}; recovery not possible"
        else:
            action, resume_from, skipped = "RESUME", failed_step, []
            summary = f"resume task at step {failed_step}"

        return TaskRecoveryPlan(
            task_id=task_id,
            action=action,
            preserved_steps=preserved,
            resume_from_step=resume_from,
            skipped_steps=skipped,
            partial_result=partial_results,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _step_meta(state: AgentState) -> dict[int, dict[str, Any]]:
        """Index the run's step plan from state.metadata by step id."""
        return {s["id"]: s for s in state.metadata.get("steps", [])}

    @staticmethod
    def _skip_or_escalate(
        blocking: bool, progress: Any, fclass: FailureClass, reason: str
    ) -> RecoveryAction:
        """SKIP a non-blocking step, ESCALATE a blocking one."""
        action = RecoveryActionType.ESCALATE if blocking else RecoveryActionType.SKIP
        tail = "blocking step -> escalate" if blocking else "non-blocking step -> skip"
        return RecoveryAction(
            action=action,
            partial_result=progress,
            failure_class=fclass,
            reason=f"{reason}; {tail}",
        )

    def _record(self, step_id: int, action: RecoveryAction) -> RecoveryAction:
        """Remember the last action taken for a step and return it."""
        self._last_action[step_id] = action.action
        return action


# ── module helpers ────────────────────────────────────────────────────────────

def _matches(message: str, keywords: tuple[str, ...]) -> bool:
    """True if any keyword is a substring of the (lower-cased) message."""
    return any(kw in message for kw in keywords)


def _read_step(record: StepRecord | dict[str, Any]) -> tuple[int, Any, bool, str]:
    """Extract (id, result, partial, tool) from a StepRecord or dict."""
    if isinstance(record, StepRecord):
        return record.id, record.result, record.partial, record.tool
    return (
        record["id"],
        record.get("result"),
        bool(record.get("partial", False)),
        record.get("tool", "unknown"),
    )


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Book scenario: a 4-step research pipeline. Step 2 (database_query) keeps
    # timing out (transient); after exhausting its retry budget the engine
    # falls back to cache_lookup, which returns stale data — the task completes
    # with a partial result, and no completed step is re-executed.

    SEP = "=" * 60

    STEPS: list[dict[str, Any]] = [
        {"id": 1, "tool": "web_search",      "blocking": True},
        {"id": 2, "tool": "database_query",  "blocking": True},
        {"id": 3, "tool": "summarize",       "blocking": True},
        {"id": 4, "tool": "generate_report", "blocking": True},
    ]

    # Simulated tool outcomes (database_query always times out).
    def run_tool(name: str) -> str:
        if name == "database_query":
            raise TimeoutError("database_query timed out after 5000ms")
        return {
            "web_search":      "3 sources found",
            "cache_lookup":    "cached AAPL prior close $188.10 (stale)",
            "summarize":       "summary drafted",
            "generate_report": "report.pdf generated",
        }[name]

    engine = RecoveryEngine(
        max_step_retries=2,
        fallback_tools={"database_query": "cache_lookup"},
    )
    state = AgentState(
        task="Research AAPL and produce a report.",
        metadata={"steps": STEPS, "max_turns": 10, "budget_usd": 0.05},
    )

    print(f"\n{SEP}")
    print("  RECOVERY ENGINE  |  task=research_pipeline  |  4 steps")
    print(f"  max_step_retries={engine.max_step_retries}  "
          f"fallback_tools={engine.fallback_tools}")
    print(SEP)

    completed: list[StepRecord] = []
    failed_step = -1

    for step in STEPS:
        sid          = step["id"]
        original     = step["tool"]
        current_tool = original
        while True:
            try:
                result = run_tool(current_tool)
            except Exception as exc:                       # noqa: BLE001
                failed_step = sid
                print(f"  [step {sid}] {current_tool:<16} FAIL   "
                      f"{type(exc).__name__}: {exc}")
                action = engine.recover_step(sid, exc, state)
                if action.action is RecoveryActionType.RETRY:
                    print(f"           classify={action.failure_class.value:<9} "
                          f"action=RETRY     {action.retry_hint}")
                    continue
                if action.action is RecoveryActionType.FALLBACK:
                    print(f"           classify={action.failure_class.value:<9} "
                          f"action=FALLBACK  -> {action.fallback_tool}")
                    current_tool = action.fallback_tool
                    continue
                print(f"           classify={action.failure_class.value:<9} "
                      f"action={action.action.value}  {action.reason}")
                break

            # success
            partial = current_tool != original
            state.tool_results.append(
                {"step": sid, "tool": current_tool, "content": result}
            )
            completed.append(StepRecord(
                id=sid, tool=current_tool, result=result,
                recovered_via=current_tool if partial else None, partial=partial,
            ))
            tag = "   [partial]" if partial else ""
            print(f"  [step {sid}] {current_tool:<16} OK     {result}{tag}")
            break

    plan = engine.recover_task("research_pipeline", completed, failed_step)

    print("-" * 60)
    print(f"  TASK RECOVERY PLAN  (task={plan.task_id})")
    print("-" * 60)
    print(f"  action           : {plan.action}")
    print(f"  preserved steps  : {plan.preserved_steps}   (completed -- NOT re-executed)")
    print(f"  resume from step : {plan.resume_from_step if plan.resume_from_step is not None else '-'}")
    print(f"  skipped steps    : {plan.skipped_steps}")
    print(f"  partial result   : {'YES' if any(r['tool'] == 'cache_lookup' for r in plan.partial_result) else 'NO'}")
    print(f"  summary          : {plan.summary}")
    print(SEP)
    print("  progress preserved: step 2's fallback result and steps 1, 3, 4 are")
    print("  reused as-is; no completed step was re-run during recovery.")
    print(SEP)
