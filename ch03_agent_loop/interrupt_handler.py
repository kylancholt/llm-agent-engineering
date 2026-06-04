"""
Human-in-the-loop interrupt management with interactive review and clean resume.

Interrupt types: HUMAN_REVIEW, BUDGET_ALERT, QUALITY_CHECK, MANUAL_STOP.

Typical loop integration:

    handler = InterruptHandler()
    handler.register_interrupt(
        InterruptType.BUDGET_ALERT,
        condition_fn=lambda s: (
            s.total_cost_usd / s.metadata.get("budget_usd", 1) >= 0.8,
            f"80%+ budget consumed",
        ),
    )
    for each turn:
        ...execute turn...
        event = handler.check(state)
        if event:
            decision = handler.handle_interrupt(event)
            state    = handler.resume(state, decision)
            if state.metadata.get("aborted"):
                break

Every interrupt and resume decision is appended to an JSONL audit log.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

# ── project root: ch03_agent_loop/../ = root ─────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ch03_agent_loop.agent_state import AgentState


# ── public enums ──────────────────────────────────────────────────────────────

class InterruptType(Enum):
    """Category of an interrupt event."""

    HUMAN_REVIEW  = "HUMAN_REVIEW"   # general human oversight request
    BUDGET_ALERT  = "BUDGET_ALERT"   # cost threshold exceeded
    QUALITY_CHECK = "QUALITY_CHECK"  # output quality gate triggered
    MANUAL_STOP   = "MANUAL_STOP"    # explicit operator pause


class ResumeAction(Enum):
    """Decision made by the human reviewer."""

    CONTINUE     = "CONTINUE"      # proceed unchanged
    MODIFY_TASK  = "MODIFY_TASK"   # change the task and continue
    ABORT        = "ABORT"         # stop the agent


# ── public data types ─────────────────────────────────────────────────────────

@dataclass
class InterruptEvent:
    """Describes a triggered interrupt.

    Attributes:
        interrupt_type:       Category of the interrupt.
        triggered_at_turn:    Turn number when the condition fired.
        reason:               Human-readable explanation from the condition function.
        state_snapshot_path:  Path to the checkpoint saved at trigger time.
        resume_instructions:  Guidance shown to the human reviewer.
    """

    interrupt_type:       InterruptType
    triggered_at_turn:    int
    reason:               str
    state_snapshot_path:  str
    resume_instructions:  str


@dataclass
class ResumeDecision:
    """Human reviewer's decision after examining an InterruptEvent.

    Attributes:
        action:         What the agent loop should do next.
        modified_task:  New task string; only used when action == MODIFY_TASK.
        human_notes:    Free-text notes from the reviewer; always logged.
    """

    action:        ResumeAction
    modified_task: str | None
    human_notes:   str


# ── private registration record ───────────────────────────────────────────────

@dataclass
class _Trigger:
    """Internal record for one registered interrupt trigger."""

    interrupt_type:      InterruptType
    condition_fn:        Callable[[AgentState], tuple[bool, str]]
    callback_fn:         Callable[[InterruptEvent], None] | None
    resume_instructions: str
    fire_once:           bool
    fired:               bool = field(default=False, repr=False)


# ── handler ───────────────────────────────────────────────────────────────────

class InterruptHandler:
    """Manages human-in-the-loop interrupts for an agent loop.

    Triggers are registered with register_interrupt() and evaluated each turn
    by check(). When a condition fires the state is checkpointed, an
    InterruptEvent is returned, and handle_interrupt() prompts the operator for
    a ResumeDecision. resume() applies that decision to the AgentState.

    All events are appended to an JSONL audit log.

    Args:
        log_path:       Path to the JSONL audit file (default "interrupts.jsonl").
        checkpoint_dir: Directory where interrupt-triggered snapshots are saved
                        (default "checkpoints").
    """

    def __init__(
        self,
        log_path:       str | Path = "interrupts.jsonl",
        checkpoint_dir: str | Path = "checkpoints",
    ) -> None:
        self._log_path      = Path(log_path)
        self._checkpoint_dir = Path(checkpoint_dir)
        self._triggers:      list[_Trigger] = []
        self._event_history: list[InterruptEvent] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_interrupt(
        self,
        interrupt_type:      InterruptType,
        condition_fn:        Callable[[AgentState], tuple[bool, str]],
        callback_fn:         Callable[[InterruptEvent], None] | None = None,
        resume_instructions: str = "Review the current state and decide how to proceed.",
        fire_once:           bool = True,
    ) -> None:
        """Register a trigger that will be evaluated every turn.

        Args:
            interrupt_type:      Category logged in the audit trail.
            condition_fn:        Called with the current AgentState each turn.
                                 Must return (should_fire: bool, reason: str).
                                 Return (False, "") when the condition is not met.
            callback_fn:         Optional side-effect function called immediately
                                 when the interrupt fires (e.g. send a Slack alert).
            resume_instructions: Guidance displayed to the human reviewer in the
                                 handle_interrupt() UI.
            fire_once:           If True (default) the trigger fires at most once
                                 per InterruptHandler instance; re-register to reset.
        """
        self._triggers.append(
            _Trigger(
                interrupt_type=interrupt_type,
                condition_fn=condition_fn,
                callback_fn=callback_fn,
                resume_instructions=resume_instructions,
                fire_once=fire_once,
            )
        )

    # ------------------------------------------------------------------
    # Per-turn check
    # ------------------------------------------------------------------

    def check(self, state: AgentState) -> InterruptEvent | None:
        """Evaluate all registered triggers against the current state.

        Called once per agent turn. Returns the first trigger that fires,
        or None if no conditions are met. When a trigger fires:
          - The current state is checkpointed to disk.
          - The event is appended to the JSONL audit log.
          - The optional callback_fn is invoked.

        Args:
            state: Current AgentState (read-only from the handler's perspective;
                   only state.save() is called to create the snapshot).

        Returns:
            InterruptEvent if a trigger fires; None otherwise.
        """
        for trigger in self._triggers:
            if trigger.fire_once and trigger.fired:
                continue
            try:
                should_fire, reason = trigger.condition_fn(state)
            except Exception as exc:
                reason      = f"condition_fn raised {type(exc).__name__}: {exc}"
                should_fire = False

            if not should_fire:
                continue

            trigger.fired = True
            snapshot_path = self._save_snapshot(state)

            event = InterruptEvent(
                interrupt_type=trigger.interrupt_type,
                triggered_at_turn=state.turn_count,
                reason=reason,
                state_snapshot_path=snapshot_path,
                resume_instructions=trigger.resume_instructions,
            )
            self._event_history.append(event)
            self._append_log({
                "event_type":          "interrupt_triggered",
                "interrupt_type":      event.interrupt_type.value,
                "triggered_at_turn":   event.triggered_at_turn,
                "reason":              event.reason,
                "state_snapshot_path": event.state_snapshot_path,
            })

            if trigger.callback_fn is not None:
                try:
                    trigger.callback_fn(event)
                except Exception:
                    pass  # never let a callback crash the agent

            return event

        return None

    # ------------------------------------------------------------------
    # Interactive review
    # ------------------------------------------------------------------

    def handle_interrupt(self, event: InterruptEvent) -> ResumeDecision:
        """Present the interrupt to the operator and collect a ResumeDecision.

        Renders a summary panel, then prompts for:
          [1] CONTINUE     — proceed unchanged
          [2] MODIFY_TASK  — enter a new task string and continue
          [3] ABORT        — stop the agent loop

        Args:
            event: The InterruptEvent returned by check().

        Returns:
            ResumeDecision with action, optional modified task, and human notes.
        """
        sep = "+" + "=" * 62 + "+"
        print(f"\n{sep}")
        print(f"|  INTERRUPT  {event.interrupt_type.value:<48}|")
        print(sep)
        print(f"|  Turn          : {event.triggered_at_turn:<44}|")
        reason_lines = _wrap(event.reason, 44)
        print(f"|  Reason        : {reason_lines[0]:<44}|")
        for line in reason_lines[1:]:
            print(f"|                  {line:<44}|")
        snap = Path(event.state_snapshot_path).name
        print(f"|  State saved   : {snap:<44}|")
        instr_lines = _wrap(event.resume_instructions, 44)
        print(f"|  Instructions  : {instr_lines[0]:<44}|")
        for line in instr_lines[1:]:
            print(f"|                  {line:<44}|")
        print(f"+{'-' * 62}+")
        print("|  Options:                                                    |")
        print("|    [1] CONTINUE     -- proceed with current task             |")
        print("|    [2] MODIFY_TASK  -- change the task and continue          |")
        print("|    [3] ABORT        -- stop the agent here                   |")
        print(f"{sep}")

        # ── action choice ─────────────────────────────────────────────
        action_map = {"1": ResumeAction.CONTINUE, "2": ResumeAction.MODIFY_TASK,
                      "3": ResumeAction.ABORT}
        while True:
            try:
                raw = input("\n  Decision [1/2/3]: ").strip()
            except EOFError:
                raw = "1"
            if raw in action_map:
                action = action_map[raw]
                break
            print(f"  Invalid choice '{raw}' — enter 1, 2, or 3.")

        # ── modified task ──────────────────────────────────────────────
        modified_task: str | None = None
        if action == ResumeAction.MODIFY_TASK:
            try:
                new_task = input("  New task (Enter to keep current): ").strip()
            except EOFError:
                new_task = ""
            modified_task = new_task if new_task else None

        # ── human notes ────────────────────────────────────────────────
        try:
            notes = input("  Notes (optional, press Enter to skip): ").strip()
        except EOFError:
            notes = ""

        decision = ResumeDecision(
            action=action,
            modified_task=modified_task,
            human_notes=notes,
        )
        self._append_log({
            "event_type":      "resume_decision",
            "action":          decision.action.value,
            "modified_task":   decision.modified_task,
            "human_notes":     decision.human_notes,
            "at_turn":         event.triggered_at_turn,
        })
        return decision

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def resume(
        self,
        state:    AgentState,
        decision: ResumeDecision,
    ) -> AgentState:
        """Apply a ResumeDecision to the current AgentState.

        CONTINUE     — updates metadata and timestamp; task unchanged.
        MODIFY_TASK  — replaces state.task, appends a notification message to
                       the conversation history, updates metadata.
        ABORT        — sets metadata["aborted"] = True and records the reason.

        The updated state is checkpointed and the resume is logged.

        Args:
            state:    Current AgentState (mutated in-place).
            decision: ResumeDecision returned by handle_interrupt().

        Returns:
            The same AgentState object with the decision applied.
        """
        now = time.time()

        if decision.action == ResumeAction.ABORT:
            state.metadata["aborted"]     = True
            state.metadata["abort_notes"] = decision.human_notes or "No notes provided."
            state.last_updated            = now
            self._append_log({
                "event_type":  "resume",
                "action":      "ABORT",
                "human_notes": decision.human_notes,
                "at_turn":     state.turn_count,
            })
            return state

        if decision.action == ResumeAction.MODIFY_TASK and decision.modified_task:
            old_task   = state.task
            state.task = decision.modified_task
            note_text  = (
                f"[Human reviewer modified the task at turn {state.turn_count}]\n"
                f"Previous task: {old_task}\n"
                f"New task: {decision.modified_task}"
            )
            if decision.human_notes:
                note_text += f"\nReviewer notes: {decision.human_notes}"
            state.messages.append({"role": "user", "content": note_text})

        # Common bookkeeping for CONTINUE and MODIFY_TASK
        notes_list: list[str] = state.metadata.setdefault("human_review_notes", [])
        if decision.human_notes:
            notes_list.append(decision.human_notes)

        state.metadata["last_resume_action"] = decision.action.value
        state.last_updated = now

        # Checkpoint the resumed state
        resumed_path = self._save_snapshot(state)
        self._append_log({
            "event_type":          "resume",
            "action":              decision.action.value,
            "modified_task":       decision.modified_task,
            "human_notes":         decision.human_notes,
            "at_turn":             state.turn_count,
            "resumed_snapshot":    resumed_path,
        })
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_snapshot(self, state: AgentState) -> str:
        """Save a checkpoint and return the path."""
        stem = f"interrupt_t{state.turn_count:03d}"
        return state.save(str(self._checkpoint_dir / stem))

    def _append_log(self, entry: dict[str, Any]) -> None:
        """Append a timestamped JSONL entry to the audit log."""
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


# ── module-level helpers ──────────────────────────────────────────────────────

def _wrap(text: str, width: int) -> list[str]:
    """Break text into lines of at most `width` characters (word-aware)."""
    if len(text) <= width:
        return [text]
    lines: list[str] = []
    while len(text) > width:
        cut = text.rfind(" ", 0, width)
        cut = cut if cut > 0 else width
        lines.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        lines.append(text)
    return lines


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    BUDGET    = 0.05
    MAX_TURNS = 10
    MODEL     = "claude-sonnet-4-6"

    # ── setup ─────────────────────────────────────────────────────────────────
    state = AgentState(
        task="Find the AAPL stock price, calculate a 50-share portfolio value, "
             "and write a one-paragraph investment note.",
        metadata={"budget_usd": BUDGET, "max_turns": MAX_TURNS, "model": MODEL},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file  = f"{tmpdir}/interrupts.jsonl"
        ckpt_dir  = f"{tmpdir}/checkpoints"

        handler = InterruptHandler(log_path=log_file, checkpoint_dir=ckpt_dir)

        # Register a HUMAN_REVIEW interrupt that fires when budget crosses 50%
        handler.register_interrupt(
            interrupt_type=InterruptType.HUMAN_REVIEW,
            condition_fn=lambda s: (
                s.total_cost_usd / s.metadata.get("budget_usd", 1.0) >= 0.50,
                f"Budget {s.total_cost_usd / s.metadata['budget_usd'] * 100:.1f}%"
                f" consumed (${s.total_cost_usd:.4f} of ${s.metadata['budget_usd']:.2f})",
            ),
            resume_instructions=(
                "Budget has crossed 50%. Review spend rate and task progress "
                "before authorising further execution."
            ),
            fire_once=True,
        )

        # Costs that put us past 50% at turn 4:
        # turns 1-3: $0.004 + $0.008 + $0.009 = $0.021  (42%)
        # turn 4:    + $0.006 = $0.027  (54%)  <- triggers
        TURN_COSTS = [0.004, 0.008, 0.009, 0.006, 0.004, 0.003]

        sep = "=" * 68
        print(f"\n{sep}")
        print(f"  InterruptHandler demo  |  budget=${BUDGET:.2f}  |  {MODEL}")
        print(sep)

        for turn_n, turn_cost in enumerate(TURN_COSTS, start=1):
            # Simulate turn work
            state.turn_count     = turn_n
            state.total_cost_usd = round(state.total_cost_usd + turn_cost, 8)
            state.total_tokens  += int(turn_cost / 3e-6)   # rough token estimate
            state.last_updated   = time.time()
            state.messages.extend([
                {"role": "user",      "content": f"[turn {turn_n}] task input"},
                {"role": "assistant", "content": f"[turn {turn_n}] model response"},
            ])

            pct = state.total_cost_usd / BUDGET * 100
            print(f"  [turn {turn_n:>2}]  cost=${state.total_cost_usd:.4f}  "
                  f"({pct:.1f}%)  {state.snapshot()}")

            # Check for interrupts
            event = handler.check(state)
            if event:
                print(f"\n  --> Interrupt fired: {event.interrupt_type.value}")
                decision = handler.handle_interrupt(event)
                state    = handler.resume(state, decision)

                if state.metadata.get("aborted"):
                    print(f"\n  Agent aborted at turn {turn_n}.")
                    print(f"  Notes: {state.metadata.get('abort_notes', '')}")
                    break

                print(f"\n  Resume action   : {decision.action.value}")
                if decision.modified_task:
                    print(f"  New task        : {decision.modified_task}")
                if decision.human_notes:
                    print(f"  Reviewer notes  : {decision.human_notes}")
                print(f"  Continuing from turn {turn_n}...\n")

        # ── show audit log ─────────────────────────────────────────────────
        print(f"\n{sep}")
        print(f"  Audit log: {log_file}")
        print(f"  {'-'*50}")
        from pathlib import Path as _P
        log_content = _P(log_file).read_text(encoding="utf-8") if _P(log_file).exists() else "(empty)"
        for line in log_content.splitlines():
            entry = json.loads(line)
            ts    = entry.pop("timestamp", "")[:19]
            print(f"  {ts}  {json.dumps(entry)}")
        print(sep)
