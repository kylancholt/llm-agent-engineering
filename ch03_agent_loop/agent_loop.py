"""
Production agent loop template — the heart of the book.

State machine:
  IDLE -> RUNNING -> FINAL_ANSWER | BUDGET_EXCEEDED | MAX_TURNS | INTERRUPTED | ERROR

Uses the Anthropic Messages API with *native tool_use* (not the text-based
ReAct format) for reliable, structured tool dispatch. Each turn:
  1. Call the LLM with the current message history and tool definitions.
  2. Execute any tool_use blocks returned by the model.
  3. Record cost with CostGuard (AgentCostTracker).
  4. Emit a structured log line.
  5. Auto-checkpoint state to disk.
  6. Evaluate termination conditions.

Human-in-the-loop interruption is supported via SIGINT (Ctrl-C) or by
calling AgentLoop.interrupt() from another thread.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import signal
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import anthropic

# ── project root: ch03_agent_loop/../ = root ─────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ch01_why_agents_break.cost_tracker import AgentCostTracker, AlertLevel


# ── .env loader ───────────────────────────────────────────────────────────────

def _load_env(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (idempotent)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_load_env(_ROOT / ".env")

# ── structured logger ─────────────────────────────────────────────────────────
logging.basicConfig(format="%(message)s", level=logging.INFO)
_log = logging.getLogger(__name__)

# ── pricing tier map ──────────────────────────────────────────────────────────
_COST_TIER: dict[str, str] = {
    "haiku": "claude-haiku-4-5",
    "opus":  "claude-opus-4-7",
}


def _cost_model(model: str) -> str:
    """Map any Claude model name to the nearest AgentCostTracker pricing tier."""
    lc = model.lower()
    for key, tier in _COST_TIER.items():
        if key in lc:
            return tier
    return "claude-sonnet-4-6"


# ── state machine types ───────────────────────────────────────────────────────

class MachineState(Enum):
    """Lifecycle state of the agent loop."""

    IDLE    = "IDLE"
    RUNNING = "RUNNING"


class TerminationReason(Enum):
    """How and why the loop exited RUNNING state."""

    FINAL_ANSWER    = "FINAL_ANSWER"     # model finished without pending tool calls
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"  # CostGuard triggered HALT
    MAX_TURNS       = "MAX_TURNS"        # turn counter hit the hard limit
    INTERRUPTED     = "INTERRUPTED"      # SIGINT or programmatic interrupt
    ERROR           = "ERROR"            # unrecoverable API or runtime error


# ── public data types ─────────────────────────────────────────────────────────

@dataclass
class LoopState:
    """Full mutable loop state — serialisable for checkpointing and resumption.

    Attributes:
        task: Original task string.
        turn: Current (or last completed) turn number.
        messages: Serialised message history (JSON-safe dicts throughout).
        total_cost_usd: Cumulative USD spent so far.
        total_input_tokens: Cumulative input tokens across all turns.
        total_output_tokens: Cumulative output tokens across all turns.
        machine_state: Current MachineState enum value.
        last_tool_names: Tool names invoked in the most recent turn.
        last_text: Last text block emitted by the model.
    """

    task:                 str
    turn:                 int
    messages:             list[dict[str, Any]]
    total_cost_usd:       float
    total_input_tokens:   int
    total_output_tokens:  int
    machine_state:        MachineState
    last_tool_names:      list[str]
    last_text:            str


@dataclass
class LoopResult:
    """Return value of AgentLoop.run().

    Attributes:
        answer: Model's final answer, or last text before non-success termination.
        termination_reason: Why the loop exited.
        turns_used: Total turns completed.
        total_cost_usd: Total USD spent for this run.
        checkpoint_path: Path to the last checkpoint file written, or None.
        state_at_termination: Full LoopState at the moment the loop ended.
    """

    answer:                str
    termination_reason:    TerminationReason
    turns_used:            int
    total_cost_usd:        float
    checkpoint_path:       str | None
    state_at_termination:  LoopState


# ── agent loop ────────────────────────────────────────────────────────────────

class AgentLoop:
    """Production agent loop with state machine, cost guard, and auto-checkpoint.

    Native tool_use (Anthropic's structured function-calling) is used instead
    of text-based ReAct parsing, making tool dispatch unambiguous and robust.

    Termination conditions (checked in this order each turn):
      FINAL_ANSWER    stop_reason == "end_turn" with no pending tool calls
      INTERRUPTED     SIGINT or AgentLoop.interrupt() was called
      BUDGET_EXCEEDED CostGuard reports >= 95 % of budget_usd consumed
      MAX_TURNS       turn counter reached max_turns

    Args:
        model: Anthropic model ID (default "claude-sonnet-4-6").
        budget_usd: Per-run USD cap; halts at 95 % consumed (default $0.10).
        max_turns: Hard limit on tool-use cycles (default 10).
        checkpoint_dir: Directory for per-turn checkpoint files (default "checkpoints").
        max_output_tokens: Max tokens per model response (default 4096).
    """

    def __init__(
        self,
        model:              str        = "claude-sonnet-4-6",
        budget_usd:         float      = 0.10,
        max_turns:          int        = 10,
        checkpoint_dir:     str | Path = "checkpoints",
        max_output_tokens:  int        = 4096,
    ) -> None:
        self.model             = model
        self.budget_usd        = budget_usd
        self.max_turns         = max_turns
        self.checkpoint_dir    = Path(checkpoint_dir)
        self.max_output_tokens = max_output_tokens
        self._client           = anthropic.Anthropic()
        self._interrupted      = threading.Event()
        self._register_sigint()

    def interrupt(self) -> None:
        """Request graceful termination after the current turn completes."""
        self._interrupted.set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        task:  str,
        tools: dict[str, Callable[..., str]],
    ) -> LoopResult:
        """Execute the agent loop for the given task.

        Builds Anthropic tool definitions from the callables, then runs the
        IDLE -> RUNNING -> <terminal> state machine, checkpointing after each
        non-terminal turn.

        Args:
            task: Natural-language task description.
            tools: Mapping of tool_name -> callable(**kwargs) -> str. Each
                   callable's first docstring line is used as its description.

        Returns:
            LoopResult with answer, termination reason, token totals, cost,
            path to the last checkpoint, and the final LoopState.
        """
        self._interrupted.clear()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        tracker   = AgentCostTracker(budget_usd=self.budget_usd, model=_cost_model(self.model))
        tool_defs = [_build_tool_def(name, fn) for name, fn in tools.items()]

        state = LoopState(
            task=task,
            turn=0,
            messages=[{"role": "user", "content": task}],
            total_cost_usd=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            machine_state=MachineState.RUNNING,
            last_tool_names=[],
            last_text="",
        )

        termination:  TerminationReason = TerminationReason.ERROR
        answer:       str               = ""
        last_ckpt:    str | None        = None

        for turn in range(1, self.max_turns + 1):
            state.turn = turn

            # ── call LLM ──────────────────────────────────────────────────
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_output_tokens,
                    tools=tool_defs,
                    messages=state.messages,
                )
            except anthropic.APIError as exc:
                answer      = f"API error at turn {turn}: {exc}"
                termination = TerminationReason.ERROR
                _log.error("[turn %2d] ERROR %s", turn, exc)
                break

            # ── cost accounting ───────────────────────────────────────────
            in_tok  = resp.usage.input_tokens
            out_tok = resp.usage.output_tokens
            status  = tracker.record_turn(in_tok, out_tok, turn_id=turn)
            state.total_cost_usd      = tracker.get_report().total_cost_usd
            state.total_input_tokens  += in_tok
            state.total_output_tokens += out_tok

            # ── extract content ───────────────────────────────────────────
            tool_calls  = [b for b in resp.content if b.type == "tool_use"]
            text_blocks = [b for b in resp.content if b.type == "text"]
            state.last_text       = " ".join(b.text for b in text_blocks).strip()
            state.last_tool_names = [b.name for b in tool_calls]

            # ── structured log ────────────────────────────────────────────
            remaining = max(0.0, self.budget_usd - state.total_cost_usd)
            tools_str = ",".join(state.last_tool_names) or "(none)"
            _log.info(
                "[turn %2d] tool=%-22s tokens=%5d  cost=$%.5f  budget_remaining=$%.5f",
                turn, tools_str,
                in_tok + out_tok,
                status.turn_cost_usd,
                remaining,
            )

            # ── append assistant message to history ────────────────────────
            state.messages.append({
                "role":    "assistant",
                "content": _serialise_blocks(resp.content),
            })

            # ── FINAL_ANSWER: end_turn with no tool calls ─────────────────
            if resp.stop_reason == "end_turn" and not tool_calls:
                answer      = state.last_text
                termination = TerminationReason.FINAL_ANSWER
                last_ckpt   = self._checkpoint(state, termination)
                break

            # ── execute tool calls and append results ─────────────────────
            if tool_calls:
                tool_result_blocks: list[dict[str, Any]] = []
                for tc in tool_calls:
                    output = _exec_tool(tc.name, dict(tc.input), tools)
                    tool_result_blocks.append({
                        "type":        "tool_result",
                        "tool_use_id": tc.id,
                        "content":     output,
                    })
                state.messages.append({"role": "user", "content": tool_result_blocks})

            # ── checkpoint after each non-terminal turn ────────────────────
            last_ckpt = self._checkpoint(state, None)

            # ── termination checks for the *next* turn ────────────────────
            if self._interrupted.is_set():
                answer      = state.last_text or "Interrupted by operator."
                termination = TerminationReason.INTERRUPTED
                _log.warning("[turn %2d] INTERRUPTED — halting after this turn.", turn)
                break

            if status.alert_level is AlertLevel.HALT:
                answer      = state.last_text or "Budget exhausted."
                termination = TerminationReason.BUDGET_EXCEEDED
                _log.warning("[turn %2d] BUDGET_EXCEEDED — $%.5f of $%.2f consumed.",
                             turn, state.total_cost_usd, self.budget_usd)
                break

            if turn >= self.max_turns:
                answer      = state.last_text or "Max turns reached."
                termination = TerminationReason.MAX_TURNS
                _log.warning("[turn %2d] MAX_TURNS reached (%d).", turn, self.max_turns)
                break

        state.machine_state = MachineState.IDLE
        _log.info(
            "[done]    reason=%-18s  turns=%d  total=$%.5f  budget=$%.2f",
            termination.value, state.turn, state.total_cost_usd, self.budget_usd,
        )

        return LoopResult(
            answer=answer,
            termination_reason=termination,
            turns_used=state.turn,
            total_cost_usd=state.total_cost_usd,
            checkpoint_path=last_ckpt,
            state_at_termination=state,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _checkpoint(
        self,
        state:       LoopState,
        termination: TerminationReason | None,
    ) -> str:
        """Serialise the current LoopState to a JSON checkpoint file.

        Returns the absolute path of the written file.
        The full message history is saved to allow resumption.
        """
        path = self.checkpoint_dir / f"turn_{state.turn:03d}.json"
        data: dict[str, Any] = {
            "turn":                state.turn,
            "task":                state.task,
            "machine_state":       state.machine_state.value,
            "termination_reason":  termination.value if termination else None,
            "total_cost_usd":      state.total_cost_usd,
            "total_input_tokens":  state.total_input_tokens,
            "total_output_tokens": state.total_output_tokens,
            "last_tool_names":     state.last_tool_names,
            "last_text":           state.last_text,
            "messages":            state.messages,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return str(path)

    def _register_sigint(self) -> None:
        """Register a SIGINT handler to set the interrupt flag gracefully."""
        try:
            signal.signal(signal.SIGINT, self._on_sigint)
        except (ValueError, OSError):
            pass  # called from non-main thread or OS doesn't support it

    def _on_sigint(self, signum: int, frame: Any) -> None:
        _log.warning("\n[interrupt] SIGINT received — will halt after this turn.")
        self._interrupted.set()


# ── module-level helpers ──────────────────────────────────────────────────────

def _build_tool_def(name: str, fn: Callable[..., str]) -> dict[str, Any]:
    """Build an Anthropic tool-definition dict from a Python callable.

    Infers parameter types from type annotations; defaults to 'string'.
    Uses the first docstring line as the tool description.

    Args:
        name: Tool name (as it will appear to the model).
        fn: Python callable to wrap.

    Returns:
        Dict with "name", "description", and "input_schema" keys.
    """
    sig  = inspect.signature(fn)
    doc  = (fn.__doc__ or "").strip()
    desc = doc.splitlines()[0] if doc else name

    _PY_TO_JSON: dict[type, str] = {
        str:   "string",
        int:   "integer",
        float: "number",
        bool:  "boolean",
        list:  "array",
        dict:  "object",
    }

    properties: dict[str, Any] = {}
    required:   list[str]      = []

    for pname, param in sig.parameters.items():
        ann  = param.annotation
        jtype = _PY_TO_JSON.get(ann, "string")
        properties[pname] = {"type": jtype}
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    return {
        "name":         name,
        "description":  desc,
        "input_schema": {
            "type":       "object",
            "properties": properties,
            "required":   required,
        },
    }


def _serialise_blocks(blocks: list[Any]) -> list[dict[str, Any]]:
    """Convert Anthropic SDK content blocks to JSON-serialisable dicts."""
    out: list[dict[str, Any]] = []
    for b in blocks:
        if hasattr(b, "type"):
            if b.type == "text":
                out.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                out.append({
                    "type":  "tool_use",
                    "id":    b.id,
                    "name":  b.name,
                    "input": dict(b.input),
                })
        elif isinstance(b, dict):
            out.append(b)
    return out


def _exec_tool(
    name:   str,
    kwargs: dict[str, Any],
    tools:  dict[str, Callable[..., str]],
) -> str:
    """Look up and run a tool; return its string output or a descriptive error."""
    if name not in tools:
        return f"Error: unknown tool '{name}'. Available: {', '.join(tools)}"
    try:
        return str(tools[name](**kwargs))
    except TypeError as exc:
        return f"Error: bad arguments for '{name}': {exc}"
    except Exception as exc:
        return f"Error ({type(exc).__name__}): {exc}"


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set. Add it to .env and retry.")
        sys.exit(1)

    from ch02_architectures.react_agent import web_search, calculator, read_file

    BUDGET = 0.05
    TASK = (
        "Find the current AAPL stock price and calculate the value of 50 shares. "
        "Also read data.csv to get our latest monthly revenue figure. "
        "Write a two-sentence investment note comparing the 50-share position "
        "to one month of that revenue."
    )

    loop = AgentLoop(
        model="claude-sonnet-4-6",
        budget_usd=BUDGET,
        max_turns=10,
        checkpoint_dir="checkpoints",
    )

    sep = "-" * 70
    print(f"\n{sep}")
    print(f"  AgentLoop production demo")
    print(f"  Budget: ${BUDGET:.2f}  |  Max turns: {loop.max_turns}  |  Model: {loop.model}")
    print(f"  Task  : {TASK}")
    print(f"{sep}")

    result = loop.run(
        task=TASK,
        tools={"web_search": web_search, "calculator": calculator, "read_file": read_file},
    )

    print(f"\n{sep}")
    print("  FINAL ANSWER")
    print(f"{sep}")
    for line in result.answer.splitlines():
        print(f"  {line}")

    pct = (result.total_cost_usd / BUDGET) * 100
    print(f"\n  Termination : {result.termination_reason.value}")
    print(f"  Turns used  : {result.turns_used}")
    print(f"\nTask complete. Total cost: ${result.total_cost_usd:.5f} / budget ${BUDGET:.2f} ({pct:.1f}% used)")
    if result.checkpoint_path:
        print(f"Last checkpoint : {result.checkpoint_path}")
