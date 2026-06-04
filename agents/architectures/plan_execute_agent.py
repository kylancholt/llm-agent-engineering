"""
Plan-and-Execute agent — Anthropic SDK, zero framework dependencies.

Phase 1  Planning  : one Claude call produces a JSON step list with tool names
                     and explicit inter-step dependencies.
Phase 2  Execution : each step resolves its tool input via a Claude call, runs
                     the tool, and retries once on failure with the error in the
                     prompt. Null-tool steps are pure-reasoning calls.

Shares AgentResult with react_agent so callers can treat both agents uniformly.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import anthropic

# ── sys.path: project root → ch01_why_agents_break + agents/ visible ─────────
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from ch01_why_agents_break.cost_tracker import AgentCostTracker, AlertLevel
from agents.architectures.react_agent import (   # shared result type + demo tools
    AgentResult,
    web_search,
    calculator,
    read_file,
)


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

# ── pricing fallback ──────────────────────────────────────────────────────────
_PRICING_MODELS = frozenset({"claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"})


def _cost_model(model: str) -> str:
    """Map any Claude model name to the nearest supported pricing tier."""
    if model in _PRICING_MODELS:
        return model
    lc = model.lower()
    if "haiku" in lc:
        return "claude-haiku-4-5"
    if "opus" in lc:
        return "claude-opus-4-7"
    return "claude-sonnet-4-6"


# ── system prompts ────────────────────────────────────────────────────────────

_PLAN_SYS = """\
You are a strategic planning agent. Break a task into an ordered list of concrete steps.

Output ONLY a valid JSON object — no markdown, no commentary:
{
  "steps": [
    {
      "id": 1,
      "description": "concise description of what this step does",
      "tool": "tool_name or null for reasoning/synthesis",
      "depends_on": [],
      "estimated_tokens": 400
    }
  ]
}

Rules:
- 2 to 6 steps total.
- Each step uses ONE tool from the provided list, or null for reasoning.
- depends_on: IDs of steps that must succeed before this one.
- The LAST step must be a synthesis step (tool: null) that combines all results.
- estimated_tokens: realistic estimate 200-800.
- Only use the exact tool names provided; do not invent new tools.
"""

_INPUT_SYS = (
    "Output ONLY a valid JSON object for the tool input. "
    "Use the exact parameter names from the tool description. "
    "No explanation, no markdown, just the JSON."
)

_REASON_SYS = (
    "You are an analytical agent. Complete the requested reasoning step "
    "concisely using only the provided context."
)


# ── internal data type ────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Execution record for one plan step.

    Attributes:
        step_id: Matches the id in the generated plan.
        description: Step description from the plan.
        tool_name: Tool used, or None for reasoning steps.
        tool_input: Resolved JSON input dict (empty for reasoning steps).
        output: Tool output or reasoning text.
        tokens_used: Input + output tokens for this step's API call(s).
        cost_usd: USD cost for this step.
        success: False if the final attempt raised an error.
        retried: True when the first attempt failed and a retry was made.
        blocked: True when a required dependency failed.
    """

    step_id: int
    description: str
    tool_name: str | None
    tool_input: dict[str, Any]
    output: str
    tokens_used: int
    cost_usd: float
    success: bool
    retried: bool
    blocked: bool


# ── agent ─────────────────────────────────────────────────────────────────────

class PlanExecuteAgent:
    """Plan-and-Execute agent using the Anthropic Messages API.

    Phase 1 — Planning: one Claude call produces a JSON plan with steps,
    tools, and dependencies.

    Phase 2 — Execution: each step calls Claude to resolve the tool input,
    then runs the tool. On error the step retries once with the failure text
    appended to the prompt. Null-tool steps are pure reasoning calls. A step
    whose dependency failed is marked blocked and skipped.

    Args:
        model: Anthropic model ID (default "claude-sonnet-4-6").
        max_steps: Cap on the number of plan steps executed (default 8).
        budget_usd: Per-run spend cap; halts at 95 % consumed (default $0.05).
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_steps: int = 8,
        budget_usd: float = 0.05,
    ) -> None:
        self.model = model
        self.max_steps = max_steps
        self.budget_usd = budget_usd
        self._client = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        tools: dict[str, Callable[..., str]],
    ) -> AgentResult:
        """Plan then execute the task, returning a unified AgentResult.

        Args:
            task: Natural-language description of what to accomplish.
            tools: Mapping of tool_name -> callable(**kwargs) -> str.
                   First docstring line of each callable is shown to the planner.

        Returns:
            AgentResult with final answer, token totals, cost, and full trace
            (planning phase + one entry per executed step).
        """
        tracker = AgentCostTracker(
            budget_usd=self.budget_usd,
            model=_cost_model(self.model),
        )
        trace: list[dict[str, Any]] = []

        # ── Phase 1: plan ─────────────────────────────────────────────────────
        plan = self._generate_plan(task, tools, tracker, trace)
        if plan is None:
            return AgentResult(
                answer="Planning failed — could not parse a valid JSON plan.",
                turns_used=len(trace),
                total_tokens=sum(t.get("tokens", 0) for t in trace),
                cost_usd=tracker.get_report().total_cost_usd,
                success=False,
                trace=trace,
            )

        steps: list[dict[str, Any]] = plan.get("steps", [])[: self.max_steps]

        # ── Phase 2: execute steps ────────────────────────────────────────────
        completed: dict[int, StepResult] = {}
        failed_ids: set[int] = set()

        for step in steps:
            step_id: int = step.get("id", 0)
            depends_on: list[int] = step.get("depends_on", [])

            # Skip if any hard dependency failed
            blocking = [d for d in depends_on if d in failed_ids]
            if blocking:
                br = StepResult(
                    step_id=step_id,
                    description=step.get("description", ""),
                    tool_name=step.get("tool"),
                    tool_input={},
                    output=f"Blocked: step(s) {blocking} failed.",
                    tokens_used=0,
                    cost_usd=0.0,
                    success=False,
                    retried=False,
                    blocked=True,
                )
                failed_ids.add(step_id)
                completed[step_id] = br
                trace.append({
                    "turn_id":     step_id,
                    "thought":     step.get("description", ""),
                    "action":      "BLOCKED",
                    "tool_name":   None,
                    "tool_input":  None,
                    "observation": br.output,
                    "tokens":      0,
                    "cost_usd":    0.0,
                })
                continue

            sr = self._execute_step(step, task, tools, completed, tracker, trace)
            completed[step_id] = sr
            if not sr.success:
                failed_ids.add(step_id)

            # Budget HALT
            if tracker.get_report().budget_used_pct >= 95.0:
                break

        # ── determine final answer ────────────────────────────────────────────
        successful = [r for r in completed.values() if r.success and not r.blocked]
        if successful:
            answer = max(successful, key=lambda r: r.step_id).output
        else:
            answer = "No steps completed successfully."

        report = tracker.get_report()
        return AgentResult(
            answer=answer,
            turns_used=len(trace),
            total_tokens=sum(t.get("tokens", 0) for t in trace),
            cost_usd=report.total_cost_usd,
            success=bool(successful),
            trace=trace,
        )

    # ------------------------------------------------------------------
    # Private: planning
    # ------------------------------------------------------------------

    def _generate_plan(
        self,
        task: str,
        tools: dict[str, Callable[..., str]],
        tracker: AgentCostTracker,
        trace: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Call Claude once to produce a JSON execution plan.

        Returns the parsed plan dict, or None on API / parse failure.
        Appends one entry to trace regardless of outcome.
        """
        tool_desc = "\n".join(
            f"  {name}: {(fn.__doc__ or '').splitlines()[0].strip()}"
            for name, fn in tools.items()
        )
        user_msg = f"Task: {task}\n\nAvailable tools:\n{tool_desc}"

        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=_PLAN_SYS,
                messages=[{"role": "user", "content": user_msg}],
            )
        except anthropic.APIError as exc:
            trace.append(_trace_entry("plan", "Generate plan", "plan", None, None,
                                      f"API error: {exc}", 0, 0.0))
            return None

        in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
        status = tracker.record_turn(in_tok, out_tok, turn_id="plan")
        text = resp.content[0].text.strip()
        plan = _extract_json(text)

        trace.append(_trace_entry(
            "plan", "Generate execution plan", "plan", None, None,
            json.dumps(plan, indent=2) if plan else text[:600],
            in_tok + out_tok, status.turn_cost_usd,
        ))
        return plan

    # ------------------------------------------------------------------
    # Private: step execution
    # ------------------------------------------------------------------

    def _execute_step(
        self,
        step: dict[str, Any],
        task: str,
        tools: dict[str, Callable[..., str]],
        completed: dict[int, StepResult],
        tracker: AgentCostTracker,
        trace: list[dict[str, Any]],
    ) -> StepResult:
        """Resolve input, run tool, retry once on failure, record trace."""
        step_id: int = step.get("id", 0)
        tool_name: str | None = step.get("tool")
        description: str = step.get("description", "")
        context = _format_context(completed)

        # Reasoning-only step (tool = null)
        if not tool_name:
            return self._run_reasoning(step, task, context, tracker, trace)

        # Unknown tool
        if tool_name not in tools:
            err = f"Unknown tool '{tool_name}'. Available: {', '.join(tools)}"
            trace.append(_trace_entry(step_id, description, tool_name, {}, None,
                                      err, 0, 0.0))
            return StepResult(step_id=step_id, description=description,
                              tool_name=tool_name, tool_input={}, output=err,
                              tokens_used=0, cost_usd=0.0, success=False,
                              retried=False, blocked=False)

        cost_before = tracker.get_report().total_cost_usd
        total_tokens = 0
        last_input: dict[str, Any] = {}
        last_output = ""
        retried = False

        for attempt in range(2):          # attempt 0 = first try; 1 = retry
            is_retry = attempt > 0
            if is_retry:
                retry_ctx = f"{context}\n\nPrevious attempt failed: {last_output}"
                ctx = retry_ctx
                retried = True
            else:
                ctx = context

            tool_input, in_tok, out_tok = self._resolve_input(
                step, task, tools[tool_name], ctx, tracker, is_retry
            )
            total_tokens += in_tok + out_tok
            last_input = tool_input
            last_output = _exec_tool(tool_name, tool_input, tools)

            if not last_output.startswith("Error"):
                break   # success — no retry needed

        success = not last_output.startswith("Error")
        cost_after = tracker.get_report().total_cost_usd
        step_cost = round(cost_after - cost_before, 8)

        label = f"{step_id}-retry" if retried and not success else step_id
        trace.append(_trace_entry(label, description, tool_name, last_input,
                                  last_input, last_output, total_tokens, step_cost))

        return StepResult(
            step_id=step_id,
            description=description,
            tool_name=tool_name,
            tool_input=last_input,
            output=last_output,
            tokens_used=total_tokens,
            cost_usd=step_cost,
            success=success,
            retried=retried,
            blocked=False,
        )

    def _resolve_input(
        self,
        step: dict[str, Any],
        task: str,
        tool_fn: Callable[..., str],
        context: str,
        tracker: AgentCostTracker,
        is_retry: bool = False,
    ) -> tuple[dict[str, Any], int, int]:
        """Ask Claude for the JSON input object for this step's tool call.

        Returns (tool_input_dict, input_tokens, output_tokens).
        Falls back to {"input": raw_text} if JSON parsing fails.
        """
        prefix = "RETRY — previous attempt failed. Try a different input.\n\n" if is_retry else ""
        tool_doc = (tool_fn.__doc__ or "").splitlines()[0].strip()
        prompt = (
            f"{prefix}"
            f"Task: {task}\n\n"
            f"Step {step['id']}: {step['description']}\n"
            f"Tool to call: {step['tool']}\n"
            f"Tool description: {tool_doc}\n\n"
            f"Context from completed steps:\n{context}\n\n"
            f"Output the JSON input object for '{step['tool']}'."
        )
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=256,
                system=_INPUT_SYS,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError:
            return {"input": "API call failed"}, 0, 0

        in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
        tid = f"step-{step['id']}-input" + ("-retry" if is_retry else "")
        tracker.record_turn(in_tok, out_tok, turn_id=tid)

        text = resp.content[0].text.strip()
        tool_input = _extract_json(text) or {"input": text}
        return tool_input, in_tok, out_tok

    def _run_reasoning(
        self,
        step: dict[str, Any],
        task: str,
        context: str,
        tracker: AgentCostTracker,
        trace: list[dict[str, Any]],
    ) -> StepResult:
        """Execute a null-tool (reasoning-only) step via a direct Claude call."""
        step_id: int = step.get("id", 0)
        description: str = step.get("description", "")
        prompt = (
            f"Task: {task}\n\n"
            f"Step {step_id}: {description}\n\n"
            f"Context from completed steps:\n{context}\n\n"
            "Complete this step using only the information above."
        )
        cost_before = tracker.get_report().total_cost_usd

        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=512,
                system=_REASON_SYS,
                messages=[{"role": "user", "content": prompt}],
            )
            in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
            tracker.record_turn(in_tok, out_tok, turn_id=f"step-{step_id}-reason")
            output = resp.content[0].text.strip()
            success = True
        except anthropic.APIError as exc:
            in_tok = out_tok = 0
            output = f"API error: {exc}"
            success = False

        tokens = in_tok + out_tok
        cost_usd = round(tracker.get_report().total_cost_usd - cost_before, 8)
        trace.append(_trace_entry(step_id, description, "reason", None, None,
                                  output, tokens, cost_usd))
        return StepResult(
            step_id=step_id, description=description, tool_name=None,
            tool_input={}, output=output, tokens_used=tokens, cost_usd=cost_usd,
            success=success, retried=False, blocked=False,
        )


# ── module-level helpers ──────────────────────────────────────────────────────

def _trace_entry(
    turn_id: int | str,
    thought: str,
    action: str | None,
    tool_input: dict[str, Any] | None,
    _unused: Any,
    observation: str,
    tokens: int,
    cost_usd: float,
) -> dict[str, Any]:
    """Build a trace dict compatible with AgentResult.trace format."""
    is_tool = action not in (None, "plan", "reason", "BLOCKED", "synthesize")
    return {
        "turn_id":     turn_id,
        "thought":     thought,
        "action":      action,
        "tool_name":   action if is_tool else None,
        "tool_input":  tool_input,
        "observation": observation,
        "tokens":      tokens,
        "cost_usd":    cost_usd,
    }


def _exec_tool(
    name: str,
    kwargs: dict[str, Any],
    tools: dict[str, Callable[..., str]],
) -> str:
    """Run a tool by name; return its output or a descriptive error string."""
    if name not in tools:
        return f"Error: unknown tool '{name}'. Available: {', '.join(tools)}"
    try:
        return str(tools[name](**kwargs))
    except TypeError as exc:
        return f"Error: bad arguments for '{name}': {exc}"
    except Exception as exc:
        return f"Error ({type(exc).__name__}): {exc}"


def _format_context(completed: dict[int, StepResult]) -> str:
    """Render completed step results as a compact context string."""
    if not completed:
        return "No previous steps completed yet."
    lines: list[str] = []
    for sid, r in sorted(completed.items()):
        tag = "BLOCKED" if r.blocked else ("OK" if r.success else "FAILED")
        snippet = r.output[:400].replace("\n", " ")
        lines.append(f"  Step {sid} [{tag}]: {snippet}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract the first valid JSON object from a model response."""
    text = text.strip()
    # Try direct parse
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Markdown code block
    m = re.search(r"```(?:json)?\s*(\{.+?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Any JSON object (greedy to capture nested braces)
    m = re.search(r"(\{.+\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


# ── entry point ───────────────────────────────────────────────────────────────

def _print_plan(plan: dict[str, Any]) -> None:
    """Print the generated plan in a readable table."""
    steps = plan.get("steps", [])
    print(f"\n  Generated plan  ({len(steps)} steps):")
    print(f"  {'ID':<4} {'Tool':<14} {'Deps':<10} Description")
    print(f"  {'-'*64}")
    for s in steps:
        tool = s.get("tool") or "(reasoning)"
        deps = str(s.get("depends_on", [])) if s.get("depends_on") else "none"
        print(f"  {s['id']:<4} {tool:<14} {deps:<10} {s['description']}")


def _print_result(result: AgentResult) -> None:
    """Print the full trace and metrics for a completed run."""
    for t in result.trace:
        tid = t["turn_id"]
        if tid == "plan":
            continue                        # plan already printed above
        print(f"\n  [Step {tid}]  {t['thought']}")
        if t["tool_name"]:
            print(f"    Tool   : {t['tool_name']}")
            print(f"    Input  : {t['tool_input']}")
        obs = (t.get("observation") or "").replace("\n", " ")
        print(f"    Output : {obs[:200]}")
        retry_note = "  (retried)" if str(tid).endswith("-retry") else ""
        print(f"    Status : tokens={t['tokens']}  cost=${t['cost_usd']:.6f}{retry_note}")

    print(f"\n  *** FINAL ANSWER ***")
    for line in result.answer.splitlines():
        print(f"  {line}")

    print(f"\n  Metrics")
    print(f"  {'Turns used':<20}: {result.turns_used}")
    print(f"  {'Total tokens':<20}: {result.total_tokens}")
    print(f"  {'Cost':<20}: ${result.cost_usd:.6f}")
    print(f"  {'Success':<20}: {result.success}")


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set. Add it to .env and retry.")
        sys.exit(1)

    TASKS = [
        {
            "label": "Task 1 (3-step) — stock portfolio brief",
            "task": (
                "Find the current AAPL stock price. "
                "Calculate the total value of a 50-share portfolio at that price. "
                "Write a one-paragraph investment note."
            ),
            "tools": {"web_search": web_search, "calculator": calculator},
        },
        {
            "label": "Task 2 (5-step) — revenue performance report",
            "task": (
                "Read the file data.csv to get monthly revenue figures. "
                "Read notes.txt for board context. "
                "Calculate the total revenue across all months. "
                "Calculate the average monthly revenue. "
                "Write a two-sentence performance summary combining the numbers and the board notes."
            ),
            "tools": {"read_file": read_file, "calculator": calculator},
        },
    ]

    agent = PlanExecuteAgent(model="claude-sonnet-4-6", max_steps=8, budget_usd=0.15)

    for item in TASKS:
        sep = "=" * 68
        print(f"\n{sep}")
        print(f"  {item['label']}")
        print(f"  Task : {item['task']}")
        print(sep)

        # Intercept the plan before full execution to print it first
        # We'll rely on the trace entry to extract and print the plan
        result = agent.run(task=item["task"], tools=item["tools"])

        # Print plan from trace
        plan_entry = next((t for t in result.trace if t["turn_id"] == "plan"), None)
        if plan_entry:
            raw = plan_entry.get("observation", "")
            try:
                _print_plan(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                print(f"\n  Plan: {raw[:200]}")

        print(f"\n  EXECUTION")
        _print_result(result)
        print(f"\n{sep}")
