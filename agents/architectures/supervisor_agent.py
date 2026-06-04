"""
Supervisor + Subagent delegation pattern — Anthropic SDK, no framework deps.

Routing layer  (supervisor_model = haiku)  : decides per-delegation whether to
    answer directly or hand off to a specialised subagent.
Subagents      (subagent_model   = sonnet) : RESEARCHER, ANALYZER, and EXECUTOR
    are ReActAgent instances with filtered tool subsets; WRITER is a direct
    generation call (no tools needed).
Aggregation    (supervisor_model = haiku)  : after all delegations the supervisor
    synthesises the collected results into a single final answer.

Cost model: haiku for routing + synthesis ($0.80/$4.00 per 1M tokens),
            sonnet for subagent execution ($3.00/$15.00 per 1M tokens).
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import anthropic

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from agents.architectures.react_agent import (
    AgentResult,
    ReActAgent,
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


# ── per-model cost helper ─────────────────────────────────────────────────────

def _cost(model: str, in_tok: int, out_tok: int) -> float:
    """Compute USD cost for one API call given the model and token counts."""
    lc = model.lower()
    if "haiku" in lc:
        return in_tok * 0.80e-6 + out_tok * 4.00e-6
    if "opus" in lc:
        return in_tok * 15.0e-6 + out_tok * 75.0e-6
    return in_tok * 3.0e-6 + out_tok * 15.0e-6


# ── subagent taxonomy ─────────────────────────────────────────────────────────

class SubagentType(str, Enum):
    """Specialised subagent roles the supervisor can delegate to."""

    RESEARCHER = "RESEARCHER"
    ANALYZER   = "ANALYZER"
    WRITER     = "WRITER"
    EXECUTOR   = "EXECUTOR"


_SUBAGENT_DESCRIPTIONS: dict[SubagentType, str] = {
    SubagentType.RESEARCHER: "searches the web and reads files to gather information",
    SubagentType.ANALYZER:   "performs calculations and quantitative data analysis",
    SubagentType.WRITER:     "composes text, summaries, and reports (no tool access)",
    SubagentType.EXECUTOR:   "executes any available tool to complete actions",
}

# Tool names each subagent may access.  None = all tools (EXECUTOR).
_TOOL_ALLOW: dict[SubagentType, frozenset[str] | None] = {
    SubagentType.RESEARCHER: frozenset({"web_search", "read_file"}),
    SubagentType.ANALYZER:   frozenset({"calculator", "read_file"}),
    SubagentType.WRITER:     frozenset(),   # direct generation; no tools
    SubagentType.EXECUTOR:   None,
}


# ── system prompts ────────────────────────────────────────────────────────────

def _build_route_sys() -> str:
    desc_lines = "\n".join(
        f"  {st.value:<12}: {desc}"
        for st, desc in _SUBAGENT_DESCRIPTIONS.items()
    )
    return f"""\
You are a supervisor agent routing a task to specialised subagents.

Available subagents:
{desc_lines}

Output ONLY valid JSON — no markdown, no commentary.

To delegate to a subagent:
{{"action": "DELEGATE", "delegate_to": "RESEARCHER", "subtask": "what exactly the subagent should do", "required_output": "what you need back", "reasoning": "why this subagent"}}

To answer using results already gathered:
{{"action": "ANSWER_DIRECTLY", "reasoning": "why you have enough information"}}

Choose ANSWER_DIRECTLY only when sufficient information has been collected."""


_ROUTE_SYS = _build_route_sys()

_WRITE_SYS = (
    "You are a professional writer. Produce clear, concise, well-structured text "
    "based only on the information provided. Do not invent data."
)

_SYNTH_SYS = (
    "You are a supervisor synthesising results from specialised subagents. "
    "Combine the collected outputs into one coherent, precise final answer."
)


# ── internal data types ───────────────────────────────────────────────────────

@dataclass
class DelegationRecord:
    """Log entry for one completed delegation.

    Attributes:
        subagent_type: Which subagent handled this delegation.
        subtask: The specific task description sent to the subagent.
        result: The subagent's final answer.
        success: Whether the subagent completed without errors.
        tokens_used: Combined input + output tokens for this delegation.
        cost_usd: USD cost for this delegation.
    """

    subagent_type: SubagentType
    subtask: str
    result: str
    success: bool
    tokens_used: int
    cost_usd: float


# ── agent ─────────────────────────────────────────────────────────────────────

class SupervisorAgent:
    """Supervisor agent with specialised subagent delegation.

    The supervisor (haiku) routes sub-tasks to specialised ReActAgent subagents
    (sonnet), collecting their results until it has enough to synthesise a final
    answer.

    Tool access per subagent type:
      RESEARCHER   web_search, read_file
      ANALYZER     calculator, read_file
      WRITER       (direct generation — no tools)
      EXECUTOR     all tools

    Args:
        supervisor_model: Model for routing decisions and synthesis
                          (default "claude-haiku-4-5" — cheap and fast).
        subagent_model:   Model for subagent task execution
                          (default "claude-sonnet-4-6" — capable).
        max_delegations:  Maximum subagent calls before forcing synthesis
                          (default 5).
        budget_usd:       Total per-run spend cap in USD (default $0.05).
    """

    def __init__(
        self,
        supervisor_model: str = "claude-haiku-4-5",
        subagent_model: str   = "claude-sonnet-4-6",
        max_delegations: int  = 5,
        budget_usd: float     = 0.05,
    ) -> None:
        self.supervisor_model = supervisor_model
        self.subagent_model   = subagent_model
        self.max_delegations  = max_delegations
        self.budget_usd       = budget_usd
        self._client = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        tools: dict[str, Callable[..., str]],
    ) -> AgentResult:
        """Route the task through the supervisor/subagent loop.

        Args:
            task: High-level task description.
            tools: All available tools; subsets are distributed to subagents
                   according to their type.

        Returns:
            AgentResult with synthesised answer, full routing trace
            (routing decisions + flattened subagent turns + aggregation),
            aggregated token counts, and total cost across all models.
        """
        trace: list[dict[str, Any]] = []
        delegations: list[DelegationRecord] = []
        total_cost = 0.0
        total_tokens = 0

        for n in range(1, self.max_delegations + 1):
            if total_cost >= self.budget_usd * 0.95:
                break

            # ── routing decision (haiku) ──────────────────────────────────────
            remaining = self.max_delegations - n + 1
            decision, r_in, r_out = self._route(
                task, tools, delegations, n, remaining, trace
            )
            total_cost   += _cost(self.supervisor_model, r_in, r_out)
            total_tokens += r_in + r_out

            action = decision.get("action", "ANSWER_DIRECTLY")
            if action != "DELEGATE":
                break

            # ── delegate to subagent (sonnet) ─────────────────────────────────
            subagent_budget = max(
                0.005,
                (self.budget_usd - total_cost) / max(1, remaining),
            )
            record = self._delegate(
                decision, task, tools, delegations, subagent_budget, n, trace
            )
            delegations.append(record)
            total_cost   += record.cost_usd
            total_tokens += record.tokens_used

        # ── synthesise final answer (haiku) ───────────────────────────────────
        answer, s_in, s_out = self._synthesize(task, delegations, trace)
        total_cost   += _cost(self.supervisor_model, s_in, s_out)
        total_tokens += s_in + s_out

        if not answer:
            answer = "No answer produced (all delegations failed or budget exhausted)."

        return AgentResult(
            answer=answer,
            turns_used=len(trace),
            total_tokens=total_tokens,
            cost_usd=round(total_cost, 8),
            success=bool(delegations) and any(d.success for d in delegations),
            trace=trace,
        )

    # ------------------------------------------------------------------
    # Private: routing
    # ------------------------------------------------------------------

    def _route(
        self,
        task: str,
        tools: dict[str, Callable[..., str]],
        delegations: list[DelegationRecord],
        delegation_num: int,
        remaining: int,
        trace: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], int, int]:
        """Ask the supervisor (haiku) for the next routing decision.

        Returns (decision_dict, input_tokens, output_tokens).
        Appends one 'route-N' entry to trace.
        """
        context = _format_delegations(delegations)
        nudge   = " Consider ANSWER_DIRECTLY — most delegations used." if remaining <= 1 else ""
        user_msg = (
            f"Task: {task}\n\n"
            f"Available tools: {', '.join(tools) or '(none)'}\n"
            f"Delegation {delegation_num}/{self.max_delegations}.{nudge}\n\n"
            f"Completed delegations:\n{context}\n\n"
            "Decide the next action."
        )

        try:
            resp = self._client.messages.create(
                model=self.supervisor_model,
                max_tokens=512,
                system=_ROUTE_SYS,
                messages=[{"role": "user", "content": user_msg}],
            )
        except anthropic.APIError as exc:
            fallback: dict[str, Any] = {
                "action": "ANSWER_DIRECTLY",
                "reasoning": f"API error: {exc}",
            }
            trace.append(_route_entry(delegation_num, fallback, 0, 0.0))
            return fallback, 0, 0

        in_tok  = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        text    = resp.content[0].text.strip()
        decision = _extract_json(text) or {
            "action": "ANSWER_DIRECTLY",
            "reasoning": text[:200],
        }

        route_cost = _cost(self.supervisor_model, in_tok, out_tok)
        trace.append(_route_entry(delegation_num, decision, in_tok + out_tok, route_cost))
        return decision, in_tok, out_tok

    # ------------------------------------------------------------------
    # Private: delegation dispatch
    # ------------------------------------------------------------------

    def _delegate(
        self,
        decision: dict[str, Any],
        task: str,
        tools: dict[str, Callable[..., str]],
        delegations: list[DelegationRecord],
        budget: float,
        delegation_num: int,
        trace: list[dict[str, Any]],
    ) -> DelegationRecord:
        """Dispatch a subtask to the appropriate specialised subagent."""
        raw_type = decision.get("delegate_to", "EXECUTOR")
        subtask  = decision.get("subtask", task)
        req_out  = decision.get("required_output", "")

        try:
            sa_type = SubagentType(raw_type)
        except ValueError:
            sa_type = SubagentType.EXECUTOR

        context  = _format_delegations(delegations)
        enriched = (
            f"{subtask}\n\n"
            f"Required output: {req_out}\n"
            f"Overall task for context: {task}\n\n"
            f"Prior delegation results:\n{context}"
        )

        if sa_type == SubagentType.WRITER:
            result = self._run_writer(enriched, delegation_num, trace)
        else:
            tool_subset = _get_tool_subset(sa_type, tools)
            result = self._run_react_subagent(
                sa_type, enriched, tool_subset, budget, delegation_num, trace
            )

        return DelegationRecord(
            subagent_type=sa_type,
            subtask=subtask,
            result=result.answer,
            success=result.success,
            tokens_used=result.total_tokens,
            cost_usd=result.cost_usd,
        )

    def _run_react_subagent(
        self,
        sa_type: SubagentType,
        task_with_context: str,
        tool_subset: dict[str, Callable[..., str]],
        budget: float,
        delegation_n: int,
        trace: list[dict[str, Any]],
    ) -> AgentResult:
        """Run a ReActAgent subagent; flatten its trace into the supervisor trace."""
        subagent = ReActAgent(
            model=self.subagent_model,
            max_turns=5,
            budget_usd=budget,
        )
        result = subagent.run(task=task_with_context, tools=tool_subset)

        prefix = f"sub-{sa_type.value}-{delegation_n}"
        for st in result.trace:
            entry = dict(st)
            entry["turn_id"] = f"{prefix}-{st['turn_id']}"
            trace.append(entry)

        return result

    def _run_writer(
        self,
        task_with_context: str,
        delegation_n: int,
        trace: list[dict[str, Any]],
    ) -> AgentResult:
        """Execute the WRITER subagent as a direct generation call (no tools)."""
        try:
            resp = self._client.messages.create(
                model=self.subagent_model,
                max_tokens=1024,
                system=_WRITE_SYS,
                messages=[{"role": "user", "content": task_with_context}],
            )
        except anthropic.APIError as exc:
            err = f"Writer API error: {exc}"
            trace.append({
                "turn_id":     f"sub-WRITER-{delegation_n}-1",
                "thought":     "Direct generation",
                "action":      "write",
                "tool_name":   None,
                "tool_input":  None,
                "observation": err,
                "tokens":      0,
                "cost_usd":    0.0,
            })
            return AgentResult(
                answer=err, turns_used=1, total_tokens=0,
                cost_usd=0.0, success=False, trace=[],
            )

        in_tok  = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        text    = resp.content[0].text.strip()
        w_cost  = _cost(self.subagent_model, in_tok, out_tok)

        trace.append({
            "turn_id":     f"sub-WRITER-{delegation_n}-1",
            "thought":     "Direct text generation (no tools)",
            "action":      "write",
            "tool_name":   None,
            "tool_input":  None,
            "observation": text,
            "tokens":      in_tok + out_tok,
            "cost_usd":    round(w_cost, 8),
        })
        return AgentResult(
            answer=text,
            turns_used=1,
            total_tokens=in_tok + out_tok,
            cost_usd=round(w_cost, 8),
            success=True,
            trace=[],
        )

    # ------------------------------------------------------------------
    # Private: final synthesis
    # ------------------------------------------------------------------

    def _synthesize(
        self,
        task: str,
        delegations: list[DelegationRecord],
        trace: list[dict[str, Any]],
    ) -> tuple[str, int, int]:
        """Aggregate delegation results into a final answer (haiku call).

        Returns (answer, input_tokens, output_tokens).
        Appends one 'aggregate' entry to trace.
        """
        if not delegations:
            trace.append({
                "turn_id": "aggregate", "thought": "Nothing to aggregate",
                "action": "synthesize", "tool_name": None, "tool_input": None,
                "observation": "No delegations completed.", "tokens": 0, "cost_usd": 0.0,
            })
            return "No delegations completed.", 0, 0

        context  = _format_delegations(delegations)
        user_msg = (
            f"Original task: {task}\n\n"
            f"Subagent results:\n{context}\n\n"
            "Write a complete, precise final answer to the original task."
        )

        try:
            resp = self._client.messages.create(
                model=self.supervisor_model,
                max_tokens=1024,
                system=_SYNTH_SYS,
                messages=[{"role": "user", "content": user_msg}],
            )
        except anthropic.APIError as exc:
            err = f"Synthesis failed: {exc}"
            trace.append({
                "turn_id": "aggregate", "thought": "Synthesis",
                "action": "synthesize", "tool_name": None, "tool_input": None,
                "observation": err, "tokens": 0, "cost_usd": 0.0,
            })
            return err, 0, 0

        in_tok  = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        answer  = resp.content[0].text.strip()
        s_cost  = _cost(self.supervisor_model, in_tok, out_tok)

        trace.append({
            "turn_id":     "aggregate",
            "thought":     "Synthesising all delegation results",
            "action":      "synthesize",
            "tool_name":   None,
            "tool_input":  None,
            "observation": answer,
            "tokens":      in_tok + out_tok,
            "cost_usd":    round(s_cost, 8),
        })
        return answer, in_tok, out_tok


# ── module-level helpers ──────────────────────────────────────────────────────

def _format_delegations(delegations: list[DelegationRecord]) -> str:
    """Render completed delegations as a readable context string."""
    if not delegations:
        return "  (none yet)"
    lines: list[str] = []
    for i, d in enumerate(delegations, 1):
        tag     = "OK" if d.success else "FAILED"
        snippet = d.result.replace("\n", " ")[:400]
        lines.append(f"  [{i}] {d.subagent_type.value} [{tag}] — {d.subtask}")
        lines.append(f"       Result: {snippet}")
    return "\n".join(lines)


def _get_tool_subset(
    sa_type: SubagentType,
    all_tools: dict[str, Callable[..., str]],
) -> dict[str, Callable[..., str]]:
    """Return the tools this subagent type is allowed to use."""
    allowed = _TOOL_ALLOW.get(sa_type)
    if allowed is None:
        return dict(all_tools)
    return {k: v for k, v in all_tools.items() if k in allowed}


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract the first valid JSON object from a model response string."""
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    m = re.search(r"```(?:json)?\s*(\{.+?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"(\{.+\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def _route_entry(
    n: int,
    decision: dict[str, Any],
    tokens: int,
    cost_usd: float,
) -> dict[str, Any]:
    """Build a trace entry for a routing decision."""
    action  = decision.get("action", "ANSWER_DIRECTLY")
    sub     = decision.get("delegate_to", "")
    label   = f"DELEGATE_TO_{sub}" if action == "DELEGATE" else "ANSWER_DIRECTLY"
    return {
        "turn_id":     f"route-{n}",
        "thought":     decision.get("reasoning", ""),
        "action":      label,
        "tool_name":   None,
        "tool_input":  {
            "delegate_to": sub,
            "subtask":     decision.get("subtask", ""),
        },
        "observation": json.dumps(decision),
        "tokens":      tokens,
        "cost_usd":    round(cost_usd, 8),
    }


# ── entry point ───────────────────────────────────────────────────────────────

def _print_run(result: AgentResult, label: str, task: str) -> None:
    """Print a formatted supervisor run report to stdout."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {label}")
    print(f"  Task : {task}")
    print(sep)

    current_sub: str | None = None
    delegations_seen: set[str] = set()

    for t in result.trace:
        tid = str(t["turn_id"])

        # ── routing decision ──────────────────────────────────────────────────
        if tid.startswith("route-"):
            n = tid.split("-", 1)[1]
            print(f"\n  [Routing {n}]  {t['action']}")
            print(f"    Reason  : {t['thought'][:100]}")
            di = _extract_json(t.get("observation") or "")
            if di and di.get("subtask"):
                print(f"    Subtask : {di['subtask'][:100]}")
            if di and di.get("required_output"):
                print(f"    Needs   : {di['required_output'][:80]}")

        # ── subagent turn ─────────────────────────────────────────────────────
        elif tid.startswith("sub-"):
            parts   = tid.split("-")            # sub / TYPE / N / turn_id
            sa_type = parts[1] if len(parts) > 1 else "?"
            del_n   = parts[2] if len(parts) > 2 else "?"
            sub_key = f"{sa_type}-{del_n}"

            if sub_key not in delegations_seen:
                delegations_seen.add(sub_key)
                print(f"\n  --- {sa_type} subagent (delegation #{del_n}) ---")

            if t.get("tool_name"):
                obs = (t.get("observation") or "")[:100].replace("\n", " ")
                print(f"    {t['tool_name']}({t.get('tool_input', {})}) -> {obs}")
            elif t.get("action") == "write":
                obs = (t.get("observation") or "")[:120].replace("\n", " ")
                print(f"    [write] {obs}")
            elif t.get("observation") == "FINAL_ANSWER":
                print(f"    [done]  {t.get('thought', '')[:80]}")

        # ── aggregation ───────────────────────────────────────────────────────
        elif tid == "aggregate":
            print(f"\n  [Aggregate]  tokens={t['tokens']}  cost=${t['cost_usd']:.6f}")

    print(f"\n  *** FINAL ANSWER ***")
    for line in result.answer.splitlines():
        print(f"  {line}")

    # ── metrics ───────────────────────────────────────────────────────────────
    subs = {
        "-".join(str(t["turn_id"]).split("-")[:3])
        for t in result.trace
        if str(t["turn_id"]).startswith("sub-")
        and len(str(t["turn_id"]).split("-")) >= 3
    }
    types_used = sorted({s.split("-")[1] for s in subs if s.split("-")[1] != "?"})
    print(f"\n  Metrics")
    print(f"  {'Subagent types':<22}: {', '.join(types_used) or 'none'}")
    print(f"  {'Delegations':<22}: {len(subs)}")
    print(f"  {'Trace entries':<22}: {result.turns_used}")
    print(f"  {'Total tokens':<22}: {result.total_tokens}")
    print(f"  {'Total cost':<22}: ${result.cost_usd:.6f}")
    print(f"  {'Success':<22}: {result.success}")
    print(sep)


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set. Add it to .env and retry.")
        sys.exit(1)

    ALL_TOOLS = {
        "web_search": web_search,
        "calculator": calculator,
        "read_file":  read_file,
    }

    # Task requires at least 3 delegations to 3 different subagent types:
    #   RESEARCHER  -> find AAPL stock price
    #   ANALYZER    -> calculate 75-share portfolio value
    #   WRITER      -> compose professional investment memo
    TASK = (
        "Build a professional investment memo: "
        "(1) research the current AAPL stock price, "
        "(2) calculate the total value of a 75-share portfolio at that price, "
        "(3) also read data.csv to note our latest revenue figures, "
        "(4) write a concise two-paragraph investment memo combining the "
        "stock valuation and the revenue context."
    )

    agent = SupervisorAgent(
        supervisor_model="claude-haiku-4-5",
        subagent_model="claude-sonnet-4-6",
        max_delegations=5,
        budget_usd=0.20,
    )

    result = agent.run(task=TASK, tools=ALL_TOOLS)
    _print_run(result, label="SupervisorAgent demo", task=TASK)
