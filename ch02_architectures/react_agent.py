"""
ReAct (Reason + Act) agent — raw Anthropic SDK, zero framework dependencies.

Loop: Thought -> Action -> Observation, repeated until Final Answer or a
stop condition (budget exhausted, max_turns reached, API error).

Integrates with AgentCostTracker from ch01_why_agents_break for real-time
budget enforcement: the run halts automatically when spend reaches 95 % of
budget_usd.
"""
from __future__ import annotations

import ast
import json
import operator
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import anthropic

# ── project root on sys.path ──────────────────────────────────────────────────
# parents[0] = ch02_architectures/  parents[1] = project root
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ch01_why_agents_break.cost_tracker import AgentCostTracker, AlertLevel

# ── .env loader ───────────────────────────────────────────────────────────────

def _load_dotenv(path: Path) -> None:
    """Parse KEY=VALUE pairs from a .env file into os.environ (no-op if absent)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(_ROOT / ".env")

# ── pricing-model fallback ────────────────────────────────────────────────────
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


# ── ReAct system prompt ───────────────────────────────────────────────────────
_SYSTEM = """\
You are a ReAct agent. Solve the task step by step using the available tools.

Each turn respond in EXACTLY this format:

Thought: <reason about what to do or what you have learned>
Action: <tool_name>
Action Input: {"key": "value"}

When you have enough information for a complete answer:

Thought: <concluding reasoning>
Final Answer: <complete, precise answer>

Rules:
- Always start with "Thought:".
- After a Thought use EITHER a tool (Action + Action Input) OR Final Answer — never both.
- Action Input must be a single-line valid JSON object.
- Never fabricate Observations — wait for the system to supply them.
"""

# ── regex patterns ────────────────────────────────────────────────────────────
_RE_THOUGHT = re.compile(
    r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer):|$)", re.S | re.I
)
_RE_ACTION       = re.compile(r"^Action:\s*(\S+)",       re.M | re.I)
_RE_ACTION_INPUT = re.compile(r"Action Input:\s*(\{.+?\})", re.S | re.I)
_RE_FINAL        = re.compile(r"Final Answer:\s*(.+)$",     re.S | re.I)


# ── public data types ─────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """Output of a single ReActAgent.run() call.

    Attributes:
        answer: Final answer string, or a halt/error description on failure.
        turns_used: Number of Thought->Action->Observation cycles completed.
        total_tokens: Sum of input + output tokens across all API calls.
        cost_usd: Cumulative USD cost for this run.
        success: True only when a Final Answer was reached.
        trace: Per-turn dicts with thought, action, observation, and metrics.
    """

    answer: str
    turns_used: int
    total_tokens: int
    cost_usd: float
    success: bool
    trace: list[dict[str, Any]] = field(default_factory=list)


# ── agent ─────────────────────────────────────────────────────────────────────

class ReActAgent:
    """ReAct agent backed by the Anthropic Messages API.

    Runs a Thought -> Action -> Observation loop over a caller-supplied tool
    registry until it emits a Final Answer or a stop condition fires.

    Stop conditions (in priority order):
      1. Model emits "Final Answer:" -> success
      2. budget_usd * 95 % consumed  -> budget HALT
      3. max_turns reached           -> forced stop
      4. Anthropic API error         -> error stop

    Args:
        model: Anthropic model ID (default "claude-sonnet-4-6").
        max_turns: Maximum Thought->Action cycles (default 10).
        budget_usd: Per-run spend cap in USD; halts at 95 % (default $0.05).
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_turns: int = 10,
        budget_usd: float = 0.05,
    ) -> None:
        self.model = model
        self.max_turns = max_turns
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
        """Execute the ReAct loop for the given task.

        Args:
            task: Natural-language task description.
            tools: Mapping of tool_name -> callable(**kwargs) -> str.
                   The callable's first docstring line is shown to the model.

        Returns:
            AgentResult with answer, token totals, cost, and full trace.
        """
        tracker = AgentCostTracker(
            budget_usd=self.budget_usd,
            model=_cost_model(self.model),
        )

        tool_desc = "\n".join(
            f"  {name}: {(fn.__doc__ or '').splitlines()[0].strip()}"
            for name, fn in tools.items()
        )
        messages: list[dict[str, str]] = [
            {
                "role": "user",
                "content": (
                    f"Task: {task}\n\n"
                    f"Available tools:\n{tool_desc}\n\n"
                    "Begin."
                ),
            }
        ]

        trace: list[dict[str, Any]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        answer = ""
        success = False

        for turn in range(1, self.max_turns + 1):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=_SYSTEM,
                    messages=messages,
                )
            except anthropic.APIError as exc:
                answer = f"API error at turn {turn}: {exc}"
                break

            in_tok  = response.usage.input_tokens
            out_tok = response.usage.output_tokens
            total_input_tokens  += in_tok
            total_output_tokens += out_tok

            status = tracker.record_turn(in_tok, out_tok, turn_id=turn)

            text   = response.content[0].text.strip()
            parsed = _parse_react(text)

            turn_record: dict[str, Any] = {
                "turn_id":     turn,
                "thought":     parsed["thought"],
                "action":      parsed["action"],
                "tool_name":   parsed["action"],
                "tool_input":  parsed["tool_input"],
                "observation": None,
                "tokens":      in_tok + out_tok,
                "cost_usd":    round(status.turn_cost_usd, 6),
            }

            if parsed["final_answer"]:
                answer  = parsed["final_answer"]
                success = True
                turn_record["observation"] = "FINAL_ANSWER"
                trace.append(turn_record)
                break

            if parsed["action"]:
                observation = _exec_tool(
                    parsed["action"], parsed["tool_input"] or {}, tools
                )
                turn_record["observation"] = observation
                trace.append(turn_record)
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user",      "content": f"Observation: {observation}"})
            else:
                # Model did not follow the format
                answer = text
                turn_record["observation"] = "FORMAT_ERROR"
                trace.append(turn_record)
                break

            if status.alert_level is AlertLevel.HALT:
                answer = (
                    f"Budget HALT at turn {turn}: "
                    f"${status.cumulative_cost_usd:.5f} of ${self.budget_usd:.2f} used."
                )
                break

        if not answer:
            answer = f"Stopped after {self.max_turns} turns without a Final Answer."

        return AgentResult(
            answer=answer,
            turns_used=len(trace),
            total_tokens=total_input_tokens + total_output_tokens,
            cost_usd=tracker.get_report().total_cost_usd,
            success=success,
            trace=trace,
        )


# ── private helpers ───────────────────────────────────────────────────────────

def _parse_react(text: str) -> dict[str, Any]:
    """Extract Thought / Action / Action Input / Final Answer from a model turn."""
    thought = ""
    m = _RE_THOUGHT.search(text)
    if m:
        thought = m.group(1).strip()

    m = _RE_FINAL.search(text)
    if m:
        return {
            "thought": thought,
            "action": None,
            "tool_input": None,
            "final_answer": m.group(1).strip(),
        }

    action: str | None = None
    m = _RE_ACTION.search(text)
    if m:
        action = m.group(1).strip()

    tool_input: dict[str, Any] | None = None
    m = _RE_ACTION_INPUT.search(text)
    if m:
        raw = m.group(1).strip()
        try:
            tool_input = json.loads(raw)
        except json.JSONDecodeError:
            tool_input = {"input": raw}

    return {
        "thought": thought,
        "action": action,
        "tool_input": tool_input,
        "final_answer": None,
    }


def _exec_tool(
    name: str,
    kwargs: dict[str, Any],
    tools: dict[str, Callable[..., str]],
) -> str:
    """Look up and call a tool; return its string output or a descriptive error."""
    if name not in tools:
        return f"Error: unknown tool '{name}'. Available: {', '.join(tools)}"
    try:
        return str(tools[name](**kwargs))
    except TypeError as exc:
        return f"Error: bad arguments for '{name}': {exc}"
    except Exception as exc:
        return f"Error ({type(exc).__name__}): {exc}"


# ── example tools for __main__ ────────────────────────────────────────────────

_SEARCH_DB: dict[str, str] = {
    "aapl":             "Apple Inc. (AAPL) last close: $189.30 (NASDAQ, delayed).",
    "python":           "Python 3.13 released Oct 2024. Key additions: free-threaded mode, JIT compiler.",
    "gdp":              "Global GDP 2024 estimate: $110 trillion USD (World Bank).",
    "inflation":        "US CPI inflation 2024: ~3.2 % year-over-year (BLS).",
    "numpy":            "NumPy 2.0 released May 2024. Breaking change: new dtype promotion rules.",
    "anthropic":        "Anthropic released Claude 4 family in 2025: Haiku 4.5, Sonnet 4.6, Opus 4.7.",
}


def web_search(query: str) -> str:
    """Search the web for current facts. Args: query (str)"""
    q = query.lower()
    for key, result in _SEARCH_DB.items():
        if key in q:
            return result
    return f"No specific result for '{query}'. Try a more targeted query."


_AST_OPS: dict[type, Any] = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
    ast.USub:     operator.neg,
    ast.UAdd:     operator.pos,
}


def _ast_eval(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _AST_OPS:
        return _AST_OPS[type(node.op)](_ast_eval(node.left), _ast_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _AST_OPS:
        return _AST_OPS[type(node.op)](_ast_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate a maths expression safely. Args: expression (str) e.g. '(156 * 8) - (225 / 15)'"""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _ast_eval(tree.body)
        return f"{result:g}"
    except Exception as exc:
        return f"Error: {exc}"


_FILE_DB: dict[str, str] = {
    "data.csv": (
        "month,revenue_k\n"
        "2024-01,842\n2024-02,917\n2024-03,889\n"
        "2024-04,954\n2024-05,1023\n2024-06,998"
    ),
    "notes.txt": (
        "Q2 board notes: Revenue up 12 % QoQ. "
        "Cloud division exceeded targets by 8 %. "
        "Headcount freeze extended through Q3."
    ),
    "config.json": '{"model": "claude-sonnet-4-6", "budget_usd": 0.05, "max_turns": 10}',
}


def read_file(path: str) -> str:
    """Read a local file and return its full contents. Args: path (str) — filename"""
    return _FILE_DB.get(
        path,
        f"Error: '{path}' not found. Available: {', '.join(_FILE_DB)}",
    )


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set. Add it to .env and retry.")
        sys.exit(1)

    ALL_TOOLS = {
        "web_search": web_search,
        "calculator": calculator,
        "read_file":  read_file,
    }

    TASKS = [
        {
            "label": "Task 1 (simple) — pure arithmetic",
            "desc":  "Single tool, one or two turns.",
            "task":  "Calculate (347 * 28) + (1024 / 16) and give the exact numeric result.",
            "tools": {"calculator": calculator},
        },
        {
            "label": "Task 2 (medium) — web search + calculation",
            "desc":  "Two different tools, multi-step reasoning.",
            "task":  (
                "Find the current AAPL stock price. "
                "Then calculate the total value of a portfolio of 42 shares at that price."
            ),
            "tools": {"web_search": web_search, "calculator": calculator},
        },
        {
            "label": "Task 3 (complex) — multi-file synthesis",
            "desc":  "Three tool calls, data aggregation, and synthesis.",
            "task":  (
                "Read data.csv to get monthly revenue figures. "
                "Read notes.txt for business context. "
                "Calculate the average monthly revenue from the CSV data. "
                "Write a two-sentence financial summary combining the trend and the board notes."
            ),
            "tools": ALL_TOOLS,
        },
    ]

    agent = ReActAgent(model="claude-sonnet-4-6", max_turns=10, budget_usd=0.10)

    for item in TASKS:
        sep = "=" * 68
        print(f"\n{sep}")
        print(f"  {item['label']}  |  {item['desc']}")
        print(f"  Task : {item['task']}")
        print(sep)

        result = agent.run(task=item["task"], tools=item["tools"])

        for t in result.trace:
            print(f"\n  [Turn {t['turn_id']}]")
            print(f"  Thought : {t['thought']}")
            if t["tool_name"] and t["observation"] != "FINAL_ANSWER":
                print(f"  Action  : {t['tool_name']}  Input: {t['tool_input']}")
                print(f"  Obs     : {t['observation']}")
            else:
                if t["tool_name"]:
                    print(f"  Action  : {t['tool_name']}  Input: {t['tool_input']}")

        print(f"\n  *** FINAL ANSWER ***")
        # Wrap long answers
        for line in result.answer.splitlines():
            print(f"  {line}")

        print(f"\n  Metrics")
        print(f"  {'Turns used':<18}: {result.turns_used}")
        print(f"  {'Total tokens':<18}: {result.total_tokens}")
        print(f"  {'Cost':<18}: ${result.cost_usd:.6f}")
        print(f"  {'Success':<18}: {result.success}")
