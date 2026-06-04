"""
Compares Direct, ReAct, PlanExecute, and Supervisor architectures on a
standardised 10-task set (3 simple · 4 medium · 3 complex).

Evaluation:
  simple / medium   substring match against a known expected token
  complex           haiku LLM-as-judge (one call per architecture result)

Results are printed as an ASCII comparison table and saved to
benchmarks/architecture_benchmark_results.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import anthropic

# ── project root on sys.path ──────────────────────────────────────────────────
# benchmarks/../ = project root
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from agents.architectures.react_agent        import AgentResult, ReActAgent, web_search, calculator, read_file
from agents.architectures.plan_execute_agent import PlanExecuteAgent
from agents.architectures.supervisor_agent   import SupervisorAgent


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


# ── task set ──────────────────────────────────────────────────────────────────
# Tool return values (simulated, used to calibrate expected answers):
#   web_search("aapl")  -> "...AAPL) last close: $189.30..."
#   calculator("...")   -> exact numeric string
#   data.csv revenues   -> 842, 917, 889, 954, 1023, 998  sum=5623  avg≈937
#   notes.txt           -> "...Cloud division exceeded targets by 8%..."

@dataclass(frozen=True)
class BenchmarkTask:
    """One benchmark task.

    Attributes:
        id: 1-based task number.
        text: Natural-language task sent to the agent.
        complexity: "simple" | "medium" | "complex".
        expected: Substring to find (simple/medium) or judge criteria (complex).
        check_mode: "contains" | "judge".
    """

    id: int
    text: str
    complexity: str
    expected: str
    check_mode: str


TASKS: list[BenchmarkTask] = [
    # ── Simple (3) — pure arithmetic, no tools required ───────────────────────
    BenchmarkTask(
        id=1,
        text="Calculate 256 * 47 and give only the numeric result.",
        complexity="simple",
        expected="12032",
        check_mode="contains",
    ),
    BenchmarkTask(
        id=2,
        text="What is 15 percent of 840? Give only the numeric answer.",
        complexity="simple",
        expected="126",
        check_mode="contains",
    ),
    BenchmarkTask(
        id=3,
        text="A stock costs $189.30 per share. What is the total cost of 10 shares?",
        complexity="simple",
        expected="1893",
        check_mode="contains",
    ),
    # ── Medium (4) — require tool calls ───────────────────────────────────────
    BenchmarkTask(
        id=4,
        text="Search for the current AAPL stock price and calculate the value of 15 shares.",
        complexity="medium",
        expected="2839",          # 15 * $189.30 = $2839.50
        check_mode="contains",
    ),
    BenchmarkTask(
        id=5,
        text="Read the file data.csv and calculate the sum of all revenue_k values.",
        complexity="medium",
        expected="5623",          # 842+917+889+954+1023+998
        check_mode="contains",
    ),
    BenchmarkTask(
        id=6,
        text=(
            "Read data.csv and report which month had the highest revenue value "
            "and what that value was."
        ),
        complexity="medium",
        expected="1023",          # 2024-05 = 1023 k
        check_mode="contains",
    ),
    BenchmarkTask(
        id=7,
        text="Find the AAPL stock price and calculate exactly 8 percent of that price.",
        complexity="medium",
        expected="15.1",          # 189.30 * 0.08 = 15.144
        check_mode="contains",
    ),
    # ── Complex (3) — multi-step, LLM-as-judge ────────────────────────────────
    BenchmarkTask(
        id=8,
        text=(
            "Research the AAPL stock price, calculate the value of a 25-share portfolio, "
            "read data.csv to find the average monthly revenue in thousands, "
            "and write a two-sentence comparison of the stock investment "
            "versus one month of average revenue."
        ),
        complexity="complex",
        expected=(
            "Mentions AAPL price near $189.30, portfolio value near $4732, "
            "average monthly revenue near $937k, and draws a comparison between them."
        ),
        check_mode="judge",
    ),
    BenchmarkTask(
        id=9,
        text=(
            "Read data.csv and notes.txt. "
            "Calculate the total and average monthly revenue. "
            "Write a three-sentence business performance summary that incorporates "
            "the board meeting notes."
        ),
        complexity="complex",
        expected=(
            "Mentions total revenue near 5623 (thousand), average near 937 (thousand), "
            "references the cloud division or revenue growth from the board notes, "
            "and presents a coherent business summary."
        ),
        check_mode="judge",
    ),
    BenchmarkTask(
        id=10,
        text=(
            "Look up the AAPL stock price and search for Python programming language information. "
            "Calculate the value of a 20-share AAPL portfolio. "
            "Write a one-paragraph technology investment brief covering "
            "AAPL stock and Python's relevance to modern technology."
        ),
        complexity="complex",
        expected=(
            "Mentions AAPL price near $189.30, 20-share portfolio value near $3786, "
            "includes Python-related information, and presents a coherent "
            "technology investment perspective."
        ),
        check_mode="judge",
    ),
]

# ── rough cost hints (USD per task per architecture) ──────────────────────────
_COST_HINTS: dict[tuple[str, str], float] = {
    ("simple",  "Direct"):       0.0010,
    ("simple",  "ReAct"):        0.0025,
    ("simple",  "PlanExecute"):  0.0045,
    ("simple",  "Supervisor"):   0.0035,
    ("medium",  "Direct"):       0.0010,
    ("medium",  "ReAct"):        0.0055,
    ("medium",  "PlanExecute"):  0.0085,
    ("medium",  "Supervisor"):   0.0075,
    ("complex", "Direct"):       0.0015,
    ("complex", "ReAct"):        0.0110,
    ("complex", "PlanExecute"):  0.0170,
    ("complex", "Supervisor"):   0.0140,
}
_JUDGE_COST_HINT: float = 0.0004   # haiku judge call per complex task × architecture


# ── result dataclasses ────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    """Outcome of running one task with one architecture.

    Attributes:
        task_id: Matches BenchmarkTask.id.
        complexity: Task complexity tier.
        architecture: Agent architecture name.
        answer: Agent's final answer (first 300 chars stored).
        correct: Whether the evaluation passed.
        turns: Agent turns used.
        tokens: Total tokens consumed.
        cost_usd: Agent run cost.
        judge_cost_usd: LLM-judge cost (0 for contains-checked tasks).
        elapsed_s: Wall-clock seconds for the agent run.
    """

    task_id: int
    complexity: str
    architecture: str
    answer: str
    correct: bool
    turns: int
    tokens: int
    cost_usd: float
    judge_cost_usd: float
    elapsed_s: float


@dataclass
class ArchMetrics:
    """Aggregate performance metrics for one architecture.

    Attributes:
        architecture: Architecture name.
        tasks_run: Total tasks attempted.
        tasks_correct: Tasks that passed evaluation.
        success_rate: tasks_correct / tasks_run in [0, 1].
        avg_turns: Mean turns per task.
        avg_tokens_per_task: Mean tokens per task.
        avg_cost_per_task: Mean USD cost per task (agent + judge).
        total_cost_usd: Total cost across all tasks.
    """

    architecture: str
    tasks_run: int
    tasks_correct: int
    success_rate: float
    avg_turns: float
    avg_tokens_per_task: float
    avg_cost_per_task: float
    total_cost_usd: float


@dataclass
class BenchmarkReport:
    """Full benchmark results across all architectures and tasks.

    Attributes:
        timestamp: ISO-8601 UTC run timestamp.
        architectures: Aggregate metrics keyed by architecture name.
        task_results: All individual TaskResult records.
        total_benchmark_cost_usd: Combined agent + judge spend.
        recommendation: Auto-generated best-architecture string.
    """

    timestamp: str
    architectures: dict[str, ArchMetrics]
    task_results: list[TaskResult]
    total_benchmark_cost_usd: float
    recommendation: str


# ── DirectAgent — baseline (4th architecture) ─────────────────────────────────

class DirectAgent:
    """Single-shot baseline: one Claude call, no tools, no loop.

    Provides a cost and accuracy lower bound. Succeeds on pure-reasoning
    tasks; fails on tasks that require fetching real data via tools.

    Args:
        model: Claude model ID (default "claude-sonnet-4-6").
    """

    _SYSTEM = (
        "Answer the task directly and concisely. "
        "If the task asks for a numeric result, compute it precisely."
    )

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.model   = model
        self._client = anthropic.Anthropic()

    def run(
        self,
        task: str,
        tools: dict[str, Callable[..., str]],   # accepted for API compatibility
    ) -> AgentResult:
        """Single-shot inference. The tools dict is intentionally unused."""
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self._SYSTEM,
                messages=[{"role": "user", "content": task}],
            )
        except anthropic.APIError as exc:
            return AgentResult(
                answer=f"API error: {exc}", turns_used=1, total_tokens=0,
                cost_usd=0.0, success=False, trace=[],
            )

        in_tok  = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        text    = resp.content[0].text.strip()
        cost    = in_tok * 3.0e-6 + out_tok * 15.0e-6   # sonnet pricing

        return AgentResult(
            answer=text,
            turns_used=1,
            total_tokens=in_tok + out_tok,
            cost_usd=round(cost, 8),
            success=True,
            trace=[{
                "turn_id":     1,
                "thought":     "Single-shot direct answer (no tools)",
                "action":      None,
                "tool_name":   None,
                "tool_input":  None,
                "observation": text,
                "tokens":      in_tok + out_tok,
                "cost_usd":    round(cost, 8),
            }],
        )


# ── LLM-as-judge ─────────────────────────────────────────────────────────────

def _judge(
    task: str,
    criteria: str,
    answer: str,
    client: anthropic.Anthropic,
    model: str = "claude-haiku-4-5",
) -> tuple[bool, float]:
    """Evaluate answer quality against criteria using a lightweight LLM judge.

    Args:
        task: Original task text.
        criteria: Expected content or quality description.
        answer: The agent's final answer.
        client: Anthropic client instance.
        model: Judge model (haiku by default — cheap and sufficient).

    Returns:
        (passed: bool, judge_cost_usd: float)
    """
    prompt = (
        f"Task: {task}\n\n"
        f"Evaluation criteria: {criteria}\n\n"
        f"Answer (first 600 chars):\n{answer[:600]}\n\n"
        "Does this answer satisfy the criteria? "
        "Reply with exactly one word: PASS or FAIL"
    )
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=5,
            system="You are a strict answer evaluator. Reply with only PASS or FAIL.",
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = resp.content[0].text.strip().upper()
        cost    = (resp.usage.input_tokens  * 0.80e-6
                 + resp.usage.output_tokens * 4.00e-6)
        return verdict.startswith("PASS"), cost
    except anthropic.APIError:
        return False, 0.0


# ── benchmark class ───────────────────────────────────────────────────────────

class ArchitectureBenchmark:
    """Benchmarks four agent architectures on the standardised TASKS set.

    Architectures compared:
      Direct       single-shot, no tools (baseline)
      ReAct        reason + act loop with tool access
      PlanExecute  structured plan then step-by-step execution
      Supervisor   haiku routing + sonnet subagents

    Args:
        per_task_budget: USD cap per agent run (default $0.08).
        judge_model:     Model used for LLM-as-judge evaluation (default haiku).
    """

    ARCH_ORDER: list[str] = ["Direct", "ReAct", "PlanExecute", "Supervisor"]

    def __init__(
        self,
        per_task_budget: float = 0.08,
        judge_model: str = "claude-haiku-4-5",
    ) -> None:
        self.per_task_budget = per_task_budget
        self.judge_model     = judge_model
        self._client         = anthropic.Anthropic()
        self._tools: dict[str, Callable[..., str]] = {
            "web_search": web_search,
            "calculator": calculator,
            "read_file":  read_file,
        }
        self._agents: dict[str, Any] = {
            "Direct": DirectAgent(model="claude-sonnet-4-6"),
            "ReAct":  ReActAgent(
                model="claude-sonnet-4-6",
                max_turns=8,
                budget_usd=per_task_budget,
            ),
            "PlanExecute": PlanExecuteAgent(
                model="claude-sonnet-4-6",
                max_steps=6,
                budget_usd=per_task_budget,
            ),
            "Supervisor": SupervisorAgent(
                supervisor_model="claude-haiku-4-5",
                subagent_model="claude-sonnet-4-6",
                max_delegations=4,
                budget_usd=per_task_budget * 1.5,
            ),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_cost(self) -> float:
        """Return a rough USD estimate for the full benchmark run."""
        total = 0.0
        for task in TASKS:
            for arch in self.ARCH_ORDER:
                total += _COST_HINTS.get((task.complexity, arch), 0.005)
            if task.check_mode == "judge":
                total += _JUDGE_COST_HINT * len(self.ARCH_ORDER)
        return total

    def run_all(self) -> BenchmarkReport:
        """Execute all tasks against all architectures; return a BenchmarkReport.

        Progress is streamed to stdout as each task/architecture combination
        completes.

        Returns:
            BenchmarkReport with per-architecture metrics, all TaskResult
            records, total cost, and a recommendation string.
        """
        all_results: list[TaskResult] = []

        for task in TASKS:
            print(
                f"\n[{task.id:>2}/{len(TASKS)}] {task.complexity:<7} "
                f"| {task.text[:62]}"
            )

            for arch in self.ARCH_ORDER:
                result = self._run_task(task, arch, self._agents[arch])
                all_results.append(result)

                tag = "YES" if result.correct else " NO"
                print(
                    f"  {arch:<14} turns={result.turns:<3} "
                    f"tok={result.tokens:<6} "
                    f"${result.cost_usd:.5f}  "
                    f"correct={tag}  "
                    f"({result.elapsed_s:.1f}s)"
                )

        report = self._build_report(all_results)
        self._save(report)
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_task(
        self,
        task: BenchmarkTask,
        arch: str,
        agent: Any,
    ) -> TaskResult:
        """Run one agent on one task, evaluate, and return a TaskResult."""
        t0 = time.monotonic()
        try:
            result: AgentResult = agent.run(task.text, self._tools)
        except Exception as exc:           # catch broad to keep benchmark running
            result = AgentResult(
                answer=f"Unhandled exception: {exc}",
                turns_used=0, total_tokens=0, cost_usd=0.0,
                success=False, trace=[],
            )
        elapsed = time.monotonic() - t0

        correct, judge_cost = self._evaluate(task, result)

        return TaskResult(
            task_id=task.id,
            complexity=task.complexity,
            architecture=arch,
            answer=result.answer[:300],
            correct=correct,
            turns=result.turns_used,
            tokens=result.total_tokens,
            cost_usd=result.cost_usd,
            judge_cost_usd=judge_cost,
            elapsed_s=round(elapsed, 2),
        )

    def _evaluate(
        self,
        task: BenchmarkTask,
        result: AgentResult,
    ) -> tuple[bool, float]:
        """Check the agent result against the task's expected value."""
        if task.check_mode == "contains":
            # Normalise: remove digit-grouping commas, lowercase
            norm = result.answer.replace(",", "").lower()
            hit  = task.expected.replace(",", "").lower() in norm
            return hit, 0.0

        # "judge" — LLM-as-judge via haiku
        return _judge(
            task.text, task.expected, result.answer,
            self._client, self.judge_model,
        )

    def _build_report(self, results: list[TaskResult]) -> BenchmarkReport:
        """Aggregate TaskResults into a BenchmarkReport."""
        arch_metrics: dict[str, ArchMetrics] = {}
        total_cost = 0.0

        for arch in self.ARCH_ORDER:
            rows = [r for r in results if r.architecture == arch]
            if not rows:
                continue

            n       = len(rows)
            correct = sum(r.correct for r in rows)
            costs   = [r.cost_usd + r.judge_cost_usd for r in rows]
            arch_total = sum(costs)
            total_cost += arch_total

            arch_metrics[arch] = ArchMetrics(
                architecture=arch,
                tasks_run=n,
                tasks_correct=correct,
                success_rate=correct / n,
                avg_turns=sum(r.turns  for r in rows) / n,
                avg_tokens_per_task=sum(r.tokens for r in rows) / n,
                avg_cost_per_task=arch_total / n,
                total_cost_usd=arch_total,
            )

        return BenchmarkReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            architectures=arch_metrics,
            task_results=results,
            total_benchmark_cost_usd=round(total_cost, 6),
            recommendation=self._recommend(arch_metrics),
        )

    def _recommend(self, metrics: dict[str, ArchMetrics]) -> str:
        """Generate a one-line best-architecture recommendation."""
        if not metrics:
            return "No data."

        # Best = highest success_rate; tie-break on lowest cost
        best     = max(metrics.values(), key=lambda m: (m.success_rate, -m.avg_cost_per_task))
        baseline = metrics.get("Direct")
        priciest = max(metrics.values(), key=lambda m: m.avg_cost_per_task)

        parts: list[str] = [f"Best cost/quality: {best.architecture}"]

        if baseline and best.architecture != "Direct":
            ds = (best.success_rate - baseline.success_rate) * 100
            if ds > 0:
                parts.append(f"+{ds:.1f}% success vs Direct")

        if priciest.architecture != best.architecture:
            dc = (
                (priciest.avg_cost_per_task - best.avg_cost_per_task)
                / priciest.avg_cost_per_task * 100
            )
            parts.append(f"-{dc:.1f}% cost vs {priciest.architecture}")

        return "  ".join(parts)

    def _save(self, report: BenchmarkReport) -> None:
        """Save the report to benchmarks/architecture_benchmark_results.json."""
        out = Path(__file__).parent / "architecture_benchmark_results.json"
        data: dict[str, Any] = {
            "timestamp":                report.timestamp,
            "total_benchmark_cost_usd": report.total_benchmark_cost_usd,
            "recommendation":           report.recommendation,
            "architectures":            {k: asdict(v) for k, v in report.architectures.items()},
            "task_results":             [asdict(r) for r in report.task_results],
        }
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\n  Saved: {out}")


# ── ASCII comparison table ────────────────────────────────────────────────────

def _ascii_table(report: BenchmarkReport) -> str:
    """Build a padded ASCII comparison table from a BenchmarkReport."""
    # Column widths (content only, not including border chars)
    W = (14, 9, 9, 11, 10, 9)
    div = "+" + "+".join("-" * (w + 2) for w in W) + "+"

    def row(*cells: str) -> str:
        parts = [f" {c:<{W[i]}} " if i == 0 else f" {c:>{W[i]}} " for i, c in enumerate(cells)]
        return "|" + "|".join(parts) + "|"

    lines: list[str] = [
        div,
        row("Architecture", "Success%", "AvgTurns", "AvgTokens", "AvgCost$", "Total$"),
        div,
    ]
    for arch in ArchitectureBenchmark.ARCH_ORDER:
        m = report.architectures.get(arch)
        if m is None:
            continue
        lines.append(row(
            arch,
            f"{m.success_rate*100:.1f}%",
            f"{m.avg_turns:.1f}",
            f"{m.avg_tokens_per_task:,.0f}",
            f"{m.avg_cost_per_task:.5f}",
            f"{m.total_cost_usd:.4f}",
        ))
    lines.append(div)
    lines.append(f"  Total benchmark cost: ${report.total_benchmark_cost_usd:.4f}")
    return "\n".join(lines)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set. Add it to .env and retry.")
        sys.exit(1)

    bench   = ArchitectureBenchmark(per_task_budget=0.08)
    est     = bench.estimate_cost()
    n_arch  = len(ArchitectureBenchmark.ARCH_ORDER)
    n_tasks = len(TASKS)
    simple  = sum(1 for t in TASKS if t.complexity == "simple")
    medium  = sum(1 for t in TASKS if t.complexity == "medium")
    complex_= sum(1 for t in TASKS if t.complexity == "complex")
    n_judge = complex_ * n_arch

    sep = "=" * 62
    print(sep)
    print("  Architecture Benchmark  —  LLM Agent Engineering")
    print(sep)
    print(f"  Tasks         : {n_tasks}  ({simple} simple · {medium} medium · {complex_} complex)")
    print(f"  Architectures : {', '.join(ArchitectureBenchmark.ARCH_ORDER)}")
    print(f"  Agent runs    : {n_tasks * n_arch}  ({n_tasks} tasks × {n_arch} architectures)")
    print(f"  Judge calls   : {n_judge}  (haiku, complex tasks only)")
    print(f"  Budget/task   : ${bench.per_task_budget:.2f} per architecture")
    print(f"  Est. cost     : ~${est:.3f}")
    print(sep)

    confirm = input("  Proceed? [y/N] ").strip().lower()
    if confirm != "y":
        print("  Aborted.")
        sys.exit(0)

    print("\n  Running benchmark...")

    report = bench.run_all()

    print(f"\n\n{sep}")
    print("  BENCHMARK RESULTS")
    print(sep)
    print()
    print(_ascii_table(report))
    print()
    print(f"  Recommendation: {report.recommendation}")
    print(sep)
