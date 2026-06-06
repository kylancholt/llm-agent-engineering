"""
Parallel execution of subagents with asyncio.

Where MultiAgentSupervisor.run() (see supervisor.py) executes dependency waves
with a thread pool, ParallelRunner is the asyncio-native engine for the same
job: it runs many subagents concurrently under a concurrency cap, enforces a
per-agent timeout, respects inter-subtask dependencies, survives partial
failures, and reports live progress on a single in-place status line.

Core ideas:
  - Concurrency is bounded by an asyncio.Semaphore (``max_concurrent``).
  - Each subtask waits on its dependencies' completion (asyncio.Event) before
    acquiring a slot, so dependency order is preserved without a topo sort in
    the parallel path. Waiters do not hold a slot, so there is no deadlock.
  - A subagent is run via ``agent_factory``. Sync factories are offloaded with
    ``asyncio.to_thread`` (the supervisor's subagent calls are blocking HTTP);
    coroutine factories are awaited directly.
  - A failing subagent does not abort the run — its failure is recorded in
    ``partial_results`` and dependents of a failed subtask are skipped.
  - ``benchmark_vs_sequential`` runs the same work both ways and reports the
    wall-clock speedup and the (near-zero) cost delta — parallelism buys time,
    not tokens.

The runner is decoupled from any specific result type: ``agent_factory`` may
return any object. If that object exposes ``success: bool``, ``cost_usd:
float``, and ``error: str | None`` (as supervisor.SubtaskResult does), those
fields are used for failure detection and cost aggregation.

Requires ANTHROPIC_API_KEY in the environment or the project-root .env file
(used by the subagent factory, not by the runner itself).
"""
from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ── project root: ch08_multi_agent/.. = root ────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ch08_multi_agent.supervisor import Subtask, SubtaskResult


# ── .env loader (mirrors the rest of the project) ───────────────────────────────

def _load_env(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (idempotent)."""
    import os
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_load_env(_ROOT / ".env")


# ── defaults ────────────────────────────────────────────────────────────────────

_DEFAULT_MAX_CONCURRENT = 5
_DEFAULT_TIMEOUT_S      = 120

# A subagent execution function: (subtask, completed_dependency_results) -> result.
# ``dependency_results`` maps each dependency's subtask id to its result object.
# The factory may be a plain function (run in a worker thread) or a coroutine
# function (awaited directly).
AgentFactory = Callable[[Subtask, dict[str, Any]], Any]


# ── public types ──────────────────────────────────────────────────────────────

@dataclass
class AgentOutcome:
    """Per-subtask outcome recorded by the runner (success or failure).

    Attributes:
        task_id: The subtask's id.
        agent: The subtask's assigned agent name (used as the progress label).
        failure: True when the subagent errored, timed out, returned a result
            flagged ``success == False``, or was skipped due to a failed
            dependency.
        result: The object returned by ``agent_factory`` (None on failure).
        error: Failure description (None on success).
        elapsed_ms: Wall-clock duration of this subagent's execution.
        cost_usd: Cost reported by the result object (0.0 if unavailable).
        timed_out: True when the per-agent timeout fired.
        skipped: True when the subtask was skipped (a dependency failed).
    """
    task_id:    str
    agent:      str
    failure:    bool
    result:     Any | None
    error:      str | None
    elapsed_ms: float
    cost_usd:   float
    timed_out:  bool = False
    skipped:    bool = False


@dataclass
class ParallelResult:
    """Aggregate result of ParallelRunner.run_parallel().

    Attributes:
        results: Mapping of task_id -> result object, for successful subtasks
            only.
        partial_results: Mapping of task_id -> AgentOutcome for *every* subtask,
            each carrying a ``failure`` flag. This is the partial-failure view:
            inspect it to see which subagents failed and why.
        wall_time_ms: Measured wall-clock duration of the whole parallel run.
        sequential_equivalent_ms: Sum of every subagent's individual execution
            time — what the same work would have cost run back-to-back.
        speedup_factor: sequential_equivalent_ms / wall_time_ms.
        total_cost_usd: Sum of every subagent's cost (successes and failures).
    """
    results:                  dict[str, Any]
    partial_results:          dict[str, AgentOutcome]
    wall_time_ms:             float
    sequential_equivalent_ms: float
    speedup_factor:           float
    total_cost_usd:           float

    @property
    def had_failures(self) -> bool:
        """True if any subtask failed, timed out, or was skipped."""
        return any(oc.failure for oc in self.partial_results.values())

    @property
    def success_count(self) -> int:
        """Number of subtasks that completed successfully."""
        return sum(1 for oc in self.partial_results.values() if not oc.failure)

    @property
    def failure_count(self) -> int:
        """Number of subtasks that failed, timed out, or were skipped."""
        return sum(1 for oc in self.partial_results.values() if oc.failure)


@dataclass
class BenchmarkComparison:
    """Side-by-side parallel-vs-sequential measurement of the same task set.

    Attributes:
        task_order: Subtask ids in input order (used for stable reporting).
        parallel_wall_ms: Wall time of the parallel run.
        sequential_wall_ms: Wall time of the sequential run.
        speedup_factor: sequential_wall_ms / parallel_wall_ms.
        speedup_pct: Percentage wall-time reduction from going parallel.
        parallel_cost_usd: Total cost of the parallel run.
        sequential_cost_usd: Total cost of the sequential run.
        cost_delta_usd: parallel_cost_usd - sequential_cost_usd (≈ 0; parallel
            work costs the same tokens, it just finishes sooner).
        parallel_result: The full ParallelResult from the parallel run.
        sequential_outcomes: Per-subtask outcomes from the sequential run.
    """
    task_order:          list[str]
    parallel_wall_ms:    float
    sequential_wall_ms:  float
    speedup_factor:      float
    speedup_pct:         float
    parallel_cost_usd:   float
    sequential_cost_usd: float
    cost_delta_usd:      float
    parallel_result:     ParallelResult
    sequential_outcomes: dict[str, AgentOutcome]

    def report(self) -> str:
        """Render the book-style benchmark table as a string."""
        sep = "=" * 64
        n = len(self.task_order)
        lines: list[str] = [
            sep,
            f"  PARALLEL vs SEQUENTIAL BENCHMARK  ({n} subagents)",
            sep,
            "  Per-agent execution time (parallel run):",
        ]
        for tid in self.task_order:
            oc = self.parallel_result.partial_results[tid]
            flag = "" if not oc.failure else "  [FAIL]"
            lines.append(
                f"    [{oc.agent}]  {oc.elapsed_ms / 1000.0:>5.2f} s   "
                f"${oc.cost_usd:.5f}{flag}"
            )
        delta = self.cost_delta_usd
        sign = "+" if delta >= 0 else "-"
        pct = (delta / self.sequential_cost_usd * 100.0) if self.sequential_cost_usd else 0.0
        lines += [
            "  " + "-" * 60,
            f"  Parallel wall time   : {self.parallel_wall_ms / 1000.0:>6.2f} s",
            f"  Sequential wall time : {self.sequential_wall_ms / 1000.0:>6.2f} s",
            f"  Speedup              : {self.speedup_factor:.2f}x  "
            f"({self.speedup_pct:.1f}% faster)",
            "  " + "-" * 60,
            f"  Parallel cost        : ${self.parallel_cost_usd:.5f}",
            f"  Sequential cost      : ${self.sequential_cost_usd:.5f}",
            f"  Cost delta           : {sign}${abs(delta):.5f}  ({sign}{abs(pct):.1f}%)",
            sep,
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()


# ── runner ──────────────────────────────────────────────────────────────────────

class ParallelRunner:
    """Runs subagents concurrently with asyncio, bounded and dependency-aware.

    Args:
        max_concurrent: Maximum number of subagents executing at once.
        timeout_per_agent_seconds: Hard per-subagent wall-clock limit; a
            subagent exceeding it is recorded as a timeout failure.

    Usage::

        runner = ParallelRunner(max_concurrent=5, timeout_per_agent_seconds=120)
        result = await runner.run_parallel(subtasks, agent_factory)
        print(result.speedup_factor, result.total_cost_usd)
    """

    def __init__(
        self,
        max_concurrent:            int = _DEFAULT_MAX_CONCURRENT,
        timeout_per_agent_seconds: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.max_concurrent = max_concurrent
        self.timeout_per_agent_seconds = timeout_per_agent_seconds

        # Progress-line state (reset per run).
        self._progress_enabled = False
        self._progress_order: list[str] = []
        self._progress_label: dict[str, str] = {}
        self._progress_state: dict[str, tuple[str, float | None]] = {}
        self._progress_maxlen = 0

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------

    async def run_parallel(
        self,
        subtasks:      list[Subtask],
        agent_factory: AgentFactory,
        progress:      bool = True,
    ) -> ParallelResult:
        """Execute subtasks concurrently, honouring dependencies and limits.

        Each subtask waits for its dependencies, then competes for one of
        ``max_concurrent`` slots. Failures are isolated: a failed subtask is
        recorded and its dependents are skipped, but unrelated subtasks run to
        completion.

        Args:
            subtasks: Subtasks to execute. ``dependencies`` referencing ids not
                present in this list are ignored.
            agent_factory: Callable that runs one subtask (see AgentFactory).
            progress: When True, render a live in-place status line.

        Returns:
            A ParallelResult with successful results, the full per-subtask
            outcome view (``partial_results``), timing, speedup, and cost.
        """
        self._init_progress(subtasks, enabled=progress)

        by_id = {st.id: st for st in subtasks}
        events: dict[str, asyncio.Event] = {st.id: asyncio.Event() for st in subtasks}
        outcomes: dict[str, AgentOutcome] = {}
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_one(st: Subtask) -> None:
            oc: AgentOutcome | None = None
            try:
                deps = [d for d in st.dependencies if d in by_id]
                for dep in deps:
                    await events[dep].wait()

                failed_dep = next((d for d in deps if outcomes[d].failure), None)
                if failed_dep is not None:
                    oc = AgentOutcome(
                        task_id=st.id, agent=st.assigned_agent, failure=True,
                        result=None,
                        error=f"skipped: dependency '{failed_dep}' failed",
                        elapsed_ms=0.0, cost_usd=0.0, skipped=True,
                    )
                    self._set_progress(st.id, "skipped", 0.0)
                    return

                dep_results = {d: outcomes[d].result for d in deps}
                async with semaphore:
                    self._set_progress(st.id, "running", None)
                    oc = await self._execute(agent_factory, st, dep_results)

                status = (
                    "timeout"  if oc.timed_out else
                    "complete" if not oc.failure else
                    "failed"
                )
                self._set_progress(st.id, status, oc.elapsed_ms / 1000.0)
            finally:
                if oc is None:  # defensive: never leave a dependent waiting
                    oc = AgentOutcome(
                        task_id=st.id, agent=st.assigned_agent, failure=True,
                        result=None, error="internal runner error",
                        elapsed_ms=0.0, cost_usd=0.0,
                    )
                    self._set_progress(st.id, "failed", 0.0)
                outcomes[st.id] = oc
                events[st.id].set()

        wall_start = time.perf_counter()
        await asyncio.gather(*(run_one(st) for st in subtasks))
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        self._finish_progress()

        results = {tid: oc.result for tid, oc in outcomes.items() if not oc.failure}
        seq_equiv = sum(oc.elapsed_ms for oc in outcomes.values())
        speedup = seq_equiv / max(wall_ms, 1e-9)
        total_cost = sum(oc.cost_usd for oc in outcomes.values())

        return ParallelResult(
            results=results,
            partial_results=outcomes,
            wall_time_ms=round(wall_ms, 1),
            sequential_equivalent_ms=round(seq_equiv, 1),
            speedup_factor=round(speedup, 2),
            total_cost_usd=round(total_cost, 6),
        )

    async def _execute(
        self,
        agent_factory: AgentFactory,
        subtask:       Subtask,
        dep_results:   dict[str, Any],
    ) -> AgentOutcome:
        """Run a single subagent under the per-agent timeout and capture its outcome.

        Sync factories are offloaded to a worker thread; coroutine factories are
        awaited directly. Timeouts and exceptions are converted into failure
        outcomes rather than propagated.

        Args:
            agent_factory: The subagent execution callable.
            subtask: The subtask being executed.
            dep_results: Completed dependency results, keyed by subtask id.

        Returns:
            An AgentOutcome describing success or the specific failure.
        """
        start = time.perf_counter()
        try:
            if asyncio.iscoroutinefunction(agent_factory):
                result = await asyncio.wait_for(
                    agent_factory(subtask, dep_results),
                    timeout=self.timeout_per_agent_seconds,
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(agent_factory, subtask, dep_results),
                    timeout=self.timeout_per_agent_seconds,
                )
            elapsed = (time.perf_counter() - start) * 1000.0
            success = _result_success(result)
            return AgentOutcome(
                task_id=subtask.id, agent=subtask.assigned_agent,
                failure=not success, result=result,
                error=None if success else _result_error(result),
                elapsed_ms=elapsed, cost_usd=_result_cost(result),
            )
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - start) * 1000.0
            return AgentOutcome(
                task_id=subtask.id, agent=subtask.assigned_agent, failure=True,
                result=None,
                error=f"timeout after {self.timeout_per_agent_seconds}s",
                elapsed_ms=elapsed, cost_usd=0.0, timed_out=True,
            )
        except Exception as exc:  # noqa: BLE001 -- isolate this subagent's failure
            elapsed = (time.perf_counter() - start) * 1000.0
            return AgentOutcome(
                task_id=subtask.id, agent=subtask.assigned_agent, failure=True,
                result=None, error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=elapsed, cost_usd=0.0,
            )

    # ------------------------------------------------------------------
    # Sequential execution (for benchmarking)
    # ------------------------------------------------------------------

    async def _run_sequential(
        self,
        subtasks:      list[Subtask],
        agent_factory: AgentFactory,
    ) -> tuple[dict[str, AgentOutcome], float]:
        """Run subtasks one at a time in dependency order; return outcomes + wall ms."""
        order = _topological_order(subtasks)
        present = {st.id for st in subtasks}
        outcomes: dict[str, AgentOutcome] = {}

        wall_start = time.perf_counter()
        for st in order:
            deps = [d for d in st.dependencies if d in present]
            failed_dep = next((d for d in deps if outcomes[d].failure), None)
            if failed_dep is not None:
                outcomes[st.id] = AgentOutcome(
                    task_id=st.id, agent=st.assigned_agent, failure=True,
                    result=None, error=f"skipped: dependency '{failed_dep}' failed",
                    elapsed_ms=0.0, cost_usd=0.0, skipped=True,
                )
                continue
            dep_results = {d: outcomes[d].result for d in deps}
            outcomes[st.id] = await self._execute(agent_factory, st, dep_results)
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        return outcomes, wall_ms

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    async def benchmark_vs_sequential(
        self,
        tasks:         list[Subtask],
        agent_factory: AgentFactory,
    ) -> BenchmarkComparison:
        """Run the same tasks both in parallel and sequentially and compare.

        The parallel run is measured first (with live progress), then the
        identical work is run sequentially. The comparison reports the
        wall-clock speedup and the cost delta between the two modes.

        Args:
            tasks: Subtasks to benchmark.
            agent_factory: The subagent execution callable (see AgentFactory).

        Returns:
            A BenchmarkComparison with both wall times, speedup, and cost delta.
        """
        parallel = await self.run_parallel(tasks, agent_factory, progress=True)
        seq_outcomes, seq_wall = await self._run_sequential(tasks, agent_factory)

        seq_cost = sum(oc.cost_usd for oc in seq_outcomes.values())
        par_wall = parallel.wall_time_ms
        speedup = seq_wall / max(par_wall, 1e-9)
        speedup_pct = (1.0 - par_wall / max(seq_wall, 1e-9)) * 100.0

        return BenchmarkComparison(
            task_order=[st.id for st in tasks],
            parallel_wall_ms=round(par_wall, 1),
            sequential_wall_ms=round(seq_wall, 1),
            speedup_factor=round(speedup, 2),
            speedup_pct=round(speedup_pct, 1),
            parallel_cost_usd=round(parallel.total_cost_usd, 6),
            sequential_cost_usd=round(seq_cost, 6),
            cost_delta_usd=round(parallel.total_cost_usd - seq_cost, 6),
            parallel_result=parallel,
            sequential_outcomes=seq_outcomes,
        )

    # ------------------------------------------------------------------
    # Progress reporting (single in-place status line)
    # ------------------------------------------------------------------

    def _init_progress(self, subtasks: list[Subtask], enabled: bool) -> None:
        """Reset progress state for a new run and render the initial line."""
        self._progress_enabled = enabled
        self._progress_order = [st.id for st in subtasks]
        self._progress_label = {st.id: st.assigned_agent for st in subtasks}
        self._progress_state = {st.id: ("pending", None) for st in subtasks}
        self._progress_maxlen = 0
        if enabled:
            self._render()

    def _set_progress(self, task_id: str, status: str, elapsed_s: float | None) -> None:
        """Update one subtask's progress state and re-render the status line."""
        self._progress_state[task_id] = (status, elapsed_s)
        if self._progress_enabled:
            self._render()

    def _render(self) -> None:
        """Render the in-place status line: '[a] complete 2.1s | [b] running... | ...'."""
        parts = [
            _format_progress(self._progress_label[tid], *self._progress_state[tid])
            for tid in self._progress_order
        ]
        line = " | ".join(parts)
        self._progress_maxlen = max(self._progress_maxlen, len(line))
        sys.stdout.write("\r" + line.ljust(self._progress_maxlen))
        sys.stdout.flush()

    def _finish_progress(self) -> None:
        """Terminate the in-place line with a newline."""
        if self._progress_enabled:
            sys.stdout.write("\n")
            sys.stdout.flush()


# ── module helpers ────────────────────────────────────────────────────────────

def _format_progress(label: str, status: str, elapsed_s: float | None) -> str:
    """Format one agent's progress segment for the status line."""
    if status == "running":
        return f"[{label}] running..."
    if status == "complete":
        return f"[{label}] complete {elapsed_s:.1f}s"
    if status == "failed":
        return f"[{label}] failed {elapsed_s:.1f}s"
    if status == "timeout":
        return f"[{label}] timeout"
    if status == "skipped":
        return f"[{label}] skipped"
    return f"[{label}] pending"


def _topological_order(subtasks: list[Subtask]) -> list[Subtask]:
    """Order subtasks so every subtask follows its dependencies (stable).

    Args:
        subtasks: Subtasks to order.

    Returns:
        Subtasks in a dependency-respecting order, preserving input order among
        independent subtasks.

    Raises:
        ValueError: If the dependency graph contains a cycle.
    """
    present = {st.id for st in subtasks}
    deps = {st.id: [d for d in st.dependencies if d in present] for st in subtasks}
    done: set[str] = set()
    order: list[Subtask] = []

    while len(order) < len(subtasks):
        progressed = False
        for st in subtasks:
            if st.id in done:
                continue
            if all(d in done for d in deps[st.id]):
                order.append(st)
                done.add(st.id)
                progressed = True
        if not progressed:
            remaining = [st.id for st in subtasks if st.id not in done]
            raise ValueError(f"Cyclic subtask dependencies among: {remaining}")
    return order


def _result_success(result: Any) -> bool:
    """True if the factory's result is non-None and not flagged as a failure."""
    if result is None:
        return False
    return bool(getattr(result, "success", True))


def _result_cost(result: Any) -> float:
    """Extract ``cost_usd`` from a result object, defaulting to 0.0."""
    try:
        return float(getattr(result, "cost_usd", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _result_error(result: Any) -> str | None:
    """Extract an error message from a failed result object."""
    if result is None:
        return "no result produced"
    return getattr(result, "error", None) or "subagent reported failure"


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    from ch08_multi_agent.supervisor import MultiAgentSupervisor, SubagentConfig

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set. Add it to .env and retry.")
        raise SystemExit(1)

    # Three independent research subagents — fully parallelisable, so the
    # parallel wall time approaches the slowest single agent while the
    # sequential wall time is the sum of all three.
    configs: dict[str, SubagentConfig] = {
        "researcher_notion": SubagentConfig(
            model="claude-sonnet-4-6", tools=[], max_turns=1, budget_usd=0.40,
            specialization_description=(
                "Summarises one note-taking app's strengths and weaknesses in "
                "exactly three short bullet points."
            ),
        ),
        "researcher_obsidian": SubagentConfig(
            model="claude-sonnet-4-6", tools=[], max_turns=1, budget_usd=0.40,
            specialization_description=(
                "Summarises one note-taking app's strengths and weaknesses in "
                "exactly three short bullet points."
            ),
        ),
        "analyst_market": SubagentConfig(
            model="claude-sonnet-4-6", tools=[], max_turns=1, budget_usd=0.40,
            specialization_description=(
                "Lists current trends in the personal note-taking app market in "
                "exactly three short bullet points."
            ),
        ),
    }

    supervisor = MultiAgentSupervisor(
        supervisor_model="claude-sonnet-4-6", subagent_configs=configs
    )

    def agent_factory(subtask: Subtask, dep_results: dict[str, Any]) -> SubtaskResult:
        """Run one subtask via the supervisor's subagent machinery."""
        config = configs[subtask.assigned_agent]
        context = MultiAgentSupervisor._dependency_context(subtask, dep_results)
        return supervisor._run_subagent(
            subtask.assigned_agent, config, subtask, context
        )

    tasks: list[Subtask] = [
        Subtask("t1", "Summarise Notion's strengths and weaknesses as a "
                      "note-taking app.", "researcher_notion", []),
        Subtask("t2", "Summarise Obsidian's strengths and weaknesses as a "
                      "note-taking app.", "researcher_obsidian", []),
        Subtask("t3", "List three current trends in the personal note-taking "
                      "app market.", "analyst_market", []),
    ]

    runner = ParallelRunner(max_concurrent=5, timeout_per_agent_seconds=120)

    print("\nBenchmarking 3 subagents: parallel vs sequential...\n")
    comparison = asyncio.run(runner.benchmark_vs_sequential(tasks, agent_factory))
    print("\n" + comparison.report())
