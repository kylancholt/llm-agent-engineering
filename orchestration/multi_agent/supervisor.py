"""
Multi-agent supervisor with typed subagent delegation.

A supervisor takes a single high-level task and runs the full
plan -> dispatch -> execute -> synthesise pipeline against a roster of
specialist subagents:

  1. decompose(task)      -- the supervisor LLM splits the task into typed
                             Subtasks, each assigned to a named specialist and
                             carrying an explicit dependency list.
  2. dispatch(decomp)     -- the dependency graph is topologically layered into
                             execution "waves"; subtasks in the same wave have
                             no unmet dependencies and run in parallel.
  3. run(task)            -- waves execute in order (parallel within a wave),
                             each subagent runs under its own model, turn cap,
                             and dollar budget; finally the supervisor LLM
                             synthesises the subagent outputs into one answer.

Each specialist is described declaratively by a SubagentConfig (model, tools,
turn cap, budget, and a natural-language specialization). The supervisor uses
those specialization descriptions to decide which agent owns which subtask.

Pricing/cost accounting reuses ch01_why_agents_break.AgentCostTracker. Actual
API model IDs (e.g. claude-sonnet-4-6) are mapped to the tracker's pricing tiers
via _cost_model, mirroring ch03_agent_loop.

Requires ANTHROPIC_API_KEY in the environment or the project-root .env file.
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

# ── project root: orchestration/multi_agent/../../ = root ───────────────────────
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from ch01_why_agents_break.cost_tracker import AgentCostTracker, AlertLevel


# ── .env loader (mirrors ch03_agent_loop pattern) ───────────────────────────────

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

_SUPERVISOR_MODEL_DEFAULT = "claude-sonnet-4-6"
_SUBAGENT_MODEL_DEFAULT   = "claude-sonnet-4-6"

_DECOMPOSE_MAX_TOKENS = 5000
_SYNTH_MAX_TOKENS     = 5000
_SUBAGENT_MAX_TOKENS  = 4096
_DEFAULT_EFFORT       = "medium"   # low | medium | high | max
_MAX_CONTINUATIONS    = 4          # pause_turn resumes per LLM call (server tools)

# Per-subtask token assumption used only for the pre-execution cost *estimate*.
_EST_INPUT_TOKENS  = 2500
_EST_OUTPUT_TOKENS = 1200
_EST_SUPERVISOR_IN  = 3000
_EST_SUPERVISOR_OUT = 1500

# Declarative tool name -> Anthropic server-tool definition.
_SERVER_TOOLS: dict[str, dict[str, Any]] = {
    "web_search": {"type": "web_search_20260209", "name": "web_search"},
    "web_fetch":  {"type": "web_fetch_20260209",  "name": "web_fetch"},
}


def _cost_model(model: str) -> str:
    """Map any Claude model ID to the nearest AgentCostTracker pricing tier."""
    lc = model.lower()
    if "haiku" in lc:
        return "claude-haiku-4-5"
    if "opus" in lc:
        return "claude-sonnet-4-6"
    return "claude-sonnet-4-6"


def _price(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost of a single (input, output) token pair for a model."""
    tracker = AgentCostTracker(budget_usd=1e9, model=_cost_model(model))
    status = tracker.record_turn(input_tokens, output_tokens, turn_id=0)
    return status.turn_cost_usd


# ── public types ──────────────────────────────────────────────────────────────

@dataclass
class SubagentConfig:
    """Declarative configuration for one specialist subagent.

    Attributes:
        model: Claude model ID this specialist runs on (e.g. "claude-sonnet-4-6").
        tools: Declarative capability names granted to the specialist. Known
            server tools ("web_search", "web_fetch") are wired to the Anthropic
            Messages API automatically; unknown names are ignored.
        max_turns: Hard cap on LLM round-trips for a single subtask.
        budget_usd: Spend ceiling for a single subtask; the loop halts when the
            specialist's cumulative cost crosses the HALT threshold.
        specialization_description: Natural-language description of what this
            specialist is good at. The supervisor reads this when deciding which
            agent should own each subtask, so make it concrete.
    """
    model:                      str       = _SUBAGENT_MODEL_DEFAULT
    tools:                      list[str] = field(default_factory=list)
    max_turns:                  int       = 3
    budget_usd:                 float     = 0.50
    specialization_description: str       = ""


@dataclass
class Subtask:
    """A single unit of delegated work produced by decompose().

    Attributes:
        id: Stable identifier (e.g. "t1") referenced by other subtasks' deps.
        description: Self-contained instruction for the assigned specialist.
        assigned_agent: Name of the SubagentConfig that will execute this.
        dependencies: IDs of subtasks that must finish before this one runs;
            their outputs are injected into this subtask's context.
        required_output_format: Short instruction describing the shape the
            specialist's output should take (e.g. "bullet points", "JSON").
    """
    id:                     str
    description:            str
    assigned_agent:         str
    dependencies:           list[str] = field(default_factory=list)
    required_output_format: str       = "concise prose"


@dataclass
class TaskDecomposition:
    """The supervisor's plan for a task, returned by decompose().

    Attributes:
        subtasks: Ordered list of Subtasks covering the task.
        dependency_graph: Mapping of subtask id -> list of prerequisite ids.
        estimated_total_cost: Heuristic pre-execution cost estimate (USD),
            derived from assumed per-subtask token usage at each assigned
            agent's price plus supervisor decompose/synthesise overhead.
    """
    subtasks:              list[Subtask]
    dependency_graph:      dict[str, list[str]]
    estimated_total_cost:  float


@dataclass
class DispatchPlan:
    """Execution schedule for a TaskDecomposition, returned by dispatch().

    Attributes:
        waves: Ordered list of waves; each wave is a list of subtask ids that
            can run concurrently (all dependencies satisfied by earlier waves).
        max_parallelism: Largest wave size — the peak concurrency the run will
            use.
        total_subtasks: Total number of subtasks scheduled.
        agent_assignments: Mapping of subtask id -> assigned agent name.
    """
    waves:             list[list[str]]
    max_parallelism:   int
    total_subtasks:    int
    agent_assignments: dict[str, str]


@dataclass
class SubtaskResult:
    """Outcome of executing one Subtask via its specialist subagent.

    Attributes:
        subtask_id: The executed subtask's id.
        agent: Name of the specialist that ran it.
        output: The specialist's final text output ("" on hard failure).
        success: True when the specialist produced an answer without an
            unrecoverable API error.
        error: Error message when success is False (else None).
        input_tokens: Total input tokens consumed across the subtask's turns.
        output_tokens: Total output tokens generated across the subtask's turns.
        cost_usd: Measured cost of the subtask.
        turns: Number of LLM round-trips used.
        wall_time_ms: Wall-clock duration of the subtask.
    """
    subtask_id:    str
    agent:         str
    output:        str
    success:       bool
    error:         str | None
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    turns:         int
    wall_time_ms:  float


@dataclass
class SupervisorResult:
    """Aggregate result of MultiAgentSupervisor.run().

    Attributes:
        final_answer: The synthesised answer combining all subagent outputs.
        subtask_results: Mapping of subtask id -> SubtaskResult.
        total_cost_usd: Decompose + all subagents + synthesis cost.
        total_turns: Total LLM round-trips across the whole pipeline.
        wall_time_ms: End-to-end wall-clock duration.
        agents_used: Sorted list of distinct specialist names that ran.
    """
    final_answer:    str
    subtask_results: dict[str, SubtaskResult]
    total_cost_usd:  float
    total_turns:     int
    wall_time_ms:    float
    agents_used:     list[str]


@dataclass
class _CallStats:
    """Internal accounting for a single supervisor LLM call."""
    input_tokens:  int
    output_tokens: int
    turns:         int
    cost_usd:      float


# ── supervisor ──────────────────────────────────────────────────────────────────

class MultiAgentSupervisor:
    """Orchestrates a roster of typed specialist subagents for one task.

    The supervisor itself is an LLM (``supervisor_model``) responsible for
    planning (decompose) and synthesis. Each specialist is described by a
    SubagentConfig keyed by name in ``subagent_configs``.

    Usage::

        configs = {
            "researcher": SubagentConfig(
                model="claude-sonnet-4-6",
                specialization_description="Researches one product in depth.",
            ),
            "writer": SubagentConfig(
                model="claude-sonnet-4-6",
                specialization_description="Writes comparison documents.",
            ),
        }
        supervisor = MultiAgentSupervisor("claude-sonnet-4-6", configs)
        result = supervisor.run("Compare product A and product B, then ...")
        print(result.final_answer)

    Args:
        supervisor_model: Claude model ID used for decomposition and synthesis.
        subagent_configs: Mapping of specialist name -> SubagentConfig.

    Raises:
        ValueError: If no subagents are configured.
    """

    def __init__(
        self,
        supervisor_model:  str = _SUPERVISOR_MODEL_DEFAULT,
        subagent_configs:  dict[str, SubagentConfig] | None = None,
    ) -> None:
        if not subagent_configs:
            raise ValueError("At least one subagent must be configured.")
        self.supervisor_model = supervisor_model
        self.subagent_configs = subagent_configs
        self._client          = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # 1. Decomposition
    # ------------------------------------------------------------------

    def decompose(self, task: str) -> TaskDecomposition:
        """Split a task into typed, assigned, dependency-aware subtasks.

        Calls the supervisor LLM with the specialist roster and a structured
        output schema, then validates the returned subtasks (known agents,
        unique ids, references to existing dependency ids).

        Args:
            task: Natural-language task to decompose.

        Returns:
            A TaskDecomposition with subtasks, dependency graph, and a heuristic
            cost estimate.

        Raises:
            ValueError: If the model returns no usable subtasks.
        """
        decomposition, _ = self._decompose(task)
        return decomposition

    def _decompose(self, task: str) -> tuple[TaskDecomposition, _CallStats]:
        """decompose() plus the supervisor call's cost/turn accounting."""
        roster = "\n".join(
            f"  - {name}: {cfg.specialization_description or '(no description)'}"
            for name, cfg in self.subagent_configs.items()
        )
        system = (
            "You are a planning supervisor that decomposes a user task into "
            "subtasks for a team of specialist agents.\n\n"
            "Available specialists:\n"
            f"{roster}\n\n"
            "Rules:\n"
            "  - Produce between 2 and 6 subtasks.\n"
            "  - Assign each subtask to exactly one specialist by name.\n"
            "  - Give each subtask a stable id like 't1', 't2', 't3'.\n"
            "  - 'dependencies' lists the ids of subtasks that must finish "
            "before this one (their outputs are given to this subtask). "
            "Independent subtasks have an empty dependency list.\n"
            "  - 'required_output_format' is a short instruction on the shape "
            "of the expected output (e.g. 'concise bullet points').\n"
            "  - Subtasks that can run in parallel should NOT depend on each "
            "other. Aggregation/writing subtasks should depend on the research "
            "they consume."
        )
        user = f"Decompose this task for the team:\n\n{task}"
        schema = _decomposition_schema(list(self.subagent_configs.keys()))

        data, in_tok, out_tok, turns = self._call_structured(
            system, user, schema, _DECOMPOSE_MAX_TOKENS
        )

        raw_subtasks = data.get("subtasks") or []
        if not raw_subtasks:
            raise ValueError("Decomposition produced no subtasks.")

        subtasks: list[Subtask] = []
        seen_ids: set[str] = set()
        for item in raw_subtasks:
            sid = str(item["id"])
            if sid in seen_ids:
                continue  # ignore duplicate ids defensively
            seen_ids.add(sid)
            agent = str(item["assigned_agent"])
            if agent not in self.subagent_configs:
                # Schema enum should prevent this; fall back to the first agent.
                agent = next(iter(self.subagent_configs))
            subtasks.append(Subtask(
                id=sid,
                description=str(item["description"]),
                assigned_agent=agent,
                dependencies=[str(d) for d in item.get("dependencies", [])],
                required_output_format=str(
                    item.get("required_output_format", "concise prose")
                ),
            ))

        # Drop dependency references to ids that don't exist.
        valid_ids = {st.id for st in subtasks}
        for st in subtasks:
            st.dependencies = [d for d in st.dependencies if d in valid_ids and d != st.id]

        graph = {st.id: list(st.dependencies) for st in subtasks}
        estimate = self._estimate_cost(subtasks)
        decomposition = TaskDecomposition(
            subtasks=subtasks,
            dependency_graph=graph,
            estimated_total_cost=round(estimate, 6),
        )
        stats = _CallStats(
            input_tokens=in_tok,
            output_tokens=out_tok,
            turns=turns,
            cost_usd=_price(self.supervisor_model, in_tok, out_tok),
        )
        return decomposition, stats

    def _estimate_cost(self, subtasks: list[Subtask]) -> float:
        """Heuristic pre-execution cost estimate for a set of subtasks."""
        total = 0.0
        for st in subtasks:
            cfg = self.subagent_configs[st.assigned_agent]
            est_turns = max(1, min(cfg.max_turns, 2))
            total += _price(
                cfg.model,
                _EST_INPUT_TOKENS * est_turns,
                _EST_OUTPUT_TOKENS * est_turns,
            )
        # Supervisor overhead: one decompose call + one synthesis call.
        total += 2 * _price(
            self.supervisor_model, _EST_SUPERVISOR_IN, _EST_SUPERVISOR_OUT
        )
        return total

    # ------------------------------------------------------------------
    # 2. Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, decomposition: TaskDecomposition) -> DispatchPlan:
        """Layer the dependency graph into parallel execution waves.

        Uses Kahn-style wave extraction: each wave contains every subtask whose
        dependencies are all satisfied by earlier waves. Subtasks within a wave
        run concurrently.

        Args:
            decomposition: The plan produced by decompose().

        Returns:
            A DispatchPlan with ordered waves and parallelism metadata.

        Raises:
            ValueError: If the dependency graph contains a cycle.
        """
        graph = decomposition.dependency_graph
        remaining = set(graph.keys())
        completed: set[str] = set()
        waves: list[list[str]] = []

        while remaining:
            ready = sorted(
                sid for sid in remaining
                if all(dep in completed for dep in graph[sid])
            )
            if not ready:
                raise ValueError(
                    f"Cyclic or unresolvable dependencies among: {sorted(remaining)}"
                )
            waves.append(ready)
            completed.update(ready)
            remaining.difference_update(ready)

        assignments = {st.id: st.assigned_agent for st in decomposition.subtasks}
        return DispatchPlan(
            waves=waves,
            max_parallelism=max((len(w) for w in waves), default=0),
            total_subtasks=len(graph),
            agent_assignments=assignments,
        )

    # ------------------------------------------------------------------
    # 3. Run
    # ------------------------------------------------------------------

    def run(self, task: str, verbose: bool = True) -> SupervisorResult:
        """Execute the full plan -> dispatch -> execute -> synthesise pipeline.

        Decomposes the task, schedules it into waves, runs each wave (parallel
        within the wave) under per-subagent model/turn/budget limits, then asks
        the supervisor LLM to synthesise a single final answer from the
        subagent outputs.

        Args:
            task: Natural-language task to solve.
            verbose: When True, print the decomposition plan, dispatch schedule,
                per-subtask results, and a final metrics summary.

        Returns:
            A SupervisorResult with the final answer, per-subtask results, and
            aggregate cost/turn/wall-time metrics.
        """
        t0 = time.perf_counter()

        decomposition, dstats = self._decompose(task)
        if verbose:
            self._print_decomposition(task, decomposition)

        plan = self.dispatch(decomposition)
        if verbose:
            self._print_dispatch_plan(plan)

        subtask_by_id = {st.id: st for st in decomposition.subtasks}
        results: dict[str, SubtaskResult] = {}

        for wave_idx, wave in enumerate(plan.waves, start=1):
            if verbose:
                names = ", ".join(f"{sid}->{plan.agent_assignments[sid]}" for sid in wave)
                print(f"\n  Wave {wave_idx}/{len(plan.waves)} (parallel: {names})")

            with ThreadPoolExecutor(max_workers=max(1, len(wave))) as pool:
                futures = {}
                for sid in wave:
                    st  = subtask_by_id[sid]
                    cfg = self.subagent_configs[st.assigned_agent]
                    ctx = self._dependency_context(st, results)
                    fut = pool.submit(self._run_subagent, st.assigned_agent, cfg, st, ctx)
                    futures[fut] = sid

                for fut in as_completed(futures):
                    res = fut.result()
                    results[res.subtask_id] = res
                    if verbose:
                        self._print_subtask_result(res)

        final_answer, sstats = self._synthesize(task, decomposition, results)

        sub_cost  = sum(r.cost_usd for r in results.values())
        sub_turns = sum(r.turns for r in results.values())
        result = SupervisorResult(
            final_answer=final_answer,
            subtask_results=results,
            total_cost_usd=round(dstats.cost_usd + sstats.cost_usd + sub_cost, 6),
            total_turns=dstats.turns + sstats.turns + sub_turns,
            wall_time_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            agents_used=sorted({st.assigned_agent for st in decomposition.subtasks}),
        )

        if verbose:
            self._print_summary(result)
        return result

    # ------------------------------------------------------------------
    # Subagent execution
    # ------------------------------------------------------------------

    def _run_subagent(
        self,
        agent_name: str,
        config:     SubagentConfig,
        subtask:    Subtask,
        dep_context: str,
    ) -> SubtaskResult:
        """Run one specialist on one subtask under its model/turn/budget limits.

        Server tools declared in ``config.tools`` are attached to the request;
        if the API rejects them, the call retries once without tools rather than
        failing the subtask. The loop honours ``pause_turn`` (server-tool
        continuation), the per-subtask turn cap, and the dollar budget.

        Args:
            agent_name: Name of the specialist.
            config: The specialist's configuration.
            subtask: The subtask to execute.
            dep_context: Pre-formatted outputs of completed dependencies.

        Returns:
            A SubtaskResult capturing output, success, cost, and timing.
        """
        t0 = time.perf_counter()
        tracker = AgentCostTracker(budget_usd=max(config.budget_usd, 1e-9),
                                   model=_cost_model(config.model))

        system = (
            f"You are the '{agent_name}' specialist.\n"
            f"{config.specialization_description}\n\n"
            f"Required output format: {subtask.required_output_format}.\n"
            "Stay strictly within your subtask. Be concise and concrete."
        )
        user_parts = [f"Subtask: {subtask.description}"]
        if dep_context:
            user_parts.append("\nContext from completed dependencies:\n" + dep_context)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "\n".join(user_parts)}
        ]

        tools = _anthropic_tools(config.tools)
        use_tools = bool(tools)

        answer = ""
        in_total = out_total = 0
        turns = 0
        success = True
        error: str | None = None

        turn = 0
        while turn < config.max_turns:
            turn += 1
            turns = turn
            kwargs: dict[str, Any] = {
                "model":         config.model,
                "max_tokens":    _SUBAGENT_MAX_TOKENS,
                "system":        system,
                "messages":      messages,
                "thinking":      {"type": "adaptive"},
                "output_config": {"effort": _DEFAULT_EFFORT},
            }
            if use_tools:
                kwargs["tools"] = tools

            try:
                resp = self._client.messages.create(**kwargs)
            except anthropic.BadRequestError as exc:
                if use_tools:
                    use_tools = False          # tools unavailable: retry plain
                    turn -= 1
                    continue
                success, error = False, f"BadRequest: {exc}"
                break
            except anthropic.APIError as exc:
                success, error = False, f"APIError: {exc}"
                break

            in_total  += resp.usage.input_tokens
            out_total += resp.usage.output_tokens
            status = tracker.record_turn(
                resp.usage.input_tokens, resp.usage.output_tokens, turn_id=turn
            )

            text = _extract_text(resp)
            if text:
                answer = text
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "pause_turn" and turn < config.max_turns:
                continue  # server tool wants to keep going — resume
            if status.alert_level is AlertLevel.HALT:
                error = error or "budget halt"
                break
            break  # end_turn / max_tokens / refusal / no client tools to run

        if not answer and success:
            success, error = False, error or "no output produced"

        return SubtaskResult(
            subtask_id=subtask.id,
            agent=agent_name,
            output=answer,
            success=success,
            error=error,
            input_tokens=in_total,
            output_tokens=out_total,
            cost_usd=round(tracker.get_report().total_cost_usd, 6),
            turns=turns,
            wall_time_ms=round((time.perf_counter() - t0) * 1000.0, 1),
        )

    @staticmethod
    def _dependency_context(subtask: Subtask, results: dict[str, SubtaskResult]) -> str:
        """Format the outputs of a subtask's completed dependencies."""
        chunks: list[str] = []
        for dep in subtask.dependencies:
            res = results.get(dep)
            if res is None:
                continue
            label = f"[{dep} — {res.agent}]"
            body = res.output if res.success else f"(failed: {res.error})"
            chunks.append(f"{label}\n{body}")
        return "\n\n".join(chunks)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _synthesize(
        self,
        task:          str,
        decomposition: TaskDecomposition,
        results:       dict[str, SubtaskResult],
    ) -> tuple[str, _CallStats]:
        """Combine all subagent outputs into one coherent final answer."""
        system = (
            "You are the supervisor. Synthesise the specialists' results into a "
            "single, coherent, well-structured answer to the user's original "
            "task. Resolve overlaps, keep it self-contained, and do not mention "
            "the internal subtask machinery."
        )
        sections = [f"Original task:\n{task}\n", "Specialist results:"]
        for st in decomposition.subtasks:
            res = results.get(st.id)
            body = res.output if (res and res.success) else "(no result)"
            sections.append(f"\n## {st.id} — {st.assigned_agent}: {st.description}\n{body}")
        user = "\n".join(sections)

        text, in_tok, out_tok, turns = self._call_text(system, user, _SYNTH_MAX_TOKENS)
        stats = _CallStats(
            input_tokens=in_tok,
            output_tokens=out_tok,
            turns=turns,
            cost_usd=_price(self.supervisor_model, in_tok, out_tok),
        )
        return text, stats

    # ------------------------------------------------------------------
    # Low-level supervisor LLM calls
    # ------------------------------------------------------------------

    def _call_structured(
        self,
        system:     str,
        user:       str,
        schema:     dict[str, Any],
        max_tokens: int,
    ) -> tuple[dict[str, Any], int, int, int]:
        """Call the supervisor model with a JSON-schema-constrained response.

        Returns:
            (parsed_dict, input_tokens, output_tokens, turns).

        Raises:
            ValueError: If the response cannot be parsed as JSON.
        """
        text, in_tok, out_tok, turns = self._pause_aware_call(
            system, user, max_tokens,
            output_config={"format": {"type": "json_schema", "schema": schema},
                            "effort": _DEFAULT_EFFORT},
        )
        return _parse_json(text), in_tok, out_tok, turns

    def _call_text(
        self,
        system:     str,
        user:       str,
        max_tokens: int,
    ) -> tuple[str, int, int, int]:
        """Call the supervisor model for a free-form text response."""
        return self._pause_aware_call(
            system, user, max_tokens, output_config={"effort": _DEFAULT_EFFORT}
        )

    def _pause_aware_call(
        self,
        system:        str,
        user:          str,
        max_tokens:    int,
        output_config: dict[str, Any],
    ) -> tuple[str, int, int, int]:
        """Run a supervisor call, resuming across pause_turn continuations.

        Returns:
            (text, input_tokens, output_tokens, turns).
        """
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        in_total = out_total = 0
        text = ""
        turns = 0

        for turn in range(1, _MAX_CONTINUATIONS + 1):
            turns = turn
            resp = self._client.messages.create(
                model=self.supervisor_model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config=output_config,
            )
            in_total  += resp.usage.input_tokens
            out_total += resp.usage.output_tokens
            piece = _extract_text(resp)
            if piece:
                text = piece
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "pause_turn":
                continue
            break

        return text, in_total, out_total, turns

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    @staticmethod
    def _print_decomposition(task: str, decomp: TaskDecomposition) -> None:
        sep = "=" * 78
        print(f"\n{sep}")
        print("  TASK DECOMPOSITION")
        print(sep)
        print(f"  Task: {task}")
        print(f"  Subtasks: {len(decomp.subtasks)} | "
              f"estimated cost: ${decomp.estimated_total_cost:.5f}")
        print(f"  {'-'*74}")
        for st in decomp.subtasks:
            deps = ", ".join(st.dependencies) or "(none)"
            print(f"  [{st.id}] -> {st.assigned_agent}")
            print(f"        {st.description}")
            print(f"        deps={deps} | output={st.required_output_format}")
        print(sep)

    @staticmethod
    def _print_dispatch_plan(plan: DispatchPlan) -> None:
        sep = "=" * 78
        print(f"\n{sep}")
        print("  DISPATCH PLAN")
        print(sep)
        print(f"  {plan.total_subtasks} subtasks in {len(plan.waves)} wave(s) | "
              f"max parallelism: {plan.max_parallelism}")
        for i, wave in enumerate(plan.waves, start=1):
            entries = ", ".join(f"{sid}({plan.agent_assignments[sid]})" for sid in wave)
            mode = "parallel" if len(wave) > 1 else "single"
            print(f"  Wave {i} [{mode}]: {entries}")
        print(sep)

    @staticmethod
    def _print_subtask_result(res: SubtaskResult) -> None:
        flag = "OK " if res.success else "ERR"
        snippet = res.output.replace("\n", " ")[:120]
        print(f"    [{flag}] {res.subtask_id} ({res.agent}) "
              f"turns={res.turns} cost=${res.cost_usd:.5f} "
              f"time={res.wall_time_ms:.0f}ms")
        if res.success:
            print(f"          {snippet}{'…' if len(res.output) > 120 else ''}")
        else:
            print(f"          error: {res.error}")

    @staticmethod
    def _print_summary(result: SupervisorResult) -> None:
        sep = "=" * 78
        print(f"\n{sep}")
        print("  FINAL ANSWER")
        print(sep)
        print(result.final_answer)
        print(f"\n{sep}")
        print("  RUN METRICS")
        print(sep)
        print(f"  Agents used    : {', '.join(result.agents_used)}")
        print(f"  Subtasks       : {len(result.subtask_results)}")
        print(f"  Total turns    : {result.total_turns}")
        print(f"  Total cost     : ${result.total_cost_usd:.5f}")
        print(f"  Wall time      : {result.wall_time_ms:.0f} ms")
        print(sep)


# ── module helpers ────────────────────────────────────────────────────────────

def _decomposition_schema(agent_names: list[str]) -> dict[str, Any]:
    """Build the JSON schema constraining decompose() output."""
    return {
        "type": "object",
        "properties": {
            "subtasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id":          {"type": "string"},
                        "description": {"type": "string"},
                        "assigned_agent": {"type": "string", "enum": agent_names},
                        "dependencies": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "required_output_format": {"type": "string"},
                    },
                    "required": [
                        "id", "description", "assigned_agent",
                        "dependencies", "required_output_format",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["subtasks"],
        "additionalProperties": False,
    }


def _anthropic_tools(tool_names: list[str]) -> list[dict[str, Any]]:
    """Map declarative tool names to Anthropic server-tool definitions."""
    return [_SERVER_TOOLS[n] for n in tool_names if n in _SERVER_TOOLS]


def _extract_text(resp: Any) -> str:
    """Join all text blocks in a Messages API response into one string."""
    return "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    ).strip()


def _parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model text, tolerating surrounding noise.

    Args:
        text: Model output expected to contain a JSON object.

    Returns:
        The parsed dict.

    Raises:
        ValueError: If no JSON object can be parsed.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse decomposition JSON: {exc}") from exc
    raise ValueError("Model returned no JSON object for decomposition.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set. Add it to .env and retry.")
        raise SystemExit(1)

    # Two research specialists feed one writer — a fan-out / fan-in pattern.
    subagent_configs: dict[str, SubagentConfig] = {
        "researcher": SubagentConfig(
            model="claude-sonnet-4-6",
            tools=[],                       # parametric knowledge; add "web_search" for live data
            max_turns=2,
            budget_usd=0.60,
            specialization_description=(
                "Researches a single product or company in depth: positioning, "
                "key features, pricing model, offline support, extensibility, "
                "and ideal user. Returns concise, factual bullet points about "
                "one subject only."
            ),
        ),
        "writer": SubagentConfig(
            model="claude-sonnet-4-6",
            tools=[],
            max_turns=2,
            budget_usd=0.60,
            specialization_description=(
                "Writes structured head-to-head comparison documents, "
                "synthesising research provided by other agents into a clear, "
                "decision-oriented narrative with a short recommendation."
            ),
        ),
    }

    supervisor = MultiAgentSupervisor(
        supervisor_model="claude-sonnet-4-6",
        subagent_configs=subagent_configs,
    )

    task = (
        "Research the note-taking apps Notion and Obsidian, then write a "
        "head-to-head comparison covering pricing, offline support, "
        "extensibility, and the ideal user for each."
    )

    print("\nRunning MultiAgentSupervisor on a 3-subagent task...")
    result = supervisor.run(task, verbose=True)
