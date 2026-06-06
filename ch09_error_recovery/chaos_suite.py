"""
Chaos test suite for agent resilience.

Injects five categories of controlled failure into ``AgentLoop.run()`` and
measures whether the agent produces an acceptable response despite each fault.

Five scenarios:

  tool_failure         -- randomly raises an exception inside a tool call
                          (the loop's ``_exec_tool`` catches it and passes the
                          error string to the LLM, which can retry)
  latency_spike        -- adds Gaussian-distributed sleep before each tool
                          returns, stressing timeout-sensitivity
  budget_exhaustion    -- caps ``budget_usd`` to a tiny value so the loop
                          terminates with BUDGET_EXCEEDED before the task is done
  context_corruption   -- replaces one tool result with garbled bytes to simulate
                          memory corruption or a scrambled API response
  partial_tool_failure -- truncates tool results with a configurable probability
                          to simulate incomplete or dropped API payloads

Quality is estimated heuristically (no LLM judge): FINAL_ANSWER with a
non-trivial answer scores 1.0, BUDGET_EXCEEDED scores 0.45, etc.
Resilience score = fraction of scenarios whose quality >= 0.60.

Requires ANTHROPIC_API_KEY (loaded from project-root .env).
"""
from __future__ import annotations

import functools
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# -- project root: ch09_error_recovery/../ = root -----------------------------
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ch03_agent_loop.agent_loop import (  # noqa: E402
    AgentLoop,
    LoopResult,
    TerminationReason,
)


# -- env loader ----------------------------------------------------------------

def _load_env(path: Path) -> None:
    """Read KEY=VALUE pairs from a .env file into os.environ (idempotent)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_load_env(_ROOT / ".env")


# -- error taxonomy for tool_failure scenario ----------------------------------

_ERROR_MAP: dict[str, type[Exception]] = {
    "RuntimeError":   RuntimeError,
    "TimeoutError":   TimeoutError,
    "ConnectionError": ConnectionError,
    "ValueError":     ValueError,
    "IOError":        IOError,
}


# -- result types --------------------------------------------------------------

@dataclass
class ChaosResult:
    """Outcome of a single chaos scenario run.

    Attributes:
        scenario:             Name of the chaos scenario injected.
        task_completed:       True when final_answer_quality >= 0.60.
        recovery_triggered:   True when at least one injected fault was visible
                              in the message history (tool error, corruption, etc.).
        steps_completed:      Tool calls that returned clean (non-error) results.
        steps_failed:         Tool calls that returned an injected-fault string.
        final_answer_quality: Heuristic score 0.0-1.0 (no LLM judge).
        cost_usd:             Total USD charged to the Anthropic API for this run.
        duration_s:           Wall-clock seconds for the run.
        termination_reason:   TerminationReason.value string from the loop.
        notes:                Short description of the injected chaos parameters.
    """

    scenario:             str
    task_completed:       bool
    recovery_triggered:   bool
    steps_completed:      int
    steps_failed:         int
    final_answer_quality: float
    cost_usd:             float
    duration_s:           float
    termination_reason:   str
    notes:                str


@dataclass
class ChaosReport:
    """Consolidated report for a full five-scenario chaos run.

    Attributes:
        results:          Per-scenario ChaosResult list (same order as SCENARIOS).
        resilience_score: Fraction of scenarios whose quality >= 0.60.
        scenarios_passed: Count of passing scenarios.
        scenarios_failed: Count of failing scenarios.
        total_cost_usd:   Sum of per-scenario costs.
        total_duration_s: Sum of per-scenario wall-clock seconds.
        summary:          One-line human-readable description.
    """

    results:          list[ChaosResult]
    resilience_score: float
    scenarios_passed: int
    scenarios_failed: int
    total_cost_usd:   float
    total_duration_s: float
    summary:          str


# -- chaos suite ---------------------------------------------------------------

class AgentChaosSuite:
    """Injects controlled failures into AgentLoop and measures resilience.

    Args:
        seed: RNG seed for reproducible failure injection (default 42).

    Usage::

        suite  = AgentChaosSuite(seed=42)
        report = suite.run_all(agent, task, tools)
        print(report.summary)
    """

    SCENARIOS: list[str] = [
        "tool_failure",
        "latency_spike",
        "budget_exhaustion",
        "context_corruption",
        "partial_tool_failure",
    ]

    _DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
        "tool_failure":         {"failure_rate": 0.50, "error_type": "RuntimeError"},
        "latency_spike":        {"mean_s": 0.40,  "std_s": 0.15},
        "budget_exhaustion":    {"budget_usd": 0.0001},
        "context_corruption":   {"corrupt_at_call": 1},
        "partial_tool_failure": {"corrupt_rate": 0.70},
    }

    # Quality >= this threshold -> task_completed = True
    _QUALITY_THRESHOLD: float = 0.60

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_scenario(
        self,
        scenario: str,
        agent:    AgentLoop,
        task:     str,
        params:   dict[str, Any],
    ) -> ChaosResult:
        """Run one chaos scenario against a fresh agent instance.

        A fresh ``AgentLoop`` is created for each scenario (same model/turns/
        token limits as ``agent``, different budget for budget_exhaustion and a
        scenario-namespaced checkpoint directory) to avoid state leakage.

        Args:
            scenario: One of the five scenario names (see SCENARIOS).
            agent:    Base ``AgentLoop`` whose config is cloned per run.
            task:     Natural-language task for the agent.
            params:   Chaos parameters.  Must include key ``"tools"``
                      (``dict[str, Callable[..., str]]``) with the clean tool
                      callables to wrap.

        Returns:
            ChaosResult with outcome metrics.

        Raises:
            ValueError: If ``scenario`` is not in SCENARIOS.
        """
        if scenario not in self.SCENARIOS:
            raise ValueError(
                f"Unknown scenario {scenario!r}. Valid: {self.SCENARIOS}"
            )

        base_tools: dict[str, Callable[..., str]] = params.get("tools", {})
        ckpt_dir = str(agent.checkpoint_dir) + f"/{scenario}"

        chaos_agent, tools, notes = self._setup(
            scenario, agent, base_tools, params, ckpt_dir
        )

        t0     = time.perf_counter()
        result = chaos_agent.run(task, tools)
        dur_s  = time.perf_counter() - t0

        completed, failed, recovery = self._analyze_messages(
            result.state_at_termination.messages
        )
        quality = self._estimate_quality(result, completed, failed, scenario)
        passed  = quality >= self._QUALITY_THRESHOLD

        return ChaosResult(
            scenario=scenario,
            task_completed=passed,
            recovery_triggered=recovery,
            steps_completed=completed,
            steps_failed=failed,
            final_answer_quality=round(quality, 3),
            cost_usd=result.total_cost_usd,
            duration_s=round(dur_s, 2),
            termination_reason=result.termination_reason.value,
            notes=notes,
        )

    def run_all(
        self,
        agent: AgentLoop,
        task:  str,
        tools: dict[str, Callable[..., str]],
    ) -> ChaosReport:
        """Run all five chaos scenarios and return a consolidated ChaosReport.

        Default chaos parameters (see _DEFAULT_PARAMS) are used for each
        scenario.  The ``tools`` dict must contain clean (un-wrapped) callables;
        wrapping is applied per-scenario internally.

        Args:
            agent: Base ``AgentLoop`` configuration to clone per scenario.
            task:  Task description given to the agent in every scenario.
            tools: Clean tool callables (dict of name -> callable).

        Returns:
            ChaosReport with per-scenario results and a composite resilience_score.
        """
        results: list[ChaosResult] = []
        for scenario in self.SCENARIOS:
            params = dict(self._DEFAULT_PARAMS[scenario])
            params["tools"] = tools
            cr = self.run_scenario(scenario, agent, task, params)
            results.append(cr)

        passed     = sum(1 for r in results if r.task_completed)
        total      = len(results)
        score      = passed / total if total > 0 else 0.0
        total_cost = sum(r.cost_usd for r in results)
        total_dur  = sum(r.duration_s for r in results)

        tier = (
            "EXCELLENT" if score >= 0.80 else
            "GOOD"      if score >= 0.60 else
            "DEGRADED"  if score >= 0.40 else
            "FRAGILE"
        )

        summary = (
            f"{passed}/{total} scenarios passed ({score:.0%}) -- "
            f"resilience tier: {tier}  |  "
            f"total cost: ${total_cost:.5f}  |  "
            f"total time: {total_dur:.1f}s"
        )

        return ChaosReport(
            results=results,
            resilience_score=round(score, 4),
            scenarios_passed=passed,
            scenarios_failed=total - passed,
            total_cost_usd=round(total_cost, 6),
            total_duration_s=round(total_dur, 2),
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Scenario setup
    # ------------------------------------------------------------------

    def _setup(
        self,
        scenario:   str,
        agent:      AgentLoop,
        base_tools: dict[str, Callable[..., str]],
        params:     dict[str, Any],
        ckpt_dir:   str,
    ) -> tuple[AgentLoop, dict[str, Callable[..., str]], str]:
        """Build the chaos agent and wrapped tool dict for one scenario."""

        if scenario == "tool_failure":
            rate  = float(params.get("failure_rate", 0.50))
            etype = str(params.get("error_type", "RuntimeError"))
            tools = self._wrap_tool_failure(base_tools, rate, etype)
            notes = f"failure_rate={rate}, error_type={etype}"
            chaos_agent = self._fresh_agent(agent, ckpt_dir=ckpt_dir)

        elif scenario == "latency_spike":
            mean_s = float(params.get("mean_s", 0.40))
            std_s  = float(params.get("std_s",  0.15))
            tools  = self._wrap_latency_spike(base_tools, mean_s, std_s)
            notes  = f"mean_s={mean_s}, std_s={std_s}"
            chaos_agent = self._fresh_agent(agent, ckpt_dir=ckpt_dir)

        elif scenario == "budget_exhaustion":
            bud   = float(params.get("budget_usd", 0.0001))
            tools = dict(base_tools)
            notes = f"budget_usd={bud}"
            chaos_agent = self._fresh_agent(agent, budget_usd=bud, ckpt_dir=ckpt_dir)

        elif scenario == "context_corruption":
            at    = int(params.get("corrupt_at_call", 1))
            tools = self._wrap_context_corruption(base_tools, at)
            notes = f"corrupt_at_call={at}"
            chaos_agent = self._fresh_agent(agent, ckpt_dir=ckpt_dir)

        elif scenario == "partial_tool_failure":
            rate  = float(params.get("corrupt_rate", 0.70))
            tools = self._wrap_partial_failure(base_tools, rate)
            notes = f"corrupt_rate={rate}"
            chaos_agent = self._fresh_agent(agent, ckpt_dir=ckpt_dir)

        else:
            raise ValueError(f"Unknown scenario: {scenario!r}")

        return chaos_agent, tools, notes

    # ------------------------------------------------------------------
    # Tool wrappers
    # ------------------------------------------------------------------

    def _wrap_tool_failure(
        self,
        tools:        dict[str, Callable[..., str]],
        failure_rate: float,
        error_type:   str,
    ) -> dict[str, Callable[..., str]]:
        """Randomly raise ``error_type`` with probability ``failure_rate``.

        The agent loop's ``_exec_tool`` catches any exception and returns
        ``"Error (<type>): <msg>"`` to the LLM, which can then retry the call.
        """
        ErrClass = _ERROR_MAP.get(error_type, RuntimeError)
        return {
            name: _make_failure_wrapper(fn, failure_rate, ErrClass, name, self._rng)
            for name, fn in tools.items()
        }

    def _wrap_latency_spike(
        self,
        tools:  dict[str, Callable[..., str]],
        mean_s: float,
        std_s:  float,
    ) -> dict[str, Callable[..., str]]:
        """Sleep for ``max(0, gauss(mean_s, std_s))`` seconds before each tool call."""
        return {
            name: _make_latency_wrapper(fn, mean_s, std_s, self._rng)
            for name, fn in tools.items()
        }

    def _wrap_context_corruption(
        self,
        tools:           dict[str, Callable[..., str]],
        corrupt_at_call: int,
    ) -> dict[str, Callable[..., str]]:
        """Return a garbled result on the N-th tool call across all tools.

        Simulates memory or transmission corruption: the first ``corrupt_at_call``
        calls return normally; call number ``corrupt_at_call`` returns a ROT-13-
        garbled prefix followed by the clean suffix.
        """
        call_count: list[int] = [0]
        return {
            name: _make_corruption_wrapper(fn, corrupt_at_call, call_count)
            for name, fn in tools.items()
        }

    def _wrap_partial_failure(
        self,
        tools:        dict[str, Callable[..., str]],
        corrupt_rate: float,
    ) -> dict[str, Callable[..., str]]:
        """Truncate results to one-third of their length with probability ``corrupt_rate``.

        Simulates dropped bytes or a partial API payload: the tool succeeds (no
        exception) but the LLM receives incomplete data.
        """
        return {
            name: _make_partial_wrapper(fn, corrupt_rate, self._rng)
            for name, fn in tools.items()
        }

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _analyze_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[int, int, bool]:
        """Scan the message history to count tool outcomes.

        A tool result is classified as *failed* when its content:
        - starts with "Error" (loop's _exec_tool exception handler)
        - contains "[CHAOS"   (tool_failure tag)
        - contains "[CORRUPT" (context_corruption tag)
        - contains "[PARTIAL" (partial_tool_failure tag)

        Returns:
            (steps_completed, steps_failed, recovery_triggered)
        """
        completed = failed = 0
        recovery  = False

        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                text = str(block.get("content", ""))
                is_fault = (
                    text.startswith("Error")
                    or "[CHAOS"   in text
                    or "[CORRUPT" in text
                    or "[PARTIAL" in text
                )
                if is_fault:
                    failed  += 1
                    recovery = True
                else:
                    completed += 1

        return completed, failed, recovery

    @staticmethod
    def _estimate_quality(
        result:          LoopResult,
        steps_completed: int,
        steps_failed:    int,
        scenario:        str,
    ) -> float:
        """Heuristic quality score 0.0-1.0 (no LLM judge required).

        FINAL_ANSWER with a non-trivial answer length -> 1.0 or 0.85.
        BUDGET_EXCEEDED -> 0.45 (intentional failure; never passes threshold).
        MAX_TURNS with partial work -> 0.50.
        ERROR -> 0.05.
        """
        reason = result.termination_reason
        answer = result.answer.strip()

        if reason is TerminationReason.FINAL_ANSWER:
            if len(answer) > 80:
                return 1.00
            if len(answer) > 20:
                return 0.85
            return 0.60

        if reason is TerminationReason.BUDGET_EXCEEDED:
            # Intentionally exhausted; grade on whether anything was said
            return 0.45 if answer else 0.20

        if reason is TerminationReason.MAX_TURNS:
            if steps_completed > 0 and answer:
                return 0.50
            return 0.25

        if reason is TerminationReason.INTERRUPTED:
            return 0.35 if answer else 0.15

        return 0.05  # ERROR

    @staticmethod
    def _fresh_agent(
        base:       AgentLoop,
        budget_usd: float | None = None,
        ckpt_dir:   str   | None = None,
    ) -> AgentLoop:
        """Clone ``base`` with an optional budget or checkpoint-dir override."""
        return AgentLoop(
            model=base.model,
            budget_usd=budget_usd if budget_usd is not None else base.budget_usd,
            max_turns=base.max_turns,
            checkpoint_dir=ckpt_dir or str(base.checkpoint_dir),
            max_output_tokens=base.max_output_tokens,
        )


# -- wrapper factory functions (module-level for clean closures) ---------------

def _make_failure_wrapper(
    fn:           Callable[..., str],
    failure_rate: float,
    ErrClass:     type[Exception],
    tool_name:    str,
    rng:          random.Random,
) -> Callable[..., str]:
    """Return a wrapped ``fn`` that randomly raises ``ErrClass``."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        if rng.random() < failure_rate:
            raise ErrClass(
                f"[CHAOS:tool_failure] {ErrClass.__name__} injected in '{tool_name}'"
            )
        return fn(*args, **kwargs)
    return wrapper


def _make_latency_wrapper(
    fn:     Callable[..., str],
    mean_s: float,
    std_s:  float,
    rng:    random.Random,
) -> Callable[..., str]:
    """Return a wrapped ``fn`` that sleeps before executing."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        delay = max(0.0, rng.gauss(mean_s, std_s))
        time.sleep(delay)
        return fn(*args, **kwargs)
    return wrapper


def _make_corruption_wrapper(
    fn:              Callable[..., str],
    corrupt_at_call: int,
    call_count:      list[int],
) -> Callable[..., str]:
    """Return a wrapped ``fn`` that garbles the N-th call's result."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        result = fn(*args, **kwargs)
        call_count[0] += 1
        if call_count[0] == corrupt_at_call:
            garbled = "".join(
                chr((ord(c) + 13) % 95 + 32) if c.isalpha() else c
                for c in result[:60]
            )
            return f"[CORRUPT_MEM] {garbled} [/CORRUPT_MEM] {result[60:]}"
        return result
    return wrapper


def _make_partial_wrapper(
    fn:           Callable[..., str],
    corrupt_rate: float,
    rng:          random.Random,
) -> Callable[..., str]:
    """Return a wrapped ``fn`` that truncates its result with probability ``corrupt_rate``."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        result = fn(*args, **kwargs)
        if rng.random() < corrupt_rate:
            cut = max(1, len(result) // 3)
            return result[:cut] + "...[PARTIAL DATA -- transmission error]"
        return result
    return wrapper


# -- entry point ---------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set. Add it to .env and retry.")
        sys.exit(1)

    # Suppress the per-turn log lines from AgentLoop so chaos output stays readable.
    logging.getLogger("ch03_agent_loop.agent_loop").setLevel(logging.WARNING)

    SEP  = "=" * 68
    SEP2 = "-" * 40

    # -- demo tools (deterministic, no external calls) ------------------------

    def web_search(query: str) -> str:
        """Search the web and return relevant results for the query."""
        return (
            f"Results for '{query}': "
            "AAPL closed at $189.30 today, up 1.2% from yesterday. "
            "52-week range $142.00-$198.23. Analyst consensus: Buy."
        )

    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression and return the numeric result."""
        allowed = set("0123456789 +-*/()., ")
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        try:
            return str(round(eval(expression), 4))  # noqa: S307
        except Exception as exc:
            return f"Error: {exc}"

    TOOLS = {"web_search": web_search, "calculator": calculator}
    TASK  = (
        "Use web_search to find the current AAPL stock price, "
        "then use calculator to compute the total value of 42 shares. "
        "Report the stock price and the 42-share portfolio value."
    )

    print(f"\n{SEP}")
    print("  AGENT CHAOS SUITE  |  5 scenarios  |  model=claude-sonnet-4-6")
    print(f"  task: {TASK[:70]}...")
    print(SEP)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_agent = AgentLoop(
            model="claude-sonnet-4-6",
            budget_usd=0.30,
            max_turns=8,
            checkpoint_dir=tmpdir,
        )

        suite  = AgentChaosSuite(seed=42)
        report = suite.run_all(base_agent, TASK, TOOLS)

        print(f"\n  SCENARIO RESULTS")
        print(SEP2)
        for cr in report.results:
            status = "PASS" if cr.task_completed else "FAIL"
            print(
                f"  [{status}] {cr.scenario:<22}  "
                f"quality={cr.final_answer_quality:.2f}  "
                f"ok={cr.steps_completed}/fail={cr.steps_failed}  "
                f"reason={cr.termination_reason}"
            )
            print(f"         params  : {cr.notes}")
            print(
                f"         cost    : ${cr.cost_usd:.5f}  "
                f"duration: {cr.duration_s:.1f}s"
            )
            if cr.recovery_triggered:
                print(
                    "         recovery: triggered -- agent handled injected fault"
                )
            print()

        print(SEP2)
        print("  CHAOS REPORT")
        print(SEP2)
        print(f"  {report.summary}")
        print(f"\n  resilience_score   : {report.resilience_score:.0%}")
        print(f"  scenarios_passed   : {report.scenarios_passed} / {len(report.results)}")
        print(f"  scenarios_failed   : {report.scenarios_failed} / {len(report.results)}")
        print(f"  total_cost_usd     : ${report.total_cost_usd:.5f}")
        print(f"  total_duration_s   : {report.total_duration_s:.1f}s")
        print(SEP)
