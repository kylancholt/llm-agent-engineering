"""
Tracks cumulative cost of a running LLM agent in real time.

Pricing (per 1M tokens) — hardcoded for three Claude model tiers:
  claude-haiku-4-5  : input $0.80  / output $4.00
  claude-sonnet-4-6 : input $3.00  / output $15.00
  claude-opus-4-7   : input $15.00 / output $75.00
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# $/1M tokens per model
_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":  {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00},
    "claude-opus-4-7":   {"input": 15.00, "output": 75.00},
}

_WARN_PCT: float     = 50.0
_CRITICAL_PCT: float = 80.0
_HALT_PCT: float     = 95.0


# ------------------------------------------------------------------
# Public types
# ------------------------------------------------------------------

class AlertLevel(Enum):
    """Budget consumption thresholds."""

    OK       = "OK"
    WARN     = "WARN"
    CRITICAL = "CRITICAL"
    HALT     = "HALT"


@dataclass
class CostStatus:
    """Returned by AgentCostTracker.record_turn() after each agent turn.

    Attributes:
        turn_cost_usd: Cost incurred by this turn alone.
        cumulative_cost_usd: Total spend from turn 1 to this turn.
        budget_remaining_usd: Budget not yet consumed (floored at 0).
        budget_used_pct: Percentage of the total budget consumed.
        alert_level: Current severity level based on consumption thresholds.
    """

    turn_cost_usd: float
    cumulative_cost_usd: float
    budget_remaining_usd: float
    budget_used_pct: float
    alert_level: AlertLevel

    def __str__(self) -> str:
        flag = f"  [{self.alert_level.value}]" if self.alert_level is not AlertLevel.OK else ""
        return (
            f"  turn ${self.turn_cost_usd:>9.5f} | "
            f"cum ${self.cumulative_cost_usd:>8.5f} | "
            f"{self.budget_used_pct:>6.2f}%{flag}"
        )


@dataclass
class TurnRecord:
    """Per-turn entry stored in CostReport.turns.

    Attributes:
        turn_id: Caller-supplied identifier (int or str).
        tool_name: Tool invoked this turn; None for non-tool turns.
        input_tokens: Input tokens consumed.
        output_tokens: Output tokens generated.
        cost_usd: Cost for this turn.
        cumulative_cost_usd: Running total after this turn.
        alert_level: Budget alert level after this turn.
    """

    turn_id: int | str
    tool_name: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cumulative_cost_usd: float
    alert_level: AlertLevel


@dataclass
class CostReport:
    """Full cost breakdown returned by AgentCostTracker.get_report().

    Attributes:
        model: Model used for pricing.
        budget_usd: Total budget configured.
        turns_recorded: Number of turns recorded.
        total_cost_usd: Total spend so far.
        budget_used_pct: Percentage of budget consumed.
        avg_cost_per_turn_usd: Mean cost per recorded turn.
        projected_next_turn_usd: Estimated cost of the next turn (OLS trend).
        projected_turns_remaining: Estimated turns before budget exhausted;
            None when budget is effectively unlimited.
        projected_completion_cost_usd: Estimated total cost if the same
            number of turns is run again at the current trend.
        turns: Per-turn breakdown list.
    """

    model: str
    budget_usd: float
    turns_recorded: int
    total_cost_usd: float
    budget_used_pct: float
    avg_cost_per_turn_usd: float
    projected_next_turn_usd: float
    projected_turns_remaining: float | None
    projected_completion_cost_usd: float
    turns: list[TurnRecord] = field(default_factory=list)

    def __str__(self) -> str:
        sep = "=" * 72
        remain = (
            f"{self.projected_turns_remaining:.1f}"
            if self.projected_turns_remaining is not None
            else "unlimited"
        )
        header = (
            f"\n{sep}\n"
            f"  COST REPORT  ({self.model})\n"
            f"{sep}\n"
            f"  Budget              : ${self.budget_usd:.4f}\n"
            f"  Total cost          : ${self.total_cost_usd:.5f}\n"
            f"  Budget used         : {self.budget_used_pct:.2f}%\n"
            f"  Turns recorded      : {self.turns_recorded}\n"
            f"  Avg cost / turn     : ${self.avg_cost_per_turn_usd:.5f}\n"
            f"\n  Projection (OLS trend over {self.turns_recorded} turns):\n"
            f"    Estimated next turn    : ${self.projected_next_turn_usd:.5f}\n"
            f"    Turns until HALT       : {remain}\n"
            f"    Completion estimate    : ${self.projected_completion_cost_usd:.5f}"
            f"  (if {self.turns_recorded} more turns run)\n"
            f"\n  Per-turn breakdown:\n"
            f"  {'ID':<6} {'Tool':<18} {'In':>7} {'Out':>7}"
            f"  {'$/turn':>9}  {'$cumul':>9}  Alert\n"
            f"  {'-'*68}\n"
        )
        rows: list[str] = []
        for t in self.turns:
            tool = (t.tool_name or "-")[:17]
            rows.append(
                f"  {str(t.turn_id):<6} {tool:<18} {t.input_tokens:>7} {t.output_tokens:>7}"
                f"  ${t.cost_usd:>8.5f}  ${t.cumulative_cost_usd:>8.5f}  {t.alert_level.value}"
            )
        return header + "\n".join(rows) + f"\n{sep}\n"


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------

class AgentCostTracker:
    """Tracks cumulative token cost for a running LLM agent.

    Call record_turn() once per agent turn. The tracker accumulates cost,
    compares against alert thresholds, and can produce a full CostReport
    including a linear-trend projection.

    Args:
        budget_usd: Total allowed spend (default $1.00).
        model: One of the supported Claude model identifiers
            (default "claude-sonnet-4-6").

    Raises:
        ValueError: If model is not in the supported pricing table.
    """

    def __init__(
        self,
        budget_usd: float = 1.0,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        if model not in _PRICING:
            raise ValueError(
                f"Unknown model '{model}'. Supported: {sorted(_PRICING)}"
            )
        self.budget_usd = budget_usd
        self.model = model
        self._prices = _PRICING[model]
        self._cumulative_cost_usd: float = 0.0
        self._turns: list[TurnRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_turn(
        self,
        input_tokens: int,
        output_tokens: int,
        turn_id: int | str,
        tool_name: str | None = None,
    ) -> CostStatus:
        """Record one agent turn and return the updated cost status.

        Args:
            input_tokens: Input tokens consumed this turn.
            output_tokens: Output tokens generated this turn.
            turn_id: Caller-supplied identifier for this turn.
            tool_name: Tool invoked this turn (None if not a tool call).

        Returns:
            CostStatus with per-turn cost, cumulative spend, remaining budget,
            percentage consumed, and the current alert level.
        """
        turn_cost = (
            input_tokens  * self._prices["input"]  / 1_000_000
            + output_tokens * self._prices["output"] / 1_000_000
        )
        self._cumulative_cost_usd += turn_cost
        used_pct = (self._cumulative_cost_usd / self.budget_usd) * 100.0
        remaining = max(0.0, self.budget_usd - self._cumulative_cost_usd)
        level = _classify_alert(used_pct)

        self._turns.append(TurnRecord(
            turn_id=turn_id,
            tool_name=tool_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(turn_cost, 8),
            cumulative_cost_usd=round(self._cumulative_cost_usd, 8),
            alert_level=level,
        ))

        return CostStatus(
            turn_cost_usd=round(turn_cost, 8),
            cumulative_cost_usd=round(self._cumulative_cost_usd, 8),
            budget_remaining_usd=round(remaining, 8),
            budget_used_pct=round(used_pct, 2),
            alert_level=level,
        )

    def get_report(self) -> CostReport:
        """Build a full cost report with per-turn breakdown and trend projection.

        The projection uses ordinary least-squares regression over the recorded
        turn costs. projected_completion_cost_usd assumes the same number of
        turns again at the extrapolated rate.

        Returns:
            CostReport dataclass (also has a human-readable __str__).
        """
        n = len(self._turns)
        if n == 0:
            return CostReport(
                model=self.model,
                budget_usd=self.budget_usd,
                turns_recorded=0,
                total_cost_usd=0.0,
                budget_used_pct=0.0,
                avg_cost_per_turn_usd=0.0,
                projected_next_turn_usd=0.0,
                projected_turns_remaining=None,
                projected_completion_cost_usd=0.0,
                turns=[],
            )

        costs = [t.cost_usd for t in self._turns]
        avg = sum(costs) / n

        slope, intercept = _ols(list(range(1, n + 1)), costs)
        projected_next = max(0.0, slope * (n + 1) + intercept)

        remaining = max(0.0, self.budget_usd - self._cumulative_cost_usd)
        turns_remaining: float | None = (
            None if projected_next == 0.0
            else round(remaining / projected_next, 1)
        )

        projected_future = sum(
            max(0.0, slope * (n + i) + intercept) for i in range(1, n + 1)
        )
        projected_completion = self._cumulative_cost_usd + projected_future

        return CostReport(
            model=self.model,
            budget_usd=self.budget_usd,
            turns_recorded=n,
            total_cost_usd=round(self._cumulative_cost_usd, 8),
            budget_used_pct=round((self._cumulative_cost_usd / self.budget_usd) * 100.0, 2),
            avg_cost_per_turn_usd=round(avg, 8),
            projected_next_turn_usd=round(projected_next, 8),
            projected_turns_remaining=turns_remaining,
            projected_completion_cost_usd=round(projected_completion, 6),
            turns=list(self._turns),
        )

    def save(self, path: str) -> None:
        """Serialise the current report to a JSON file.

        Enum values are written as their string representation.

        Args:
            path: Destination file path. Parent directory must exist.
        """
        report = self.get_report()
        data = asdict(report)
        _enum_to_value(data)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def reset(self) -> None:
        """Clear all recorded turns and reset the cumulative cost counter."""
        self._cumulative_cost_usd = 0.0
        self._turns.clear()


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _classify_alert(budget_used_pct: float) -> AlertLevel:
    if budget_used_pct >= _HALT_PCT:
        return AlertLevel.HALT
    if budget_used_pct >= _CRITICAL_PCT:
        return AlertLevel.CRITICAL
    if budget_used_pct >= _WARN_PCT:
        return AlertLevel.WARN
    return AlertLevel.OK


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Ordinary least squares: returns (slope, intercept) for y = slope*x + intercept.

    Falls back to (0, mean(ys)) when fewer than 2 points are provided or
    when the design matrix is singular.
    """
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0)
    sum_x  = sum(xs)
    sum_y  = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x * x for x in xs)
    denom  = n * sum_x2 - sum_x ** 2
    if denom == 0.0:
        return 0.0, sum_y / n
    slope     = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def _enum_to_value(obj: Any) -> None:
    """Recursively replace Enum instances with their .value strings in-place."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, Enum):
                obj[k] = v.value
            else:
                _enum_to_value(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, Enum):
                obj[i] = v.value
            else:
                _enum_to_value(v)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Token counts grow roughly 3-4x across the run, simulating a complex task
    # that progressively builds context and calls heavier tools.
    # With budget=$0.50 and claude-sonnet-4-6 pricing the escalation is:
    #   Turns 1-5  → OK
    #   Turn 6     → WARN     (>50%)
    #   Turn 7     → CRITICAL (>80%)
    #   Turn 8     → HALT     (>95%)
    scenario: list[tuple[int | str, str | None, int, int]] = [
        # (turn_id, tool_name, input_tokens, output_tokens)
        (1, "read_file",         500,    150),
        (2, "parse_data",      2_000,    600),
        (3, "compute_stats",   5_000,  1_500),
        (4, "web_search",     12_000,  3_500),
        (5, "web_search",     10_000,  3_000),
        (6, "summarise",       4_000,  1_500),   # crosses 50% -> WARN
        (7, "deep_analysis",  20_000,  6_500),   # crosses 80% -> CRITICAL
        (8, "generate_report",10_000,  3_500),   # crosses 95% -> HALT
    ]

    tracker = AgentCostTracker(budget_usd=0.50, model="claude-sonnet-4-6")

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  AgentCostTracker demo  |  budget=$0.50  |  claude-sonnet-4-6")
    print(f"{sep}")
    print(f"  {'ID':<4} {'Tool':<18} {'In':>7} {'Out':>6}  "
          f"{'$/turn':>9}  {'$cumul':>9}  {'Used%':>6}  Alert")
    print(f"  {'-'*68}")

    for turn_id, tool_name, in_tok, out_tok in scenario:
        status = tracker.record_turn(
            in_tok, out_tok, turn_id=turn_id, tool_name=tool_name
        )
        escalation = (
            "  <-- WARN"     if status.alert_level is AlertLevel.WARN     else
            "  <-- CRITICAL" if status.alert_level is AlertLevel.CRITICAL  else
            "  <-- HALT"     if status.alert_level is AlertLevel.HALT      else ""
        )
        print(
            f"  {str(turn_id):<4} {(tool_name or '-'):<18} {in_tok:>7} {out_tok:>6}  "
            f"${status.turn_cost_usd:>8.5f}  ${status.cumulative_cost_usd:>8.5f}  "
            f"{status.budget_used_pct:>5.1f}%{escalation}"
        )
        if status.alert_level is AlertLevel.HALT:
            print(f"\n  *** BUDGET HALT at turn {turn_id}: "
                  f"${status.cumulative_cost_usd:.5f} of ${tracker.budget_usd:.2f} used ***\n")
            break

    print(tracker.get_report())
