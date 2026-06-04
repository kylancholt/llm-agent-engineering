"""
Real-time budget enforcement for LLM agent loops.

Designed as a single-responsibility module with no external dependencies:
stdlib only, no I/O in the hot path, record() < 2ms guaranteed.

The guard accumulates per-turn cost, classifies the current alert level,
and signals should_halt when the configured halt_threshold is crossed.

Pricing table mirrors ch01_why_agents_break/cost_tracker.py ($/1M tokens):
  claude-haiku-4-5  : $0.80 in / $4.00 out
  claude-sonnet-4-6 : $3.00 in / $15.00 out
  claude-opus-4-7   : $15.00 in / $75.00 out
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── pricing ($/1M tokens) ─────────────────────────────────────────────────────
_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":  {"input":  0.80, "output":  4.00},
    "claude-sonnet-4-6": {"input":  3.00, "output": 15.00},
    "claude-opus-4-7":   {"input": 15.00, "output": 75.00},
}
_DEFAULT_MODEL = "claude-sonnet-4-6"


# ── public types ──────────────────────────────────────────────────────────────

class AlertLevel(Enum):
    """Budget consumption severity, in ascending order."""

    OK       = "OK"
    WARN     = "WARN"
    CRITICAL = "CRITICAL"
    HALT     = "HALT"


@dataclass
class GuardStatus:
    """Return value of CostGuard.record() — one per agent turn.

    Attributes:
        turn_cost_usd:        Cost incurred this turn alone.
        cumulative_cost_usd:  Total spend from turn 1 to this turn.
        budget_remaining_usd: Budget not yet consumed (floored at 0).
        alert_level:          Current severity level.
        should_halt:          True when the agent loop must stop immediately.
        message:              Human-readable status line for this turn.
    """

    turn_cost_usd:        float
    cumulative_cost_usd:  float
    budget_remaining_usd: float
    alert_level:          AlertLevel
    should_halt:          bool
    message:              str


# ── cost guard ────────────────────────────────────────────────────────────────

class CostGuard:
    """Real-time budget enforcement for LLM agent loops.

    Call record() once per turn. The guard accumulates cost, classifies
    alert level, and sets should_halt when halt_threshold is exceeded.

    record() is designed for < 2ms latency: arithmetic, dict lookups, and
    list appends only — no I/O, no locks, no system calls.

    Args:
        budget_usd:          Total USD budget for the run.
        warn_threshold:      Budget fraction that triggers WARN (default 0.50).
        critical_threshold:  Budget fraction that triggers CRITICAL (default 0.80).
        halt_threshold:      Budget fraction that triggers HALT (default 0.95).

    Raises:
        ValueError: If thresholds are not strictly ordered 0 < warn < critical < halt <= 1.
    """

    def __init__(
        self,
        budget_usd:         float,
        warn_threshold:     float = 0.50,
        critical_threshold: float = 0.80,
        halt_threshold:     float = 0.95,
    ) -> None:
        if not (0 < warn_threshold < critical_threshold < halt_threshold <= 1.0):
            raise ValueError(
                "Thresholds must satisfy 0 < warn < critical < halt <= 1.0. "
                f"Got: warn={warn_threshold}, critical={critical_threshold}, halt={halt_threshold}"
            )
        self.budget_usd         = budget_usd
        self.warn_threshold     = warn_threshold
        self.critical_threshold = critical_threshold
        self.halt_threshold     = halt_threshold

        self._cumulative_usd: float       = 0.0
        self._costs_per_turn: list[float] = []
        self._last_model:     str         = _DEFAULT_MODEL
        self._turn_count:     int         = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        input_tokens:  int,
        output_tokens: int,
        model:         str = _DEFAULT_MODEL,
    ) -> GuardStatus:
        """Record one agent turn and return the updated budget status.

        Hot-path method: only arithmetic, a dict lookup, and a list append.
        Verified < 2ms in the __main__ demo via time.perf_counter.

        Args:
            input_tokens:  Tokens consumed as model input this turn.
            output_tokens: Tokens generated as model output this turn.
            model:         Claude model ID (for per-model pricing).
                           Falls back to claude-sonnet-4-6 if unknown.

        Returns:
            GuardStatus with costs, remaining budget, alert level, halt
            flag, and a human-readable message.
        """
        pricing   = _PRICING.get(model, _PRICING[_DEFAULT_MODEL])
        turn_cost = (input_tokens  * pricing["input"]  / 1_000_000
                   + output_tokens * pricing["output"] / 1_000_000)

        self._cumulative_usd  += turn_cost
        self._costs_per_turn.append(turn_cost)
        self._last_model       = model
        self._turn_count      += 1

        fraction  = self._cumulative_usd / self.budget_usd
        remaining = max(0.0, self.budget_usd - self._cumulative_usd)
        level, halt, msg = self._classify(fraction, remaining)

        return GuardStatus(
            turn_cost_usd=        round(turn_cost, 8),
            cumulative_cost_usd=  round(self._cumulative_usd, 8),
            budget_remaining_usd= round(remaining, 8),
            alert_level=          level,
            should_halt=          halt,
            message=              msg,
        )

    def check_budget(self, estimated_next_turn_tokens: int) -> bool:
        """Estimate whether one more turn is affordable.

        Assumes a 70 / 30 input / output token split and uses the last
        recorded model's pricing for the estimate.

        Args:
            estimated_next_turn_tokens: Expected combined token count
                (input + output) for the upcoming turn.

        Returns:
            True if the estimated cost fits within the remaining budget.
        """
        pricing  = _PRICING.get(self._last_model, _PRICING[_DEFAULT_MODEL])
        in_tok   = int(estimated_next_turn_tokens * 0.70)
        out_tok  = int(estimated_next_turn_tokens * 0.30)
        est_cost = (in_tok  * pricing["input"]  / 1_000_000
                  + out_tok * pricing["output"] / 1_000_000)
        return est_cost <= max(0.0, self.budget_usd - self._cumulative_usd)

    def get_summary(self) -> dict[str, Any]:
        """Return a full cost breakdown with statistics and projections.

        Projections use the average of the last 3 turns (or all turns if
        fewer than 3) as the estimated per-turn burn rate.

        Returns:
            Dict with keys:
              total_turns             — turns recorded
              total_cost_usd          — cumulative spend
              budget_usd              — configured budget
              budget_remaining_usd    — remaining budget
              budget_used_pct         — percentage consumed (0–100)
              alert_level             — current AlertLevel string value
              should_halt             — halt flag
              costs_per_turn          — list of per-turn costs
              avg_cost_per_turn_usd   — mean across all turns
              projected_5_turns_usd   — estimated spend for 5 more turns
              projected_10_turns_usd  — estimated spend for 10 more turns
              turns_until_halt        — estimated remaining turns before HALT
                                        (None if no turns recorded yet)
        """
        n         = self._turn_count
        remaining = max(0.0, self.budget_usd - self._cumulative_usd)
        used_pct  = (self._cumulative_usd / self.budget_usd) * 100.0

        recent     = self._costs_per_turn[-3:] if self._costs_per_turn else []
        avg_recent = sum(recent) / len(recent) if recent else 0.0
        avg_all    = (self._cumulative_usd / n) if n > 0 else 0.0

        turns_until_halt: float | None = None
        if avg_recent > 0:
            halt_budget    = self.budget_usd * self.halt_threshold
            budget_to_halt = max(0.0, halt_budget - self._cumulative_usd)
            turns_until_halt = round(budget_to_halt / avg_recent, 1)

        fraction = (self._cumulative_usd / self.budget_usd) if self.budget_usd > 0 else 0.0
        level, halt, _ = self._classify(fraction, remaining)

        return {
            "total_turns":            n,
            "total_cost_usd":         round(self._cumulative_usd, 8),
            "budget_usd":             self.budget_usd,
            "budget_remaining_usd":   round(remaining, 8),
            "budget_used_pct":        round(used_pct, 2),
            "alert_level":            level.value,
            "should_halt":            halt,
            "costs_per_turn":         [round(c, 8) for c in self._costs_per_turn],
            "avg_cost_per_turn_usd":  round(avg_all, 8),
            "projected_5_turns_usd":  round(avg_recent * 5, 8),
            "projected_10_turns_usd": round(avg_recent * 10, 8),
            "turns_until_halt":       turns_until_halt,
        }

    def reset(self) -> None:
        """Clear all accumulated state; reuse this guard for a new run."""
        self._cumulative_usd = 0.0
        self._costs_per_turn.clear()
        self._last_model = _DEFAULT_MODEL
        self._turn_count = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify(
        self,
        fraction:  float,
        remaining: float,
    ) -> tuple[AlertLevel, bool, str]:
        """Map budget-used fraction to (AlertLevel, should_halt, message).

        Thresholds are checked in descending severity order so the highest
        applicable level always wins.
        """
        pct = fraction * 100.0
        cum = self._cumulative_usd
        bud = self.budget_usd

        if fraction >= self.halt_threshold:
            return (
                AlertLevel.HALT, True,
                f"HALT: {pct:.1f}% of ${bud:.3f} consumed (${cum:.5f})"
                f" -- budget limit reached, agent must stop.",
            )
        if fraction >= self.critical_threshold:
            return (
                AlertLevel.CRITICAL, False,
                f"Critical: {pct:.1f}% of ${bud:.3f} consumed (${cum:.5f})"
                f" -- reduce token usage immediately.",
            )
        if fraction >= self.warn_threshold:
            return (
                AlertLevel.WARN, False,
                f"Warning: {pct:.1f}% of ${bud:.3f} consumed (${cum:.5f})"
                f" -- monitor usage, ${remaining:.5f} remaining.",
            )
        return (
            AlertLevel.OK, False,
            f"OK: {pct:.1f}% of ${bud:.3f} consumed (${cum:.5f})"
            f", ${remaining:.5f} remaining.",
        )


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import datetime

    MODEL  = "claude-sonnet-4-6"
    BUDGET = 0.04

    # Token scenario: 10 turns with growing context that produce
    # the full OK -> WARN -> CRITICAL -> HALT escalation.
    # (Pre-verified against sonnet pricing.)
    #
    # Turns 1-4   OK        9.4% -> 49.5%
    # Turn  5     WARN     52.1%           (crosses 50%)
    # Turns 6-7   WARN     61.5% -> 76.9%
    # Turn  8     CRITICAL 88.9%           (crosses 80%)
    # Turn  9     CRITICAL 94.9%
    # Turn  10    HALT     99.4%           (crosses 95%)
    SCENARIO: list[tuple[str, int, int]] = [
        ("message",      500,  150),
        ("tool_call",   1000,  300),
        ("tool_result",  800,  250),
        ("tool_call",    300,  100),
        ("tool_result",  100,   50),
        ("tool_call",    500,  150),
        ("tool_result",  800,  250),
        ("tool_call",    600,  200),
        ("tool_result",  300,  100),
        ("final_answer", 200,   80),
    ]

    guard = CostGuard(
        budget_usd=BUDGET,
        warn_threshold=0.50,
        critical_threshold=0.80,
        halt_threshold=0.95,
    )

    sep = "=" * 76
    print(f"\n{sep}")
    print(f"  CostGuard demo  |  budget=${BUDGET:.2f}  |  model={MODEL}")
    print(sep)
    print(
        f"  {'Turn':>4}  {'Type':<14}  {'In':>6}  {'Out':>5}  "
        f"{'$/turn':>9}  {'$cumul':>9}  {'Used%':>6}  {'Level':<10}  Latency"
    )
    print(f"  {'-'*72}")

    prev_level = AlertLevel.OK
    max_latency_us: float = 0.0

    for i, (turn_type, in_tok, out_tok) in enumerate(SCENARIO, start=1):
        ts = time.perf_counter()
        status = guard.record(in_tok, out_tok, model=MODEL)
        latency_us = (time.perf_counter() - ts) * 1_000_000

        # Verify < 2ms SLA on every call
        assert latency_us < 2_000, (
            f"Turn {i}: record() took {latency_us:.0f}us — violates <2ms SLA"
        )
        max_latency_us = max(max_latency_us, latency_us)

        pct    = (status.cumulative_cost_usd / BUDGET) * 100
        marker = ""
        if status.alert_level != prev_level:
            marker = f"  <-- {status.alert_level.value}"
        prev_level = status.alert_level

        print(
            f"  {i:>4}  {turn_type:<14}  {in_tok:>6}  {out_tok:>5}  "
            f"${status.turn_cost_usd:>8.5f}  ${status.cumulative_cost_usd:>8.5f}  "
            f"{pct:>5.1f}%  {status.alert_level.value:<10}  "
            f"{latency_us:>6.2f}us{marker}"
        )

        if status.should_halt:
            print(f"\n  *** {status.message}")
            break

    print(f"\n  Max latency: {max_latency_us:.2f}us  (SLA: <2000us)  PASS")

    # ── summary ────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  SUMMARY")
    print(f"  {'-'*50}")
    summary = guard.get_summary()
    skip = {"costs_per_turn"}
    for k, v in summary.items():
        if k not in skip:
            print(f"  {k:<30}: {v}")
    print(f"  {'costs_per_turn':<30}: {[f'${c:.5f}' for c in summary['costs_per_turn']]}")

    print(f"\n  check_budget(1000 tokens): {guard.check_budget(1000)}")
    print(f"  check_budget(100 tokens) : {guard.check_budget(100)}")
    print(f"{sep}\n")
