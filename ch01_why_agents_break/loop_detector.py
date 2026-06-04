"""
Detects and interrupts infinite loops in a running LLM agent.

Loop patterns detected (in priority order):
  MAX_TURNS_EXCEEDED — hard turn limit reached regardless of content
  EXACT_REPEAT       — identical (tool_name, serialised_args) in the last N calls
  SEMANTIC_REPEAT    — arg-string similarity >= threshold across the last N calls
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import Any


# ~$3/M tokens (Claude Sonnet blended in/out approximation)
_COST_PER_TOKEN_USD: float = 3e-6


class LoopType(Enum):
    """Category of detected loop pattern."""

    NONE = "none"
    EXACT_REPEAT = "exact_repeat"
    SEMANTIC_REPEAT = "semantic_repeat"
    MAX_TURNS_EXCEEDED = "max_turns_exceeded"


class Action(Enum):
    """Recommended action for the calling agent runtime."""

    CONTINUE = "CONTINUE"
    WARN = "WARN"
    HALT = "HALT"


@dataclass
class LoopStatus:
    """Result returned by LoopDetector.check() for a single agent turn.

    Attributes:
        is_loop: True when any loop pattern was detected.
        loop_type: The matched pattern (NONE when clean).
        turns_elapsed: Total turns processed by this detector instance.
        cost_burned_usd: Cumulative estimated cost in USD at this turn.
        action: Recommended next step — CONTINUE, WARN, or HALT.
    """

    is_loop: bool
    loop_type: LoopType
    turns_elapsed: int
    cost_burned_usd: float
    action: Action

    def __str__(self) -> str:
        tag = f"{self.action.value:<8}"
        pattern = f"[{self.loop_type.value}]" if self.is_loop else "ok"
        return (
            f"  Turn {self.turns_elapsed:>3} | {tag} | "
            f"{pattern:<28} | cost ${self.cost_burned_usd:.6f}"
        )


class LoopDetector:
    """Detects and interrupts infinite loops in a running LLM agent.

    Call check() once per agent turn. The detector maintains its own
    turn counter and tool-call history; call reset() between agent runs.

    Detection logic:
      - MAX_TURNS_EXCEEDED fires when turns_elapsed > max_turns → HALT
      - EXACT_REPEAT fires when the last window_size tool calls are all
        identical (same tool_name + same serialised args) → HALT
      - SEMANTIC_REPEAT fires when arg-string similarity >= threshold
        across the same window → WARN

    Args:
        max_turns: Hard limit on total turns before HALT (default 10).
        similarity_threshold: SequenceMatcher ratio for SEMANTIC_REPEAT
            detection; 0.0 = always trigger, 1.0 = exact match only
            (default 0.95).
        window_size: Number of consecutive matching calls required to
            confirm a loop (default 3; minimum effective value is 2).
    """

    def __init__(
        self,
        max_turns: int = 10,
        similarity_threshold: float = 0.95,
        window_size: int = 3,
    ) -> None:
        self.max_turns = max_turns
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size

        self._turns_elapsed: int = 0
        self._total_cost_usd: float = 0.0
        self._tool_call_history: list[tuple[str, str]] = []
        self._total_checks: int = 0
        self._loops_detected: int = 0
        self._halts_triggered: int = 0
        self._loop_costs: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, turn: dict[str, Any]) -> LoopStatus:
        """Evaluate one agent turn for loop patterns.

        Must be called for every turn, including non-tool turns, so that
        the turn counter and cost accumulator stay accurate.

        Args:
            turn: Turn dict. Recognised keys:
                type        (str)  — "tool_call" | "message" | "tool_result" | ...
                tool_name   (str)  — name of the tool being called
                tool_args   (dict) — arguments passed to the tool
                token_count (int)  — tokens consumed this turn

        Returns:
            LoopStatus describing the detected pattern and recommended action.
        """
        self._turns_elapsed += 1
        self._total_checks += 1
        self._total_cost_usd += int(turn.get("token_count", 0)) * _COST_PER_TOKEN_USD

        if self._turns_elapsed > self.max_turns:
            return self._make_status(LoopType.MAX_TURNS_EXCEEDED, Action.HALT)

        if turn.get("type") != "tool_call":
            return self._make_status(LoopType.NONE, Action.CONTINUE)

        current = _make_key(turn.get("tool_name", ""), turn.get("tool_args", {}))
        loop_type, action = self._classify(current)

        # Append *after* classification so the current call is not compared to itself
        self._tool_call_history.append(current)

        return self._make_status(loop_type, action)

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics over all check() calls so far.

        Returns:
            Dict with keys:
              total_checks          — total calls to check()
              loops_detected        — turns classified as any loop pattern
              halts_triggered       — turns that returned Action.HALT
              avg_cost_per_loop_usd — mean cost-at-detection across loop events
              total_cost_usd        — total accumulated estimated spend
        """
        avg = (sum(self._loop_costs) / len(self._loop_costs)) if self._loop_costs else 0.0
        return {
            "total_checks": self._total_checks,
            "loops_detected": self._loops_detected,
            "halts_triggered": self._halts_triggered,
            "avg_cost_per_loop_usd": round(avg, 6),
            "total_cost_usd": round(self._total_cost_usd, 6),
        }

    def reset(self) -> None:
        """Clear all accumulated state for reuse across agent runs."""
        self._turns_elapsed = 0
        self._total_cost_usd = 0.0
        self._tool_call_history.clear()
        self._total_checks = 0
        self._loops_detected = 0
        self._halts_triggered = 0
        self._loop_costs.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify(self, current: tuple[str, str]) -> tuple[LoopType, Action]:
        """Compare current key against the recent window in tool-call history.

        Needs window_size - 1 prior calls that all match the current call
        to declare a loop. Returns NONE if history is too short to fill
        the window yet.
        """
        needed = self.window_size - 1
        if needed <= 0:
            return LoopType.NONE, Action.CONTINUE

        recent = self._tool_call_history[-needed:]
        if len(recent) < needed:
            return LoopType.NONE, Action.CONTINUE

        if all(entry == current for entry in recent):
            return LoopType.EXACT_REPEAT, Action.HALT

        if all(
            SequenceMatcher(None, entry[1], current[1]).ratio() >= self.similarity_threshold
            for entry in recent
        ):
            return LoopType.SEMANTIC_REPEAT, Action.WARN

        return LoopType.NONE, Action.CONTINUE

    def _make_status(self, loop_type: LoopType, action: Action) -> LoopStatus:
        is_loop = loop_type is not LoopType.NONE
        if is_loop:
            self._loops_detected += 1
            self._loop_costs.append(self._total_cost_usd)
            if action is Action.HALT:
                self._halts_triggered += 1
        return LoopStatus(
            is_loop=is_loop,
            loop_type=loop_type,
            turns_elapsed=self._turns_elapsed,
            cost_burned_usd=self._total_cost_usd,
            action=action,
        )


def _make_key(tool_name: str, tool_args: dict[str, Any]) -> tuple[str, str]:
    """Serialise a tool call to a comparable, hashable key."""
    return tool_name, json.dumps(tool_args, sort_keys=True)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    sep = "=" * 66

    # 15 turns defined; loop starts at turn 7 (first repeated call).
    # With window_size=3 the detector needs 2 prior identical calls before
    # firing, so EXACT_REPEAT is confirmed at turn 9 (3rd repetition).
    turns: list[dict[str, Any]] = [
        # --- Turns 1-6: normal varied activity ---
        {"type": "message",     "content": "Analyse AAPL performance.",            "token_count": 350},
        {"type": "tool_call",   "tool_name": "read_file",     "tool_args": {"path": "portfolio.csv"},      "token_count": 800},
        {"type": "tool_result", "tool_name": "read_file",     "result": "...",                             "token_count": 400},
        {"type": "tool_call",   "tool_name": "compute_stats", "tool_args": {"column": "close_price"},      "token_count": 600},
        {"type": "tool_result", "tool_name": "compute_stats", "result": "mean=183.4",                      "token_count": 300},
        {"type": "tool_call",   "tool_name": "web_search",    "tool_args": {"query": "AAPL Q3 earnings"},  "token_count": 700},
        # --- Turns 7-15: loop begins (agent retries the same query on timeout) ---
        {"type": "tool_call",   "tool_name": "web_search",    "tool_args": {"query": "AAPL current price"}, "token_count": 700},
        {"type": "tool_call",   "tool_name": "web_search",    "tool_args": {"query": "AAPL current price"}, "token_count": 700},
        {"type": "tool_call",   "tool_name": "web_search",    "tool_args": {"query": "AAPL current price"}, "token_count": 700},
        {"type": "tool_call",   "tool_name": "web_search",    "tool_args": {"query": "AAPL current price"}, "token_count": 700},
        {"type": "tool_call",   "tool_name": "web_search",    "tool_args": {"query": "AAPL current price"}, "token_count": 700},
        {"type": "tool_call",   "tool_name": "web_search",    "tool_args": {"query": "AAPL current price"}, "token_count": 700},
        {"type": "message",     "content": "Still waiting for price data...",      "token_count": 250},
        {"type": "tool_call",   "tool_name": "web_search",    "tool_args": {"query": "AAPL current price"}, "token_count": 700},
        {"type": "tool_call",   "tool_name": "web_search",    "tool_args": {"query": "AAPL current price"}, "token_count": 700},
    ]

    detector = LoopDetector(max_turns=15, similarity_threshold=0.95, window_size=3)

    print(f"\n{sep}")
    print("  LoopDetector demo  |  loop starts turn 7, window_size=3")
    print(f"{sep}")
    print(f"  {'Turn':>4}  {'Action':<8}  {'Pattern':<28}  Cost (USD)")
    print(f"  {'-'*4}  {'-'*8}  {'-'*28}  {'-'*12}")

    for sequence_idx, turn in enumerate(turns, start=1):
        turn_type = turn.get("type", "?")
        status = detector.check(turn)
        marker = " <-- LOOP DETECTED" if status.is_loop else ""
        print(f"{status}{marker}  [{turn_type}]")

        if status.action is Action.HALT:
            print(f"\n  *** AGENT HALTED at turn {status.turns_elapsed} ***")
            print(f"  (turns {sequence_idx + 1}-15 never executed in production)\n")
            break

    print(f"{sep}")
    print("  STATS")
    print(f"  {'-'*40}")
    for key, val in detector.get_stats().items():
        print(f"  {key:<30}: {val}")
    print(f"{sep}\n")
