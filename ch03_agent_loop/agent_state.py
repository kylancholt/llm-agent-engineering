"""
Serialisable agent state with validation, checkpointing, snapshotting, and
field-level diff for debug.

Stdlib only — no external dependencies, no I/O outside save() / load().
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    """Full mutable state of a running agent — serialisable for checkpointing.

    All fields are public. Mutate them directly between turns, then call
    save() to persist or snapshot() for a one-line status summary.

    Attributes:
        task:            Original task string given to the agent.
        messages:        Full conversation history as serialisable dicts.
                         Each entry must have "role" and "content" keys.
        tool_results:    Accumulated tool execution results (one per call).
        turn_count:      Turns completed so far.
        total_tokens:    Cumulative input + output tokens.
        total_cost_usd:  Cumulative spend in USD.
        start_time:      Unix timestamp when the state was created.
        last_updated:    Unix timestamp of the most recent mutation.
        metadata:        Caller-supplied extras (model, budget_usd, max_turns…).
    """

    task:            str
    messages:        list[dict[str, Any]]  = field(default_factory=list)
    tool_results:    list[dict[str, Any]]  = field(default_factory=list)
    turn_count:      int                   = 0
    total_tokens:    int                   = 0
    total_cost_usd:  float                 = 0.0
    start_time:      float                 = field(default_factory=time.time)
    last_updated:    float                 = field(default_factory=time.time)
    metadata:        dict[str, Any]        = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._last_save_path: str | None = None
        self.validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Verify structural integrity of the messages list.

        Checks that every entry is a dict with both "role" and "content" keys.

        Raises:
            ValueError: On the first invalid message encountered.
        """
        if not isinstance(self.messages, list):
            raise ValueError("AgentState.messages must be a list.")
        for i, msg in enumerate(self.messages):
            if not isinstance(msg, dict):
                raise ValueError(
                    f"messages[{i}] must be a dict, got {type(msg).__name__}."
                )
            for key in ("role", "content"):
                if key not in msg:
                    raise ValueError(
                        f"messages[{i}] is missing required key '{key}'."
                    )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full state to a JSON-compatible dict.

        Only the nine canonical fields are included; no derived or private
        attributes are written. Round-trips losslessly through json.dumps /
        json.loads.

        Returns:
            Dict with exactly nine keys matching the dataclass fields.
        """
        return {
            "task":           self.task,
            "messages":       self.messages,
            "tool_results":   self.tool_results,
            "turn_count":     self.turn_count,
            "total_tokens":   self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "start_time":     self.start_time,
            "last_updated":   self.last_updated,
            "metadata":       self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        """Deserialise a dict produced by to_dict() back into an AgentState.

        Extra keys in data (e.g. added by save()) are silently ignored to
        maintain forward compatibility.

        Args:
            data: Dict with at minimum the nine canonical field keys.

        Returns:
            A validated AgentState instance.

        Raises:
            KeyError: If a required field is missing from data.
            ValueError: If the resulting messages list fails validation.
        """
        required = {
            "task", "messages", "tool_results", "turn_count",
            "total_tokens", "total_cost_usd", "start_time",
            "last_updated", "metadata",
        }
        missing = required - data.keys()
        if missing:
            raise KeyError(f"Missing required fields in data: {sorted(missing)}")

        return cls(
            task=           data["task"],
            messages=       data["messages"],
            tool_results=   data["tool_results"],
            turn_count=     data["turn_count"],
            total_tokens=   data["total_tokens"],
            total_cost_usd= data["total_cost_usd"],
            start_time=     data["start_time"],
            last_updated=   data["last_updated"],
            metadata=       data["metadata"],
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> str:
        """Serialise state to a JSON file; embed a UTC timestamp in the filename.

        The given path is treated as a stem (without extension). The actual
        file is written to ``{directory}/{stem}_{YYYYMMDD_HHMMSS}.json``.
        The resolved path is stored in self._last_save_path and returned.

        Args:
            path: Base path / filename prefix. Extension is optional and
                  ignored if present; .json is always used.

        Returns:
            Absolute path of the written file as a string.
        """
        base   = Path(path).with_suffix("")   # strip any existing extension
        stamp  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target = base.parent / f"{base.name}_{stamp}.json"

        target.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        data["_saved_utc"] = datetime.now(timezone.utc).isoformat()  # convenience field
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")

        self._last_save_path = str(target)
        return self._last_save_path

    @staticmethod
    def load(path: str) -> "AgentState":
        """Load an AgentState from a JSON checkpoint file.

        Args:
            path: Path to a .json file previously written by save().

        Returns:
            A validated AgentState instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            KeyError / ValueError: If the file content is invalid.
        """
        raw = Path(path).read_text(encoding="utf-8")
        return AgentState.from_dict(json.loads(raw))

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def snapshot(self) -> str:
        """Return a compact one-line status summary.

        Example output:
            "Turn 3/10 | $0.0021/$0.05 | 3 tool calls completed"

        max_turns and budget_usd are read from self.metadata; "?" is shown
        when either is absent.

        Returns:
            Single-line string suitable for logging or progress display.
        """
        max_turns   = self.metadata.get("max_turns",  "?")
        budget      = self.metadata.get("budget_usd")
        n_tools     = len(self.tool_results)
        tool_word   = "tool call" if n_tools == 1 else "tool calls"
        budget_str  = f"${budget:.2f}" if isinstance(budget, (int, float)) else "?"

        return (
            f"Turn {self.turn_count}/{max_turns} | "
            f"${self.total_cost_usd:.4f}/{budget_str} | "
            f"{n_tools} {tool_word} completed"
        )

    # ------------------------------------------------------------------
    # Field-level diff (debug utility)
    # ------------------------------------------------------------------

    def diff(self, other: "AgentState") -> dict[str, Any]:
        """Return a dict of fields that differ between self and other.

        Scalar fields report {"from": old_value, "to": new_value}.
        List fields report counts and whether the shared prefix changed.
        metadata reports added / removed / changed keys.

        An empty dict means the two states are identical (e.g. after a
        lossless save / load round-trip).

        Args:
            other: The AgentState to compare against.

        Returns:
            Dict of differences; empty if the states are equal.
        """
        changes: dict[str, Any] = {}

        # ── scalar fields ─────────────────────────────────────────────
        for fname in ("task", "turn_count", "total_tokens",
                      "total_cost_usd", "start_time", "last_updated"):
            sv, ov = getattr(self, fname), getattr(other, fname)
            if sv != ov:
                changes[fname] = {"from": sv, "to": ov}

        # ── list fields ───────────────────────────────────────────────
        for lname in ("messages", "tool_results"):
            sv: list[Any] = getattr(self, lname)
            ov: list[Any] = getattr(other, lname)
            if sv != ov:
                n_common = min(len(sv), len(ov))
                changes[lname] = {
                    "from_count":     len(sv),
                    "to_count":       len(ov),
                    "delta":          len(ov) - len(sv),
                    "prefix_changed": sv[:n_common] != ov[:n_common],
                }

        # ── metadata ──────────────────────────────────────────────────
        if self.metadata != other.metadata:
            added   = sorted(k for k in other.metadata if k not in self.metadata)
            removed = sorted(k for k in self.metadata  if k not in other.metadata)
            changed = sorted(
                k for k in self.metadata
                if k in other.metadata and self.metadata[k] != other.metadata[k]
            )
            changes["metadata"] = {
                "added":   added,
                "removed": removed,
                "changed": changed,
            }

        return changes


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    sep = "=" * 68

    # ── build initial state ────────────────────────────────────────────────
    state = AgentState(
        task="Find the AAPL stock price and calculate a 50-share portfolio.",
        metadata={"max_turns": 10, "budget_usd": 0.05, "model": "claude-sonnet-4-6"},
    )

    # ── simulate 3 turns ──────────────────────────────────────────────────
    TURNS: list[dict[str, Any]] = [
        {
            "msgs": [
                {"role": "user",      "content": "Find AAPL stock price."},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1",
                    "name": "web_search", "input": {"query": "AAPL stock price"}}]},
            ],
            "results": [{"tool_use_id": "t1", "content": "AAPL last close: $189.30"}],
            "tokens":  523, "cost": 0.00157,
        },
        {
            "msgs": [
                {"role": "user",      "content": [{"type": "tool_result",
                    "tool_use_id": "t1", "content": "AAPL last close: $189.30"}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t2",
                    "name": "calculator", "input": {"expression": "50 * 189.30"}}]},
            ],
            "results": [{"tool_use_id": "t2", "content": "9465.0"}],
            "tokens":  412, "cost": 0.00124,
        },
        {
            "msgs": [
                {"role": "user",      "content": [{"type": "tool_result",
                    "tool_use_id": "t2", "content": "9465.0"}]},
                {"role": "assistant", "content": "50 AAPL shares at $189.30 = $9,465.00."},
            ],
            "results": [],
            "tokens":  634, "cost": 0.00190,
        },
    ]

    print(f"\n{sep}")
    print("  AgentState demo  |  3 turns  |  save -> load -> diff")
    print(sep)

    for i, turn in enumerate(TURNS, start=1):
        state.messages.extend(turn["msgs"])
        state.tool_results.extend(turn["results"])
        state.turn_count    = i
        state.total_tokens += turn["tokens"]
        state.total_cost_usd = round(state.total_cost_usd + turn["cost"], 8)
        state.last_updated  = time.time()
        print(f"  [turn {i}] {state.snapshot()}")

    # ── save to disk ──────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = state.save(f"{tmpdir}/agent_state")
        print(f"\n  Checkpoint : {ckpt_path}")

        # ── reload ────────────────────────────────────────────────────────
        loaded = AgentState.load(ckpt_path)

        # ── round-trip verification ───────────────────────────────────────
        delta = state.diff(loaded)
        if not delta:
            print("  Round-trip : PASS  (diff is empty -- states are identical)")
        else:
            print(f"  Round-trip : FAIL  unexpected diff -> {delta}")

        # ── final snapshot from loaded state ──────────────────────────────
        print(f"  Snapshot   : {loaded.snapshot()}")

    # ── validation demo ───────────────────────────────────────────────────
    print(f"\n  Validation tests:")
    try:
        bad = AgentState(task="t", messages=[{"role": "user"}])  # missing 'content'
        print("  FAIL: should have raised ValueError")
    except ValueError as e:
        print(f"  PASS: invalid message rejected -> {e}")

    try:
        bad2 = AgentState(task="t", messages=["not a dict"])
        print("  FAIL: should have raised ValueError")
    except ValueError as e:
        print(f"  PASS: non-dict message rejected -> {e}")

    # ── diff demo ─────────────────────────────────────────────────────────
    old = AgentState(task="test", metadata={"budget_usd": 0.05})
    new = AgentState(
        task="test",
        messages=[{"role": "user", "content": "hello"}],
        turn_count=1,
        total_tokens=300,
        total_cost_usd=0.0009,
        start_time=old.start_time,  # keep same to isolate other diffs
        last_updated=old.last_updated,
        metadata={"budget_usd": 0.05},
    )
    d = old.diff(new)
    print(f"\n  Diff (old -> new after 1 turn):")
    for k, v in d.items():
        print(f"    {k}: {v}")

    print(f"\n{sep}\n")
