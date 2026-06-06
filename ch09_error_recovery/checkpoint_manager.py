"""
Checkpoint manager for agent state persistence and crash recovery.

Saves AgentState snapshots to disk at each completed step, enabling crash
recovery without re-executing finished work. Key features:

- Content-addressed deduplication: if the canonical state fields have not
  changed since the last checkpoint (same SHA-256 prefix), the write is
  skipped entirely.  ``start_time`` and ``last_updated`` are excluded from the
  hash — they change on every mutation but represent no new work.
- Resume: ``resume()`` returns the latest checkpoint state together with the
  set of completed step ids so the orchestrator knows which steps to skip.
- Pruning: at most ``max_checkpoints_per_task`` files are kept per task
  directory; older checkpoints are removed after each save.

Checkpoint files live at::

    {checkpoint_dir}/{task_id}/ckpt_{step_id:05d}_{ts_ms:013d}_{hash16}.json

The ``_ckpt_*`` metadata embedded in each file is stripped before
deserialising into AgentState, so the format tolerates future AgentState
fields transparently.

Stdlib only — no external dependencies, no LLM calls.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── project root: ch09_error_recovery/../ = root ──────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ch03_agent_loop.agent_state import AgentState  # noqa: E402


# ── hash contract ─────────────────────────────────────────────────────────────

# Only content-bearing fields participate in the dedup hash.
# Excluded: start_time, last_updated (pure timing metadata, change every turn).
_HASH_FIELDS: tuple[str, ...] = (
    "task", "messages", "tool_results",
    "turn_count", "total_tokens", "total_cost_usd", "metadata",
)

_CKPT_VERSION = 1


# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class CheckpointInfo:
    """Lightweight descriptor for a checkpoint file on disk.

    Attributes:
        checkpoint_path: Absolute path to the JSON file.
        task_id: Task this checkpoint belongs to.
        step_id: Step that was completed when this checkpoint was written.
        timestamp: Unix timestamp of the save (seconds since epoch).
        state_hash: 16-hex-char SHA-256 prefix of the canonical state fields.
            Used for deduplication; also embedded in the filename so
            ``list_checkpoints`` never needs to open any file.
        size_bytes: File size in bytes as reported by the filesystem stat.
    """

    checkpoint_path: str
    task_id:         str
    step_id:         int
    timestamp:       float
    state_hash:      str
    size_bytes:      int


@dataclass
class ResumeResult:
    """Result of a resume operation.

    Attributes:
        state: AgentState restored from the latest checkpoint.  Its
            ``tool_results`` already contains the outputs of every completed
            step — no re-execution is needed for those steps.
        last_checkpoint: Metadata for the checkpoint that was loaded.
        completed_steps: Step ids reflected in ``state``.  The orchestrator
            must skip all steps whose id appears in this list.
        skippable_steps: Alias of ``completed_steps`` for clarity at call
            sites that phrase the question as "which steps can I skip?".
    """

    state:            AgentState
    last_checkpoint:  CheckpointInfo
    completed_steps:  list[int]
    skippable_steps:  list[int]


# ── manager ───────────────────────────────────────────────────────────────────

class CheckpointManager:
    """Saves and restores AgentState checkpoints with deduplication and pruning.

    Args:
        checkpoint_dir: Root directory for all checkpoint files.  Each task
            gets its own subdirectory: ``{checkpoint_dir}/{task_id}/``.
            Created on first save if it does not exist.
        max_checkpoints_per_task: Maximum number of checkpoint files kept per
            task.  After each save, files beyond this limit are deleted (oldest
            step first).  Set to 0 to disable auto-pruning.

    The manager maintains an in-memory cache of the most-recently-saved hash
    per task.  The cache is cold after a process restart; it is lazily warmed
    from the last file on disk the first time ``save`` is called.
    """

    def __init__(
        self,
        checkpoint_dir:           str = "checkpoints/",
        max_checkpoints_per_task: int = 10,
    ) -> None:
        self.checkpoint_dir           = Path(checkpoint_dir)
        self.max_checkpoints_per_task = max_checkpoints_per_task
        # task_id -> 16-char hex hash of the last-written checkpoint
        self._last_hash: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        task_id:  str,
        state:    AgentState,
        step_id:  int,
    ) -> CheckpointInfo:
        """Persist a checkpoint for ``state`` at the given ``step_id``.

        If the state's content hash matches the last saved hash for this task,
        the file write is skipped and the most recent existing CheckpointInfo is
        returned.  This avoids wasting disk I/O when a step completes without
        mutating the agent state (e.g. a read-only probe that produced no tool
        results).

        Args:
            task_id: Unique identifier for the task being checkpointed.
            state:   Current agent state snapshot to save.
            step_id: Step id that just completed successfully.

        Returns:
            CheckpointInfo for the newly created file, or for the most recent
            existing file when the write was deduplicated.
        """
        h = self._compute_hash(state)

        # Warm the dedup cache from disk on the first call after process restart.
        if task_id not in self._last_hash:
            on_disk = self.list_checkpoints(task_id)
            if on_disk:
                self._last_hash[task_id] = on_disk[-1].state_hash

        if self._last_hash.get(task_id) == h:
            on_disk = self.list_checkpoints(task_id)
            if on_disk:
                return on_disk[-1]

        ts   = time.time()
        path = self._ckpt_path(task_id, step_id, ts, h)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = state.to_dict()
        payload["_ckpt_version"] = _CKPT_VERSION
        payload["_task_id"]      = task_id
        payload["_step_id"]      = step_id
        payload["_timestamp"]    = ts
        payload["_state_hash"]   = h

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        size = path.stat().st_size

        self._last_hash[task_id] = h

        info = CheckpointInfo(
            checkpoint_path=str(path),
            task_id=task_id,
            step_id=step_id,
            timestamp=ts,
            state_hash=h,
            size_bytes=size,
        )
        self._auto_prune(task_id)
        return info

    def load(
        self,
        task_id: str,
        step_id: int | None = None,
    ) -> AgentState:
        """Deserialise a checkpoint file into an AgentState.

        Args:
            task_id: Task to load.
            step_id: Specific step checkpoint to load.  When ``None``, the
                checkpoint with the highest step_id is loaded.

        Returns:
            A fully validated AgentState.

        Raises:
            FileNotFoundError: No checkpoints exist for ``task_id``.
            ValueError: A specific ``step_id`` was requested but not found.
        """
        checkpoints = self.list_checkpoints(task_id)
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found for task '{task_id}' "
                f"in {self.checkpoint_dir}"
            )

        if step_id is None:
            info = checkpoints[-1]
        else:
            candidates = [c for c in checkpoints if c.step_id == step_id]
            if not candidates:
                raise ValueError(
                    f"No checkpoint for task '{task_id}' at step {step_id}"
                )
            info = candidates[-1]   # latest timestamp if duplicates exist

        raw  = Path(info.checkpoint_path).read_text(encoding="utf-8")
        data = json.loads(raw)
        # Strip checkpoint-specific underscore-prefixed fields.
        for key in [k for k in data if k.startswith("_")]:
            del data[key]
        return AgentState.from_dict(data)

    def list_checkpoints(self, task_id: str) -> list[CheckpointInfo]:
        """Return all checkpoints for ``task_id``, ordered by step_id ascending.

        Metadata is parsed from filename and filesystem stat only — no file
        reads are performed, making this safe to call frequently.

        Args:
            task_id: Task to inspect.

        Returns:
            Sorted list of CheckpointInfo; empty when the task directory does
            not exist or contains no valid checkpoint files.
        """
        task_dir = self._task_dir(task_id)
        if not task_dir.exists():
            return []

        infos: list[CheckpointInfo] = []
        for p in task_dir.glob("ckpt_*.json"):
            info = self._parse_filename(p, task_id)
            if info is not None:
                infos.append(info)

        infos.sort(key=lambda c: (c.step_id, c.timestamp))
        return infos

    def resume(self, task_id: str) -> ResumeResult:
        """Restore from the latest checkpoint and report which steps to skip.

        The ``completed_steps`` / ``skippable_steps`` in the returned result
        contain the step ids of every checkpoint on disk for ``task_id``.  The
        orchestrator should skip all steps whose id appears in
        ``skippable_steps`` — their tool results are already present in the
        restored state's ``tool_results`` list.

        Args:
            task_id: Task to resume.

        Returns:
            ResumeResult with the restored AgentState and completed step ids.

        Raises:
            FileNotFoundError: No checkpoints exist for the task.
        """
        checkpoints = self.list_checkpoints(task_id)
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints to resume from for task '{task_id}'"
            )

        last  = checkpoints[-1]
        state = self.load(task_id, last.step_id)
        done  = [c.step_id for c in checkpoints]

        return ResumeResult(
            state=state,
            last_checkpoint=last,
            completed_steps=done,
            skippable_steps=done,
        )

    def prune(self, task_id: str, keep_last_n: int) -> int:
        """Delete old checkpoints, retaining the ``keep_last_n`` most recent.

        Checkpoints are ordered by step_id (then timestamp for ties).  The
        highest-step files are kept; earlier ones are removed.

        Args:
            task_id:    Task to prune.
            keep_last_n: Number of checkpoint files to retain.  Pass 0 to
                delete all checkpoints for the task.

        Returns:
            Number of files actually deleted.
        """
        checkpoints = self.list_checkpoints(task_id)
        to_delete   = checkpoints[:-keep_last_n] if keep_last_n > 0 else checkpoints

        deleted = 0
        for info in to_delete:
            try:
                Path(info.checkpoint_path).unlink(missing_ok=True)
                deleted += 1
            except OSError:
                pass
        return deleted

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _task_dir(self, task_id: str) -> Path:
        return self.checkpoint_dir / task_id

    def _ckpt_path(
        self, task_id: str, step_id: int, ts: float, state_hash: str
    ) -> Path:
        """Build the canonical path for a checkpoint file.

        Format: ``ckpt_{step_id:05d}_{ts_ms:013d}_{hash16}.json``
        """
        ts_ms = int(ts * 1000)
        name  = f"ckpt_{step_id:05d}_{ts_ms:013d}_{state_hash}.json"
        return self._task_dir(task_id) / name

    @staticmethod
    def _compute_hash(state: AgentState) -> str:
        """Return a 16-char hex SHA-256 prefix over the canonical state fields.

        Excludes ``start_time`` and ``last_updated`` so that pure timing
        mutations don't defeat deduplication.
        """
        d         = state.to_dict()
        canonical = {k: d[k] for k in _HASH_FIELDS}
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True).encode()
        ).hexdigest()[:16]

    @staticmethod
    def _parse_filename(path: Path, task_id: str) -> CheckpointInfo | None:
        """Parse a CheckpointInfo from filename + stat; return None on failure.

        Expected stem: ``ckpt_{step_id:05d}_{ts_ms:013d}_{hash16}``
        """
        try:
            parts = path.stem.split("_")
            # ["ckpt", "00001", "0000001234567", "abcdef0123456789"]
            if len(parts) < 4 or parts[0] != "ckpt":
                return None
            step_id    = int(parts[1])
            ts         = int(parts[2]) / 1000.0
            state_hash = parts[3]
            size       = path.stat().st_size
        except (ValueError, OSError, IndexError):
            return None

        return CheckpointInfo(
            checkpoint_path=str(path),
            task_id=task_id,
            step_id=step_id,
            timestamp=ts,
            state_hash=state_hash,
            size_bytes=size,
        )

    def _auto_prune(self, task_id: str) -> None:
        if self.max_checkpoints_per_task > 0:
            self.prune(task_id, self.max_checkpoints_per_task)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Book scenario: an 8-step analysis pipeline.
    # Steps 1-5 complete successfully; a crash occurs at step 6 before its
    # checkpoint is written.  A fresh manager resumes from step 5 and reports
    # that steps 1-5 need not be re-executed.

    import tempfile

    SEP = "=" * 68

    STEPS: list[dict[str, Any]] = [
        {"id": 1, "tool": "fetch_data",     "blocking": True},
        {"id": 2, "tool": "parse_sources",  "blocking": True},
        {"id": 3, "tool": "run_analysis",   "blocking": True},
        {"id": 4, "tool": "cross_validate", "blocking": True},
        {"id": 5, "tool": "draft_summary",  "blocking": True},
        {"id": 6, "tool": "generate_chart", "blocking": True},   # crash here
        {"id": 7, "tool": "format_report",  "blocking": True},
        {"id": 8, "tool": "send_email",     "blocking": True},
    ]

    TOOL_RESULTS: dict[int, str] = {
        1: "12 data points fetched from three API endpoints",
        2: "parsed: 12 records, 2 outliers flagged",
        3: "regression r²=0.94, trend: +2.1% MoM",
        4: "cross-validated: 3/3 models agree on trend direction",
        5: "executive summary drafted (287 words)",
    }

    TASK_ID = "market_report_q3"

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(
            checkpoint_dir=tmpdir,
            max_checkpoints_per_task=10,
        )

        state = AgentState(
            task="Analyse Q3 market data and produce a board-level report.",
            metadata={"steps": STEPS, "max_turns": 20, "budget_usd": 0.10},
        )

        print(f"\n{SEP}")
        print(f"  CHECKPOINT MANAGER  |  task={TASK_ID}  |  5 checkpoints + crash")
        print(f"  checkpoint_dir=<tmpdir>  max_checkpoints={mgr.max_checkpoints_per_task}")
        print(SEP)

        # ── phase 1: execute steps 1-5, checkpoint after each ────────────────
        print("\n  [phase 1] running 5 steps, saving a checkpoint after each\n")

        for sid in range(1, 6):
            step = STEPS[sid - 1]
            # simulate step execution by mutating state
            state.tool_results.append(
                {"step": sid, "tool": step["tool"], "content": TOOL_RESULTS[sid]}
            )
            state.turn_count      = sid
            state.total_tokens   += 300 + sid * 80
            state.total_cost_usd  = round(state.total_cost_usd + 0.0008 * sid, 8)
            state.last_updated    = time.time()

            t0   = time.perf_counter()
            info = mgr.save(TASK_ID, state, sid)
            ms   = (time.perf_counter() - t0) * 1000

            print(
                f"  [step {sid}] {step['tool']:<16}  save  {ms:5.1f} ms  "
                f"hash={info.state_hash[:8]}  {info.size_bytes / 1024:.1f} KB"
            )

        # ── dedup check: re-save step 5 with unchanged state ─────────────────
        print("\n  [dedup] re-saving step 5 with unchanged state (write should be skipped):")
        n_before = len(mgr.list_checkpoints(TASK_ID))

        t0       = time.perf_counter()
        info_dup = mgr.save(TASK_ID, state, 5)
        ms_dup   = (time.perf_counter() - t0) * 1000

        n_after = len(mgr.list_checkpoints(TASK_ID))
        verdict = "SKIPPED (dedup OK)" if n_after == n_before else "WRITTEN (unexpected)"
        print(f"  {ms_dup:.2f} ms   files on disk: {n_before} -> {n_after}   [{verdict}]")

        # ── simulate crash at step 6 ──────────────────────────────────────────
        print(f"\n  --- crash at step 6 (steps 6-8 never executed, checkpoints lost) ---")
        print(f"  checkpoints on disk: {[c.step_id for c in mgr.list_checkpoints(TASK_ID)]}")

        # ── phase 2: fresh manager (new process), resume ──────────────────────
        print(f"\n  [phase 2] fresh CheckpointManager instance (simulates process restart)\n")
        fresh = CheckpointManager(
            checkpoint_dir=tmpdir,
            max_checkpoints_per_task=10,
        )

        t0     = time.perf_counter()
        result = fresh.resume(TASK_ID)
        ms_load = (time.perf_counter() - t0) * 1000

        print(
            f"  load  {ms_load:.1f} ms   from step {result.last_checkpoint.step_id} "
            f"(latest checkpoint on disk)"
        )
        print()
        print("  RESUME RESULT:")
        print(
            f"    last checkpoint   : step {result.last_checkpoint.step_id}  "
            f"(hash={result.last_checkpoint.state_hash[:8]})"
        )
        print(f"    completed steps   : {result.completed_steps}  <- NOT re-executed")
        print(f"    skippable steps   : {result.skippable_steps}")
        print(f"    state.turn_count  : {result.state.turn_count}")
        print(f"    state.total_cost  : ${result.state.total_cost_usd:.4f}")
        print(
            f"    state.tool_results: {len(result.state.tool_results)} entries "
            f"preserved in restored state"
        )
        print(f"    state.task        : {result.state.task!r}")

        remaining = [s["id"] for s in STEPS if s["id"] not in result.skippable_steps]
        print(f"\n  still to execute  : steps {remaining}")
        print(
            f"  zero re-execution : steps {result.completed_steps} "
            f"({len(result.completed_steps)} tool results already in state - reused as-is)"
        )

        # ── checkpoint listing ────────────────────────────────────────────────
        print(f"\n  checkpoint listing (ordered by step):")
        for c in fresh.list_checkpoints(TASK_ID):
            print(
                f"    step {c.step_id:02d}  "
                f"hash={c.state_hash[:8]}  "
                f"{c.size_bytes / 1024:.1f} KB"
            )

        # ── prune demo ────────────────────────────────────────────────────────
        print(f"\n  [prune] keep 3 most recent checkpoints:")
        deleted = fresh.prune(TASK_ID, keep_last_n=3)
        kept    = [c.step_id for c in fresh.list_checkpoints(TASK_ID)]
        print(f"  deleted {deleted} file(s)   remaining steps: {kept}")

        print(f"\n{SEP}\n")
