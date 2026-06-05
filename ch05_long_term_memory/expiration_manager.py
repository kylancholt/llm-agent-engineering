"""
Memory lifecycle manager: TTL, frequency, and semantic expiration.

Three independent staleness signals are combined into a composite score that
drives automated memory pruning:

  TTL score       -- Age-based: 0 while age < ttl/2, rises linearly to 1.0 at
                     ttl_days, stays at 1.0 beyond. Hard-expires old memories
                     even if they are still occasionally accessed.

  Frequency score -- Usage-based: how far the access rate (accesses/week) falls
                     below min_access_frequency. Score = 0 when rate is at or
                     above threshold; score = 1 when the memory has never been
                     accessed. New memories (< 7 days) receive a grace period.

  Semantic score  -- Supersession-based: if a newer memory covers the same topic
                     with cosine similarity >= semantic_threshold, the older one
                     is considered outdated. Score equals the maximum similarity
                     found; 0 if no similar newer memory exists.

  Composite       -- Weighted mean: 0.40 * ttl + 0.30 * freq + 0.30 * semantic,
                     in [0, 1]. Labels: fresh (<0.25), aging (<0.50),
                     stale (<0.75), expired (>=0.75 or TTL exceeded).

expire_batch() classifies a list of MemoryEntry objects and returns three
buckets: to_delete (expired), to_review (stale), to_preserve (fresh/aging).

run_maintenance() applies expire_batch() against a live MemoryStore collection
and physically removes the expired entries.
"""
from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sentence_transformers import SentenceTransformer

# ── project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from memory.longterm.memory_store import MemoryEntry, MemoryStore

if TYPE_CHECKING:
    pass


# ── constants ─────────────────────────────────────────────────────────────────

_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Composite thresholds → labels
_COMPOSITE_FRESH   = 0.25
_COMPOSITE_AGING   = 0.50
_COMPOSITE_STALE   = 0.75   # >= this → expired

# Composite weights (must sum to 1.0)
_W_TTL  = 0.40
_W_FREQ = 0.30
_W_SEM  = 0.30


# ── public types ──────────────────────────────────────────────────────────────

@dataclass
class StalenessScore:
    """Full staleness breakdown for a single MemoryEntry.

    Attributes:
        ttl_score:        Age-based score in [0, 1].  0 while fresh, 1 at expiry.
        frequency_score:  Usage-based score in [0, 1]. 0 if accessed often, 1 if
                          never accessed relative to the configured threshold.
        semantic_score:   Supersession score in [0, 1]. Max cosine similarity to any
                          newer memory above semantic_threshold; 0 if none found.
        composite_score:  Weighted combination of the three signals.
        label:            "fresh" | "aging" | "stale" | "expired".
    """
    ttl_score:       float
    frequency_score: float
    semantic_score:  float
    composite_score: float
    label:           str


@dataclass
class ExpirationReport:
    """Result of ExpirationManager.expire_batch().

    Attributes:
        to_delete:        Memories labelled 'expired' -- safe to remove.
        to_review:        Memories labelled 'stale'   -- candidates for removal.
        to_preserve:      Memories labelled 'fresh' or 'aging' -- keep.
        staleness_scores: Per-memory StalenessScore keyed by memory ID.
        total:            Total memories evaluated.
        delete_count:     len(to_delete).
        review_count:     len(to_review).
        preserve_count:   len(to_preserve).
    """
    to_delete:        list[MemoryEntry]
    to_review:        list[MemoryEntry]
    to_preserve:      list[MemoryEntry]
    staleness_scores: dict[str, StalenessScore]
    total:            int
    delete_count:     int
    review_count:     int
    preserve_count:   int


@dataclass
class MaintenanceReport:
    """Result of ExpirationManager.run_maintenance().

    Attributes:
        user_id:         The user whose memories were processed.
        deleted_count:   Number of memories actually deleted from the store.
        reviewed_count:  Memories flagged stale but not deleted.
        preserved_count: Memories kept without action.
        deleted_ids:     IDs of the deleted memories.
        run_at:          UTC timestamp when maintenance started.
        duration_ms:     Wall-clock time of the full maintenance run.
    """
    user_id:         str
    deleted_count:   int
    reviewed_count:  int
    preserved_count: int
    deleted_ids:     list[str]
    run_at:          datetime
    duration_ms:     float


# ── expiration manager ────────────────────────────────────────────────────────

class ExpirationManager:
    """Compute staleness and prune expired memories.

    Usage::

        em = ExpirationManager(ttl_days=90, min_access_frequency=0.1)
        score = em.compute_staleness(entry, all_memories=corpus)
        report = em.expire_batch(corpus)
        for m in report.to_delete:
            print(f"  Deleting: {m.content[:60]}")

        # Or apply directly to a live store:
        mreport = em.run_maintenance(store, user_id="alice")
        print(f"Deleted {mreport.deleted_count}, preserved {mreport.preserved_count}")

    Args:
        ttl_days:             Hard TTL: entries older than this are always expired.
        min_access_frequency: Minimum accesses per week before frequency penalty kicks in.
        semantic_supersede:   Enable semantic scoring (requires embeddings).
        semantic_threshold:   Minimum cosine similarity to count as supersession.
        embedding_model:      sentence-transformers model for on-the-fly encoding.
    """

    def __init__(
        self,
        ttl_days:             float = 90.0,
        min_access_frequency: float = 0.1,
        semantic_supersede:   bool  = True,
        semantic_threshold:   float = 0.80,
        embedding_model:      str   = _DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.ttl_days             = ttl_days
        self.min_access_frequency = min_access_frequency
        self.semantic_supersede   = semantic_supersede
        self.semantic_threshold   = semantic_threshold
        self._model               = SentenceTransformer(embedding_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_staleness(
        self,
        entry:        MemoryEntry,
        all_memories: list[MemoryEntry] | None = None,
    ) -> StalenessScore:
        """Compute the full staleness breakdown for one memory.

        Args:
            entry:        The memory to evaluate.
            all_memories: Full corpus for semantic scoring. If None or if
                          ``semantic_supersede=False``, semantic_score = 0.

        Returns:
            StalenessScore with individual signals, composite, and label.
        """
        ttl_s  = self._ttl_score(entry)
        freq_s = self._frequency_score(entry)
        sem_s  = 0.0
        if self.semantic_supersede and all_memories:
            sem_s = self._semantic_score(entry, all_memories)

        composite = _W_TTL * ttl_s + _W_FREQ * freq_s + _W_SEM * sem_s
        composite = round(min(1.0, composite), 4)

        # Hard-expire if TTL is fully exceeded, regardless of composite
        if ttl_s >= 1.0:
            label = "expired"
        elif composite >= _COMPOSITE_STALE:
            label = "expired"
        elif composite >= _COMPOSITE_AGING:
            label = "stale"
        elif composite >= _COMPOSITE_FRESH:
            label = "aging"
        else:
            label = "fresh"

        return StalenessScore(
            ttl_score=       round(ttl_s,  4),
            frequency_score= round(freq_s, 4),
            semantic_score=  round(sem_s,  4),
            composite_score= composite,
            label=           label,
        )

    def expire_batch(
        self,
        memories: list[MemoryEntry],
    ) -> ExpirationReport:
        """Classify all memories and return the three-bucket expiration report.

        Computes staleness for every entry (including semantic scoring across the
        full batch) and partitions them into to_delete, to_review, to_preserve.

        Embeddings missing from any entry are computed in batch before scoring.

        Args:
            memories: All memories to evaluate (typically one user's full corpus).

        Returns:
            ExpirationReport with bucketed entries and per-ID staleness scores.
        """
        if not memories:
            return ExpirationReport(
                to_delete=[], to_review=[], to_preserve=[],
                staleness_scores={}, total=0,
                delete_count=0, review_count=0, preserve_count=0,
            )

        self._fill_embeddings(memories)

        scores: dict[str, StalenessScore] = {}
        for entry in memories:
            scores[entry.id] = self.compute_staleness(entry, all_memories=memories)

        to_delete:   list[MemoryEntry] = []
        to_review:   list[MemoryEntry] = []
        to_preserve: list[MemoryEntry] = []

        for entry in memories:
            label = scores[entry.id].label
            if label == "expired":
                to_delete.append(entry)
            elif label == "stale":
                to_review.append(entry)
            else:
                to_preserve.append(entry)

        return ExpirationReport(
            to_delete=        to_delete,
            to_review=        to_review,
            to_preserve=      to_preserve,
            staleness_scores= scores,
            total=            len(memories),
            delete_count=     len(to_delete),
            review_count=     len(to_review),
            preserve_count=   len(to_preserve),
        )

    def run_maintenance(
        self,
        store:   MemoryStore,
        user_id: str,
    ) -> MaintenanceReport:
        """Fetch all of a user's memories, expire the stale ones, and report.

        Loads the full corpus from ChromaDB (including embeddings), runs
        expire_batch(), and physically deletes every entry in to_delete.

        Args:
            store:   A live MemoryStore instance to read from and delete against.
            user_id: The user whose memories to maintain.

        Returns:
            MaintenanceReport summarising the actions taken.
        """
        run_at = _utcnow()
        t0     = time.perf_counter()

        memories = _load_memories_from_store(store, user_id)
        report   = self.expire_batch(memories)

        deleted_ids: list[str] = []
        for entry in report.to_delete:
            if store.delete(entry.id):
                deleted_ids.append(entry.id)

        duration_ms = (time.perf_counter() - t0) * 1000
        return MaintenanceReport(
            user_id=         user_id,
            deleted_count=   len(deleted_ids),
            reviewed_count=  report.review_count,
            preserved_count= report.preserve_count,
            deleted_ids=     deleted_ids,
            run_at=          run_at,
            duration_ms=     round(duration_ms, 1),
        )

    # ------------------------------------------------------------------
    # Staleness signal helpers
    # ------------------------------------------------------------------

    def _ttl_score(self, entry: MemoryEntry) -> float:
        """Age-based score: 0 while age < ttl/2, linear rise to 1 at ttl_days.

        Piecewise formula:
            age < ttl/2        -> 0.0
            ttl/2 <= age < ttl -> (age - ttl/2) / (ttl/2)
            age >= ttl         -> 1.0
        """
        age  = (_utcnow() - entry.created_at).total_seconds() / 86400
        half = self.ttl_days / 2.0
        if age < half:
            return 0.0
        if age >= self.ttl_days:
            return 1.0
        return (age - half) / (self.ttl_days - half)

    def _frequency_score(self, entry: MemoryEntry) -> float:
        """Usage-based score: how far the access rate falls below threshold.

        Returns 0.0 if access_count / age_weeks >= min_access_frequency,
        rising linearly to 1.0 when the memory has never been accessed.
        Memories younger than 7 days receive a grace period (score = 0.0).
        """
        age_days = (_utcnow() - entry.created_at).total_seconds() / 86400
        if age_days < 7.0:
            return 0.0     # grace period for new memories
        age_weeks   = max(age_days / 7.0, 1.0)
        actual_rate = entry.access_count / age_weeks
        return max(0.0, 1.0 - actual_rate / max(self.min_access_frequency, 1e-9))

    def _semantic_score(
        self,
        entry:        MemoryEntry,
        all_memories: list[MemoryEntry],
    ) -> float:
        """Supersession score: max cosine similarity to any newer memory.

        Iterates over all memories created AFTER entry and computes cosine
        similarity against entry's embedding. Returns the maximum value found
        if it exceeds semantic_threshold, else 0.0.

        Only memories with non-empty embeddings participate. Embeddings must
        be pre-normalised (as produced by sentence-transformers with
        normalize_embeddings=True).

        Returns:
            Float in [0, 1]: 0 = not superseded; high = strongly superseded.
        """
        if not entry.embedding:
            return 0.0

        e_emb = np.array(entry.embedding, dtype=np.float32)
        norm  = np.linalg.norm(e_emb)
        if norm < 1e-8:
            return 0.0
        e_emb = e_emb / norm   # re-normalise defensively

        max_sim = 0.0
        for other in all_memories:
            if other.id == entry.id:
                continue
            if other.created_at <= entry.created_at:
                continue       # only newer memories can supersede
            if not other.embedding:
                continue

            o_emb = np.array(other.embedding, dtype=np.float32)
            o_norm = np.linalg.norm(o_emb)
            if o_norm < 1e-8:
                continue
            o_emb  = o_emb / o_norm
            sim    = max(0.0, float(np.dot(e_emb, o_emb)))
            if sim > max_sim:
                max_sim = sim

        return max_sim if max_sim >= self.semantic_threshold else 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fill_embeddings(self, memories: list[MemoryEntry]) -> None:
        """Batch-compute missing embeddings in-place using the ST model."""
        missing = [i for i, m in enumerate(memories) if not m.embedding]
        if not missing:
            return
        texts = [memories[i].content for i in missing]
        vecs  = self._model.encode(texts, normalize_embeddings=True)
        for j, idx in enumerate(missing):
            memories[idx].embedding = vecs[j].tolist()


# ── module helpers ────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


def _parse_dt(iso: str) -> datetime:
    try:
        return datetime.fromisoformat(iso.rstrip("Z")).replace(tzinfo=None)
    except (ValueError, AttributeError):
        return _utcnow()


def _load_memories_from_store(store: MemoryStore, user_id: str) -> list[MemoryEntry]:
    """Reconstruct MemoryEntry objects from a ChromaDB get() call."""
    data = store._col.get(
        where={"user_id": user_id},
        include=["documents", "metadatas", "embeddings"],
    )
    memories: list[MemoryEntry] = []
    ids        = data.get("ids", [])
    docs       = data.get("documents") or []
    metas      = data.get("metadatas") or []
    embs_raw   = data.get("embeddings") or []

    for i, mem_id in enumerate(ids):
        meta    = metas[i] if i < len(metas) else {}
        doc     = docs[i]  if i < len(docs)  else ""
        emb_raw = embs_raw[i] if i < len(embs_raw) else []
        embedding = list(emb_raw) if emb_raw is not None else []

        memories.append(MemoryEntry(
            id=              mem_id,
            content=         doc,
            user_id=         meta.get("user_id", user_id),
            memory_type=     meta.get("memory_type", "factual"),
            embedding=       embedding,
            created_at=      _parse_dt(meta.get("created_at", "")),
            last_accessed=   _parse_dt(meta.get("last_accessed", "")),
            access_count=    int(meta.get("access_count", 0)),
            staleness_score= float(meta.get("staleness_score", 0.0)),
        ))
    return memories


def _make_entry(
    content:      str,
    memory_type:  str,
    days_ago:     float,
    access_count: int,
    user_id:      str = "demo_user",
) -> MemoryEntry:
    """Build a synthetic MemoryEntry for the demo (no embedding yet)."""
    now        = _utcnow()
    created_at = now - timedelta(days=days_ago)
    staleness  = min(1.0, max(0.0, days_ago / 30.0))
    return MemoryEntry(
        id=              str(uuid.uuid4()),
        content=         content,
        user_id=         user_id,
        memory_type=     memory_type,
        embedding=       [],
        created_at=      created_at,
        last_accessed=   created_at + timedelta(days=min(days_ago, 1)),
        access_count=    access_count,
        staleness_score= staleness,
    )


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Build 30 synthetic memories ───────────────────────────────────────────
    # 6 groups x 5 memories = 30 total.
    #
    # Group A (5): Fresh superseding facts  -- 3-7d old, frequent access.
    #              These ALSO act as the "newer version" for Group F entries.
    # Group B (5): Aging  -- 40-50d, moderate access.
    # Group C (5): Stale  -- 65-80d, rare access.
    # Group D (5): Expired by TTL -- 95-120d, no access (age >= ttl_days=90).
    # Group E (5): Expired by frequency -- 55-75d, zero accesses.
    # Group F (5): Semantically superseded -- 88-96d, 0-1 access,
    #              outdated versions of the Group A facts.

    _SEP = "=" * 72
    print(_SEP)
    print("  ExpirationManager demo  |  30 synthetic memories  |  ttl=90d")
    print(_SEP)
    print("\n  Building corpus...")

    # --- Group A: fresh AND superseding (3-7 days, 3-7 accesses) ---
    group_a = [
        _make_entry("The API rate limit was raised to 120 requests per minute after tier upgrade.",
                    "factual",  3.0, 7),
        _make_entry("LLM API budget increased to $1 500 per month following usage growth.",
                    "factual",  4.0, 5),
        _make_entry("Production database now holds 4.5 million records after 4 months of growth.",
                    "factual",  5.0, 6),
        _make_entry("The agent now supports 9 tools including web_search, python_repl, and bash.",
                    "factual",  6.0, 4),
        _make_entry("CI/CD pipeline optimised: total deployment time reduced to under 5 minutes.",
                    "factual",  7.0, 3),
    ]

    # --- Group B: aging (40-50 days, 2-4 accesses) ---
    group_b = [
        _make_entry("The ReAct pattern interleaves reasoning steps with tool-use actions.",
                    "semantic", 42.0, 3),
        _make_entry("Redis cluster was scaled from 3 to 6 nodes to handle peak load.",
                    "episodic", 45.0, 2),
        _make_entry("Sentence-transformers all-MiniLM-L6-v2 produces 384-dimensional embeddings.",
                    "factual",  47.0, 4),
        _make_entry("The context window for claude-sonnet-4-6 is 200 000 tokens.",
                    "factual",  49.0, 2),
        _make_entry("Implemented exponential backoff for API retries with jitter to avoid thundering herd.",
                    "episodic", 50.0, 3),
    ]

    # --- Group C: stale (65-80 days, 0-1 access) ---
    group_c = [
        _make_entry("Docker layer caching reduced CI build time from 8 min to 90 seconds.",
                    "episodic", 66.0, 1),
        _make_entry("BM25 sparse retrieval boosts exact keyword matches missed by dense embeddings.",
                    "semantic", 70.0, 0),
        _make_entry("The staging environment uses t3.medium instances on AWS us-east-1.",
                    "factual",  74.0, 1),
        _make_entry("Prompt caching on claude-sonnet-4-6 reduces cost by up to 90% for repeated prefixes.",
                    "semantic", 77.0, 0),
        _make_entry("The on-call rotation follows a 1-week cycle with 4 engineers in the pool.",
                    "factual",  80.0, 1),
    ]

    # --- Group D: expired by TTL (95-120 days, never accessed) ---
    group_d = [
        _make_entry("Initial agent loop prototype completed; used simple text-based tool calling.",
                    "episodic", 95.0,  0),
        _make_entry("Python 3.11 beta installed on dev machines for performance testing.",
                    "factual",  100.0, 0),
        _make_entry("The team used Notion for documentation before migrating to Confluence.",
                    "factual",  105.0, 0),
        _make_entry("Load testing with Locust showed the API saturates at 400 rps on 2 vCPUs.",
                    "episodic", 110.0, 0),
        _make_entry("Embeddings were first stored in Pinecone before switching to ChromaDB.",
                    "episodic", 120.0, 0),
    ]

    # --- Group E: expired by frequency (55-75 days, zero accesses) ---
    group_e = [
        _make_entry("GDPR compliance review scheduled for Q2; DPA agreement pending signature.",
                    "factual",  55.0, 0),
        _make_entry("Investigated using LangGraph for multi-agent orchestration.",
                    "episodic", 60.0, 0),
        _make_entry("Considered adopting Weaviate as alternative to ChromaDB for scale.",
                    "episodic", 65.0, 0),
        _make_entry("Spike on using async generators for streaming agent responses completed.",
                    "episodic", 70.0, 0),
        _make_entry("Evaluated OpenAI Assistants API for comparison with Anthropic tool_use.",
                    "episodic", 75.0, 0),
    ]

    # --- Group F: semantically superseded by Group A (88-96 days, 0-1 access) ---
    group_f = [
        _make_entry("The API rate limit is 60 requests per minute per user.",
                    "factual", 92.0, 0),
        _make_entry("The team budget for LLM APIs is $500 per month.",
                    "factual", 88.0, 1),
        _make_entry("The production database holds approximately 1 million records.",
                    "factual", 96.0, 0),
        _make_entry("The agent currently supports 3 tools: search, read_file, write_file.",
                    "factual", 90.0, 0),
        _make_entry("The deployment pipeline takes around 15 minutes end-to-end.",
                    "factual", 93.0, 1),
    ]

    all_memories: list[MemoryEntry] = (
        group_a + group_b + group_c + group_d + group_e + group_f
    )
    assert len(all_memories) == 30, f"Expected 30, got {len(all_memories)}"

    print(f"  Corpus built: {len(all_memories)} entries across 6 groups")
    print(f"    Group A (fresh+superseding, 3-7d):   {len(group_a)}")
    print(f"    Group B (aging, 40-50d):              {len(group_b)}")
    print(f"    Group C (stale, 65-80d):              {len(group_c)}")
    print(f"    Group D (expired TTL, 95-120d):       {len(group_d)}")
    print(f"    Group E (expired freq, 55-75d):       {len(group_e)}")
    print(f"    Group F (superseded by A, 88-96d):   {len(group_f)}")

    # ── Run expiration ────────────────────────────────────────────────────────
    em = ExpirationManager(
        ttl_days=90,
        min_access_frequency=0.1,
        semantic_supersede=True,
        semantic_threshold=0.80,
    )

    print(f"\n  Computing staleness scores (including semantic; embeddings computed in batch)...")
    t0     = time.perf_counter()
    report = em.expire_batch(all_memories)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # ── Per-entry detail table ────────────────────────────────────────────────
    print(f"\n{_SEP}")
    print("  STALENESS DETAIL (all 30 memories)")
    print(_SEP)
    print(
        f"  {'Grp':<4}  {'Age':>5}  {'Acc':>3}  "
        f"{'TTL':>5}  {'Freq':>5}  {'Sem':>5}  {'Comp':>5}  {'Label':<9}  "
        f"Content (first 48 chars)"
    )
    print(f"  {'-'*100}")

    group_labels = (
        ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5 +
        ["E"] * 5 + ["F"] * 5
    )

    now = _utcnow()
    for i, mem in enumerate(all_memories):
        sc       = report.staleness_scores[mem.id]
        age_days = (now - mem.created_at).total_seconds() / 86400
        grp      = group_labels[i]
        print(
            f"  {grp:<4}  {age_days:>5.1f}  {mem.access_count:>3}  "
            f"{sc.ttl_score:>5.2f}  {sc.frequency_score:>5.2f}  "
            f"{sc.semantic_score:>5.2f}  {sc.composite_score:>5.3f}  "
            f"{sc.label:<9}  {mem.content[:48]}"
        )

    # ── Summary report ────────────────────────────────────────────────────────
    print(f"\n{_SEP}")
    print("  EXPIRATION REPORT")
    print(f"  {'-'*55}")
    print(f"  Total evaluated  : {report.total}")
    print(f"  To delete        : {report.delete_count}  (label=expired)")
    print(f"  To review        : {report.review_count}  (label=stale)")
    print(f"  To preserve      : {report.preserve_count}  (label=fresh/aging)")
    print(f"  Elapsed          : {elapsed_ms:.0f}ms")

    print(f"\n  Entries to DELETE:")
    for m in report.to_delete:
        sc = report.staleness_scores[m.id]
        age_days = (now - m.created_at).total_seconds() / 86400
        reason = (
            "TTL"       if sc.ttl_score >= 1.0  else
            "semantic"  if sc.semantic_score >= em.semantic_threshold else
            "frequency" if sc.frequency_score >= 0.90 else "composite"
        )
        print(f"    [{reason:<9}]  age={age_days:>5.0f}d  comp={sc.composite_score:.3f}  "
              f"{m.content[:55]}")

    print(f"\n  Entries to REVIEW (stale):")
    for m in report.to_review:
        sc = report.staleness_scores[m.id]
        age_days = (now - m.created_at).total_seconds() / 86400
        print(f"    age={age_days:>5.0f}d  comp={sc.composite_score:.3f}  {m.content[:60]}")

    # ── Signal breakdown ──────────────────────────────────────────────────────
    scores_list = list(report.staleness_scores.values())
    avg_ttl  = sum(s.ttl_score       for s in scores_list) / len(scores_list)
    avg_freq = sum(s.frequency_score for s in scores_list) / len(scores_list)
    avg_sem  = sum(s.semantic_score  for s in scores_list) / len(scores_list)
    avg_comp = sum(s.composite_score for s in scores_list) / len(scores_list)

    by_label: dict[str, int] = {}
    for s in scores_list:
        by_label[s.label] = by_label.get(s.label, 0) + 1

    print(f"\n{_SEP}")
    print("  SIGNAL AVERAGES")
    print(f"  {'-'*40}")
    print(f"  Avg TTL score       : {avg_ttl:.3f}  (weight={_W_TTL})")
    print(f"  Avg frequency score : {avg_freq:.3f}  (weight={_W_FREQ})")
    print(f"  Avg semantic score  : {avg_sem:.3f}  (weight={_W_SEM})")
    print(f"  Avg composite score : {avg_comp:.3f}")
    print(f"\n  Label distribution  : {dict(sorted(by_label.items()))}")
    print(_SEP)
