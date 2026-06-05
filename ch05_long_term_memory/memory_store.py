"""
Long-term memory store backed by ChromaDB with sentence-transformer embeddings.

Supports three memory types that map to classical memory science:
  episodic  -- personal experiences and events ("I completed the first agent run").
  semantic  -- general knowledge and concepts ("Transformers use attention mechanisms").
  factual   -- specific, verifiable facts ("The API rate limit is 60 req/min").

Each memory entry tracks access history (last_accessed, access_count) and
a staleness score that grows linearly as the memory sits idle, reaching 1.0
at stale_threshold_days. Staleness is computed dynamically from last_accessed,
not stored, so it is always current.

ChromaDB persists the index to disk; re-instantiating MemoryStore over the
same persist_dir recovers all previously written memories.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ── constants ─────────────────────────────────────────────────────────────────

MEMORY_TYPES: frozenset[str] = frozenset({"episodic", "semantic", "factual"})

_DEFAULT_MODEL        = "all-MiniLM-L6-v2"
_DEFAULT_COLLECTION   = "agent_memories"
_STALE_DAYS_DEFAULT   = 30.0   # idle days at which staleness_score hits 1.0

# Freshness thresholds based on staleness_score
_FRESH_LIMIT  = 0.33   # [0,    0.33) -> "fresh"
_AGING_LIMIT  = 0.67   # [0.33, 0.67) -> "aging"
                        # [0.67, 1.0]  -> "stale"


# ── public types ──────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single stored memory with full access-tracking metadata.

    Attributes:
        id:              UUID string assigned at write time.
        content:         Stored text.
        user_id:         Owner identifier.
        memory_type:     "episodic" | "semantic" | "factual".
        embedding:       Dense vector from the sentence-transformer model.
        created_at:      UTC datetime when first written.
        last_accessed:   UTC datetime of the most recent retrieval or write.
        access_count:    Total number of times this entry was retrieved.
        staleness_score: 0.0 = freshly accessed, 1.0 = idle >= stale_threshold_days.
    """
    id:              str
    content:         str
    user_id:         str
    memory_type:     str
    embedding:       list[float]
    created_at:      datetime
    last_accessed:   datetime
    access_count:    int
    staleness_score: float


@dataclass
class RetrievedMemory:
    """A memory returned by a similarity search.

    Attributes:
        entry:            The MemoryEntry (access metadata already updated).
        similarity_score: Cosine similarity in [0, 1]; 1.0 = identical.
        age_days:         Days elapsed since the memory was created.
        freshness_label:  "fresh" | "aging" | "stale".
    """
    entry:            MemoryEntry
    similarity_score: float
    age_days:         float
    freshness_label:  str


@dataclass
class MemoryStats:
    """Aggregate statistics for one user's memories.

    Attributes:
        total_memories: Total stored entries.
        by_type:        Entry count per memory_type.
        avg_age_days:   Mean age of all memories (days since created_at).
        stale_count:    Entries with current staleness_score >= _AGING_LIMIT.
    """
    total_memories: int
    by_type:        dict[str, int]
    avg_age_days:   float
    stale_count:    int


# ── memory store ──────────────────────────────────────────────────────────────

class MemoryStore:
    """Long-term memory store backed by ChromaDB.

    All users share one ChromaDB collection; ``user_id`` is stored as metadata
    and used as a filter on every query so users never see each other's memories.

    Usage::

        store = MemoryStore(persist_dir="memory_store")
        entry = store.write("The rate limit is 60 req/min", user_id="alice", memory_type="factual")
        results = store.retrieve("API limits", user_id="alice", top_k=3)
        for r in results:
            print(f"{r.similarity_score:.3f}  [{r.freshness_label}]  {r.entry.content}")

    Args:
        persist_dir:          Path where ChromaDB writes its on-disk index.
        collection_name:      ChromaDB collection name.
        embedding_model:      Sentence-transformers model ID.
        stale_threshold_days: Idle days at which staleness_score reaches 1.0.
    """

    def __init__(
        self,
        persist_dir:          str   = "memory_store",
        collection_name:      str   = _DEFAULT_COLLECTION,
        embedding_model:      str   = _DEFAULT_MODEL,
        stale_threshold_days: float = _STALE_DAYS_DEFAULT,
    ) -> None:
        self.persist_dir          = persist_dir
        self.collection_name      = collection_name
        self.stale_threshold_days = stale_threshold_days

        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
            normalize_embeddings=True,   # enables cosine = 1 - distance
        )
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col    = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(
        self,
        content:     str,
        user_id:     str,
        memory_type: str,
        metadata:    dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Store a new memory and return its MemoryEntry.

        Args:
            content:     Text to embed and store.
            user_id:     Owner identifier.
            memory_type: One of "episodic", "semantic", or "factual".
            metadata:    Optional extra str/int/float/bool key-value pairs.

        Returns:
            The newly created MemoryEntry.

        Raises:
            ValueError: If memory_type is not a recognised type.
        """
        if memory_type not in MEMORY_TYPES:
            raise ValueError(
                f"memory_type must be one of {sorted(MEMORY_TYPES)}; got {memory_type!r}."
            )

        now       = _utcnow()
        entry_id  = str(uuid.uuid4())
        embedding = self._embed_fn([content])[0]

        chroma_meta: dict[str, Any] = {
            "user_id":       user_id,
            "memory_type":   memory_type,
            "created_at":    now.isoformat(),
            "last_accessed": now.isoformat(),
            "access_count":  0,
        }
        if metadata:
            _reserved = set(chroma_meta.keys())
            for k, v in metadata.items():
                if k not in _reserved and isinstance(v, (str, int, float, bool)):
                    chroma_meta[k] = v

        self._col.add(
            ids=[entry_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[chroma_meta],
        )

        return MemoryEntry(
            id=entry_id,
            content=content,
            user_id=user_id,
            memory_type=memory_type,
            embedding=list(embedding),
            created_at=now,
            last_accessed=now,
            access_count=0,
            staleness_score=0.0,
        )

    def retrieve(
        self,
        query:   str,
        user_id: str,
        top_k:   int = 5,
    ) -> list[RetrievedMemory]:
        """Find the top_k most similar memories for a user.

        Updates last_accessed and increments access_count in ChromaDB for each
        returned entry. Staleness is recomputed at call time from last_accessed.

        Args:
            query:   Natural-language query string.
            user_id: Only memories belonging to this user are searched.
            top_k:   Maximum number of results to return.

        Returns:
            List of RetrievedMemory, sorted by similarity descending.
        """
        # Count user's entries to avoid n_results > collection size error
        count_resp = self._col.get(
            where={"user_id": user_id},
            include=["documents"],
        )
        actual_k = min(top_k, len(count_resp["ids"]))
        if actual_k == 0:
            return []

        results = self._col.query(
            query_texts=[query],
            n_results=actual_k,
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        now  = _utcnow()
        ids       = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        embs      = (results.get("embeddings") or [[]])[0]

        retrieved: list[RetrievedMemory] = []
        for i, mem_id in enumerate(ids):
            meta      = metadatas[i]
            new_count = int(meta.get("access_count", 0)) + 1

            updated_meta = {
                **meta,
                "last_accessed": now.isoformat(),
                "access_count":  new_count,
            }
            self._col.update(ids=[mem_id], metadatas=[updated_meta])

            created_at = _parse_dt(meta["created_at"])
            age_days   = (now - created_at).total_seconds() / 86400
            staleness  = _compute_staleness(
                _parse_dt(meta["last_accessed"]), now, self.stale_threshold_days
            )
            sim_score  = max(0.0, round(1.0 - float(distances[i]), 4))
            embedding  = list(embs[i]) if i < len(embs) else []

            entry = MemoryEntry(
                id=mem_id,
                content=documents[i],
                user_id=meta["user_id"],
                memory_type=meta["memory_type"],
                embedding=embedding,
                created_at=created_at,
                last_accessed=now,
                access_count=new_count,
                staleness_score=staleness,
            )
            retrieved.append(RetrievedMemory(
                entry=entry,
                similarity_score=sim_score,
                age_days=round(age_days, 2),
                freshness_label=_freshness_label(staleness),
            ))

        return retrieved

    def delete(self, memory_id: str) -> bool:
        """Remove a memory by its ID.

        Args:
            memory_id: UUID of the entry to remove.

        Returns:
            True if the entry existed and was deleted; False if not found.
        """
        existing = self._col.get(ids=[memory_id], include=["documents"])
        if not existing["ids"]:
            return False
        self._col.delete(ids=[memory_id])
        return True

    def update(self, memory_id: str, new_content: str) -> MemoryEntry | None:
        """Replace a memory's content and recompute its embedding.

        last_accessed is updated to now; access_count is preserved.

        Args:
            memory_id:   UUID of the entry to modify.
            new_content: Replacement text.

        Returns:
            The updated MemoryEntry, or None if memory_id was not found.
        """
        existing = self._col.get(ids=[memory_id], include=["metadatas"])
        if not existing["ids"]:
            return None

        now          = _utcnow()
        meta         = existing["metadatas"][0]
        new_embedding = self._embed_fn([new_content])[0]
        updated_meta  = {**meta, "last_accessed": now.isoformat()}

        self._col.update(
            ids=[memory_id],
            documents=[new_content],
            embeddings=[new_embedding],
            metadatas=[updated_meta],
        )

        staleness = _compute_staleness(
            _parse_dt(meta["last_accessed"]), now, self.stale_threshold_days
        )
        return MemoryEntry(
            id=memory_id,
            content=new_content,
            user_id=meta["user_id"],
            memory_type=meta["memory_type"],
            embedding=list(new_embedding),
            created_at=_parse_dt(meta["created_at"]),
            last_accessed=now,
            access_count=int(meta.get("access_count", 0)),
            staleness_score=staleness,
        )

    def get_stats(self, user_id: str) -> MemoryStats:
        """Return aggregate statistics for one user's memories.

        Staleness for each entry is computed dynamically from last_accessed,
        so stats reflect the current state even without a fresh retrieve() call.

        Args:
            user_id: Owner to compute statistics for.

        Returns:
            MemoryStats with totals, type breakdown, average age, and stale count.
        """
        all_data = self._col.get(
            where={"user_id": user_id},
            include=["metadatas"],
        )
        metas = all_data["metadatas"]
        if not metas:
            return MemoryStats(
                total_memories=0, by_type={}, avg_age_days=0.0, stale_count=0
            )

        now         = _utcnow()
        by_type:    dict[str, int] = {}
        age_sum     = 0.0
        stale_count = 0

        for meta in metas:
            mtype         = meta.get("memory_type", "unknown")
            by_type[mtype] = by_type.get(mtype, 0) + 1

            age_sum += (now - _parse_dt(meta["created_at"])).total_seconds() / 86400

            staleness = _compute_staleness(
                _parse_dt(meta["last_accessed"]), now, self.stale_threshold_days
            )
            if staleness >= _AGING_LIMIT:
                stale_count += 1

        return MemoryStats(
            total_memories=len(metas),
            by_type=by_type,
            avg_age_days=round(age_sum / len(metas), 3),
            stale_count=stale_count,
        )

    def clear_user(self, user_id: str) -> int:
        """Delete all memories belonging to user_id.

        Args:
            user_id: Owner whose memories to purge.

        Returns:
            Number of memories deleted.
        """
        resp = self._col.get(where={"user_id": user_id}, include=["documents"])
        ids  = resp["ids"]
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    # ------------------------------------------------------------------
    # Internal helpers for the __main__ demo
    # ------------------------------------------------------------------

    def _backdate_last_accessed(self, memory_id: str, days_ago: float) -> None:
        """Simulate an aged memory by backdating last_accessed (demo use only)."""
        existing = self._col.get(ids=[memory_id], include=["metadatas"])
        if not existing["ids"]:
            return
        past = (_utcnow() - timedelta(days=days_ago)).isoformat()
        self._col.update(
            ids=[memory_id],
            metadatas=[{**existing["metadatas"][0], "last_accessed": past}],
        )


# ── module-level helpers ──────────────────────────────────────────────────────

def _utcnow() -> datetime:
    """Return a naive UTC datetime (timezone-free for consistent storage)."""
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


def _parse_dt(iso_str: str) -> datetime:
    """Parse an ISO 8601 string to a naive UTC datetime."""
    try:
        return datetime.fromisoformat(iso_str.rstrip("Z")).replace(tzinfo=None)
    except (ValueError, AttributeError):
        return _utcnow()


def _compute_staleness(
    last_accessed:    datetime,
    now:              datetime,
    threshold_days:   float,
) -> float:
    """Linear staleness in [0.0, 1.0] based on idle time since last access."""
    idle_days = (now - last_accessed).total_seconds() / 86400
    return min(1.0, max(0.0, idle_days / threshold_days))


def _freshness_label(staleness: float) -> str:
    """Map staleness score to a human-readable freshness label."""
    if staleness < _FRESH_LIMIT:
        return "fresh"
    if staleness < _AGING_LIMIT:
        return "aging"
    return "stale"


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import shutil
    import sys
    import time

    # ── setup ─────────────────────────────────────────────────────────────────
    _PERSIST = str(Path(__file__).parent / "memory_store")
    _USER    = "user_alice_001"
    _STALE_DAYS = 30.0   # 30-day threshold

    # Clean start for repeatable demo
    if Path(_PERSIST).exists():
        shutil.rmtree(_PERSIST)

    store = MemoryStore(
        persist_dir=_PERSIST,
        stale_threshold_days=_STALE_DAYS,
    )

    # ── 10 memories to write ──────────────────────────────────────────────────
    _MEMORIES: list[tuple[str, str]] = [
        # --- episodic (personal events) ---
        ("episodic",
         "Ho completato il primo agente AI autonomo che ha risolto un bug di produzione "
         "senza intervento umano, usando tool_use per leggere i log e applicare il fix."),
        ("episodic",
         "Durante il code review di martedi', abbiamo scoperto un memory leak nel modulo "
         "di caching: gli oggetti TTL non venivano deallocati al timeout."),
        ("episodic",
         "L'incontro con il team di sicurezza ha evidenziato 3 vulnerabilita' critiche "
         "nel sistema di autenticazione: SQL injection, JWT senza scadenza, e CORS wildcard."),
        ("episodic",
         "Il deploy in produzione di venerd' ha causato un'interruzione di 12 minuti "
         "a causa di una migrazione di schema non retrocompatibile."),
        # --- semantic (general knowledge) ---
        ("semantic",
         "I transformer usano meccanismi di self-attention per pesare l'importanza "
         "di ogni token rispetto agli altri nel contesto, in parallelo anzi che in sequenza."),
        ("semantic",
         "Il pattern RAG (Retrieval-Augmented Generation) combina un retriever vettoriale "
         "con un LLM generativo, riducendo le allucinazioni ancorando le risposte a fatti recuperati."),
        ("semantic",
         "I sistemi multi-agente richiedono protocolli di comunicazione ben definiti: "
         "messaggi strutturati, timeout espliciti e idempotenza per gestire i retry senza duplicati."),
        # --- factual (specific facts) ---
        ("factual",
         "Il budget mensile per le API LLM e' di $500, con alert automatico al 70% "
         "e blocco al 95%. Il modello principale e' claude-sonnet-4-6."),
        ("factual",
         "Il team e' composto da 4 ingegneri: Alice (tech lead), Bob (backend), "
         "Carol (ML), Dave (DevOps). La cadenza degli sprint e' bisettimanale."),
        ("factual",
         "Il database PostgreSQL del progetto ha 2.3M di record nella tabella orders, "
         "con un indice B-tree su user_id e un indice GIN per la ricerca full-text."),
    ]

    sep = "=" * 72
    print(sep)
    print(f"  MemoryStore demo  |  user={_USER}  |  stale_threshold={int(_STALE_DAYS)}d")
    print(sep)
    print(f"\n  Writing {len(_MEMORIES)} memories...")
    print(f"  {'#':<3}  {'Type':<10}  {'ID prefix':<10}  Content (first 60 chars)")
    print(f"  {'-'*66}")

    entry_ids: list[str] = []
    t_write = time.perf_counter()
    for idx, (mtype, content) in enumerate(_MEMORIES, start=1):
        entry = store.write(content=content, user_id=_USER, memory_type=mtype)
        entry_ids.append(entry.id)
        print(f"  {idx:<3}  {mtype:<10}  {entry.id[:8]}...  {content[:60]}")
    write_ms = (time.perf_counter() - t_write) * 1000
    print(f"\n  Wrote {len(_MEMORIES)} entries in {write_ms:.0f}ms")

    # ── simulate aging: backdate entries unlikely to be retrieved ─────────────
    # Indices 3,8,9 ("deploy","team","PostgreSQL") -> 25d idle -> stale (0.83)
    # Indices 4,5   ("transformer","RAG")          -> 12d idle -> aging (0.40)
    # These entries are thematically distant from the 3 query topics below.
    for i in [3, 8, 9]:
        store._backdate_last_accessed(entry_ids[i], days_ago=25)
    for i in [4, 5]:
        store._backdate_last_accessed(entry_ids[i], days_ago=12)

    print(f"\n  Simulated aging:")
    print(f"    Entries #4,#9,#10 backdated 25 days  -> staleness=0.83 (stale)")
    print(f"    Entries #5,#6     backdated 12 days  -> staleness=0.40 (aging)")

    # ── PRE-QUERY stats ───────────────────────────────────────────────────────
    def _print_stats(label: str) -> None:
        st = store.get_stats(user_id=_USER)
        print(f"\n{sep}")
        print(f"  STATS {label}")
        print(f"  {'-'*50}")
        print(f"  Total memories   : {st.total_memories}")
        print(f"  By type          : {dict(sorted(st.by_type.items()))}")
        print(f"  Avg age (days)   : {st.avg_age_days:.3f}")
        print(f"  Stale count      : {st.stale_count}  "
              f"(staleness >= {_AGING_LIMIT} = {int(_AGING_LIMIT * _STALE_DAYS)}d idle)")

    _print_stats("(before retrieval)")

    # ── retrieval queries ─────────────────────────────────────────────────────
    _QUERIES = [
        ("vulnerabilita' di sicurezza nel sistema di autenticazione",   3),
        ("architettura transformer e meccanismo di attention nei LLM",  3),
        ("budget mensile API e configurazione del modello LLM",         3),
    ]

    print(f"\n{sep}")
    print("  RETRIEVAL QUERIES")
    print(sep)

    for q_idx, (query, k) in enumerate(_QUERIES, start=1):
        print(f"\n  Query {q_idx}: \"{query}\"")
        print(f"  top_k={k}")
        print(f"  {'Rank':<5}  {'Score':>6}  {'Fresh':>7}  {'Age(d)':>7}  "
              f"{'Type':<10}  {'AccessN':>7}  Content")
        print(f"  {'-'*70}")

        t0 = time.perf_counter()
        results = store.retrieve(query=query, user_id=_USER, top_k=k)
        latency_ms = (time.perf_counter() - t0) * 1000

        for rank, r in enumerate(results, start=1):
            print(
                f"  {rank:<5}  {r.similarity_score:>6.4f}  {r.freshness_label:>7}  "
                f"{r.age_days:>7.2f}  {r.entry.memory_type:<10}  "
                f"{r.entry.access_count:>7}  {r.entry.content[:52]}..."
            )
        print(f"\n  Latency: {latency_ms:.0f}ms  |  {len(results)} result(s) returned")

    # ── POST-QUERY stats (some stale entries may have been reset by retrieve) ─
    _print_stats("(after retrieval  -- retrieved entries reset to fresh)")

    # ── delete + update smoke-test ────────────────────────────────────────────
    print(f"\n{sep}")
    print("  DELETE / UPDATE SMOKE-TEST")
    print(f"  {'-'*50}")

    target_id = entry_ids[0]
    deleted   = store.delete(target_id)
    print(f"  delete({target_id[:8]}...): {deleted}")
    print(f"  Stats after delete -- total: {store.get_stats(_USER).total_memories}")

    update_target = entry_ids[1]
    updated = store.update(
        update_target,
        new_content=(
            "AGGIORNATO: Il memory leak nel modulo di caching e' stato risolto "
            "usando weak references e un cleanup thread ogni 60s."
        ),
    )
    if updated:
        print(f"  update({update_target[:8]}...): OK -- new content: {updated.content[:60]}...")

    print(f"\n  Persist dir: {_PERSIST}")
    print(f"  (ChromaDB index survives process restart)")
    print(sep)
