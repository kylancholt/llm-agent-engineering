"""
Semantic cache for agent pipelines.

Supports four cache types with two underlying lookup strategies:

  Exact lookup (SHA-256 key)
    tool_result  -- cache tool responses keyed by SHA-256(tool_name + JSON(input))
    embedding    -- cache pre-computed embedding vectors keyed by SHA-256(text)

  Semantic lookup (cosine similarity on stored embeddings)
    routing      -- cache routing decisions; similar queries reuse earlier decisions
    plan_step    -- cache planning steps; similar sub-goals reuse earlier results

Usage::

    cache = AgentSemanticCache(similarity_threshold=0.85, ttl_seconds=3600)

    # Tool result (exact)
    key = AgentSemanticCache.make_tool_key("web_search", {"query": "AAPL price"})
    hit = cache.get(key, "tool_result")
    if hit is None:
        result = call_tool(...)
        cache.set(key, result, "tool_result")

    # Routing decision (semantic)
    hit = cache.get(task_text, "routing")
    if hit is None:
        decision = router.route(task_text)
        cache.set(task_text, decision, "routing")

    cache.print_stats()
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

# ── project root & .env ───────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]


def _load_env(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (idempotent)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_load_env(_ROOT / ".env")

# ── constants ─────────────────────────────────────────────────────────────────
_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Which cache types use semantic (embedding) lookup vs SHA-256 exact lookup
_SEMANTIC_TYPES: frozenset[str] = frozenset({"routing", "plan_step"})
_EXACT_TYPES:    frozenset[str] = frozenset({"tool_result", "embedding"})

# Estimated USD saved per cache hit by type (based on typical call costs)
_COST_PER_HIT: dict[str, float] = {
    "tool_result": 0.00030,  # Sonnet call + tool execution
    "routing":     0.00008,  # Haiku classify call
    "embedding":   0.00000,  # local model, no API cost
    "plan_step":   0.00020,  # Sonnet planning call
}

# ── public dataclasses ────────────────────────────────────────────────────────

@dataclass
class CacheHit:
    """Returned by AgentSemanticCache.get() on a cache hit."""

    value: Any
    """The cached value."""

    similarity_score: float
    """1.0 for exact hits; cosine similarity in [0, 1] for semantic hits."""

    original_key: str
    """The key that was used when the entry was stored via set()."""

    age_seconds: float
    """Seconds elapsed since the entry was stored."""

    hit_type: str
    """'exact' for SHA-256 matches; 'semantic' for cosine-similarity matches."""


@dataclass
class CacheStats:
    """Per-cache-type statistics returned by AgentSemanticCache.get_stats()."""

    cache_type: str
    requests: int
    hits: int
    misses: int
    hit_rate: float          # hits / requests, or 0.0 if no requests
    bytes_saved: int         # cumulative approximate bytes of cached responses
    api_calls_saved: int     # equals hits (one API call avoided per hit)
    cost_saved_usd: float    # hits x _COST_PER_HIT[cache_type]


# ── internal dataclasses ──────────────────────────────────────────────────────

@dataclass
class _CacheEntry:
    """Internal storage record for one cached value."""

    key: str                            # original key passed to set()
    value: Any
    cache_type: str
    created_at: float                   # time.time() at insertion
    size_bytes: int                     # len(repr(value)) approximation
    embedding: list[float] | None = None  # normalised; only for semantic types
    access_count: int = 0


@dataclass
class _TypeStats:
    """Mutable stats accumulator, one per cache type."""

    requests: int = 0
    hits: int = 0
    bytes_saved: int = 0
    cost_saved_usd: float = 0.0


# ── cache ─────────────────────────────────────────────────────────────────────

class AgentSemanticCache:
    """
    Semantic cache for agent pipelines.

    Combines exact SHA-256 lookup (for deterministic keys like tool results and
    pre-computed embeddings) with cosine-similarity semantic lookup (for natural-
    language keys like routing queries and planning steps).

    Args:
        similarity_threshold: Minimum cosine similarity to count as a semantic
            cache hit.  Range [0, 1]; default 0.85.
        ttl_seconds: Maximum age of a cache entry in seconds.  Expired entries
            are evicted lazily on access.  Default 3600 (1 hour).
        max_entries: Hard cap on the total number of stored entries.  When
            reached, the oldest entry is evicted before each new insertion.
            Default 1000.
        embedding_model: sentence-transformers model ID used to encode semantic
            keys.  Default 'all-MiniLM-L6-v2'.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        ttl_seconds: float = 3600,
        max_entries: int = 1000,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries

        # storage_key -> entry (storage_key is SHA-256 hex for both strategies)
        self._entries: dict[str, _CacheEntry] = {}
        # cache_type -> [storage_key, ...]  (insertion order preserved)
        self._type_index: dict[str, list[str]] = defaultdict(list)
        # per-type stats
        self._stats: dict[str, _TypeStats] = defaultdict(_TypeStats)

        print(f"Loading sentence-transformers model '{embedding_model}'...")
        self._model = SentenceTransformer(embedding_model)

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def make_tool_key(tool_name: str, tool_input: dict) -> str:
        """
        Build a canonical cache key for a tool result.

        The key is the raw string ``tool_name + JSON(tool_input, sorted_keys)``.
        Pass this string to get() / set() with cache_type='tool_result'; the
        cache will SHA-256 it internally for storage.

        Args:
            tool_name: Name of the tool (e.g. 'web_search').
            tool_input: Dict of tool parameters.

        Returns:
            A stable string key suitable for exact cache lookup.
        """
        return tool_name + ":" + json.dumps(tool_input, sort_keys=True)

    # ── public API ────────────────────────────────────────────────────────────

    def get(self, key: str, cache_type: str) -> CacheHit | None:
        """
        Look up *key* in the cache.

        For exact types (tool_result, embedding): computes SHA-256 of the key
        and returns the stored value if found and not expired.

        For semantic types (routing, plan_step): encodes the key with the
        embedding model and returns the stored entry with the highest cosine
        similarity if it exceeds similarity_threshold.

        Args:
            key: Raw key string (tool key string or natural-language text).
            cache_type: One of 'tool_result', 'routing', 'embedding', 'plan_step'.

        Returns:
            CacheHit on success; None on miss or expiry.
        """
        self._stats[cache_type].requests += 1
        self._evict_expired()

        if cache_type in _SEMANTIC_TYPES:
            return self._semantic_get(key, cache_type)
        return self._exact_get(key, cache_type)

    def set(self, key: str, value: Any, cache_type: str) -> None:
        """
        Store *value* under *key* in the cache.

        For semantic types the key text is encoded with the embedding model and
        the normalised vector is stored alongside the value to enable future
        cosine-similarity lookups.  For exact types only the SHA-256 digest is
        stored.

        If a duplicate key already exists it is overwritten in-place.

        Args:
            key: Raw key string.
            value: Arbitrary serialisable value to cache.
            cache_type: One of 'tool_result', 'routing', 'embedding', 'plan_step'.
        """
        self._evict_expired()

        # Respect capacity limit (evict oldest first)
        while len(self._entries) >= self.max_entries:
            self._evict_oldest()

        storage_key = self._storage_key(key, cache_type)

        if cache_type in _SEMANTIC_TYPES:
            q_emb = self._embed(key)
            embedding: list[float] | None = q_emb.tolist()
        else:
            embedding = None

        entry = _CacheEntry(
            key=key,
            value=value,
            cache_type=cache_type,
            created_at=time.time(),
            size_bytes=len(repr(value)),
            embedding=embedding,
        )

        # Update type index (avoid duplicate storage_key entries)
        if storage_key not in self._entries:
            self._type_index[cache_type].append(storage_key)

        self._entries[storage_key] = entry

    def invalidate_by_type(self, cache_type: str) -> int:
        """
        Remove all cache entries of *cache_type*.

        Args:
            cache_type: One of 'tool_result', 'routing', 'embedding', 'plan_step'.

        Returns:
            Number of entries removed.
        """
        keys = list(self._type_index.get(cache_type, []))
        for k in keys:
            self._entries.pop(k, None)
        self._type_index[cache_type] = []
        return len(keys)

    def get_stats(self) -> dict[str, CacheStats]:
        """
        Return per-cache-type statistics.

        Returns:
            Dict mapping cache_type to CacheStats.  Types that have received
            no requests are omitted.
        """
        result: dict[str, CacheStats] = {}
        for ct, s in self._stats.items():
            misses = s.requests - s.hits
            rate = s.hits / s.requests if s.requests else 0.0
            result[ct] = CacheStats(
                cache_type=ct,
                requests=s.requests,
                hits=s.hits,
                misses=misses,
                hit_rate=rate,
                bytes_saved=s.bytes_saved,
                api_calls_saved=s.hits,
                cost_saved_usd=s.cost_saved_usd,
            )
        return result

    def print_stats(self) -> None:
        """Print a formatted per-type stats table to stdout."""
        stats = self.get_stats()
        if not stats:
            print("No cache activity recorded.")
            return

        # Sort by a canonical order so the table is stable
        _order = {"tool_result": 0, "routing": 1, "embedding": 2, "plan_step": 3}
        rows = sorted(stats.values(), key=lambda s: _order.get(s.cache_type, 9))

        totals = CacheStats(
            cache_type="TOTAL",
            requests=sum(s.requests for s in rows),
            hits=sum(s.hits for s in rows),
            misses=sum(s.misses for s in rows),
            hit_rate=0.0,
            bytes_saved=sum(s.bytes_saved for s in rows),
            api_calls_saved=sum(s.api_calls_saved for s in rows),
            cost_saved_usd=sum(s.cost_saved_usd for s in rows),
        )
        if totals.requests:
            totals.hit_rate = totals.hits / totals.requests

        sep = "-" * 73
        hdr = (
            f"{'Cache Type':<13}  "
            f"{'Requests':>8}  "
            f"{'Hits':>5}  "
            f"{'Misses':>6}  "
            f"{'Hit Rate':>8}  "
            f"{'$ Saved':>9}  "
            f"{'Calls Saved':>11}  "
            f"{'Bytes':>8}"
        )

        print()
        print("=== Semantic Cache Stats ===")
        print(hdr)
        print(sep)

        for s in rows:
            print(
                f"{s.cache_type:<13}  "
                f"{s.requests:>8}  "
                f"{s.hits:>5}  "
                f"{s.misses:>6}  "
                f"{s.hit_rate:>7.1%}  "
                f"${s.cost_saved_usd:>8.5f}  "
                f"{s.api_calls_saved:>11}  "
                f"{s.bytes_saved:>8,}"
            )

        print(sep)
        print(
            f"{'TOTAL':<13}  "
            f"{totals.requests:>8}  "
            f"{totals.hits:>5}  "
            f"{totals.misses:>6}  "
            f"{totals.hit_rate:>7.1%}  "
            f"${totals.cost_saved_usd:>8.5f}  "
            f"{totals.api_calls_saved:>11}  "
            f"{totals.bytes_saved:>8,}"
        )

        n_stored = len(self._entries)
        print(f"\nCache state: {n_stored} entries stored, {self.max_entries} max capacity")
        print()

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _storage_key(key: str, cache_type: str) -> str:
        """Compute the internal SHA-256 storage key."""
        raw = cache_type + "\x00" + key
        return hashlib.sha256(raw.encode()).hexdigest()

    def _embed(self, text: str) -> np.ndarray:
        """Encode *text* and return a normalised float32 ndarray."""
        return self._model.encode([text], normalize_embeddings=True)[0]

    def _exact_get(self, key: str, cache_type: str) -> CacheHit | None:
        """SHA-256 lookup; returns CacheHit or None."""
        storage_key = self._storage_key(key, cache_type)
        entry = self._entries.get(storage_key)

        if entry is None:
            return None

        age = time.time() - entry.created_at
        if age > self.ttl_seconds:
            self._entries.pop(storage_key, None)
            return None

        # Record hit
        entry.access_count += 1
        s = self._stats[cache_type]
        s.hits += 1
        s.bytes_saved += entry.size_bytes
        s.cost_saved_usd += _COST_PER_HIT.get(cache_type, 0.0)

        return CacheHit(
            value=entry.value,
            similarity_score=1.0,
            original_key=entry.key,
            age_seconds=age,
            hit_type="exact",
        )

    def _semantic_get(self, key: str, cache_type: str) -> CacheHit | None:
        """Cosine-similarity lookup among all entries of *cache_type*."""
        q_emb = self._embed(key)

        best_score = -1.0
        best_entry: _CacheEntry | None = None

        now = time.time()
        for storage_key in self._type_index.get(cache_type, []):
            entry = self._entries.get(storage_key)
            if entry is None or entry.embedding is None:
                continue
            if now - entry.created_at > self.ttl_seconds:
                continue

            e_emb = np.array(entry.embedding, dtype=np.float32)
            score = float(q_emb @ e_emb)   # cosine for normalised vectors

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is None or best_score < self.similarity_threshold:
            return None

        age = now - best_entry.created_at
        best_entry.access_count += 1
        s = self._stats[cache_type]
        s.hits += 1
        s.bytes_saved += best_entry.size_bytes
        s.cost_saved_usd += _COST_PER_HIT.get(cache_type, 0.0)

        return CacheHit(
            value=best_entry.value,
            similarity_score=best_score,
            original_key=best_entry.key,
            age_seconds=age,
            hit_type="semantic",
        )

    def _evict_expired(self) -> None:
        """Remove all entries whose TTL has elapsed."""
        now = time.time()
        expired = [
            sk for sk, e in self._entries.items()
            if now - e.created_at > self.ttl_seconds
        ]
        for sk in expired:
            ct = self._entries[sk].cache_type
            self._entries.pop(sk)
            if sk in self._type_index.get(ct, []):
                self._type_index[ct].remove(sk)

    def _evict_oldest(self) -> None:
        """Remove the single oldest entry (by created_at)."""
        if not self._entries:
            return
        oldest_key = min(self._entries, key=lambda k: self._entries[k].created_at)
        ct = self._entries[oldest_key].cache_type
        self._entries.pop(oldest_key)
        if oldest_key in self._type_index.get(ct, []):
            self._type_index[ct].remove(oldest_key)


# ── demo ──────────────────────────────────────────────────────────────────────

def _run_demo() -> None:
    """
    Simulate 50 agent cache requests with ~30% semantic/exact overlap.

    Structure:
      Requests  1-10  : 10 unique tool_result calls  (MISS -> store)
      Requests 11-20  : 10 unique routing queries     (MISS -> store)
      Requests 21-27  :  7 unique embedding requests  (MISS -> store)
      Requests 28-35  :  8 unique plan_step queries   (MISS -> store)
      Requests 36-40  :  5 tool_result exact repeats  (HIT exact)
      Requests 41-45  :  5 routing paraphrases        (HIT semantic)
      Requests 46-48  :  3 embedding exact repeats    (HIT exact)
      Requests 49-50  :  2 plan_step paraphrases      (HIT semantic)
    """

    # ── fake values for MISS cases ─────────────────────────────────────────────
    def _tool_val(name: str, n: int) -> dict:
        return {"tool": name, "result_id": n, "data": f"result-data-{n}"}

    def _route_val(model: str, reason: str) -> dict:
        return {"model": model, "reason": reason, "confidence": 0.92}

    def _emb_val(dim: int = 384) -> list:
        return [round(float(i) / dim, 6) for i in range(dim)]

    def _plan_val(step: str) -> dict:
        return {"step": step, "subtasks": [f"sub-{i}" for i in range(3)]}

    # ── 35 unique requests (all MISS) ──────────────────────────────────────────
    unique_requests: list[tuple[str, str, Any]] = [
        # (cache_type, key, value)
        # --- tool_result (1-10) ---
        ("tool_result", AgentSemanticCache.make_tool_key("web_search",   {"query": "AAPL stock price today"}),           _tool_val("web_search",   1)),
        ("tool_result", AgentSemanticCache.make_tool_key("web_search",   {"query": "Federal Reserve interest rates"}),   _tool_val("web_search",   2)),
        ("tool_result", AgentSemanticCache.make_tool_key("web_search",   {"query": "S&P 500 index performance"}),        _tool_val("web_search",   3)),
        ("tool_result", AgentSemanticCache.make_tool_key("calculator",   {"expr": "1.5 * 8765 + 200"}),                  _tool_val("calculator",   4)),
        ("tool_result", AgentSemanticCache.make_tool_key("calculator",   {"expr": "sum([1450, 2300, 890])"}),            _tool_val("calculator",   5)),
        ("tool_result", AgentSemanticCache.make_tool_key("db_query",     {"sql": "SELECT * FROM orders LIMIT 10"}),      _tool_val("db_query",     6)),
        ("tool_result", AgentSemanticCache.make_tool_key("db_query",     {"sql": "SELECT revenue FROM q1_sales"}),       _tool_val("db_query",     7)),
        ("tool_result", AgentSemanticCache.make_tool_key("file_reader",  {"path": "/data/report_2024.pdf"}),             _tool_val("file_reader",  8)),
        ("tool_result", AgentSemanticCache.make_tool_key("code_exec",    {"code": "print(2**10)"}),                      _tool_val("code_exec",    9)),
        ("tool_result", AgentSemanticCache.make_tool_key("web_search",   {"query": "OpenAI GPT-4 vs Claude comparison"}),_tool_val("web_search",  10)),
        # --- routing (11-20) ---
        ("routing", "Should I use web_search or db_query to fetch the current AAPL stock price?",            _route_val("haiku",  "simple lookup")),
        ("routing", "Route this: classify the sentiment of a customer review",                               _route_val("haiku",  "classification")),
        ("routing", "Which tool: calculator or code_exec for solving a matrix equation?",                    _route_val("haiku",  "tool selection")),
        ("routing", "Select model: reasoning about the Federal Reserve's next rate decision",                 _route_val("sonnet", "reasoning")),
        ("routing", "Route: generate a 500-line Python module with full test coverage",                      _route_val("sonnet", "code generation")),
        ("routing", "Should I summarise or extract key entities from the Q3 earnings report?",               _route_val("sonnet", "extraction")),
        ("routing", "Which model for formatting JSON output with strict schema validation?",                  _route_val("haiku",  "formatting")),
        ("routing", "Route: build a multi-step plan to migrate a PostgreSQL database to BigQuery",           _route_val("sonnet", "complex plan")),
        ("routing", "Tool selection: web search or internal knowledge for capital of France?",               _route_val("haiku",  "factual lookup")),
        ("routing", "Choose model: write a comprehensive market analysis for a fintech startup",             _route_val("sonnet", "summarization")),
        # --- embedding (21-27) ---
        ("embedding", "What is the capital of France?",                       _emb_val()),
        ("embedding", "Explain transformer attention mechanism",               _emb_val()),
        ("embedding", "How does compound interest work?",                      _emb_val()),
        ("embedding", "Describe the AAPL earnings Q3 2024",                   _emb_val()),
        ("embedding", "What are the benefits of microservices architecture?",  _emb_val()),
        ("embedding", "Summarise the Federal Reserve meeting minutes",         _emb_val()),
        ("embedding", "Python asyncio event loop explained",                   _emb_val()),
        # --- plan_step (28-35) ---
        ("plan_step", "Analyse quarterly earnings: retrieve financials, compute ratios, draft summary",      _plan_val("earnings analysis")),
        ("plan_step", "Migrate database: export schema, transform data, validate row counts, import",        _plan_val("db migration")),
        ("plan_step", "Write unit tests: enumerate functions, generate test cases, run coverage",             _plan_val("test generation")),
        ("plan_step", "Deploy to production: run CI, build image, push registry, roll out canary",           _plan_val("deployment")),
        ("plan_step", "Research competitor: gather web data, extract features, compare with product",        _plan_val("competitor research")),
        ("plan_step", "Onboard new user: create account, send welcome email, provision resources",           _plan_val("user onboarding")),
        ("plan_step", "Debug latency spike: collect traces, identify slow spans, propose optimisations",     _plan_val("latency debug")),
        ("plan_step", "Generate report: collect data, compute aggregates, format charts, export PDF",        _plan_val("report generation")),
    ]

    # ── 15 overlap requests (should produce cache HITs) ────────────────────────
    overlap_requests: list[tuple[str, str]] = [
        # (cache_type, key)  -- no value; we just call get()
        # tool_result exact repeats (5)
        ("tool_result", AgentSemanticCache.make_tool_key("web_search",   {"query": "AAPL stock price today"})),
        ("tool_result", AgentSemanticCache.make_tool_key("web_search",   {"query": "Federal Reserve interest rates"})),
        ("tool_result", AgentSemanticCache.make_tool_key("calculator",   {"expr": "1.5 * 8765 + 200"})),
        ("tool_result", AgentSemanticCache.make_tool_key("db_query",     {"sql": "SELECT * FROM orders LIMIT 10"})),
        ("tool_result", AgentSemanticCache.make_tool_key("code_exec",    {"code": "print(2**10)"})),
        # routing semantic paraphrases (5)  -- verified >0.85 with all-MiniLM-L6-v2
        ("routing", "Which tool should I pick, web_search or db_query, to look up AAPL live stock price?"),
        ("routing", "Route: classify the sentiment of a customer review, which model?"),
        ("routing", "Route: generate a Python module of 500 lines with full test coverage"),
        ("routing", "Model selection: formatting JSON output with strict schema validation?"),
        ("routing", "Choose model: write a market analysis report for a fintech startup"),
        # embedding exact repeats (3)
        ("embedding", "What is the capital of France?"),
        ("embedding", "Explain transformer attention mechanism"),
        ("embedding", "How does compound interest work?"),
        # plan_step semantic paraphrases (2)  -- verified >0.85 with all-MiniLM-L6-v2
        ("plan_step", "Quarterly earnings analysis: retrieve financials, compute financial ratios, draft summary"),
        ("plan_step", "Debug latency spike: collect traces, find slow spans, suggest optimisations"),
    ]

    # ── run simulation ─────────────────────────────────────────────────────────
    cache = AgentSemanticCache(
        similarity_threshold=0.85,
        ttl_seconds=3600,
        max_entries=1000,
    )

    all_requests: list[tuple[str, str, Any | None]] = (
        [(ct, k, v) for ct, k, v in unique_requests]
        + [(ct, k, None)         for ct, k    in overlap_requests]
    )
    n = len(all_requests)
    hits_total = 0

    print(f"\nSimulating {n} cache requests (target: ~30% hit rate)")
    print("-" * 60)

    for i, (cache_type, key, value) in enumerate(all_requests, 1):
        hit = cache.get(key, cache_type)

        if hit is not None:
            hits_total += 1
            tag = f"HIT {hit.hit_type:<8}  sim={hit.similarity_score:.2f}"
        else:
            tag = "MISS"
            if value is not None:
                cache.set(key, value, cache_type)

        # Short display key
        short_key = (key[:42] + "...") if len(key) > 45 else key
        print(
            f"  [{i:02d}/{n}]  {cache_type:<12}  {tag:<30}  "
            f"{short_key}"
        )

    print()

    print(f"Actual hit rate: {hits_total}/{n} = {hits_total/n:.1%}")
    cache.print_stats()


if __name__ == "__main__":
    _run_demo()
