"""
Hybrid retriever: dense (cosine) + sparse (BM25) with freshness re-ranking.

Pipeline for every retrieve() call:
  1. Encode the query with sentence-transformers.
  2. Compute cosine similarity against all memory embeddings (dense score).
  3. Score each memory with BM25 term-frequency matching (sparse score).
  4. Combine: hybrid = dense_weight * cosine + sparse_weight * bm25.
  5. If rerank=True, add a freshness bonus that decays linearly with age.
  6. Return the top-k memories by final score.

BM25 parameters: k1=1.5, b=0.75 (Okapi defaults).
Freshness bonus: +fresh_bonus * (1 - staleness_score) added after combining,
where fresh_bonus defaults to 0.05 (i.e., +0.05 for a memory accessed today,
zero for a fully stale memory).

benchmark() evaluates dense-only vs hybrid on a set of labelled queries and
reports Precision@k, Recall@k, NDCG@k, and relative improvement.
"""
from __future__ import annotations

import math
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

# ── project root on sys.path for absolute imports ────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from memory.longterm.memory_store import MemoryEntry, RetrievedMemory


# ── constants ─────────────────────────────────────────────────────────────────

_DEFAULT_MODEL        = "all-MiniLM-L6-v2"
_BM25_K1              = 1.5
_BM25_B               = 0.75
_FRESHNESS_WINDOW_DAYS = 30.0   # one window = 30 days; bonus applies per window


# ── public types ──────────────────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    """Aggregated evaluation results from HybridRetriever.benchmark().

    Attributes:
        precision_at_k:       Mean Precision@k for hybrid retrieval across all queries.
        recall_at_k:          Mean Recall@k for hybrid retrieval.
        ndcg:                 Mean NDCG@k for hybrid retrieval.
        dense_only_precision: Mean Precision@k using only cosine similarity.
        hybrid_precision:     Mean Precision@k using hybrid scoring (same as precision_at_k).
        improvement_pct:      Relative Precision@k gain of hybrid over dense-only (%).
    """
    precision_at_k:       float
    recall_at_k:          float
    ndcg:                 float
    dense_only_precision: float
    hybrid_precision:     float
    improvement_pct:      float


# ── BM25 (internal) ───────────────────────────────────────────────────────────

class _BM25:
    """Lightweight Okapi BM25 over a fixed corpus.

    Args:
        documents: Corpus strings (one per memory).
        k1:        Term-saturation parameter (default 1.5).
        b:         Length-normalisation parameter (default 0.75).
    """

    def __init__(
        self,
        documents: list[str],
        k1:        float = _BM25_K1,
        b:         float = _BM25_B,
    ) -> None:
        self.k1   = k1
        self.b    = b
        self._n   = len(documents)
        self._tok: list[list[str]] = [_tokenize(d) for d in documents]

        lens          = [len(t) for t in self._tok]
        self._avgdl   = sum(lens) / max(self._n, 1)

        self._df: dict[str, int] = {}
        for tokens in self._tok:
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1

    def get_scores(self, query: str) -> list[float]:
        """Return one BM25 score per document, normalised to [0, 1] by the max.

        Args:
            query: Query string to score against the corpus.

        Returns:
            List of floats in [0, 1], one per document.
        """
        q_tokens = _tokenize(query)
        raw      = [self._score(q_tokens, i) for i in range(self._n)]
        max_s    = max(raw) if raw else 0.0
        if max_s > 0:
            return [s / max_s for s in raw]
        return raw

    def _score(self, q_tokens: list[str], doc_idx: int) -> float:
        tokens  = self._tok[doc_idx]
        doc_len = len(tokens)
        tf_map: dict[str, int] = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        score = 0.0
        for term in q_tokens:
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
            tf  = tf_map.get(term, 0)
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1.0 - self.b + self.b * doc_len / max(self._avgdl, 1.0))
            )
            score += idf * tf_norm
        return score


# ── hybrid retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """Hybrid dense + sparse retriever with optional freshness re-ranking.

    Usage::

        retriever = HybridRetriever(dense_weight=0.7, sparse_weight=0.3, rerank=True)
        results = retriever.retrieve(query="rate limiting Redis", memories=mem_list, top_k=5)
        for r in results:
            print(f"{r.similarity_score:.4f}  {r.freshness_label}  {r.entry.content[:60]}")

    Args:
        dense_weight:    Cosine similarity coefficient in the hybrid formula.
        sparse_weight:   BM25 coefficient in the hybrid formula.
        rerank:          Apply freshness bonus after combining scores.
        embedding_model: sentence-transformers model ID for query encoding.
        freshness_bonus: Maximum bonus added for a fully-fresh memory (staleness=0).
    """

    def __init__(
        self,
        dense_weight:    float = 0.7,
        sparse_weight:   float = 0.3,
        rerank:          bool  = True,
        embedding_model: str   = _DEFAULT_MODEL,
        freshness_bonus: float = 0.05,
    ) -> None:
        self.dense_weight    = dense_weight
        self.sparse_weight   = sparse_weight
        self.rerank          = rerank
        self.freshness_bonus = freshness_bonus
        self._model          = SentenceTransformer(embedding_model)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query:    str,
        memories: list[MemoryEntry],
        top_k:    int = 5,
    ) -> list[RetrievedMemory]:
        """Rank memories by hybrid score and return the top_k.

        The hybrid score is::

            score = dense_weight * cosine(q, m) + sparse_weight * bm25(q, m)

        If ``rerank=True``, a freshness bonus is added::

            score += freshness_bonus * (1 - staleness_score)

        Missing embeddings (empty list) are computed in batch before scoring.

        Args:
            query:    Natural-language query string.
            memories: Candidate MemoryEntry objects (any length).
            top_k:    Maximum number of results to return.

        Returns:
            List of RetrievedMemory sorted by descending hybrid score.
        """
        if not memories:
            return []

        q_emb      = self._encode_query(query)
        mem_embs   = self._ensure_embeddings(memories)
        dense_sc   = _dense_scores(q_emb, mem_embs)
        sparse_sc  = _BM25([m.content for m in memories]).get_scores(query)

        hybrid: list[float] = [
            self.dense_weight * d + self.sparse_weight * s
            for d, s in zip(dense_sc, sparse_sc)
        ]

        if self.rerank:
            now = _utcnow()
            for i, mem in enumerate(memories):
                age_days     = (now - mem.created_at).total_seconds() / 86400
                freshness    = max(0.0, 1.0 - age_days / _FRESHNESS_WINDOW_DAYS)
                hybrid[i]   += freshness * self.freshness_bonus

        return self._build_results(memories, hybrid, top_k)

    def dense_only_retrieve(
        self,
        query:    str,
        memories: list[MemoryEntry],
        top_k:    int = 5,
    ) -> list[RetrievedMemory]:
        """Retrieve using cosine similarity only (baseline for comparison).

        Args:
            query:    Natural-language query string.
            memories: Candidate MemoryEntry objects.
            top_k:    Maximum results.

        Returns:
            List of RetrievedMemory sorted by cosine similarity.
        """
        if not memories:
            return []

        q_emb    = self._encode_query(query)
        mem_embs = self._ensure_embeddings(memories)
        scores   = _dense_scores(q_emb, mem_embs)
        return self._build_results(memories, scores, top_k)

    def benchmark(
        self,
        queries:      list[str],
        memories:     list[MemoryEntry],
        ground_truth: dict[str, set[str]],
        k:            int = 5,
    ) -> RetrievalMetrics:
        """Compare dense-only vs hybrid retrieval across a labelled query set.

        For each query, both ``dense_only_retrieve`` and ``retrieve`` are called
        with the full corpus. Per-query Precision@k, Recall@k, and NDCG@k are
        averaged and returned as aggregated RetrievalMetrics.

        Args:
            queries:      List of query strings to evaluate.
            memories:     Full corpus of MemoryEntry objects.
            ground_truth: Maps each query string to the set of relevant memory IDs.
            k:            Cut-off depth for all metrics (default 5).

        Returns:
            RetrievalMetrics with mean scores and relative improvement.
        """
        dense_p, hybrid_p = [], []
        hybrid_r, hybrid_n = [], []

        for query in queries:
            relevant = ground_truth.get(query, set())
            if not relevant:
                continue

            d_ids = [r.entry.id for r in self.dense_only_retrieve(query, memories, k)]
            h_ids = [r.entry.id for r in self.retrieve(query, memories, k)]

            dense_p.append(_precision(d_ids, relevant, k))
            hybrid_p.append(_precision(h_ids, relevant, k))
            hybrid_r.append(_recall(h_ids, relevant, k))
            hybrid_n.append(_ndcg(h_ids, relevant, k))

        def _avg(lst: list[float]) -> float:
            return round(sum(lst) / max(len(lst), 1), 4)

        dp  = _avg(dense_p)
        hp  = _avg(hybrid_p)
        imp = round((hp - dp) / max(dp, 1e-9) * 100.0, 2)

        return RetrievalMetrics(
            precision_at_k=       hp,
            recall_at_k=          _avg(hybrid_r),
            ndcg=                  _avg(hybrid_n),
            dense_only_precision=  dp,
            hybrid_precision=      hp,
            improvement_pct=       imp,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_query(self, query: str) -> np.ndarray:
        return self._model.encode([query], normalize_embeddings=True)[0]

    def _ensure_embeddings(self, memories: list[MemoryEntry]) -> list[np.ndarray]:
        """Return an ndarray per memory; batch-compute any missing embeddings."""
        embs: list[np.ndarray | None] = []
        missing_idx: list[int]         = []

        for i, mem in enumerate(memories):
            if mem.embedding:
                embs.append(np.array(mem.embedding, dtype=np.float32))
            else:
                embs.append(None)
                missing_idx.append(i)

        if missing_idx:
            vecs = self._model.encode(
                [memories[i].content for i in missing_idx],
                normalize_embeddings=True,
            )
            for j, idx in enumerate(missing_idx):
                embs[idx]               = vecs[j]
                memories[idx].embedding = vecs[j].tolist()

        return embs  # type: ignore[return-value]

    @staticmethod
    def _build_results(
        memories: list[MemoryEntry],
        scores:   list[float],
        top_k:    int,
    ) -> list[RetrievedMemory]:
        now = _utcnow()
        ranked = sorted(
            range(len(memories)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        result: list[RetrievedMemory] = []
        for idx in ranked:
            mem      = memories[idx]
            age_days = (now - mem.created_at).total_seconds() / 86400
            result.append(RetrievedMemory(
                entry=mem,
                similarity_score=round(scores[idx], 4),
                age_days=round(age_days, 2),
                freshness_label=_freshness_label(mem.staleness_score),
            ))
        return result


# ── module helpers ────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _dense_scores(q_emb: np.ndarray, mem_embs: list[np.ndarray]) -> list[float]:
    """Cosine similarity clipped to [0, 1] (negative = orthogonal/opposite)."""
    mat = np.stack(mem_embs)               # (N, D)
    raw = mat @ q_emb                      # (N,) dot product = cosine for normalised vecs
    return [max(0.0, float(s)) for s in raw]


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


def _freshness_label(staleness: float) -> str:
    if staleness < 0.33:
        return "fresh"
    if staleness < 0.67:
        return "aging"
    return "stale"


def _precision(retrieved: list[str], relevant: set[str], k: int) -> float:
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / max(k, 1)


def _recall(retrieved: list[str], relevant: set[str], k: int) -> float:
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / max(len(relevant), 1)


def _ndcg(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg  = sum(
        1.0 / math.log2(i + 2)
        for i, doc_id in enumerate(retrieved[:k])
        if doc_id in relevant
    )
    ideal_n = min(k, len(relevant))
    idcg    = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return round(dcg / max(idcg, 1e-9), 4)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    # ── corpus: 10 topics x 5 memories = 50 entries ───────────────────────────
    # Memories are split across two age groups so the freshness bonus is visible:
    #   Topics 1-5 (indices 0-24):  created TODAY  -> staleness=0 -> bonus=+0.05
    #   Topics 6-10 (indices 25-49): created 22d ago -> staleness=0.73 -> bonus=+0.01

    _NOW  = _utcnow()
    _OLD  = _NOW - timedelta(days=22)

    _CORPUS: list[tuple[str, str, datetime]] = [  # (memory_type, content, created_at)
        # ── Topic 1: Python performance ──────────────────────────────────────
        ("semantic",
         "Python list comprehensions are 2x faster than for loops for simple "
         "operations because they execute entirely at the C level inside CPython.",
         _NOW),
        ("semantic",
         "NumPy vectorised operations replace Python loops and reduce computation "
         "time by 10-100x by dispatching to optimised BLAS/LAPACK routines.",
         _NOW),
        ("factual",
         "The multiprocessing module creates separate OS processes that each have "
         "their own GIL, enabling true CPU parallelism in Python.",
         _NOW),
        ("semantic",
         "Python @lru_cache and @functools.cache decorators memoize function results, "
         "eliminating redundant recomputation in recursive or repeated calls.",
         _NOW),
        ("episodic",
         "Profiling with cProfile revealed that 90% of the performance bottleneck "
         "came from a single O(n^2) loop inside the data-preprocessing pipeline.",
         _NOW),

        # ── Topic 2: ML training ──────────────────────────────────────────────
        ("semantic",
         "Gradient descent with momentum accumulates a velocity vector in the "
         "direction of persistent gradients, accelerating convergence in flat regions.",
         _NOW),
        ("semantic",
         "Learning rate warmup starts training with a very small learning rate and "
         "increases it gradually, preventing divergence from large random initial gradients.",
         _NOW),
        ("semantic",
         "Batch normalisation normalises each layer's inputs to zero mean and unit "
         "variance, stabilising training and allowing higher learning rates.",
         _NOW),
        ("semantic",
         "Dropout randomly zeroes a fraction of activations during training, acting "
         "as an implicit ensemble of sub-networks to reduce overfitting.",
         _NOW),
        ("factual",
         "Mixed-precision training stores weights in FP16 and accumulates gradients "
         "in FP32, halving GPU memory usage and speeding up matrix multiplications by 2-3x.",
         _NOW),

        # ── Topic 3: Database indexing ────────────────────────────────────────
        ("semantic",
         "B-tree indexes efficiently support range queries and ORDER BY clauses "
         "because leaf nodes store sorted keys with pointers to adjacent siblings.",
         _NOW),
        ("semantic",
         "Hash indexes provide O(1) lookup for equality predicates but cannot be used "
         "for range scans or partial-key searches in PostgreSQL.",
         _NOW),
        ("factual",
         "Partial indexes in PostgreSQL add a WHERE clause to the index definition, "
         "indexing only qualifying rows and shrinking the index by orders of magnitude.",
         _NOW),
        ("semantic",
         "A covering index includes every column referenced by a query so the planner "
         "can satisfy the query with an index-only scan, skipping the heap entirely.",
         _NOW),
        ("episodic",
         "VACUUM on the orders table reclaimed 4 GB after heavy UPDATE workloads "
         "left millions of dead tuples inflating the index and causing bloat.",
         _NOW),

        # ── Topic 4: Microservices ────────────────────────────────────────────
        ("semantic",
         "The strangler fig pattern migrates a monolith incrementally by routing "
         "individual endpoints to new microservices while the monolith handles the rest.",
         _NOW),
        ("semantic",
         "Service discovery allows microservices to locate each other dynamically "
         "through Consul or etcd, eliminating hardcoded service URLs from config.",
         _NOW),
        ("semantic",
         "API gateways centralise cross-cutting concerns such as authentication, "
         "rate limiting, and request routing at the cluster edge.",
         _NOW),
        ("semantic",
         "The saga pattern coordinates long-running distributed transactions via a "
         "sequence of local transactions with compensating rollback steps.",
         _NOW),
        ("semantic",
         "Circuit breakers short-circuit calls to a failing downstream service, "
         "returning cached responses or errors immediately to prevent cascading failures.",
         _NOW),

        # ── Topic 5: Security ─────────────────────────────────────────────────
        ("semantic",
         "JWT tokens are self-contained signed claims; without a revocation list or "
         "short TTL, a stolen token remains valid until its embedded expiry.",
         _NOW),
        ("semantic",
         "OAuth 2.0 PKCE includes a code_verifier in the token exchange, preventing "
         "authorisation-code interception in public clients such as mobile apps.",
         _NOW),
        ("factual",
         "bcrypt with a work factor of 12 or higher is recommended for password "
         "hashing; each factor increment doubles hashing time, resisting GPU brute force.",
         _NOW),
        ("semantic",
         "Parameterised queries separate SQL structure from user-supplied data, "
         "making SQL injection structurally impossible regardless of input content.",
         _NOW),
        ("factual",
         "A CORS wildcard origin (Access-Control-Allow-Origin: *) combined with "
         "credentials allows any website to perform authenticated cross-origin API requests.",
         _NOW),

        # ── Topic 6: Rate limiting ────────────────────────────────────────────
        ("semantic",
         "The token bucket algorithm refills tokens at a fixed rate and allows bursts "
         "up to the bucket capacity, smoothing traffic without rejecting short spikes.",
         _OLD),
        ("semantic",
         "Sliding window rate limiting tracks request timestamps in a rolling window, "
         "preventing clients from exploiting fixed-window boundary resets.",
         _OLD),
        ("factual",
         "Redis sorted sets implement distributed rate limiting by storing each request "
         "timestamp as a score and removing entries older than the window with ZREMRANGEBYSCORE.",
         _OLD),
        ("semantic",
         "Exponential backoff with full jitter prevents thundering herd by spacing out "
         "retries across many clients that received a 429 Too Many Requests response.",
         _OLD),
        ("factual",
         "Rate limit response headers X-RateLimit-Limit, X-RateLimit-Remaining, and "
         "Retry-After let clients adapt their request rate without polling.",
         _OLD),

        # ── Topic 7: CI/CD ────────────────────────────────────────────────────
        ("semantic",
         "Feature flags decouple code deployment from feature activation, enabling "
         "dark launches and A/B tests without separate code branches.",
         _OLD),
        ("semantic",
         "Blue-green deployment runs two identical environments; traffic is switched "
         "to the new environment only after smoke tests pass, enabling instant rollback.",
         _OLD),
        ("semantic",
         "Canary deployment routes a small percentage of live traffic to the new "
         "version, limiting the blast radius of a bad release before full roll-out.",
         _OLD),
        ("semantic",
         "Trunk-based development with short-lived feature branches keeps the main "
         "branch always releasable and reduces merge conflicts from long divergence.",
         _OLD),
        ("episodic",
         "Caching pip wheels in the CI layer reduced the average build time from "
         "11 minutes to 90 seconds across 40 daily pipeline runs.",
         _OLD),

        # ── Topic 8: Kubernetes ───────────────────────────────────────────────
        ("semantic",
         "Pod Disruption Budgets define a minimum available replica count, "
         "preventing node drains from taking a deployment below safe capacity.",
         _OLD),
        ("semantic",
         "Horizontal Pod Autoscaler adjusts replica count based on CPU utilisation, "
         "memory, or custom Prometheus metrics exported by the application.",
         _OLD),
        ("semantic",
         "Node affinity rules schedule pods onto nodes with specific labels such as "
         "GPU type or availability zone, using requiredDuringSchedulingIgnoredDuringExecution.",
         _OLD),
        ("semantic",
         "Init containers run to completion before app containers start, making them "
         "suitable for database migrations, secret fetching, and config validation.",
         _OLD),
        ("factual",
         "PersistentVolumeClaims request durable storage that survives pod restarts; "
         "the actual PersistentVolume is provisioned by the storage class controller.",
         _OLD),

        # ── Topic 9: Observability ────────────────────────────────────────────
        ("semantic",
         "The three pillars of observability are metrics (aggregated numbers), logs "
         "(event records), and traces (request journeys across service boundaries).",
         _OLD),
        ("factual",
         "P99 latency is the 99th-percentile response time; it reveals worst-case "
         "user experience that mean and median latency statistics systematically hide.",
         _OLD),
        ("semantic",
         "Prometheus uses a pull model, scraping metrics from /metrics HTTP endpoints "
         "on a configured interval and storing them in its local time-series database.",
         _OLD),
        ("factual",
         "Structured JSON logs let aggregation systems like Elasticsearch or Loki "
         "parse fields without regex, enabling efficient filtering and alerting.",
         _OLD),
        ("semantic",
         "OpenTelemetry propagates trace context via the traceparent HTTP header "
         "across service boundaries, allowing Jaeger or Tempo to reconstruct full traces.",
         _OLD),

        # ── Topic 10: Distributed consistency ────────────────────────────────
        ("semantic",
         "Eventual consistency guarantees all replicas converge to the same value "
         "after writes stop, without specifying how long convergence takes.",
         _OLD),
        ("semantic",
         "Two-phase commit achieves atomic distributed transactions but blocks "
         "all participants if the coordinator crashes between prepare and commit.",
         _OLD),
        ("semantic",
         "Lamport timestamps assign monotonically increasing integers to events, "
         "providing causal ordering without requiring synchronised physical clocks.",
         _OLD),
        ("factual",
         "Read-your-writes consistency ensures a client always observes its own "
         "most recent update, even if reads and writes hit different replicas.",
         _OLD),
        ("semantic",
         "CRDT data structures define a merge operation that is commutative, "
         "associative, and idempotent, enabling conflict-free replica merging.",
         _OLD),
    ]

    assert len(_CORPUS) == 50, f"Expected 50 memories, got {len(_CORPUS)}"

    # ── build MemoryEntry list ────────────────────────────────────────────────
    print("Building 50 synthetic MemoryEntry objects (embeddings computed in batch)...")
    _TOPIC_SIZE = 5
    memories: list[MemoryEntry] = []
    topic_ids: list[list[str]] = [[] for _ in range(10)]  # IDs per topic

    for i, (mtype, content, created_at) in enumerate(_CORPUS):
        topic_idx = i // _TOPIC_SIZE
        mem_id    = str(uuid.uuid4())
        age_days  = (_NOW - created_at).total_seconds() / 86400
        staleness = min(1.0, max(0.0, age_days / 30.0))
        entry = MemoryEntry(
            id=              mem_id,
            content=         content,
            user_id=         "benchmark_user",
            memory_type=     mtype,
            embedding=       [],      # computed lazily by retriever
            created_at=      created_at,
            last_accessed=   created_at,
            access_count=    0,
            staleness_score= staleness,
        )
        memories.append(entry)
        topic_ids[topic_idx].append(mem_id)

    # ── 10 queries with known ground truth (1 per topic) ──────────────────────
    _QUERIES_GT: list[tuple[str, int]] = [   # (query, topic_index 0-based)
        ("Python lru_cache profiling performance speedup",                     0),
        ("learning rate warmup dropout overfitting neural network",            1),
        ("covering index only scan PostgreSQL VACUUM bloat",                   2),
        ("strangler fig monolith microservices saga circuit breaker",          3),
        ("bcrypt password hashing SQL injection parameterised query",          4),
        ("token bucket Redis sorted sets rate limiting sliding window",        5),
        ("canary blue-green deployment feature flag CI pipeline",              6),
        ("pod disruption budget horizontal autoscaler Kubernetes node affinity", 7),
        ("P99 latency Prometheus OpenTelemetry structured log",                8),
        ("CRDT eventual consistency Lamport timestamp two-phase commit",       9),
    ]

    queries      = [q for q, _ in _QUERIES_GT]
    ground_truth = {q: set(topic_ids[t]) for q, t in _QUERIES_GT}

    # ── run benchmark ──────────────────────────────────────────────────────────
    sep = "=" * 80
    K   = 5

    retriever = HybridRetriever(
        dense_weight=0.7, sparse_weight=0.3, rerank=True, freshness_bonus=0.05
    )

    print(f"\n{sep}")
    print(
        f"  HybridRetriever benchmark  |  corpus=50  |  queries=10  |  k={K}  "
        f"|  dense={retriever.dense_weight}  sparse={retriever.sparse_weight}"
    )
    print(sep)

    # Per-query comparison table
    print(
        f"\n  {'#':<3}  {'Query (first 45 chars)':<45}  "
        f"{'Dense P@5':>9}  {'Hybr P@5':>9}  {'Delta':>7}  "
        f"{'NDCG@5':>7}  {'R@5':>5}"
    )
    print(f"  {'-'*76}")

    per_dense_p : list[float] = []
    per_hybrid_p: list[float] = []
    per_ndcg    : list[float] = []
    per_recall  : list[float] = []

    t0 = time.perf_counter()

    for q_idx, (query, topic_idx) in enumerate(_QUERIES_GT, start=1):
        relevant = ground_truth[query]

        d_ids = [r.entry.id for r in retriever.dense_only_retrieve(query, memories, K)]
        h_ids = [r.entry.id for r in retriever.retrieve(query, memories, K)]

        dp = _precision(d_ids, relevant, K)
        hp = _precision(h_ids, relevant, K)
        nr = _ndcg(h_ids, relevant, K)
        rc = _recall(h_ids, relevant, K)

        per_dense_p.append(dp)
        per_hybrid_p.append(hp)
        per_ndcg.append(nr)
        per_recall.append(rc)

        delta = hp - dp
        delta_str = f"{delta:+.2f}"

        print(
            f"  {q_idx:<3}  {query[:45]:<45}  "
            f"{dp:>9.2f}  {hp:>9.2f}  {delta_str:>7}  "
            f"{nr:>7.4f}  {rc:>5.2f}"
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Aggregate row
    def _mean(lst: list[float]) -> float:
        return sum(lst) / max(len(lst), 1)

    avg_dp = _mean(per_dense_p)
    avg_hp = _mean(per_hybrid_p)
    avg_nr = _mean(per_ndcg)
    avg_rc = _mean(per_recall)
    imp    = (avg_hp - avg_dp) / max(avg_dp, 1e-9) * 100

    print(f"  {'-'*76}")
    print(
        f"  {'MEAN':<3}  {'(10 queries)':<45}  "
        f"{avg_dp:>9.2f}  {avg_hp:>9.2f}  {imp:>+6.1f}%  "
        f"{avg_nr:>7.4f}  {avg_rc:>5.2f}"
    )

    # ── aggregate metrics ─────────────────────────────────────────────────────
    metrics = retriever.benchmark(queries, memories, ground_truth, k=K)

    print(f"\n{sep}")
    print("  AGGREGATE METRICS")
    print(f"  {'-'*55}")
    print(f"  Dense-only  Precision@{K} : {metrics.dense_only_precision:.4f}")
    print(f"  Hybrid      Precision@{K} : {metrics.hybrid_precision:.4f}")
    print(f"  Hybrid      Recall@{K}    : {metrics.recall_at_k:.4f}")
    print(f"  Hybrid      NDCG@{K}      : {metrics.ndcg:.4f}")
    print(f"  Improvement             : {metrics.improvement_pct:+.2f}%")
    print(f"  Total latency           : {elapsed_ms:.0f}ms  ({len(queries)} queries x 2 methods)")
    print(f"\n  Freshness: topics 1-5 (recent, staleness=0.00, bonus=+{retriever.freshness_bonus:.2f})")
    stale_s = min(1.0, 22.0 / 30.0)
    print(f"             topics 6-10 (22d old, staleness={stale_s:.2f}, bonus=+{(1-stale_s)*retriever.freshness_bonus:.3f})")
    print(sep)
