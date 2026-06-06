"""
Aggregate results from multiple agents into one coherent answer.

Given a list of AgentResults answering the same task, the ResultAggregator
produces a single response using one of three strategies:

  MERGE             -- an LLM (Haiku, cheap) combines the results into one
                       structured, de-duplicated document.
  CONSENSUS         -- no LLM: cluster the claims, keep points of agreement,
                       and resolve each disagreement in favour of the
                       higher-confidence agent.
  SUPERVISOR_DECIDES -- a supervisor LLM (Sonnet) adjudicates all results and
                       detected conflicts into a single authoritative answer.

Conflict detection (used by every strategy for reporting) is embedding-based:
claims from different agents that are on the same topic (share a salient
keyword) but are semantically distant (cosine similarity below
``conflict_threshold``, default 0.3) — or are on-topic but lexically opposed
(antonym/negation cue) — are flagged as contradictions. Embeddings use
sentence-transformers (all-MiniLM-L6-v2, the project default) when available,
falling back to a deterministic bag-of-words vectoriser so detection runs
offline.

  NOTE: pure embedding similarity does not reliably separate "same topic,
  opposite stance" (which can score high) from "different topic" (which scores
  low). The threshold heuristic plus the antonym cue is a pragmatic detector;
  for production-grade contradiction detection use an NLI model or an LLM judge
  (which is effectively what SUPERVISOR_DECIDES provides).

Requires ANTHROPIC_API_KEY (in the environment or the project-root .env) for
the MERGE and SUPERVISOR_DECIDES strategies only. CONSENSUS, conflict
detection, and simple_merge need no API key.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import anthropic

# ── project root: ch08_multi_agent/.. = root ────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ch01_why_agents_break.cost_tracker import AgentCostTracker
from ch08_multi_agent.supervisor import SubtaskResult


# ── .env loader (mirrors the rest of the project) ───────────────────────────────

def _load_env(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (idempotent)."""
    import os
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_load_env(_ROOT / ".env")


# ── defaults ────────────────────────────────────────────────────────────────────

_DEFAULT_MERGE_MODEL      = "claude-haiku-4-5"
_DEFAULT_SUPERVISOR_MODEL = "claude-sonnet-4-6"
_DEFAULT_EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

_DEFAULT_CONFLICT_THRESHOLD  = 0.30   # cosine below this on a same-topic pair => conflict
_DEFAULT_AGREEMENT_THRESHOLD = 0.60   # cosine at/above this => same point (agreement)
_MERGE_MAX_TOKENS      = 2048
_SUPERVISOR_MAX_TOKENS = 2048
_MIN_CLAIM_CHARS       = 15
_MAX_REPORTED_CONFLICTS = 12

_STOPWORDS: frozenset[str] = frozenset({
    "with", "that", "this", "from", "have", "will", "your", "their", "them",
    "they", "into", "also", "more", "most", "some", "when", "what", "which",
    "while", "were", "been", "being", "does", "only", "such", "than", "then",
    "very", "much", "many", "like", "well", "both", "each", "about", "across",
    "over", "under", "between", "however", "therefore", "because",
})

# Pairs of words that signal a contradiction when split across two same-topic claims.
_ANTONYMS: tuple[frozenset[str], ...] = (
    frozenset({"free", "paid"}),
    frozenset({"free", "subscription"}),
    frozenset({"free", "premium"}),
    frozenset({"offline", "online"}),
    frozenset({"offline", "internet"}),
    frozenset({"local", "cloud"}),
    frozenset({"supported", "unsupported"}),
    frozenset({"open", "proprietary"}),
)


def _cost_model(model: str) -> str:
    """Map any Claude model ID to the nearest AgentCostTracker pricing tier."""
    lc = model.lower()
    if "haiku" in lc:
        return "claude-haiku-4-5"
    if "opus" in lc:
        return "claude-opus-4-7"
    return "claude-sonnet-4-6"


def _price(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost of one (input, output) token pair for a model."""
    tracker = AgentCostTracker(budget_usd=1e9, model=_cost_model(model))
    return tracker.record_turn(input_tokens, output_tokens, turn_id=0).turn_cost_usd


# ── public types ──────────────────────────────────────────────────────────────

class AggregationStrategy(str, Enum):
    """Supported aggregation strategies."""
    MERGE              = "MERGE"
    CONSENSUS          = "CONSENSUS"
    SUPERVISOR_DECIDES = "SUPERVISOR_DECIDES"


@dataclass
class AgentResult:
    """One agent's answer to the shared task — the unit aggregated.

    Attributes:
        agent: Name of the agent that produced this result.
        content: The agent's textual answer.
        confidence: Self-/caller-assigned confidence in [0, 1]; used by
            CONSENSUS to break ties between contradictory claims.
        cost_usd: Cost the agent incurred producing this result (informational;
            not added to the aggregation cost).
    """
    agent:      str
    content:    str
    confidence: float = 0.8
    cost_usd:   float = 0.0

    @classmethod
    def from_subtask_result(
        cls, result: SubtaskResult, confidence: float = 0.8
    ) -> "AgentResult":
        """Adapt a supervisor.SubtaskResult into an AgentResult."""
        return cls(
            agent=result.agent,
            content=result.output,
            confidence=confidence,
            cost_usd=result.cost_usd,
        )


@dataclass
class Conflict:
    """A detected contradiction between two agents' claims on the same topic.

    Attributes:
        topic: Shared salient keyword(s) identifying the contested topic.
        claim_a: The first claim.
        agent_a: Agent that made claim_a.
        claim_b: The second, contradicting claim.
        agent_b: Agent that made claim_b.
        similarity: Cosine similarity between the two claims' embeddings.
    """
    topic:      str
    claim_a:    str
    agent_a:    str
    claim_b:    str
    agent_b:    str
    similarity: float


@dataclass
class ConflictResolution:
    """How a single conflict was resolved.

    Attributes:
        topic: The contested topic.
        chosen_claim: The claim that prevailed ("" when resolution is delegated
            to an LLM and the exact wording isn't extracted).
        chosen_agent: Agent whose claim prevailed (None if delegated).
        reason: Why this resolution was chosen.
        strategy: The aggregation strategy that produced this resolution.
    """
    topic:        str
    chosen_claim: str
    chosen_agent: str | None
    reason:       str
    strategy:     str


@dataclass
class AggregationResult:
    """Result of ResultAggregator.aggregate().

    Attributes:
        final_answer: The aggregated answer.
        strategy_used: Strategy name applied.
        conflicts_detected: All contradictions found across the inputs.
        conflict_resolutions: How each conflict was resolved.
        confidence: Aggregate confidence in [0, 1].
        cost_usd: API cost incurred by aggregation (0.0 for CONSENSUS).
    """
    final_answer:         str
    strategy_used:        str
    conflicts_detected:   list[Conflict]
    conflict_resolutions: list[ConflictResolution]
    confidence:           float
    cost_usd:             float


# ── embedding backend ───────────────────────────────────────────────────────────

class _Embedder:
    """Sentence embeddings with a deterministic offline fallback.

    Uses sentence-transformers (all-MiniLM-L6-v2, normalised) when importable;
    otherwise a normalised bag-of-words vectoriser. Both back ``similarity`` as
    a cosine in [0, 1] (negatives clamped to 0 for the BoW path).
    """

    def __init__(self, model_name: str = _DEFAULT_EMBEDDING_MODEL) -> None:
        self._model: Any = None
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        except Exception:  # noqa: BLE001 -- missing package, no model cache, no network
            self._model = None

    @property
    def backend(self) -> str:
        """Name of the active embedding backend."""
        return "sentence-transformers" if self._model is not None else "bag-of-words"

    def encode(self, texts: list[str]) -> list[Any]:
        """Embed a list of texts into normalised vectors (model or BoW dicts)."""
        if self._model is not None:
            return [v for v in self._model.encode(texts, normalize_embeddings=True)]
        return [_bow_vector(t) for t in texts]

    @staticmethod
    def similarity(vec_a: Any, vec_b: Any) -> float:
        """Cosine similarity between two encoded vectors (both backends)."""
        if isinstance(vec_a, dict):  # bag-of-words
            keys = set(vec_a) & set(vec_b)
            return float(sum(vec_a[k] * vec_b[k] for k in keys))
        import numpy as np  # normalised dense vectors: dot == cosine
        return float(np.dot(vec_a, vec_b))


# ── aggregator ──────────────────────────────────────────────────────────────────

class ResultAggregator:
    """Aggregates multiple AgentResults into one coherent AggregationResult.

    Args:
        merge_model: Model used by the MERGE strategy (cheap; Haiku).
        supervisor_model: Model used by SUPERVISOR_DECIDES (Sonnet).
        embedding_model: Sentence-transformers model for conflict detection.
        conflict_threshold: Cosine below which two same-topic claims are deemed
            contradictory.
        agreement_threshold: Cosine at/above which two claims are the same point.

    Usage::

        agg = ResultAggregator()
        out = agg.aggregate(results, task="...", strategy="CONSENSUS")
        print(out.final_answer, out.conflict_resolutions)
    """

    def __init__(
        self,
        merge_model:          str = _DEFAULT_MERGE_MODEL,
        supervisor_model:     str = _DEFAULT_SUPERVISOR_MODEL,
        embedding_model:      str = _DEFAULT_EMBEDDING_MODEL,
        conflict_threshold:   float = _DEFAULT_CONFLICT_THRESHOLD,
        agreement_threshold:  float = _DEFAULT_AGREEMENT_THRESHOLD,
    ) -> None:
        self.merge_model         = merge_model
        self.supervisor_model    = supervisor_model
        self.conflict_threshold  = conflict_threshold
        self.agreement_threshold = agreement_threshold
        self._embedder           = _Embedder(embedding_model)
        self._client: anthropic.Anthropic | None = None

    @property
    def embedding_backend(self) -> str:
        """Active embedding backend ('sentence-transformers' or 'bag-of-words')."""
        return self._embedder.backend

    def _get_client(self) -> anthropic.Anthropic:
        """Lazily construct the Anthropic client (only the LLM strategies need it)."""
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        results:  list[AgentResult],
        task:     str,
        strategy: str | AggregationStrategy = AggregationStrategy.MERGE,
    ) -> AggregationResult:
        """Aggregate agent results into a single answer with the given strategy.

        Conflicts are detected once and reported regardless of strategy.

        Args:
            results: The agents' results (same task).
            task: The original task, used to steer MERGE/SUPERVISOR synthesis.
            strategy: "MERGE", "CONSENSUS", or "SUPERVISOR_DECIDES" (str or enum).

        Returns:
            An AggregationResult with the final answer, conflicts, resolutions,
            confidence, and cost.

        Raises:
            ValueError: If ``strategy`` is unknown.
        """
        strat = self._normalize_strategy(strategy)

        if not results:
            return AggregationResult(
                final_answer="", strategy_used=strat.value,
                conflicts_detected=[], conflict_resolutions=[],
                confidence=0.0, cost_usd=0.0,
            )

        conflicts = self.detect_conflicts(results)

        if strat is AggregationStrategy.MERGE:
            return self._merge(results, task, conflicts)
        if strat is AggregationStrategy.CONSENSUS:
            return self._consensus(results, task, conflicts)
        return self._supervisor_decides(results, task, conflicts)

    def detect_conflicts(self, results: list[AgentResult]) -> list[Conflict]:
        """Find contradictory same-topic claims across different agents.

        Each result's content is split into claims and embedded. For every pair
        of claims from different agents that share a salient keyword (same
        topic), a conflict is flagged when the cosine similarity is below
        ``conflict_threshold``, or when it is below ``agreement_threshold`` and
        the claims carry an antonym/negation cue.

        Args:
            results: The agents' results.

        Returns:
            A de-duplicated list of Conflicts, ordered by ascending similarity
            (most divergent first), capped for readability.
        """
        claims: list[tuple[str, str, set[str]]] = []  # (agent, claim, tokens)
        for res in results:
            for claim in _extract_claims(res.content):
                claims.append((res.agent, claim, _significant_tokens(claim)))

        if len(claims) < 2:
            return []

        vectors = self._embedder.encode([c[1] for c in claims])

        found: dict[frozenset[str], Conflict] = {}
        for i in range(len(claims)):
            agent_i, claim_i, tokens_i = claims[i]
            for j in range(i + 1, len(claims)):
                agent_j, claim_j, tokens_j = claims[j]
                if agent_i == agent_j:
                    continue
                shared = tokens_i & tokens_j
                sim = self._embedder.similarity(vectors[i], vectors[j])

                # Same topic: share a salient keyword, or are embedding-close.
                same_topic = bool(shared) or sim >= self.agreement_threshold
                if not same_topic:
                    continue

                # Two complementary contradiction signals on a same-topic pair:
                #   (a) an antonym/negation cue (catches "free" vs "paid", which
                #       embeds as HIGH similarity), or
                #   (b) low embedding similarity with real topical overlap
                #       (>= 2 shared keywords, so a single shared subject word
                #       isn't enough to manufacture a false conflict).
                antonym = _antonym_hit(tokens_i, tokens_j)
                is_conflict = (
                    antonym is not None
                    or (len(shared) >= 2 and sim < self.conflict_threshold)
                )
                if not is_conflict:
                    continue

                key = frozenset({claim_i, claim_j})
                if key in found:
                    continue
                topic = (
                    f"{antonym[0]} vs {antonym[1]}" if antonym
                    else ", ".join(sorted(shared)[:3])
                )
                found[key] = Conflict(
                    topic=topic,
                    claim_a=claim_i, agent_a=agent_i,
                    claim_b=claim_j, agent_b=agent_j,
                    similarity=round(sim, 3),
                )

        ordered = sorted(found.values(), key=lambda c: c.similarity)
        return ordered[:_MAX_REPORTED_CONFLICTS]

    @staticmethod
    def simple_merge(results: list[str]) -> str:
        """Merge raw result strings into one structured document without an LLM.

        Args:
            results: Plain result texts.

        Returns:
            A markdown document with one numbered section per non-empty result.
        """
        sections: list[str] = []
        for i, text in enumerate((r for r in results if r and r.strip()), start=1):
            sections.append(f"## Source {i}\n{text.strip()}")
        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Strategy: MERGE (Haiku LLM)
    # ------------------------------------------------------------------

    def _merge(
        self, results: list[AgentResult], task: str, conflicts: list[Conflict]
    ) -> AggregationResult:
        """Combine results into one structured document via the merge LLM."""
        system = (
            "You merge several specialist answers into ONE structured, "
            "de-duplicated document that answers the user's task. Combine points "
            "of agreement, remove redundancy, and where sources conflict prefer "
            "the higher-confidence source. Use short markdown headings or "
            "bullets. Output only the merged document."
        )
        user = _format_inputs(task, results, conflicts)
        text, cost = self._call_llm(self.merge_model, system, user,
                                    _MERGE_MAX_TOKENS, use_thinking=False)

        resolutions = [
            self._resolve_by_confidence(
                c, results, strategy="MERGE",
                reason_suffix="; final wording reconciled by the MERGE LLM",
            )
            for c in conflicts
        ]
        confidence = round(_mean(r.confidence for r in results), 2)
        return AggregationResult(
            final_answer=text or self.simple_merge([r.content for r in results]),
            strategy_used="MERGE",
            conflicts_detected=conflicts,
            conflict_resolutions=resolutions,
            confidence=confidence,
            cost_usd=round(cost, 6),
        )

    # ------------------------------------------------------------------
    # Strategy: CONSENSUS (no LLM)
    # ------------------------------------------------------------------

    def _consensus(
        self, results: list[AgentResult], task: str, conflicts: list[Conflict]
    ) -> AggregationResult:
        """Keep agreement; resolve each conflict by highest agent confidence."""
        # Resolve conflicts; collect the claims that lost so we can drop them.
        resolutions: list[ConflictResolution] = []
        losing_claims: set[str] = set()
        for c in conflicts:
            res = self._resolve_by_confidence(c, results, strategy="CONSENSUS")
            resolutions.append(res)
            loser = c.claim_b if res.chosen_claim == c.claim_a else c.claim_a
            losing_claims.add(_norm(loser))

        # Gather surviving claims (agent, claim, confidence), dropping losers.
        conf_by_agent = {r.agent: r.confidence for r in results}
        surviving: list[tuple[str, str, float]] = []
        for res in results:
            for claim in _extract_claims(res.content):
                if _norm(claim) in losing_claims:
                    continue
                surviving.append((res.agent, claim, conf_by_agent.get(res.agent, 0.0)))

        # De-duplicate agreeing claims: one representative (highest confidence)
        # per agreement cluster.
        reps = self._dedupe_agreeing(surviving)

        final_answer = "\n".join(f"- {claim}" for _, claim, _ in reps)
        confidence = round(_mean(conf for _, _, conf in reps) if reps else 0.0, 2)

        return AggregationResult(
            final_answer=final_answer,
            strategy_used="CONSENSUS",
            conflicts_detected=conflicts,
            conflict_resolutions=resolutions,
            confidence=confidence,
            cost_usd=0.0,
        )

    def _dedupe_agreeing(
        self, claims: list[tuple[str, str, float]]
    ) -> list[tuple[str, str, float]]:
        """Collapse near-identical claims, keeping the highest-confidence one."""
        if not claims:
            return []
        vectors = self._embedder.encode([c[1] for c in claims])
        tokens = [_significant_tokens(c[1]) for c in claims]
        used = [False] * len(claims)
        reps: list[tuple[str, str, float]] = []
        for i in range(len(claims)):
            if used[i]:
                continue
            cluster = [i]
            used[i] = True
            for j in range(i + 1, len(claims)):
                if used[j]:
                    continue
                # Merge as agreement only when the claims are embedding-close,
                # share real topical overlap (>= 2 salient keywords, so a common
                # subject word alone can't fuse two distinct dimensions), and are
                # not lexically opposed. Bias toward mild redundancy over
                # silently dropping a dimension.
                shared = tokens[i] & tokens[j]
                if (self._embedder.similarity(vectors[i], vectors[j]) >= self.agreement_threshold
                        and len(shared) >= 2
                        and _antonym_hit(tokens[i], tokens[j]) is None):
                    cluster.append(j)
                    used[j] = True
            best = max(cluster, key=lambda k: claims[k][2])
            reps.append(claims[best])
        return reps

    # ------------------------------------------------------------------
    # Strategy: SUPERVISOR_DECIDES (Sonnet LLM)
    # ------------------------------------------------------------------

    def _supervisor_decides(
        self, results: list[AgentResult], task: str, conflicts: list[Conflict]
    ) -> AggregationResult:
        """Send all results and conflicts to a supervisor LLM to adjudicate."""
        system = (
            "You are the supervisor adjudicating multiple agent answers. Choose "
            "the most reliable information, resolve every conflict (favouring "
            "higher confidence and internal consistency), and synthesise a "
            "single authoritative answer to the task. Output only the final "
            "answer."
        )
        user = _format_inputs(task, results, conflicts)
        text, cost = self._call_llm(self.supervisor_model, system, user,
                                    _SUPERVISOR_MAX_TOKENS, use_thinking=True)

        resolutions = [
            self._resolve_by_confidence(
                c, results, strategy="SUPERVISOR_DECIDES",
                reason_suffix="; final decision made by the supervisor",
            )
            for c in conflicts
        ]
        confidences = [r.confidence for r in results]
        confidence = round(min(0.97, max(confidences) + 0.05), 2) if confidences else 0.0
        return AggregationResult(
            final_answer=text or self.simple_merge([r.content for r in results]),
            strategy_used="SUPERVISOR_DECIDES",
            conflicts_detected=conflicts,
            conflict_resolutions=resolutions,
            confidence=confidence,
            cost_usd=round(cost, 6),
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _resolve_by_confidence(
        self,
        conflict:      Conflict,
        results:       list[AgentResult],
        strategy:      str,
        reason_suffix: str = "",
    ) -> ConflictResolution:
        """Resolve a conflict in favour of the higher-confidence agent."""
        conf = {r.agent: r.confidence for r in results}
        ca, cb = conf.get(conflict.agent_a, 0.0), conf.get(conflict.agent_b, 0.0)
        if ca >= cb:
            chosen_claim, chosen_agent, hi, lo = conflict.claim_a, conflict.agent_a, ca, cb
        else:
            chosen_claim, chosen_agent, hi, lo = conflict.claim_b, conflict.agent_b, cb, ca
        reason = (
            f"selected higher-confidence source '{chosen_agent}' "
            f"({hi:.2f} vs {lo:.2f}){reason_suffix}"
        )
        return ConflictResolution(
            topic=conflict.topic, chosen_claim=chosen_claim,
            chosen_agent=chosen_agent, reason=reason, strategy=strategy,
        )

    def _call_llm(
        self,
        model:        str,
        system:       str,
        user:         str,
        max_tokens:   int,
        use_thinking: bool,
    ) -> tuple[str, float]:
        """Make one Messages API call and return (text, cost_usd).

        Args:
            model: Model ID.
            system: System prompt.
            user: User message content.
            max_tokens: Output cap.
            use_thinking: Enable adaptive thinking (Sonnet path; off for Haiku).

        Returns:
            (response_text, cost_usd).
        """
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if use_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        resp = client.messages.create(**kwargs)
        text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        ).strip()
        cost = _price(model, resp.usage.input_tokens, resp.usage.output_tokens)
        return text, cost

    @staticmethod
    def _normalize_strategy(strategy: str | AggregationStrategy) -> AggregationStrategy:
        """Coerce a strategy name/enum into an AggregationStrategy."""
        if isinstance(strategy, AggregationStrategy):
            return strategy
        try:
            return AggregationStrategy(str(strategy).strip().upper())
        except ValueError as exc:
            valid = ", ".join(s.value for s in AggregationStrategy)
            raise ValueError(
                f"Unknown strategy {strategy!r}. Valid: {valid}"
            ) from exc


# ── module helpers ────────────────────────────────────────────────────────────

def _format_inputs(
    task: str, results: list[AgentResult], conflicts: list[Conflict]
) -> str:
    """Render task + agent results (+ detected conflicts) as an LLM prompt body."""
    parts = [f"Task:\n{task}\n", "Agent results:"]
    for r in results:
        parts.append(
            f"\n### {r.agent} (confidence {r.confidence:.2f})\n{r.content.strip()}"
        )
    if conflicts:
        parts.append("\nDetected conflicts (resolve these):")
        for c in conflicts:
            parts.append(
                f"- on '{c.topic}': {c.agent_a} says \"{c.claim_a}\" vs "
                f"{c.agent_b} says \"{c.claim_b}\""
            )
    return "\n".join(parts)


def _extract_claims(text: str) -> list[str]:
    """Split a result into atomic claims (bullets / sentences)."""
    claims: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-*•").strip()
        line = re.sub(r"^\d+[.)]\s*", "", line)
        if not line:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            sentence = sentence.strip()
            if len(sentence) >= _MIN_CLAIM_CHARS:
                claims.append(sentence)
    return claims


def _significant_tokens(text: str) -> set[str]:
    """Lower-cased content words (length >= 4, excluding stopwords)."""
    return {
        t for t in re.findall(r"[a-zA-Z]{4,}", text.lower())
        if t not in _STOPWORDS
    }


def _antonym_hit(tokens_a: set[str], tokens_b: set[str]) -> tuple[str, str] | None:
    """Return the (a-side, b-side) antonym pair straddling the token sets, else None.

    Detects a contradiction cue where one claim asserts a word and the other
    asserts its opposite (e.g. 'free' in A and 'paid' in B). The returned tuple
    is ordered (token-from-A, token-from-B) so it can label the conflict topic.
    """
    for pair in _ANTONYMS:
        w1, w2 = tuple(pair)
        if w1 in tokens_a and w2 in tokens_b:
            return (w1, w2)
        if w2 in tokens_a and w1 in tokens_b:
            return (w2, w1)
    return None


def _bow_vector(text: str) -> dict[str, float]:
    """L2-normalised bag-of-words vector (deterministic offline embedding)."""
    counts: dict[str, float] = {}
    for tok in re.findall(r"[a-zA-Z]{3,}", text.lower()):
        counts[tok] = counts.get(tok, 0.0) + 1.0
    norm = sum(v * v for v in counts.values()) ** 0.5
    if norm == 0.0:
        return counts
    return {k: v / norm for k, v in counts.items()}


def _norm(text: str) -> str:
    """Normalise a claim for equality comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _mean(values: Any) -> float:
    """Arithmetic mean of an iterable, 0.0 when empty."""
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # Three simulated agent results on the same question.
    # researcher_1 and researcher_2 agree; researcher_3 partially conflicts
    # (on pricing and offline support) and has the lowest confidence.
    task = (
        "Is Obsidian a good note-taking app for a privacy-conscious user? "
        "Summarise its data storage, offline support, and pricing."
    )
    results = [
        AgentResult(
            agent="researcher_1",
            confidence=0.90,
            content=(
                "- Obsidian stores all notes as local plain-text Markdown files on the device.\n"
                "- Obsidian works fully offline and requires no account to use.\n"
                "- Obsidian is free for personal use."
            ),
        ),
        AgentResult(
            agent="researcher_2",
            confidence=0.85,
            content=(
                "- Obsidian keeps notes in local Markdown files, giving users full data ownership.\n"
                "- Obsidian functions completely offline by default.\n"
                "- The personal-use plan of Obsidian is free."
            ),
        ),
        AgentResult(
            agent="researcher_3",
            confidence=0.50,
            content=(
                "- Obsidian stores notes locally as Markdown files.\n"
                "- Obsidian requires a constant internet connection to open vaults.\n"
                "- Obsidian needs a paid subscription to unlock its core features."
            ),
        ),
    ]

    aggregator = ResultAggregator()
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  RESULT AGGREGATION DEMO   (embedding backend: {aggregator.embedding_backend})")
    print(sep)
    print(f"  Task: {task}")
    print(f"  Inputs: {len(results)} agents "
          f"({', '.join(f'{r.agent}={r.confidence:.2f}' for r in results)})")

    def show(label: str, out: AggregationResult) -> None:
        print(f"\n{sep}")
        print(f"  STRATEGY: {label}")
        print(sep)
        print(f"  confidence={out.confidence:.2f}  cost=${out.cost_usd:.5f}  "
              f"conflicts={len(out.conflicts_detected)}")
        if out.conflicts_detected:
            print("  Conflicts detected:")
            for c in out.conflicts_detected:
                print(f"    - [{c.topic}] sim={c.similarity:.2f}  "
                      f"{c.agent_a}: \"{c.claim_a}\"")
                print(f"      {' ' * len(c.topic)}        vs {c.agent_b}: \"{c.claim_b}\"")
        if out.conflict_resolutions:
            print("  Conflict resolutions:")
            for r in out.conflict_resolutions:
                print(f"    - [{r.topic}] -> {r.chosen_agent}: {r.reason}")
        print("  Final answer:")
        for line in out.final_answer.splitlines():
            print(f"    {line}")

    # CONSENSUS needs no API key — always run it.
    show("CONSENSUS", aggregator.aggregate(results, task, "CONSENSUS"))

    if has_key:
        show("MERGE", aggregator.aggregate(results, task, "MERGE"))
        show("SUPERVISOR_DECIDES", aggregator.aggregate(results, task, "SUPERVISOR_DECIDES"))
    else:
        print(f"\n{sep}")
        print("  MERGE and SUPERVISOR_DECIDES require ANTHROPIC_API_KEY "
              "(set it in .env) -- skipped.")
        print(sep)
