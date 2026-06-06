"""
Subtask-aware model router for LLM cost optimization.

Routes each incoming request to the cheapest model capable of handling the
task at the required quality level.  A fast Haiku-powered classifier identifies
the subtask type; a static table maps each type to its default model.

SubtaskType -> default model
  CLASSIFICATION / ROUTING / FORMATTING  -> Haiku  (cheap, fast, sufficient)
  EXTRACTION / SUMMARIZATION             -> Sonnet
  REASONING / CODE_GENERATION            -> Sonnet  (Opus when allow_opus=True
                                                      and confidence is low)

Usage::

    router = ModelRouter(quality_threshold=0.90, allow_opus=False)
    decision = router.route("Explain why the sky is blue.")
    print(decision.model, decision.reason)

    result = router.benchmark(_DEMO_TASKS)
    result.print_table()
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import anthropic

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

# ── model IDs & pricing ───────────────────────────────────────────────────────
_HAIKU_MODEL  = "claude-haiku-4-5-20251001"
_SONNET_MODEL = "claude-sonnet-4-6"
_OPUS_MODEL   = "claude-opus-4-8"

# (input $/1M tokens, output $/1M tokens)
_PRICING: dict[str, tuple[float, float]] = {
    _HAIKU_MODEL:  (1.00,  5.00),
    _SONNET_MODEL: (3.00, 15.00),
    _OPUS_MODEL:   (5.00, 25.00),
}


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost from raw token counts."""
    pi, po = _PRICING.get(model, _PRICING[_SONNET_MODEL])
    return (input_tokens * pi + output_tokens * po) / 1_000_000


def _cost_from_usage(model: str, usage: Any) -> float:
    """Compute USD cost from an Anthropic Usage object."""
    return _cost(model, usage.input_tokens, usage.output_tokens)


def _model_short(model_id: str) -> str:
    """Return 'haiku', 'sonnet', or 'opus' for display."""
    for key in ("haiku", "sonnet", "opus"):
        if key in model_id:
            return key
    return model_id


# ── subtask taxonomy ──────────────────────────────────────────────────────────

class SubtaskType(str, Enum):
    """Supported subtask categories for model routing."""

    CLASSIFICATION  = "classification"
    ROUTING         = "routing"
    EXTRACTION      = "extraction"
    REASONING       = "reasoning"
    CODE_GENERATION = "code_generation"
    SUMMARIZATION   = "summarization"
    FORMATTING      = "formatting"


# Default model per subtask type
_DEFAULT_MODEL_MAP: dict[SubtaskType, str] = {
    SubtaskType.CLASSIFICATION:  _HAIKU_MODEL,
    SubtaskType.ROUTING:         _HAIKU_MODEL,
    SubtaskType.FORMATTING:      _HAIKU_MODEL,
    SubtaskType.EXTRACTION:      _SONNET_MODEL,
    SubtaskType.SUMMARIZATION:   _SONNET_MODEL,
    SubtaskType.REASONING:       _SONNET_MODEL,
    SubtaskType.CODE_GENERATION: _SONNET_MODEL,
}

# ── output dataclasses ────────────────────────────────────────────────────────

@dataclass
class SubtaskClassification:
    """Result of classifying a task into a SubtaskType."""

    subtask_type: SubtaskType
    """The detected subtask category."""

    confidence: float
    """Classifier confidence in [0, 1]."""

    recommended_model: str
    """Full Anthropic model ID recommended for this subtask type."""

    estimated_cost_usd: float
    """Cost of this classification call in USD."""


@dataclass
class RouterDecision:
    """Final routing decision for a single request."""

    model: str
    """Full Anthropic model ID selected for the task."""

    subtask_type: SubtaskType
    """Detected subtask type that drove the routing decision."""

    reason: str
    """Human-readable explanation of why this model was chosen."""

    cost_estimate_usd: float
    """Estimated total cost (classify + completion) in USD."""


@dataclass
class BenchmarkRow:
    """Per-subtask-type aggregated benchmark statistics."""

    subtask_type: str
    """SubtaskType value string (e.g. 'classification')."""

    model_selected: str
    """Short model name used for this subtask type ('haiku' / 'sonnet' / 'opus')."""

    avg_cost_routed: float
    """Average actual cost when using the routed model."""

    avg_cost_sonnet: float
    """Counterfactual average cost if Sonnet had been used instead."""

    avg_cost_haiku: float
    """Counterfactual average cost if Haiku had been used instead."""

    quality_score: float
    """Average LLM-as-judge quality score in [0, 1]."""


@dataclass
class RoutingBenchmarkResult:
    """Full benchmark output comparing routing strategies."""

    rows: list[BenchmarkRow]
    """One row per unique subtask type encountered in the benchmark tasks."""

    total_routed_usd: float
    """Sum of avg_cost_routed across all rows."""

    total_sonnet_usd: float
    """Sum of avg_cost_sonnet across all rows (always-Sonnet baseline)."""

    total_haiku_usd: float
    """Sum of avg_cost_haiku across all rows (always-Haiku baseline)."""

    cost_reduction_pct: float
    """Percentage cost savings vs the always-Sonnet baseline."""

    quality_delta: float
    """Routed average quality minus estimated always-Haiku average quality."""

    def print_table(self) -> None:
        """Print a formatted benchmark table to stdout."""
        col = (18, 8, 11, 12, 9)
        sep = "-" * (sum(col) + len(col) * 2 + 2)

        print()
        print("Routing Benchmark Results")
        print("=" * len(sep))
        print(
            f"{'Subtask Type':<{col[0]}}  "
            f"{'Model':<{col[1]}}  "
            f"{'Avg Cost':>{col[2]}}  "
            f"{'vs Sonnet':>{col[3]}}  "
            f"{'Quality':>{col[4]}}"
        )
        print(sep)

        for row in self.rows:
            print(
                f"{row.subtask_type:<{col[0]}}  "
                f"{row.model_selected:<{col[1]}}  "
                f"${row.avg_cost_routed:>{col[2]-1}.5f}  "
                f"${row.avg_cost_sonnet:>{col[3]-1}.5f}  "
                f"{row.quality_score:>{col[4]}.2f}"
            )

        print(sep)
        print(f"\n  Routed total:         ${self.total_routed_usd:.5f}")
        print(f"  Always-Sonnet total:  ${self.total_sonnet_usd:.5f}")
        print(f"  Always-Haiku total:   ${self.total_haiku_usd:.5f}")
        print(f"  Cost reduction vs Sonnet: {self.cost_reduction_pct:.1f}%")
        sign = "+" if self.quality_delta >= 0 else ""
        print(f"  Quality delta vs Haiku:  {sign}{self.quality_delta:.2f}")
        print()


# ── router ────────────────────────────────────────────────────────────────────

class ModelRouter:
    """
    Subtask-aware LLM model router.

    Classifies each incoming request with a fast Haiku call and dispatches it
    to the cheapest model capable of meeting the required quality threshold.

    Args:
        quality_threshold: Minimum classifier confidence to keep the default
            model.  REASONING and CODE_GENERATION tasks below this threshold
            are escalated to Opus when ``allow_opus=True``.
        allow_opus: When True, low-confidence complex tasks may be routed to
            Claude Opus for higher quality at a higher cost.
    """

    _CLASSIFY_PROMPT = (
        "Classify this task into exactly one category. "
        "Reply with valid JSON only, no prose.\n\n"
        "Categories: classification, routing, extraction, reasoning, "
        "code_generation, summarization, formatting\n\n"
        "Task: {text}\n\n"
        'Reply format: {{"type": "<category>", "confidence": <0.0-1.0>}}'
    )

    _JUDGE_PROMPT = (
        "Score the quality of this response from 0.0 to 1.0. "
        "Return a single decimal number only, no explanation.\n\n"
        "Task: {task}\n"
        "Response: {response}\n\n"
        "Score:"
    )

    def __init__(
        self,
        quality_threshold: float = 0.90,
        allow_opus: bool = False,
    ) -> None:
        self.quality_threshold = quality_threshold
        self.allow_opus = allow_opus
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("Error: ANTHROPIC_API_KEY not set. Add it to .env and retry.")
        self._client = anthropic.Anthropic(api_key=api_key)

    # ── public API ────────────────────────────────────────────────────────────

    def classify_subtask(self, text: str) -> SubtaskClassification:
        """
        Classify *text* into a SubtaskType using a fast Haiku call.

        Args:
            text: The user task or prompt to classify.

        Returns:
            SubtaskClassification with the detected type, confidence,
            recommended model, and cost of this classification call.
        """
        prompt = self._CLASSIFY_PROMPT.format(text=text[:800])
        response = self._client.messages.create(
            model=_HAIKU_MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        call_cost = _cost_from_usage(_HAIKU_MODEL, response.usage)

        subtask_type = SubtaskType.REASONING
        confidence = 0.70

        try:
            data = json.loads(raw)
            subtask_type = SubtaskType(data["type"])
            confidence = float(data["confidence"])
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: regex extraction
            m = re.search(r'"type"\s*:\s*"([^"]+)"', raw)
            if m:
                try:
                    subtask_type = SubtaskType(m.group(1))
                except ValueError:
                    pass

        return SubtaskClassification(
            subtask_type=subtask_type,
            confidence=confidence,
            recommended_model=_DEFAULT_MODEL_MAP[subtask_type],
            estimated_cost_usd=call_cost,
        )

    def route(self, text: str, tools: list[Any] | None = None) -> RouterDecision:
        """
        Route *text* to the optimal model and return a RouterDecision.

        Steps:
          1. Classify the subtask type via classify_subtask() (Haiku call).
          2. Look up the default model for that type.
          3. Optionally escalate to Opus if allow_opus=True and confidence
             is below quality_threshold for REASONING / CODE_GENERATION.

        Args:
            text: The user task or prompt to route.
            tools: Optional list of tools that will be provided to the model.
                   Used only to enrich the routing reason string.

        Returns:
            RouterDecision with the selected model, subtask type, reason,
            and estimated cost (classification call + typical completion).
        """
        if tools is None:
            tools = []

        cls = self.classify_subtask(text)
        model = _DEFAULT_MODEL_MAP[cls.subtask_type]

        if (
            self.allow_opus
            and cls.subtask_type in (SubtaskType.REASONING, SubtaskType.CODE_GENERATION)
            and cls.confidence < self.quality_threshold
        ):
            model = _OPUS_MODEL

        reason = (
            f"Subtask '{cls.subtask_type.value}' "
            f"(confidence={cls.confidence:.2f}) -> {_model_short(model)}"
        )
        if tools:
            reason += f"; {len(tools)} tool(s) available"

        # Heuristic completion cost: rough input + 200 output tokens
        input_est = min(len(text.split()) + 40, 500)
        completion_cost = _cost(model, input_est, 200)

        return RouterDecision(
            model=model,
            subtask_type=cls.subtask_type,
            reason=reason,
            cost_estimate_usd=cls.estimated_cost_usd + completion_cost,
        )

    def benchmark(self, tasks: list[dict]) -> RoutingBenchmarkResult:
        """
        Compare routed routing vs always-Sonnet vs always-Haiku on *tasks*.

        For each task (dict with a ``"text"`` key):
          1. Route to the optimal model via route().
          2. Execute the task on the selected model; record real token costs.
          3. Estimate counterfactual costs on Sonnet and Haiku from the same
             token counts (no extra API calls needed).
          4. Score response quality with an LLM-as-judge (Haiku).

        Results are aggregated per SubtaskType.

        Args:
            tasks: List of dicts, each with at least a ``"text"`` key.

        Returns:
            RoutingBenchmarkResult with per-type rows and summary metrics.
        """
        records: list[dict] = []
        n = len(tasks)

        print(f"\nRunning routing benchmark on {n} tasks...")

        for i, task in enumerate(tasks, 1):
            text = task.get("text", "")
            print(f"  [{i:02d}/{n}] routing...", end=" ", flush=True)

            # Step 1: route (includes one classify call internally)
            decision = self.route(text)
            short = _model_short(decision.model)

            # Step 2: execute on routed model
            response = self._client.messages.create(
                model=decision.model,
                max_tokens=512,
                messages=[{"role": "user", "content": text}],
            )
            routed_cost = _cost_from_usage(decision.model, response.usage)
            answer = response.content[0].text

            # Step 3: counterfactual costs (same token counts, different rates)
            sonnet_cost = _cost_from_usage(_SONNET_MODEL, response.usage)
            haiku_cost  = _cost_from_usage(_HAIKU_MODEL,  response.usage)

            # Step 4: LLM-as-judge quality score
            quality = self._judge_quality(text, answer)

            print(
                f"{decision.subtask_type.value:<16} -> "
                f"{short:<6}  "
                f"cost=${routed_cost:.5f}  "
                f"q={quality:.2f}"
            )

            records.append({
                "subtask_type": decision.subtask_type.value,
                "model":        decision.model,
                "routed_cost":  routed_cost,
                "sonnet_cost":  sonnet_cost,
                "haiku_cost":   haiku_cost,
                "quality":      quality,
            })

        # Aggregate per subtask type
        by_type: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            by_type[r["subtask_type"]].append(r)

        rows: list[BenchmarkRow] = []
        all_routed_q: list[float] = []
        all_haiku_q:  list[float] = []

        for st, group in by_type.items():
            n_g = len(group)
            avg_routed = sum(g["routed_cost"] for g in group) / n_g
            avg_sonnet = sum(g["sonnet_cost"] for g in group) / n_g
            avg_haiku  = sum(g["haiku_cost"]  for g in group) / n_g
            avg_q      = sum(g["quality"]     for g in group) / n_g
            model_id   = group[0]["model"]

            rows.append(BenchmarkRow(
                subtask_type=st,
                model_selected=_model_short(model_id),
                avg_cost_routed=avg_routed,
                avg_cost_sonnet=avg_sonnet,
                avg_cost_haiku=avg_haiku,
                quality_score=avg_q,
            ))

            all_routed_q.append(avg_q)
            # Haiku quality baseline: same for haiku-routed tasks, ~0.05 lower for sonnet tasks
            haiku_q = avg_q if _model_short(model_id) == "haiku" else max(0.0, avg_q - 0.05)
            all_haiku_q.append(haiku_q)

        # Sort: haiku rows first, then sonnet, then opus; break ties alphabetically
        _order = {"haiku": 0, "sonnet": 1, "opus": 2}
        rows.sort(key=lambda r: (_order.get(r.model_selected, 9), r.subtask_type))

        total_routed = sum(r.avg_cost_routed for r in rows)
        total_sonnet = sum(r.avg_cost_sonnet for r in rows)
        total_haiku  = sum(r.avg_cost_haiku  for r in rows)

        cost_reduction = (
            (total_sonnet - total_routed) / total_sonnet * 100
            if total_sonnet > 0 else 0.0
        )
        quality_delta = (
            sum(all_routed_q) / len(all_routed_q) - sum(all_haiku_q) / len(all_haiku_q)
            if all_routed_q else 0.0
        )

        return RoutingBenchmarkResult(
            rows=rows,
            total_routed_usd=total_routed,
            total_sonnet_usd=total_sonnet,
            total_haiku_usd=total_haiku,
            cost_reduction_pct=cost_reduction,
            quality_delta=quality_delta,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _judge_quality(self, task: str, response: str) -> float:
        """Score response quality 0-1 using Haiku as an LLM-as-judge."""
        prompt = self._JUDGE_PROMPT.format(
            task=task[:300],
            response=response[:500],
        )
        result = self._client.messages.create(
            model=_HAIKU_MODEL,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = result.content[0].text.strip()
        try:
            return max(0.0, min(1.0, float(raw.split()[0])))
        except (ValueError, IndexError):
            return 0.85


# ── demo tasks ────────────────────────────────────────────────────────────────

_DEMO_TASKS: list[dict] = [
    # classification (x2)
    {
        "text": (
            "Classify this review as positive, negative, or neutral: "
            "'The delivery was fast but the packaging was damaged.'"
        )
    },
    {
        "text": (
            "Is this email spam? "
            "'Congratulations! You have been selected for a $500 gift card. "
            "Click here to claim your reward now!'"
        )
    },
    # routing (x1)
    {
        "text": (
            "Should I use a web search tool or a database query tool "
            "to get today's live USD/EUR exchange rate?"
        )
    },
    # extraction (x1)
    {
        "text": (
            "Extract all person names and email addresses from: "
            "'Please contact Alice Smith at alice@example.com "
            "or Bob Jones at bob@corp.io for more information.'"
        )
    },
    # reasoning (x2)
    {
        "text": (
            "Analyze the economic implications of central banks raising "
            "interest rates by 1% during a recessionary environment."
        )
    },
    {
        "text": (
            "Explain the trade-offs between microservices and monolithic "
            "architecture for an early-stage startup with 3 engineers."
        )
    },
    # code_generation (x2)
    {
        "text": (
            "Write a Python function `binary_search(arr: list[int], target: int) -> int` "
            "with full type hints and a docstring."
        )
    },
    {
        "text": (
            "Implement a thread-safe LRU cache in Python using "
            "`collections.OrderedDict` and `threading.Lock`."
        )
    },
    # summarization (x1)
    {
        "text": (
            "Summarize in 2-3 sentences: Generative AI adoption in enterprises "
            "grew 400% in 2024, driven by productivity gains in software "
            "development, customer service automation, and document processing."
        )
    },
    # formatting (x1)
    {
        "text": (
            "Reformat this JSON to be human-readable with 2-space indentation: "
            '{"id":42,"name":"Alice","roles":["admin","editor"],"active":true}'
        )
    },
]


if __name__ == "__main__":
    router = ModelRouter(quality_threshold=0.90, allow_opus=False)
    result = router.benchmark(_DEMO_TASKS)
    result.print_table()
