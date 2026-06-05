"""
Accurate token counting for messages, tool definitions, and system prompts.

Uses the Anthropic count_tokens API when a client is available; falls back
to a chars/4 estimate so callers never need to branch on availability.

Results are cached in memory: identical (text, model, tools) tuples hit the
cache without an API call, keeping hot-loop usage fast and free.

Pricing table ($/1M tokens):
  claude-haiku-4-5  : $0.80 in / $4.00 out
  claude-sonnet-4-6 : $3.00 in / $15.00 out
  claude-opus-4-7   : $15.00 in / $75.00 out
  claude-opus-4-8   : $15.00 in / $75.00 out
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── pricing ($/1M tokens) ─────────────────────────────────────────────────────
_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":       {"input":  0.80, "output":  4.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6":      {"input":  3.00, "output": 15.00},
    "claude-opus-4-7":        {"input": 15.00, "output": 75.00},
    "claude-opus-4-8":        {"input": 15.00, "output": 75.00},
}
_DEFAULT_MODEL = "claude-sonnet-4-6"

# Approximate context window sizes by model family
_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-haiku-4-5":  200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-7":   200_000,
    "claude-opus-4-8":   200_000,
}


# ── public data types ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TokenCount:
    """Breakdown of token consumption across message components.

    Attributes:
        input_tokens:   Total tokens the API would bill as input.
        system_tokens:  Tokens in the system prompt alone.
        message_tokens: Tokens in the messages list alone.
        tool_tokens:    Tokens consumed by tool/function definitions.
        total:          Alias for input_tokens (all billable input tokens).
        exact:          True when the count came from the API; False for estimate.
    """

    input_tokens:   int
    system_tokens:  int
    message_tokens: int
    tool_tokens:    int
    total:          int
    exact:          bool = True

    def __post_init__(self) -> None:
        if self.total == 0 and self.input_tokens > 0:
            # Allow callers to leave total=0 and we fill it from input_tokens
            object.__setattr__(self, "total", self.input_tokens)


@dataclass
class WindowStatus:
    """Context-window utilisation for a given message list.

    Attributes:
        used_tokens:      Tokens currently occupied.
        max_tokens:       Maximum context window for the model.
        usage_pct:        Fraction of window used (0.0 – 100.0).
        remaining_tokens: Tokens still available.
        projected_tokens: Estimated total after one more average-length turn.
        is_near_limit:    True when usage_pct >= 80.
        is_over_limit:    True when used_tokens > max_tokens.
    """

    used_tokens:      int
    max_tokens:       int
    usage_pct:        float
    remaining_tokens: int
    projected_tokens: int
    is_near_limit:    bool
    is_over_limit:    bool


# ── token counter ─────────────────────────────────────────────────────────────

class TokenCounter:
    """Count tokens accurately via the Anthropic API, with cache and fallback.

    Pass an ``anthropic.Anthropic`` client to enable exact counting via the
    ``messages.count_tokens`` endpoint.  Without a client, every method falls
    back to a chars/4 heuristic and returns ``TokenCount.exact=False``.

    Results are memoised by a SHA-256 of the serialised inputs so that
    repeated calls with identical payloads never hit the network.

    Args:
        client:        An ``anthropic.Anthropic`` instance, or ``None``.
        default_model: Model ID used when callers omit the ``model`` argument.
        cache_size:    Maximum number of results kept in the LRU-style cache.
                       Oldest entries are evicted when the limit is reached.
    """

    def __init__(
        self,
        client: Any | None = None,
        default_model: str = _DEFAULT_MODEL,
        cache_size: int = 512,
    ) -> None:
        self._client = client
        self._default_model = default_model
        self._cache_size = cache_size
        # Ordered insertion dict used as a simple LRU (evict oldest on overflow)
        self._cache: dict[str, TokenCount] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count_messages(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> TokenCount:
        """Count all tokens that would be billed for a single API call.

        When an Anthropic client is available the count is exact (one
        lightweight API call, no generation).  Without a client, each
        component is estimated via ``count_text`` and marked ``exact=False``.

        Args:
            messages: List of message dicts (``role`` + ``content``).
            system:   System prompt text.
            tools:    Tool/function definitions to include in the count.
            model:    Claude model ID; defaults to ``self.default_model``.

        Returns:
            ``TokenCount`` with per-component and total breakdowns.
        """
        tools = tools or []
        model = model or self._default_model

        cache_key = self._make_key(messages, system, tools, model)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._client is not None:
            result = self._count_via_api(messages, system, tools, model)
        else:
            result = self._count_via_estimate(messages, system, tools)

        self._store(cache_key, result)
        return result

    def count_text(self, text: str) -> int:
        """Estimate token count for a plain string.

        Uses ``chars / 4`` as the approximation — accurate to ±15 % for
        typical English prose.  No API call is made; no cache lookup needed
        for a scalar result this cheap.

        Args:
            text: Any string.

        Returns:
            Estimated token count (minimum 1 for non-empty strings).
        """
        if not text:
            return 0
        return max(1, len(text) // 4)

    def estimate_turn_cost(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        output_tokens: int = 500,
    ) -> float:
        """Estimate the USD cost of one API call with the given messages.

        Combines ``count_messages`` (exact or estimated input tokens) with
        the model's input pricing.  Output-token cost uses ``output_tokens``
        as the expected generation length.

        Args:
            messages:      Messages to include in the call.
            model:         Claude model ID; defaults to ``self.default_model``.
            system:        System prompt text.
            tools:         Tool definitions.
            output_tokens: Expected output length (default 500 tokens).

        Returns:
            Estimated cost in USD.
        """
        model = model or self._default_model
        tc = self.count_messages(messages, system=system, tools=tools, model=model)
        pricing = _PRICING.get(model, _PRICING[_DEFAULT_MODEL])
        input_cost  = tc.total       * pricing["input"]  / 1_000_000
        output_cost = output_tokens  * pricing["output"] / 1_000_000
        return round(input_cost + output_cost, 8)

    def track_window_usage(
        self,
        messages: list[dict[str, Any]],
        max_context_tokens: int,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> WindowStatus:
        """Report how much of the context window the messages currently occupy.

        The projection adds one extra "average turn" (mean tokens per message)
        to simulate what the window will look like after the next exchange.

        Args:
            messages:           Current message list.
            max_context_tokens: Hard limit for the target model/deployment.
            system:             System prompt text.
            tools:              Tool definitions.
            model:              Claude model ID.

        Returns:
            ``WindowStatus`` with usage percentage and projection.
        """
        model = model or self._default_model
        tc = self.count_messages(messages, system=system, tools=tools, model=model)
        used = tc.total
        pct  = (used / max_context_tokens * 100.0) if max_context_tokens > 0 else 0.0

        # Projection: one more turn at the average message size
        avg_per_msg = (tc.message_tokens // len(messages)) if messages else 0
        projected   = used + avg_per_msg

        return WindowStatus(
            used_tokens=      used,
            max_tokens=       max_context_tokens,
            usage_pct=        round(pct, 2),
            remaining_tokens= max(0, max_context_tokens - used),
            projected_tokens= projected,
            is_near_limit=    pct >= 80.0,
            is_over_limit=    used > max_context_tokens,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_via_api(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        model: str,
    ) -> TokenCount:
        """Call the Anthropic count_tokens endpoint and decompose the result."""
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        # Full count including all components
        full_resp = self._client.messages.count_tokens(**kwargs)
        full_tokens = full_resp.input_tokens

        # System-only count (to isolate system_tokens)
        system_tokens = 0
        if system:
            sys_resp = self._client.messages.count_tokens(
                model=model,
                messages=[{"role": "user", "content": "x"}],
                system=system,
            )
            base_resp = self._client.messages.count_tokens(
                model=model,
                messages=[{"role": "user", "content": "x"}],
            )
            system_tokens = sys_resp.input_tokens - base_resp.input_tokens

        # Tool-only overhead (subtract from full to get message portion)
        tool_tokens = 0
        if tools:
            notool_resp = self._client.messages.count_tokens(
                model=model,
                messages=messages,
                **({"system": system} if system else {}),
            )
            tool_tokens = full_tokens - notool_resp.input_tokens

        message_tokens = full_tokens - system_tokens - tool_tokens

        return TokenCount(
            input_tokens=   full_tokens,
            system_tokens=  system_tokens,
            message_tokens= max(0, message_tokens),
            tool_tokens=    max(0, tool_tokens),
            total=          full_tokens,
            exact=          True,
        )

    def _count_via_estimate(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
    ) -> TokenCount:
        """Estimate all token components without any API call."""
        system_tokens  = self.count_text(system)
        message_tokens = sum(
            self.count_text(
                m.get("content", "") if isinstance(m.get("content"), str)
                else json.dumps(m.get("content", ""))
            )
            for m in messages
        )
        tool_tokens = sum(
            self.count_text(json.dumps(t)) for t in tools
        )
        total = system_tokens + message_tokens + tool_tokens

        return TokenCount(
            input_tokens=   total,
            system_tokens=  system_tokens,
            message_tokens= message_tokens,
            tool_tokens=    tool_tokens,
            total=          total,
            exact=          False,
        )

    def _make_key(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        model: str,
    ) -> str:
        """Stable SHA-256 cache key from serialised inputs."""
        payload = json.dumps(
            {"model": model, "system": system, "messages": messages, "tools": tools},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _store(self, key: str, value: TokenCount) -> None:
        """Insert into cache, evicting the oldest entry if full."""
        if len(self._cache) >= self._cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value

    @property
    def cache_hits(self) -> int:
        """Number of entries currently in cache (not a hit counter)."""
        return len(self._cache)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parents[2] / ".env")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = None
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            print("Anthropic client initialised — exact counts enabled.\n")
        except ImportError:
            print("anthropic package not installed — using estimates only.\n")
    else:
        print("ANTHROPIC_API_KEY not found — using estimates only.\n")

    counter = TokenCounter(client=client)
    MODEL   = "claude-sonnet-4-6"
    WINDOW  = 100_000

    SYSTEM = (
        "You are a helpful assistant specialised in software engineering. "
        "Answer concisely and accurately."
    )

    # ── build 5 conversations of growing size ─────────────────────────────────
    # Each conversation is a list of alternating user/assistant messages.
    # The word count is scaled so the estimated token count is near the target.

    def _word(n: int) -> str:
        """Return a deterministic lorem-ipsum-style string of ~n words."""
        word = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod"
        words = (word.split() * ((n // 7) + 1))[:n]
        return " ".join(words)

    def _build_conv(target_tokens: int) -> list[dict[str, Any]]:
        """Build a conversation whose rough token count is near target_tokens."""
        # Each turn: ~200 chars user + ~200 chars assistant ≈ 100 tokens total
        turns = max(1, target_tokens // 100)
        msgs: list[dict[str, Any]] = []
        for i in range(turns):
            msgs.append({"role": "user",      "content": _word(40)})
            msgs.append({"role": "assistant", "content": _word(40)})
        return msgs

    TARGET_SIZES = [100, 500, 2_000, 5_000, 10_000]
    conversations = [_build_conv(t) for t in TARGET_SIZES]

    sep = "=" * 80
    print(sep)
    print(f"  Token counter demo  |  model={MODEL}  |  window={WINDOW:,} tokens")
    print(sep)
    print(
        f"  {'Target':>7}  {'Messages':>8}  "
        f"{'Estimate':>9}  {'Exact':>9}  {'Delta%':>7}  "
        f"{'Window%':>8}  {'Projected':>10}  {'NearLim':>8}  Latency"
    )
    print(f"  {'-'*76}")

    for target, conv in zip(TARGET_SIZES, conversations):
        # Estimate (no client)
        est_counter = TokenCounter(client=None)
        est = est_counter.count_messages(conv, system=SYSTEM, model=MODEL)

        # Exact (with client, if available)
        t0 = time.perf_counter()
        exact = counter.count_messages(conv, system=SYSTEM, model=MODEL)
        latency_ms = (time.perf_counter() - t0) * 1_000

        delta_pct = (
            ((exact.total - est.total) / exact.total * 100.0)
            if exact.total > 0 else 0.0
        )

        ws = counter.track_window_usage(
            conv, max_context_tokens=WINDOW, system=SYSTEM, model=MODEL
        )

        exact_label = f"{exact.total:>9,}" if exact.exact else f"{'~' + str(exact.total):>9}"

        print(
            f"  {target:>7,}  {len(conv):>8}  "
            f"{est.total:>9,}  {exact_label}  {delta_pct:>+7.1f}%  "
            f"{ws.usage_pct:>7.2f}%  {ws.projected_tokens:>10,}  "
            f"{'YES' if ws.is_near_limit else 'no':>8}  {latency_ms:>7.1f}ms"
        )

    # ── cache demo ────────────────────────────────────────────────────────────
    print(f"\n  Cache entries after 5 unique calls: {counter.cache_hits}")
    t0 = time.perf_counter()
    _ = counter.count_messages(conversations[2], system=SYSTEM, model=MODEL)
    cache_ms = (time.perf_counter() - t0) * 1_000
    print(f"  Cache hit latency (same conv #3):   {cache_ms:.3f}ms")

    # ── cost estimates ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Estimated turn cost per conversation (output=500 tokens)")
    print(f"  {'-'*50}")
    for target, conv in zip(TARGET_SIZES, conversations):
        cost = counter.estimate_turn_cost(conv, model=MODEL, system=SYSTEM)
        print(f"  target ~{target:>6,} tokens  ->  ${cost:.6f} USD")

    # ── breakdown for largest conversation ────────────────────────────────────
    largest = conversations[-1]
    tc = counter.count_messages(largest, system=SYSTEM, model=MODEL)
    ws = counter.track_window_usage(largest, max_context_tokens=WINDOW, system=SYSTEM, model=MODEL)

    print(f"\n{sep}")
    print(f"  Detailed breakdown — largest conversation ({TARGET_SIZES[-1]:,} target tokens)")
    print(f"  {'-'*50}")
    print(f"  input_tokens   : {tc.input_tokens:,}")
    print(f"  system_tokens  : {tc.system_tokens:,}")
    print(f"  message_tokens : {tc.message_tokens:,}")
    print(f"  tool_tokens    : {tc.tool_tokens:,}")
    print(f"  total          : {tc.total:,}  ({'exact' if tc.exact else 'estimated'})")
    print(f"  window usage   : {ws.usage_pct:.2f}%  ({ws.used_tokens:,} / {ws.max_tokens:,})")
    print(f"  remaining      : {ws.remaining_tokens:,} tokens")
    print(f"  projected next : {ws.projected_tokens:,} tokens")
    print(f"  near limit     : {ws.is_near_limit}")
    print(f"  over limit     : {ws.is_over_limit}")
    print(sep)
