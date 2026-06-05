"""
Retry handler with per-tool budgets and exponential backoff.

Distinguishes retriable from non-retriable failures and never wastes attempts on
errors a retry cannot fix:

  Retriable      timeout, network error, rate limit (429), server error (500-503).
  Non-retriable  schema/validation error, auth error (401/403), not found (404).

Two limits guard every tool:
  max_retries_per_call -- attempts for a single execute() call (1 try + N retries).
  budget_per_window    -- total retries allowed for a tool across a rolling time
                          window; once exhausted, further calls fail fast with
                          BUDGET_EXHAUSTED until the window slides forward.

Backoff between attempts is exponential with jitter:
  delay = min(0.5 * 2**attempt + uniform(0, 0.5), 30.0) seconds.

Error classification is driven by exception type, an optional ``status_code``
attribute, and substring matching on the message. Standard library only.
"""
from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque


# ── defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_MAX_RETRIES   = 3
_DEFAULT_BUDGET        = 10
_DEFAULT_WINDOW_SEC    = 60.0

_BACKOFF_BASE_SEC      = 0.5
_BACKOFF_MAX_SEC       = 30.0
_JITTER_MAX_SEC        = 0.5


# ── error classification ──────────────────────────────────────────────────────

class ErrorType(str, Enum):
    """Classified failure categories for retry decisions."""
    TIMEOUT        = "TIMEOUT"
    NETWORK        = "NETWORK"
    RATE_LIMIT     = "RATE_LIMIT"      # HTTP 429
    SERVER_ERROR   = "SERVER_ERROR"    # HTTP 500-503
    VALIDATION     = "VALIDATION"      # schema / input validation
    AUTH           = "AUTH"            # HTTP 401 / 403
    NOT_FOUND      = "NOT_FOUND"       # HTTP 404
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    UNKNOWN        = "UNKNOWN"
    NONE           = "NONE"            # no error (success)


# Which classified errors are worth retrying
_RETRIABLE: frozenset[ErrorType] = frozenset({
    ErrorType.TIMEOUT,
    ErrorType.NETWORK,
    ErrorType.RATE_LIMIT,
    ErrorType.SERVER_ERROR,
})

# Status-code → ErrorType mapping
_STATUS_MAP: dict[int, ErrorType] = {
    401: ErrorType.AUTH,
    403: ErrorType.AUTH,
    404: ErrorType.NOT_FOUND,
    429: ErrorType.RATE_LIMIT,
    500: ErrorType.SERVER_ERROR,
    501: ErrorType.SERVER_ERROR,
    502: ErrorType.SERVER_ERROR,
    503: ErrorType.SERVER_ERROR,
}

# Substring markers per error type (checked against the exception message, lowercased)
_MESSAGE_MARKERS: list[tuple[ErrorType, tuple[str, ...]]] = [
    (ErrorType.TIMEOUT,    ("timeout", "timed out", "deadline exceeded")),
    (ErrorType.RATE_LIMIT, ("rate limit", "429", "too many requests", "throttle")),
    (ErrorType.AUTH,       ("401", "403", "unauthorized", "unauthorised",
                            "forbidden", "permission denied", "invalid api key")),
    (ErrorType.NOT_FOUND,  ("404", "not found", "no such")),
    (ErrorType.SERVER_ERROR, ("500", "502", "503", "internal server error",
                              "bad gateway", "service unavailable")),
    (ErrorType.NETWORK,    ("connection", "network", "unreachable", "refused",
                            "reset by peer", "dns")),
    (ErrorType.VALIDATION, ("validation", "invalid input", "schema",
                            "missing required", "field required")),
]


def classify_error(exc: BaseException) -> ErrorType:
    """Classify an exception into an ErrorType for retry decisioning.

    Resolution order:
      1. A ``status_code`` attribute on the exception (mapped via _STATUS_MAP).
      2. Built-in exception types (TimeoutError, ConnectionError, ValueError...).
      3. Substring markers in the exception message.

    Args:
        exc: The caught exception.

    Returns:
        The most specific ErrorType found, or ErrorType.UNKNOWN.
    """
    # 1. explicit HTTP status code attribute
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _STATUS_MAP:
        return _STATUS_MAP[status]

    # 2. standard exception types
    if isinstance(exc, TimeoutError):
        return ErrorType.TIMEOUT
    if isinstance(exc, ConnectionError):
        return ErrorType.NETWORK
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        # default for these is validation unless the message says otherwise
        msg_type = _classify_message(str(exc))
        return msg_type if msg_type is not ErrorType.UNKNOWN else ErrorType.VALIDATION

    # 3. message-based markers
    return _classify_message(str(exc))


def _classify_message(message: str) -> ErrorType:
    """Match the (lowercased) message against known markers, in priority order."""
    low = message.lower()
    for error_type, markers in _MESSAGE_MARKERS:
        if any(m in low for m in markers):
            return error_type
    return ErrorType.UNKNOWN


# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class RetryResult:
    """Outcome of RetryHandler.execute().

    Attributes:
        tool_name:        Tool that was executed.
        success:          True when the call eventually succeeded.
        result:           Handler return value on success (None otherwise).
        attempts_used:    Total attempts made (1 = succeeded first try).
        total_latency_ms: Wall-clock time including all backoff sleeps.
        error_type:       Classified failure (ErrorType.NONE on success).
        error_message:    Last error message (None on success).
        retries_used:     Retries consumed = attempts_used - 1.
    """
    tool_name:        str
    success:          bool
    result:           Any
    attempts_used:    int
    total_latency_ms: float
    error_type:       ErrorType
    error_message:    str | None = None
    retries_used:     int = 0


# ── per-tool statistics ───────────────────────────────────────────────────────

@dataclass
class _ToolStats:
    """Mutable per-tool counters maintained across execute() calls."""
    calls:                int = 0
    successes:            int = 0
    successes_first_try:  int = 0
    successes_after_retry: int = 0
    failures:             int = 0
    total_retries:        int = 0
    budget_blocks:        int = 0
    # Sliding window of retry timestamps (monotonic seconds)
    retry_window:         Deque[float] = field(default_factory=deque)


# ── retry handler ─────────────────────────────────────────────────────────────

class RetryHandler:
    """Execute tool calls with classification-aware retries and per-tool budgets.

    Usage::

        handler = RetryHandler(max_retries_per_call=3, budget_per_window=10)
        result  = handler.execute("web_search", search_fn, "query text")
        if result.success:
            use(result.result)
        print(handler.get_stats("web_search"))

    Args:
        max_retries_per_call: Retries for one execute() call (total tries = 1 + N).
        budget_per_window:    Total retries per tool allowed within the rolling
                              window. Exhausting it fails subsequent calls fast.
        window_seconds:       Length of the rolling retry-budget window.
        sleep_fn:             Injectable sleep (defaults to time.sleep); override
                              in tests to avoid real delays.
    """

    def __init__(
        self,
        max_retries_per_call: int                    = _DEFAULT_MAX_RETRIES,
        budget_per_window:    int                    = _DEFAULT_BUDGET,
        window_seconds:       float                  = _DEFAULT_WINDOW_SEC,
        sleep_fn:             Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_retries_per_call = max_retries_per_call
        self.budget_per_window    = budget_per_window
        self.window_seconds       = window_seconds
        self._sleep               = sleep_fn
        self._stats: dict[str, _ToolStats] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        tool_name: str,
        call_fn:   Callable[..., Any],
        *args:     Any,
        **kwargs:  Any,
    ) -> RetryResult:
        """Run call_fn with retries on retriable errors, bounded by budget.

        Algorithm per attempt:
          1. Call call_fn(*args, **kwargs); on success return immediately.
          2. On exception, classify it. Non-retriable -> stop and report.
          3. Retriable -> if per-call retries and the tool budget both allow it,
             sleep with exponential backoff and retry; else stop.

        Args:
            tool_name: Logical name of the tool (keys budget and stats).
            call_fn:   The callable to execute.
            *args:     Positional arguments forwarded to call_fn.
            **kwargs:  Keyword arguments forwarded to call_fn.

        Returns:
            A RetryResult describing the final outcome and attempt accounting.
        """
        stats = self._stats.setdefault(tool_name, _ToolStats())
        stats.calls += 1

        start_t        = time.perf_counter()
        attempts       = 0
        last_error_type = ErrorType.NONE
        last_error_msg: str | None = None

        while True:
            attempts += 1
            try:
                result = call_fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 -- classified, never re-raised
                last_error_type = classify_error(exc)
                last_error_msg  = f"{type(exc).__name__}: {exc}"
            else:
                # ── success ───────────────────────────────────────────────────
                stats.successes += 1
                if attempts == 1:
                    stats.successes_first_try += 1
                else:
                    stats.successes_after_retry += 1
                return RetryResult(
                    tool_name=tool_name, success=True, result=result,
                    attempts_used=attempts,
                    total_latency_ms=round((time.perf_counter() - start_t) * 1000, 2),
                    error_type=ErrorType.NONE, error_message=None,
                    retries_used=attempts - 1,
                )

            # ── failure: decide whether to retry ─────────────────────────────
            if last_error_type not in _RETRIABLE:
                # Non-retriable: stop immediately.
                stats.failures += 1
                return self._finish_failure(
                    tool_name, stats, attempts, start_t,
                    last_error_type, last_error_msg,
                )

            # Retriable, but have we exhausted per-call retries?
            if attempts > self.max_retries_per_call:
                stats.failures += 1
                return self._finish_failure(
                    tool_name, stats, attempts, start_t,
                    last_error_type, last_error_msg,
                )

            # Retriable and retries remain: check the tool's budget window.
            if not self._budget_available(stats):
                stats.budget_blocks += 1
                stats.failures      += 1
                return RetryResult(
                    tool_name=tool_name, success=False, result=None,
                    attempts_used=attempts,
                    total_latency_ms=round((time.perf_counter() - start_t) * 1000, 2),
                    error_type=ErrorType.BUDGET_EXHAUSTED,
                    error_message=(
                        f"Retry budget exhausted for {tool_name!r} "
                        f"({self.budget_per_window} retries / {self.window_seconds:.0f}s)"
                    ),
                    retries_used=attempts - 1,
                )

            # Consume one budget unit and back off before retrying.
            self._consume_budget(stats)
            stats.total_retries += 1
            self._sleep(self._backoff_delay(attempts))

    def get_stats(self, tool_name: str) -> dict[str, Any]:
        """Return aggregate retry statistics for one tool.

        Args:
            tool_name: The tool to report on.

        Returns:
            Dict with:
              calls                   -- total execute() calls.
              successes / failures    -- terminal outcomes.
              retry_rate              -- retries / calls.
              success_after_retry_pct -- of all successes, the % that needed >=1 retry.
              budget_used             -- retries currently inside the live window.
              budget_limit            -- budget_per_window.
              budget_usage_pct        -- budget_used / budget_limit * 100.
              budget_blocks           -- calls denied because the budget was empty.
        """
        stats = self._stats.get(tool_name)
        if stats is None:
            return {
                "calls": 0, "successes": 0, "failures": 0, "retry_rate": 0.0,
                "success_after_retry_pct": 0.0, "budget_used": 0,
                "budget_limit": self.budget_per_window, "budget_usage_pct": 0.0,
                "budget_blocks": 0,
            }

        self._prune_window(stats)
        used = len(stats.retry_window)
        return {
            "calls":                   stats.calls,
            "successes":               stats.successes,
            "failures":                stats.failures,
            "retry_rate":              round(stats.total_retries / stats.calls, 3) if stats.calls else 0.0,
            "success_after_retry_pct": round(
                stats.successes_after_retry / stats.successes * 100, 1
            ) if stats.successes else 0.0,
            "budget_used":             used,
            "budget_limit":            self.budget_per_window,
            "budget_usage_pct":        round(used / self.budget_per_window * 100, 1)
                                        if self.budget_per_window else 0.0,
            "budget_blocks":           stats.budget_blocks,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _finish_failure(
        self,
        tool_name:  str,
        stats:      _ToolStats,
        attempts:   int,
        start_t:    float,
        error_type: ErrorType,
        error_msg:  str | None,
    ) -> RetryResult:
        """Assemble a failed RetryResult (non-retriable or retries exhausted)."""
        return RetryResult(
            tool_name=tool_name, success=False, result=None,
            attempts_used=attempts,
            total_latency_ms=round((time.perf_counter() - start_t) * 1000, 2),
            error_type=error_type, error_message=error_msg,
            retries_used=attempts - 1,
        )

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter, capped at _BACKOFF_MAX_SEC.

        delay = min(base * 2**attempt + uniform(0, jitter_max), max).
        attempt is 1-based at the first retry decision.
        """
        raw = _BACKOFF_BASE_SEC * (2 ** attempt) + random.uniform(0, _JITTER_MAX_SEC)
        return min(raw, _BACKOFF_MAX_SEC)

    def _budget_available(self, stats: _ToolStats) -> bool:
        """True when the tool has at least one retry left in the live window."""
        self._prune_window(stats)
        return len(stats.retry_window) < self.budget_per_window

    def _consume_budget(self, stats: _ToolStats) -> None:
        """Record one retry timestamp in the tool's window."""
        stats.retry_window.append(time.monotonic())

    def _prune_window(self, stats: _ToolStats) -> None:
        """Drop retry timestamps that have fallen outside the rolling window."""
        cutoff = time.monotonic() - self.window_seconds
        while stats.retry_window and stats.retry_window[0] < cutoff:
            stats.retry_window.popleft()


# ── module-level demo helpers ─────────────────────────────────────────────────

class _HTTPError(Exception):
    """Test exception carrying an HTTP status_code attribute."""
    def __init__(self, status_code: int, message: str = "") -> None:
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


def _make_flaky(fail_times: int, error: BaseException, ok_value: Any) -> Callable[[], Any]:
    """Return a callable that raises `error` the first `fail_times` calls, then succeeds.

    Args:
        fail_times: Number of initial calls that raise.
        error:      Exception instance to raise during the failing calls.
        ok_value:   Value returned once the failures are exhausted.

    Returns:
        A zero-argument callable with the described behaviour.
    """
    state = {"n": 0}

    def _fn() -> Any:
        state["n"] += 1
        if state["n"] <= fail_times:
            raise error
        return ok_value

    return _fn


def _always_fail(error: BaseException) -> Callable[[], Any]:
    """Return a callable that always raises `error`."""
    def _fn() -> Any:
        raise error
    return _fn


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(11)

    # Fast, deterministic sleep: record requested delays instead of waiting.
    backoff_log: list[float] = []
    def _fake_sleep(seconds: float) -> None:
        backoff_log.append(round(seconds, 3))

    handler = RetryHandler(
        max_retries_per_call=3,
        budget_per_window=4,        # small budget so exhaustion is reachable
        window_seconds=60.0,
        sleep_fn=_fake_sleep,
    )

    sep = "=" * 82
    print(sep)
    print("  RetryHandler demo  |  max_retries=3  |  budget=4/60s  |  backoff=0.5*2^n+jitter")
    print(sep)

    # ── 10 scenarios ──────────────────────────────────────────────────────────
    # Each: (label, tool, call_fn, expected note)
    scenarios: list[tuple[str, str, Callable[[], Any]]] = [
        # 1. success first try
        ("ok-first-try",       "web_search",
         _make_flaky(0, _HTTPError(503), {"hits": 3})),
        # 2. retriable timeout, recovers on 2nd attempt
        ("timeout-recover",    "web_search",
         _make_flaky(1, TimeoutError("read timed out"), {"hits": 5})),
        # 3. retriable 429 rate limit, recovers on 3rd attempt
        ("ratelimit-recover",  "web_search",
         _make_flaky(2, _HTTPError(429, "rate limit exceeded"), {"hits": 2})),
        # 4. non-retriable: 401 auth -> stop immediately
        ("auth-no-retry",      "send_email",
         _always_fail(_HTTPError(401, "invalid api key"))),
        # 5. non-retriable: validation error -> stop immediately
        ("validation-no-retry","send_email",
         _always_fail(ValueError("field required: recipient"))),
        # 6. non-retriable: 404 not found -> stop immediately
        ("notfound-no-retry",  "read_file",
         _always_fail(_HTTPError(404, "file not found"))),
        # 7. retriable network error that never recovers -> exhaust per-call retries
        ("network-exhaust",    "database_query",
         _always_fail(ConnectionError("connection refused"))),
        # 8. retriable server error, never recovers -> consumes budget
        ("server-err-1",       "database_query",
         _always_fail(_HTTPError(500, "internal server error"))),
        # 9. another failing call on same tool -> should hit BUDGET_EXHAUSTED
        ("server-err-2",       "database_query",
         _always_fail(_HTTPError(503, "service unavailable"))),
        # 10. success first try on a fresh tool
        ("ok-fresh-tool",      "translate",
         _make_flaky(0, _HTTPError(500), {"text": "ciao"})),
    ]

    print(
        f"\n  {'#':<3} {'Scenario':<20} {'Tool':<15} {'Result':<8} "
        f"{'Attempts':>8} {'Retries':>7} {'ErrorType':<16} Latency"
    )
    print(f"  {'-'*92}")

    for i, (label, tool, fn) in enumerate(scenarios, 1):
        before = len(backoff_log)
        result = handler.execute(tool, fn)
        delays = backoff_log[before:]

        res_str = "OK" if result.success else "FAIL"
        delays_str = f"  backoff={delays}" if delays else ""
        print(
            f"  {i:<3} {label:<20} {tool:<15} {res_str:<8} "
            f"{result.attempts_used:>8} {result.retries_used:>7} "
            f"{result.error_type.value:<16} {result.total_latency_ms:>6.1f}ms{delays_str}"
        )

    # ── per-tool stats ──────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  PER-TOOL STATISTICS")
    print(sep)
    print(
        f"  {'Tool':<15} {'Calls':>5} {'OK':>3} {'Fail':>4} "
        f"{'RetryRate':>9} {'OK-after-retry':>15} {'BudgetUse':>10} {'Blocks':>6}"
    )
    print(f"  {'-'*78}")

    for tool in ["web_search", "send_email", "read_file", "database_query", "translate"]:
        s = handler.get_stats(tool)
        print(
            f"  {tool:<15} {s['calls']:>5} {s['successes']:>3} {s['failures']:>4} "
            f"{s['retry_rate']:>9.2f} {s['success_after_retry_pct']:>14.1f}% "
            f"{str(s['budget_used']) + '/' + str(s['budget_limit']):>10} {s['budget_blocks']:>6}"
        )

    # ── highlight the budget-exhaustion case ────────────────────────────────────
    print(f"\n{sep}")
    print("  BUDGET ANALYSIS -- database_query")
    print(f"  {'-'*50}")
    dbs = handler.get_stats("database_query")
    print(f"  3 failing calls on database_query, budget = 4 retries / 60s:")
    print(f"    - server-err-1 (500): retriable, consumed retries until budget/per-call limit")
    print(f"    - server-err-2 (503): budget likely exhausted -> fast BUDGET_EXHAUSTED")
    print(f"  Budget usage now: {dbs['budget_used']}/{dbs['budget_limit']} "
          f"({dbs['budget_usage_pct']:.0f}%)  |  budget blocks: {dbs['budget_blocks']}")
    print(f"  Total backoff sleeps recorded across demo: {len(backoff_log)}")
    print(sep)
