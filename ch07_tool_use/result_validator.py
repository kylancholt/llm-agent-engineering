"""
Three-level result validator: guard tool outputs before they enter the context.

A tool result that reaches the model's context becomes "ground truth" for the
rest of the run, so a malformed or implausible result poisons everything
downstream. This validator screens every result through three independent
levels and produces a sanitized copy safe to inject:

  Level 1 -- Structure:    Pydantic validation against the tool's output schema.
             Missing fields or wrong types fail here (schema_errors).
  Level 2 -- Range:        Numeric fields are checked against per-tool bounds
             (range_violations). Bounds are configured via RangeRule entries.
  Level 3 -- Plausibility: Heuristic smell-tests that schemas cannot express:
             empty search results, suspiciously short/long strings, the same
             value repeated across a list, all-zero numeric payloads.

Sanitization always runs (even on failure) and produces sanitized_result:
  - extra fields not in the schema are dropped,
  - strings longer than _MAX_STRING_LEN are truncated with an ellipsis marker,
  - text is normalised to NFC Unicode and stripped of control characters.

Each validate() call targets < 2 ms of overhead; the measured latency is
returned on the ValidationResult so callers can enforce the budget.

Pydantic for schema validation; otherwise standard library only.
"""
from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError


# ── tuning constants ──────────────────────────────────────────────────────────

_MAX_STRING_LEN          = 10_000   # strings longer than this are truncated
_TRUNCATION_MARKER       = "...[truncated]"
_MIN_PLAUSIBLE_STR_LEN   = 2        # non-trivial strings should exceed this
_REPEAT_RATIO_THRESHOLD  = 0.6      # >60% identical list items is suspicious
_OVERHEAD_BUDGET_MS      = 2.0      # target per-validation overhead


# ── configuration types ───────────────────────────────────────────────────────

@dataclass
class RangeRule:
    """Numeric bound for one field path within a tool result.

    Attributes:
        field:     Dotted field path (e.g. "count" or "stats.total").
        min_value: Inclusive lower bound (None = unbounded below).
        max_value: Inclusive upper bound (None = unbounded above).
    """
    field:     str
    min_value: float | None = None
    max_value: float | None = None


@dataclass
class PlausibilityConfig:
    """Per-tool plausibility expectations.

    Attributes:
        non_empty_fields:  List/str fields that should not be empty (e.g. search
                           results). An empty value raises a plausibility flag.
        min_string_length: Minimum length for string fields before flagging.
        check_repeats:     When True, flag lists whose items are mostly identical.
        check_all_zero:    When True, flag numeric lists that are entirely zero.
    """
    non_empty_fields:  list[str] = field(default_factory=list)
    min_string_length: int       = _MIN_PLAUSIBLE_STR_LEN
    check_repeats:     bool       = True
    check_all_zero:    bool       = True


# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Outcome of ResultValidator.validate().

    Attributes:
        tool_name:         The tool whose result was validated.
        is_valid:          True when structure and range both pass. Plausibility
                           flags are warnings and do NOT by themselves fail a result.
        schema_errors:     Level 1 failures (field path: message).
        range_violations:  Level 2 failures (human-readable strings).
        plausibility_flags: Level 3 warnings (human-readable strings).
        sanitized_result:  Cleaned result dict, safe to inject (None if structure
                           failed so badly nothing could be salvaged).
        latency_ms:        Wall-clock validation overhead.
    """
    tool_name:          str
    is_valid:           bool
    schema_errors:      list[str]
    range_violations:   list[str]
    plausibility_flags: list[str]
    sanitized_result:   dict[str, Any] | None
    latency_ms:         float


# ── validator ─────────────────────────────────────────────────────────────────

class ResultValidator:
    """Validate and sanitize tool results across structure, range, plausibility.

    Range and plausibility rules are registered per tool name; tools without
    rules skip those levels (structure validation always runs).

    Usage::

        validator = ResultValidator()
        validator.configure(
            "web_search",
            range_rules=[RangeRule("count", min_value=0, max_value=1000)],
            plausibility=PlausibilityConfig(non_empty_fields=["results"]),
        )
        result = validator.validate("web_search", raw_dict, WebSearchOutput)
        if result.is_valid:
            inject(result.sanitized_result)

    Args:
        max_string_length: Strings longer than this are truncated during
                           sanitization (default 10_000).
    """

    def __init__(self, max_string_length: int = _MAX_STRING_LEN) -> None:
        self.max_string_length = max_string_length
        self._range_rules:   dict[str, list[RangeRule]]    = {}
        self._plausibility:  dict[str, PlausibilityConfig] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(
        self,
        tool_name:    str,
        range_rules:  list[RangeRule] | None      = None,
        plausibility: PlausibilityConfig | None   = None,
    ) -> None:
        """Register range and/or plausibility rules for a tool.

        Args:
            tool_name:    Tool these rules apply to.
            range_rules:  Numeric bounds for Level 2 (optional).
            plausibility: Plausibility expectations for Level 3 (optional).
        """
        if range_rules is not None:
            self._range_rules[tool_name] = range_rules
        if plausibility is not None:
            self._plausibility[tool_name] = plausibility

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        tool_name:       str,
        result:          Any,
        expected_schema: type[BaseModel],
    ) -> ValidationResult:
        """Validate a tool result through all three levels and sanitize it.

        Level 1 (structure) and Level 2 (range) determine ``is_valid``. Level 3
        (plausibility) only produces warning flags; an implausible-but-structurally
        valid result is still returned as valid, with flags attached so the caller
        can decide.

        Args:
            tool_name:       Name of the tool that produced the result.
            result:          Raw result (dict or Pydantic model instance).
            expected_schema: Pydantic model the result must satisfy.

        Returns:
            A ValidationResult with per-level findings, a sanitized copy, and the
            measured validation latency.
        """
        start = time.perf_counter()

        schema_errors:      list[str] = []
        range_violations:   list[str] = []
        plausibility_flags: list[str] = []

        # Normalise the input into a plain dict for processing.
        raw = result.model_dump() if isinstance(result, BaseModel) else result
        if not isinstance(raw, dict):
            latency = (time.perf_counter() - start) * 1000
            return ValidationResult(
                tool_name=tool_name, is_valid=False,
                schema_errors=[f"result is {type(raw).__name__}, expected object/dict"],
                range_violations=[], plausibility_flags=[],
                sanitized_result=None, latency_ms=round(latency, 4),
            )

        # ── Level 1: structure ────────────────────────────────────────────────
        validated_model: BaseModel | None = None
        try:
            validated_model = expected_schema.model_validate(raw)
        except ValidationError as exc:
            schema_errors = [
                f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}"
                for e in exc.errors()
            ]

        # Build the dict we will sanitize: the schema-coerced one if structure
        # passed (drops extras, applies defaults), else the raw dict best-effort.
        base_dict = validated_model.model_dump() if validated_model else raw

        # ── Level 2: range (only meaningful if structure passed) ──────────────
        if validated_model is not None:
            range_violations = self._check_ranges(tool_name, base_dict)

        # ── Level 3: plausibility (warnings only) ─────────────────────────────
        plausibility_flags = self._check_plausibility(tool_name, base_dict)

        # ── Sanitization (always) ─────────────────────────────────────────────
        sanitized = self._sanitize(base_dict)

        is_valid = (not schema_errors) and (not range_violations)
        latency  = (time.perf_counter() - start) * 1000

        return ValidationResult(
            tool_name=          tool_name,
            is_valid=           is_valid,
            schema_errors=      schema_errors,
            range_violations=   range_violations,
            plausibility_flags= plausibility_flags,
            sanitized_result=   sanitized,
            latency_ms=         round(latency, 4),
        )

    # ------------------------------------------------------------------
    # Level 2: range
    # ------------------------------------------------------------------

    def _check_ranges(self, tool_name: str, data: dict[str, Any]) -> list[str]:
        """Check each configured numeric field against its bounds."""
        violations: list[str] = []
        for rule in self._range_rules.get(tool_name, []):
            value = _resolve_path(data, rule.field)
            if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue  # field absent or non-numeric: not a range concern
            if rule.min_value is not None and value < rule.min_value:
                violations.append(
                    f"{rule.field}={value} below min {rule.min_value}"
                )
            if rule.max_value is not None and value > rule.max_value:
                violations.append(
                    f"{rule.field}={value} above max {rule.max_value}"
                )
        return violations

    # ------------------------------------------------------------------
    # Level 3: plausibility
    # ------------------------------------------------------------------

    def _check_plausibility(self, tool_name: str, data: dict[str, Any]) -> list[str]:
        """Run heuristic smell-tests; return human-readable warning flags."""
        cfg   = self._plausibility.get(tool_name)
        flags: list[str] = []

        if cfg is None:
            return flags

        # Empty fields that should carry content
        for fname in cfg.non_empty_fields:
            value = _resolve_path(data, fname)
            if value is None:
                continue
            if isinstance(value, (list, str, dict)) and len(value) == 0:
                flags.append(f"'{fname}' is empty (expected content)")

        # Per-field heuristics
        for key, value in data.items():
            if isinstance(value, str):
                stripped = value.strip()
                if 0 < len(stripped) < cfg.min_string_length:
                    flags.append(
                        f"'{key}' suspiciously short ({len(stripped)} chars)"
                    )
                if len(value) > self.max_string_length:
                    flags.append(
                        f"'{key}' exceeds {self.max_string_length} chars "
                        "(will be truncated)"
                    )
            elif isinstance(value, list) and value:
                if cfg.check_repeats:
                    flag = _check_repeats(key, value)
                    if flag:
                        flags.append(flag)
                if cfg.check_all_zero:
                    flag = _check_all_zero(key, value)
                    if flag:
                        flags.append(flag)

        return flags

    # ------------------------------------------------------------------
    # Sanitization
    # ------------------------------------------------------------------

    def _sanitize(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return a cleaned deep copy: truncate strings, normalise text.

        Note: extra-field removal already happened when structure validation
        coerced the dict through the schema (model_dump drops unknown keys).
        For results that failed structure we still normalise what is present.
        """
        return {k: self._sanitize_value(v) for k, v in data.items()}

    def _sanitize_value(self, value: Any) -> Any:
        """Recursively sanitize a single value."""
        if isinstance(value, str):
            return self._sanitize_string(value)
        if isinstance(value, list):
            return [self._sanitize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._sanitize_value(v) for k, v in value.items()}
        return value

    def _sanitize_string(self, text: str) -> str:
        """Normalise encoding, strip control chars, truncate to the max length."""
        # Unicode NFC normalisation
        text = unicodedata.normalize("NFC", text)
        # Strip control characters except tab/newline/carriage-return
        text = "".join(
            ch for ch in text
            if ch in ("\t", "\n", "\r") or unicodedata.category(ch)[0] != "C"
        )
        # Truncate over-long strings
        if len(text) > self.max_string_length:
            keep = self.max_string_length - len(_TRUNCATION_MARKER)
            text = text[:keep] + _TRUNCATION_MARKER
        return text


# ── module helpers ────────────────────────────────────────────────────────────

def _resolve_path(data: dict[str, Any], path: str) -> Any:
    """Resolve a dotted field path within a nested dict; None if absent."""
    current: Any = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _check_repeats(key: str, items: list[Any]) -> str | None:
    """Flag a list whose items are mostly identical (low diversity)."""
    if len(items) < 3:
        return None
    try:
        hashable = [repr(i) for i in items]
    except Exception:  # noqa: BLE001
        return None
    most_common = max(hashable.count(h) for h in set(hashable))
    ratio = most_common / len(items)
    if ratio >= _REPEAT_RATIO_THRESHOLD:
        return (
            f"'{key}' has low diversity: {most_common}/{len(items)} "
            f"items identical ({ratio*100:.0f}%)"
        )
    return None


def _check_all_zero(key: str, items: list[Any]) -> str | None:
    """Flag a numeric list that is entirely zero (often a broken computation)."""
    numeric = [i for i in items if isinstance(i, (int, float)) and not isinstance(i, bool)]
    if numeric and len(numeric) == len(items) and all(n == 0 for n in numeric):
        return f"'{key}' is all zeros ({len(items)} items) -- possible broken result"
    return None


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pydantic import Field

    # ── schemas for the demo tools ────────────────────────────────────────────
    class WebSearchOutput(BaseModel):
        results: list[str]
        count:   int

    class WeatherOutput(BaseModel):
        city:        str
        temperature: float
        humidity:    int

    class SentimentOutput(BaseModel):
        scores: list[float]
        label:  str

    # ── configure validator ───────────────────────────────────────────────────
    validator = ResultValidator()

    validator.configure(
        "web_search",
        range_rules=[RangeRule("count", min_value=0, max_value=1000)],
        plausibility=PlausibilityConfig(non_empty_fields=["results"]),
    )
    validator.configure(
        "weather",
        range_rules=[
            RangeRule("temperature", min_value=-90.0, max_value=60.0),  # plausible Earth temps
            RangeRule("humidity",    min_value=0,     max_value=100),
        ],
        plausibility=PlausibilityConfig(min_string_length=2),
    )
    validator.configure(
        "sentiment",
        plausibility=PlausibilityConfig(non_empty_fields=["scores"], check_all_zero=True),
    )

    # ── 5 scenarios ────────────────────────────────────────────────────────────
    # (label, tool, raw_result, schema, expectation)
    scenarios: list[tuple[str, str, Any, type[BaseModel], str]] = [
        (
            "valid-web-search",
            "web_search",
            {"results": ["Anthropic builds Claude", "Claude is an AI assistant",
                         "Constitutional AI paper"], "count": 3, "extra_field": "DROP ME"},
            WebSearchOutput,
            "VALID (extra_field dropped during sanitization)",
        ),
        (
            "valid-weather",
            "weather",
            {"city": "Rome", "temperature": 24.5, "humidity": 55},
            WeatherOutput,
            "VALID (all within range)",
        ),
        (
            "schema-error",
            "web_search",
            {"results": "not-a-list", "count": "three"},   # wrong types
            WebSearchOutput,
            "SCHEMA ERROR (results must be list, count must be int)",
        ),
        (
            "range-violation",
            "weather",
            {"city": "Nowhere", "temperature": 250.0, "humidity": 55},  # 250C impossible
            WeatherOutput,
            "RANGE VIOLATION (temperature 250 > 60)",
        ),
        (
            "plausibility-flag",
            "web_search",
            {"results": [], "count": 0},   # empty search results
            WebSearchOutput,
            "PLAUSIBILITY FLAG (empty results -- still 'valid' structurally)",
        ),
    ]

    sep = "=" * 84
    print(sep)
    print("  ResultValidator demo  |  3 levels: structure / range / plausibility")
    print(f"  Overhead budget: < {_OVERHEAD_BUDGET_MS} ms per validation")
    print(sep)

    latencies: list[float] = []

    for i, (label, tool, raw, schema, expectation) in enumerate(scenarios, 1):
        result = validator.validate(tool, raw, schema)
        latencies.append(result.latency_ms)

        print(f"\n  [{i}] {label}   (tool={tool})")
        print(f"      expectation : {expectation}")
        print(f"      is_valid    : {result.is_valid}")
        if result.schema_errors:
            print(f"      schema      : {result.schema_errors}")
        if result.range_violations:
            print(f"      range       : {result.range_violations}")
        if result.plausibility_flags:
            print(f"      plausibility: {result.plausibility_flags}")
        print(f"      sanitized   : {result.sanitized_result}")
        budget = "OK" if result.latency_ms < _OVERHEAD_BUDGET_MS else "OVER"
        print(f"      latency     : {result.latency_ms:.4f} ms  [{budget}]")

    # ── overhead summary ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  LATENCY SUMMARY")
    print(f"  {'-'*50}")
    avg = sum(latencies) / len(latencies)
    mx  = max(latencies)
    print(f"  Validations   : {len(latencies)}")
    print(f"  Mean latency  : {avg:.4f} ms")
    print(f"  Max  latency  : {mx:.4f} ms")
    print(f"  Budget        : < {_OVERHEAD_BUDGET_MS} ms")
    verdict = "PASS -- all under budget" if mx < _OVERHEAD_BUDGET_MS else "WARN -- some over budget"
    print(f"  Verdict       : {verdict}")

    # ── truncation demo ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  SANITIZATION: long-string truncation + control-char stripping")
    print(f"  {'-'*60}")

    class DocOutput(BaseModel):
        content: str

    long_text = "A" * 15_000 + "\x00\x07control" + "café́"   # 15k + nulls + combining
    long_result = validator.validate("doc_reader", {"content": long_text}, DocOutput)
    sanitized_content = long_result.sanitized_result["content"]
    print(f"  Original length : {len(long_text):,} chars (with NUL + control bytes)")
    print(f"  Sanitized length: {len(sanitized_content):,} chars")
    print(f"  Ends with       : ...{sanitized_content[-20:]!r}")
    print(f"  Control chars removed, NFC-normalised, truncated to {_MAX_STRING_LEN:,}")
    print(sep)
