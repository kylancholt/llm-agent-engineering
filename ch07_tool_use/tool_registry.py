"""
Type-safe tool registry with health checks for LLM agents.

A tool is registered with a Pydantic input schema, a Pydantic output schema, and
a handler callable. The registry enforces the contract on every call: inputs are
validated before the handler runs, outputs are validated after, and the handler
is bounded by a per-tool timeout. Failures are reported, never raised into the
agent loop.

Each ToolDefinition also carries operational metadata:
  retry_budget -- how many times the caller may retry this tool.
  timeout_ms   -- hard wall-clock limit; exceeding it is a timeout failure.
  sla_ms       -- soft latency target; exceeding it is a WARN, not a failure.

validate_all() runs a health check on every tool by invoking it with a minimal
valid input synthesised from the schema, then prints a status table:
  schema   = OK | FAIL   (Pydantic schemas usable as JSON Schema)
  health   = OK | WARN | FAIL   (OK within SLA, WARN over SLA, FAIL on error/timeout)
  latency  = measured call latency

get_anthropic_tools() exports the registry in the Anthropic Messages API
tool_use format.

Standard library plus Pydantic only; no Anthropic client required.
"""
from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, ValidationError


# ── defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_RETRY_BUDGET = 3
_DEFAULT_TIMEOUT_MS   = 5000
_DEFAULT_SLA_MS       = 800
_HEALTH_CHECK_RUNS    = 5    # repeated calls per tool to derive a p95 latency


# ── status enums ──────────────────────────────────────────────────────────────

class HealthStatus(str, Enum):
    """Health-check verdict for a single tool."""
    OK   = "OK"      # responded within sla_ms
    WARN = "WARN"    # responded, but slower than sla_ms (under timeout_ms)
    FAIL = "FAIL"    # raised, timed out, or returned an invalid output


class SchemaStatus(str, Enum):
    """Schema-validity verdict for a single tool."""
    OK   = "OK"
    FAIL = "FAIL"


# ── tool definition and results ───────────────────────────────────────────────

@dataclass
class ToolDefinition:
    """A fully-specified, validated tool entry in the registry.

    Attributes:
        name:          Unique tool identifier (snake_case by convention).
        description:   Natural-language description shown to the model.
        input_schema:  Pydantic model validating tool input.
        output_schema: Pydantic model validating tool output.
        handler:       Callable taking the validated input model and returning a
                       dict (or output-model instance) to validate as output.
        retry_budget:  Max retries the caller may attempt on failure.
        timeout_ms:    Hard wall-clock limit per call (milliseconds).
        sla_ms:        Soft latency target; exceeding it yields a WARN.
    """
    name:          str
    description:   str
    input_schema:  type[BaseModel]
    output_schema: type[BaseModel]
    handler:       Callable[[BaseModel], Any]
    retry_budget:  int = _DEFAULT_RETRY_BUDGET
    timeout_ms:    int = _DEFAULT_TIMEOUT_MS
    sla_ms:        int = _DEFAULT_SLA_MS


@dataclass
class ToolCallResult:
    """Outcome of ToolRegistry.call().

    Attributes:
        tool_name:    The tool that was invoked.
        success:      True when input, handler, and output all validated.
        output:       Validated output dict (None on failure).
        error:        Error message (None on success).
        latency_ms:   Wall-clock latency of the handler call.
        timed_out:    True when the handler exceeded timeout_ms.
        sla_violated: True when latency exceeded sla_ms (even if successful).
    """
    tool_name:    str
    success:      bool
    output:       dict[str, Any] | None
    error:        str | None
    latency_ms:   float
    timed_out:    bool = False
    sla_violated: bool = False


@dataclass
class ToolHealth:
    """Per-tool result row produced by validate_all()."""
    name:           str
    schema_status:  SchemaStatus
    health_status:  HealthStatus
    latency_p95_ms: float
    detail:         str


@dataclass
class RegistryReport:
    """Aggregate output of ToolRegistry.validate_all().

    Attributes:
        tools:        Per-tool health rows.
        total:        Number of tools evaluated.
        schema_ok:    Tools whose schemas validated.
        health_ok:    Tools with HealthStatus.OK.
        health_warn:  Tools with HealthStatus.WARN.
        health_fail:  Tools with HealthStatus.FAIL.
        all_healthy:  True when no tool is FAIL and all schemas are OK.
    """
    tools:       list[ToolHealth]
    total:       int
    schema_ok:   int
    health_ok:   int
    health_warn: int
    health_fail: int
    all_healthy: bool


# ── registry ──────────────────────────────────────────────────────────────────

class ToolRegistry:
    """Type-safe registry of agent tools with validation and health checks.

    Register tools either imperatively via ``register`` or declaratively with the
    ``@registry.tool(...)`` decorator. Every call through ``call`` validates the
    input against the tool's input schema, enforces the timeout, then validates
    the handler's return value against the output schema.

    Usage::

        registry = ToolRegistry()

        class In(BaseModel):
            query: str
        class Out(BaseModel):
            results: list[str]

        @registry.tool(name="web_search", description="Search the web",
                       input_schema=In, output_schema=Out, sla_ms=500)
        def web_search(inp: In) -> dict:
            return {"results": [f"hit for {inp.query}"]}

        report = registry.validate_all()
        result = registry.call("web_search", {"query": "anthropic"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool_def: ToolDefinition) -> ToolDefinition:
        """Register a fully-built ToolDefinition.

        Args:
            tool_def: The tool to add.

        Returns:
            The same ToolDefinition (for chaining).

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool_def.name in self._tools:
            raise ValueError(f"Tool {tool_def.name!r} is already registered.")
        self._tools[tool_def.name] = tool_def
        return tool_def

    def tool(
        self,
        name:          str,
        description:   str,
        input_schema:  type[BaseModel],
        output_schema: type[BaseModel],
        retry_budget:  int = _DEFAULT_RETRY_BUDGET,
        timeout_ms:    int = _DEFAULT_TIMEOUT_MS,
        sla_ms:        int = _DEFAULT_SLA_MS,
    ) -> Callable[[Callable[[BaseModel], Any]], Callable[[BaseModel], Any]]:
        """Decorator that registers the wrapped function as a tool handler.

        Args:
            name:          Unique tool identifier.
            description:   Natural-language description for the model.
            input_schema:  Pydantic model validating the tool input.
            output_schema: Pydantic model validating the tool output.
            retry_budget:  Max retries on failure.
            timeout_ms:    Hard per-call wall-clock limit.
            sla_ms:        Soft latency target (WARN when exceeded).

        Returns:
            A decorator that registers the handler and returns it unchanged.
        """
        def decorator(handler: Callable[[BaseModel], Any]) -> Callable[[BaseModel], Any]:
            self.register(ToolDefinition(
                name=          name,
                description=   description,
                input_schema=  input_schema,
                output_schema= output_schema,
                handler=       handler,
                retry_budget=  retry_budget,
                timeout_ms=    timeout_ms,
                sla_ms=        sla_ms,
            ))
            return handler
        return decorator

    # ------------------------------------------------------------------
    # Lookup / export
    # ------------------------------------------------------------------

    def get(self, tool_name: str) -> ToolDefinition | None:
        """Return the named ToolDefinition, or None if not registered."""
        return self._tools.get(tool_name)

    def names(self) -> list[str]:
        """Return all registered tool names in insertion order."""
        return list(self._tools.keys())

    def get_anthropic_tools(self) -> list[dict[str, Any]]:
        """Export all tools in the Anthropic Messages API tool_use format.

        Returns:
            A list of dicts, each with ``name``, ``description``, and
            ``input_schema`` (JSON Schema derived from the Pydantic model).
        """
        tools: list[dict[str, Any]] = []
        for td in self._tools.values():
            tools.append({
                "name":        td.name,
                "description": td.description,
                "input_schema": _to_anthropic_schema(td.input_schema),
            })
        return tools

    # ------------------------------------------------------------------
    # Calling
    # ------------------------------------------------------------------

    def call(self, tool_name: str, tool_input: dict[str, Any]) -> ToolCallResult:
        """Invoke a tool with full input/output validation and timeout.

        Pipeline:
          1. Resolve the tool (unknown name -> failure result).
          2. Validate ``tool_input`` against the input schema.
          3. Run the handler under a timeout watchdog, measuring latency.
          4. Validate the handler's return value against the output schema.

        No exception escapes: every failure is captured in the returned
        ToolCallResult so the agent loop can decide how to recover.

        Args:
            tool_name:  Name of the registered tool.
            tool_input: Raw input dict from the model.

        Returns:
            A ToolCallResult describing success or the specific failure.
        """
        td = self._tools.get(tool_name)
        if td is None:
            return ToolCallResult(
                tool_name=tool_name, success=False, output=None,
                error=f"Unknown tool {tool_name!r}", latency_ms=0.0,
            )

        # ── 1. validate input ─────────────────────────────────────────────────
        try:
            validated_in = td.input_schema.model_validate(tool_input)
        except ValidationError as exc:
            return ToolCallResult(
                tool_name=tool_name, success=False, output=None,
                error=f"Input validation failed: {_fmt_errors(exc)}",
                latency_ms=0.0,
            )

        # ── 2. run handler under timeout ──────────────────────────────────────
        raw_output, latency_ms, timed_out, handler_err = _run_with_timeout(
            td.handler, validated_in, td.timeout_ms
        )
        if timed_out:
            return ToolCallResult(
                tool_name=tool_name, success=False, output=None,
                error=f"Handler exceeded timeout of {td.timeout_ms}ms",
                latency_ms=latency_ms, timed_out=True,
            )
        if handler_err is not None:
            return ToolCallResult(
                tool_name=tool_name, success=False, output=None,
                error=f"Handler raised: {handler_err}", latency_ms=latency_ms,
            )

        # ── 3. validate output ────────────────────────────────────────────────
        try:
            payload = (
                raw_output.model_dump()
                if isinstance(raw_output, BaseModel)
                else raw_output
            )
            validated_out = td.output_schema.model_validate(payload)
        except ValidationError as exc:
            return ToolCallResult(
                tool_name=tool_name, success=False, output=None,
                error=f"Output validation failed: {_fmt_errors(exc)}",
                latency_ms=latency_ms,
            )

        return ToolCallResult(
            tool_name=tool_name,
            success=True,
            output=validated_out.model_dump(),
            error=None,
            latency_ms=round(latency_ms, 2),
            sla_violated=latency_ms > td.sla_ms,
        )

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def validate_all(self, verbose: bool = True) -> RegistryReport:
        """Validate schemas and run a health check on every registered tool.

        For each tool: confirm its schemas serialise to JSON Schema, synthesise a
        minimal valid input, call the tool _HEALTH_CHECK_RUNS times, and derive a
        p95 latency. The verdict is OK (within SLA), WARN (over SLA, under
        timeout), or FAIL (error / timeout / invalid output).

        Args:
            verbose: When True, print the formatted status table.

        Returns:
            A RegistryReport with per-tool rows and aggregate counts.
        """
        rows: list[ToolHealth] = []

        for td in self._tools.values():
            schema_status, schema_detail = self._check_schema(td)
            if schema_status is SchemaStatus.FAIL:
                rows.append(ToolHealth(
                    name=td.name, schema_status=schema_status,
                    health_status=HealthStatus.FAIL, latency_p95_ms=0.0,
                    detail=schema_detail,
                ))
                continue

            health_status, p95, detail = self._health_check(td)
            rows.append(ToolHealth(
                name=td.name, schema_status=schema_status,
                health_status=health_status, latency_p95_ms=p95, detail=detail,
            ))

        report = RegistryReport(
            tools=       rows,
            total=       len(rows),
            schema_ok=   sum(1 for r in rows if r.schema_status is SchemaStatus.OK),
            health_ok=   sum(1 for r in rows if r.health_status is HealthStatus.OK),
            health_warn= sum(1 for r in rows if r.health_status is HealthStatus.WARN),
            health_fail= sum(1 for r in rows if r.health_status is HealthStatus.FAIL),
            all_healthy= all(
                r.schema_status is SchemaStatus.OK and r.health_status is not HealthStatus.FAIL
                for r in rows
            ),
        )

        if verbose:
            self._print_report(report)
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_schema(td: ToolDefinition) -> tuple[SchemaStatus, str]:
        """Confirm both schemas serialise to JSON Schema (Anthropic-compatible)."""
        try:
            in_schema = td.input_schema.model_json_schema()
            td.output_schema.model_json_schema()
            if in_schema.get("type") != "object":
                return (SchemaStatus.FAIL, "input schema is not an object")
            return (SchemaStatus.OK, "")
        except Exception as exc:  # noqa: BLE001
            return (SchemaStatus.FAIL, f"schema error: {exc}")

    def _health_check(self, td: ToolDefinition) -> tuple[HealthStatus, float, str]:
        """Call the tool with a synthetic minimal input and measure latency.

        Returns:
            (health_status, p95_latency_ms, detail)
        """
        try:
            minimal_input = _synthesise_input(td.input_schema)
        except Exception as exc:  # noqa: BLE001
            return (HealthStatus.FAIL, 0.0, f"could not synthesise input: {exc}")

        latencies: list[float] = []
        for _ in range(_HEALTH_CHECK_RUNS):
            result = self.call(td.name, minimal_input)
            if result.timed_out:
                return (HealthStatus.FAIL, result.latency_ms, "timeout")
            if not result.success:
                return (HealthStatus.FAIL, result.latency_ms, result.error or "call failed")
            latencies.append(result.latency_ms)

        p95 = _percentile(latencies, 95)
        if p95 > td.sla_ms:
            return (HealthStatus.WARN, p95, f"p95 {p95:.0f}ms over SLA {td.sla_ms}ms")
        return (HealthStatus.OK, p95, "")

    @staticmethod
    def _print_report(report: RegistryReport) -> None:
        """Print the registry health table."""
        sep = "=" * 78
        print(sep)
        print("  TOOL REGISTRY HEALTH REPORT")
        print(sep)
        print(
            f"  {'Tool':<18} {'Schema':>7} {'Health':>7} "
            f"{'Latency p95':>12}  Detail"
        )
        print(f"  {'-'*74}")
        for r in report.tools:
            lat = f"{r.latency_p95_ms:.1f} ms" if r.latency_p95_ms else "--"
            print(
                f"  {r.name:<18} {r.schema_status.value:>7} "
                f"{r.health_status.value:>7} {lat:>12}  {r.detail}"
            )
        print(f"  {'-'*74}")
        print(
            f"  Totals: {report.total} tools | "
            f"schema OK {report.schema_ok}/{report.total} | "
            f"health OK {report.health_ok} WARN {report.health_warn} "
            f"FAIL {report.health_fail}"
        )
        verdict = "ALL HEALTHY" if report.all_healthy else "ATTENTION REQUIRED"
        print(f"  Registry status: {verdict}")
        print(sep)


# ── module helpers ────────────────────────────────────────────────────────────

def _to_anthropic_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to the Anthropic input_schema JSON-Schema dict."""
    schema = model.model_json_schema()
    # Anthropic expects a plain object schema; strip the model title.
    schema.pop("title", None)
    return {
        "type":       "object",
        "properties": schema.get("properties", {}),
        "required":   schema.get("required", []),
    }


def _synthesise_input(model: type[BaseModel]) -> dict[str, Any]:
    """Build a minimal valid input dict for a Pydantic model.

    Provides a placeholder value for every required field based on its JSON
    Schema type. Optional fields are omitted so handlers see defaults.

    Args:
        model: The input Pydantic model.

    Returns:
        A dict that satisfies the model's required fields.
    """
    schema     = model.model_json_schema()
    properties = schema.get("properties", {})
    required   = schema.get("required", [])

    sample: dict[str, Any] = {}
    for field_name in required:
        spec = properties.get(field_name, {})
        sample[field_name] = _placeholder_for(spec)
    return sample


def _placeholder_for(spec: dict[str, Any]) -> Any:
    """Return a placeholder value matching a JSON-Schema field spec."""
    # Honour an explicit enum
    if "enum" in spec and spec["enum"]:
        return spec["enum"][0]
    json_type = spec.get("type")
    if json_type == "string":
        return "health_check"
    if json_type == "integer":
        return 1
    if json_type == "number":
        return 1.0
    if json_type == "boolean":
        return True
    if json_type == "array":
        return []
    if json_type == "object":
        return {}
    # anyOf / unspecified: fall back to a string
    return "health_check"


def _run_with_timeout(
    handler:   Callable[[BaseModel], Any],
    arg:       BaseModel,
    timeout_ms: int,
) -> tuple[Any, float, bool, str | None]:
    """Run handler(arg) in a worker thread bounded by timeout_ms.

    Returns:
        (result, latency_ms, timed_out, error_message)
        On timeout: (None, timeout_ms, True, None).
        On handler exception: (None, latency_ms, False, str(exc)).
    """
    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["result"] = handler(arg)
        except Exception as exc:  # noqa: BLE001 -- surfaced to caller
            box["error"] = f"{type(exc).__name__}: {exc}"

    start  = time.perf_counter()
    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout_ms / 1000.0)
    latency_ms = (time.perf_counter() - start) * 1000.0

    if thread.is_alive():
        # Thread is daemon; it will be abandoned. Report a clean timeout.
        return (None, float(timeout_ms), True, None)
    if "error" in box:
        return (None, latency_ms, False, box["error"])
    return (box.get("result"), latency_ms, False, None)


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of a non-empty list, rounded to 2 decimals."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k       = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1))
    return round(ordered[k], 2)


def _fmt_errors(exc: ValidationError) -> str:
    """Compact one-line rendering of Pydantic validation errors."""
    parts = [
        f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}"
        for e in exc.errors()
    ]
    return "; ".join(parts[:3])


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    from pydantic import Field

    random.seed(7)
    registry = ToolRegistry()

    # ── Tool 1: web_search (fast, healthy) ────────────────────────────────────
    class WebSearchInput(BaseModel):
        query:       str = Field(description="Search query string")
        max_results: int = Field(default=5, ge=1, le=50)

    class WebSearchOutput(BaseModel):
        results: list[str]
        count:   int

    @registry.tool(
        name="web_search",
        description="Search the web and return a list of result snippets.",
        input_schema=WebSearchInput,
        output_schema=WebSearchOutput,
        sla_ms=800,
    )
    def web_search(inp: WebSearchInput) -> dict[str, Any]:
        time.sleep(random.uniform(0.05, 0.15))   # 50-150ms: well within SLA
        hits = [f"result {i+1} for '{inp.query}'" for i in range(min(inp.max_results, 3))]
        return {"results": hits, "count": len(hits)}

    # ── Tool 2: read_file (very fast, healthy) ────────────────────────────────
    class ReadFileInput(BaseModel):
        path: str = Field(description="Absolute path of the file to read")

    class ReadFileOutput(BaseModel):
        content: str
        bytes:   int

    @registry.tool(
        name="read_file",
        description="Read a UTF-8 text file and return its contents.",
        input_schema=ReadFileInput,
        output_schema=ReadFileOutput,
        sla_ms=300,
    )
    def read_file(inp: ReadFileInput) -> dict[str, Any]:
        time.sleep(random.uniform(0.01, 0.04))   # 10-40ms
        content = f"<contents of {inp.path}>"
        return {"content": content, "bytes": len(content)}

    # ── Tool 3: send_email (slow -> WARN, exceeds SLA but not timeout) ────────
    class SendEmailInput(BaseModel):
        to:      str = Field(description="Recipient email address")
        subject: str
        body:    str

    class SendEmailOutput(BaseModel):
        message_id: str
        delivered:  bool

    @registry.tool(
        name="send_email",
        description="Send an email via the transactional mail provider.",
        input_schema=SendEmailInput,
        output_schema=SendEmailOutput,
        sla_ms=200,          # tight SLA; the simulated provider is slower
        timeout_ms=3000,
    )
    def send_email(inp: SendEmailInput) -> dict[str, Any]:
        time.sleep(random.uniform(0.30, 0.45))   # 300-450ms: over the 200ms SLA
        return {"message_id": f"msg_{random.randint(1000, 9999)}", "delivered": True}

    # ── Tool 4: database_query (broken handler -> FAIL) ───────────────────────
    class DatabaseQueryInput(BaseModel):
        sql:     str = Field(description="SQL query to execute")
        timeout: int = Field(default=30, ge=1)

    class DatabaseQueryOutput(BaseModel):
        rows:         list[dict[str, Any]]
        row_count:    int

    @registry.tool(
        name="database_query",
        description="Execute a read-only SQL query and return matching rows.",
        input_schema=DatabaseQueryInput,
        output_schema=DatabaseQueryOutput,
        sla_ms=1000,
    )
    def database_query(inp: DatabaseQueryInput) -> dict[str, Any]:
        # Simulated outage: the connection pool is exhausted.
        time.sleep(random.uniform(0.02, 0.05))
        raise ConnectionError("connection pool exhausted (no available connections)")

    # ── run validate_all ──────────────────────────────────────────────────────
    print(f"\nRegistered tools: {registry.names()}\n")
    report = registry.validate_all()

    # ── demonstrate a live call with validation ────────────────────────────────
    print("\n  LIVE CALL EXAMPLES")
    print(f"  {'-'*60}")

    ok = registry.call("web_search", {"query": "anthropic claude", "max_results": 3})
    print(f"  web_search(valid)      -> success={ok.success}  "
          f"latency={ok.latency_ms:.1f}ms  output={ok.output}")

    bad = registry.call("web_search", {"max_results": 3})   # missing 'query'
    print(f"  web_search(no query)   -> success={bad.success}  error={bad.error}")

    bad_type = registry.call("web_search", {"query": "x", "max_results": 999})  # > le=50
    print(f"  web_search(max=999)    -> success={bad_type.success}  error={bad_type.error}")

    fail = registry.call("database_query", {"sql": "SELECT 1"})
    print(f"  database_query(valid)  -> success={fail.success}  error={fail.error}")

    unknown = registry.call("no_such_tool", {})
    print(f"  no_such_tool           -> success={unknown.success}  error={unknown.error}")

    # ── show Anthropic export ──────────────────────────────────────────────────
    print("\n  ANTHROPIC tool_use EXPORT (first tool)")
    print(f"  {'-'*60}")
    import json
    anthropic_tools = registry.get_anthropic_tools()
    print(f"  Exported {len(anthropic_tools)} tools. Schema for '{anthropic_tools[0]['name']}':")
    print(json.dumps(anthropic_tools[0], indent=4))
