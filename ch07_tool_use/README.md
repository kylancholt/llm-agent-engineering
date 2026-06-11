# Chapter 7: Reliable Tool Use at Scale

Production code for Chapter 7 of *LLM Agent Engineering* by Kylan C. Holt.

## Modules

- `tool_registry.py` — type-safe tool registry with Pydantic schemas, health checks (OK/WARN/FAIL), SLA monitoring, and Anthropic Messages API export
- `retry_handler.py` — error classifier distinguishing retriable (timeout/network/429/5xx) from non-retriable errors (schema/auth/404), with exponential backoff and per-window budgets
- `result_validator.py` — three-level tool result validator: Structure (Pydantic), Range (numeric bounds), Plausibility (empty/suspicious heuristics), plus sanitization in under 2ms

## Run

```bash
python tool_registry.py
```
