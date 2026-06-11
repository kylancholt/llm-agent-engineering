# Chapter 10: Observability and Debugging for Agent Systems

Production code for Chapter 10 of *LLM Agent Engineering* by Kylan C. Holt.

## Modules

- `agent_tracer.py` — turn-level tracer writing OpenTelemetry-compatible JSONL (trace_header/turn/trace_footer) with auto-diagnosis of ignored results and latency outliers
- `span_builder.py` — multi-agent span hierarchy builder with ASCII tree rendering and OTLP-JSON export for cross-agent correlation
- `structured_logger.py` — typed async JSONL logger supporting 9 event types (TURN_START/TOOL_CALL/TOOL_RESULT/etc.), sampling, and daily rotation with ~0.001ms overhead
- `dashboard.py` — Streamlit debug dashboard with Turn Replay, Cost Breakdown, Failure Rate by Tool, and Loop Detector panels

## Run

```bash
streamlit run dashboard.py
```
