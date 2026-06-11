# Chapter 1: Why Your Agent Breaks in Production

Production code for Chapter 1 of *LLM Agent Engineering* by Kylan C. Holt.

## Modules

- `failure_modes.py` — classifies 7 agent failure types (loop, context overflow, tool hallucination, memory drift, silent wrong answer, cost explosion, observability blindness) from JSONL execution traces
- `loop_detector.py` — detects infinite loops via exact repeat, semantic repeat, or max-turns exceeded, with CONTINUE/WARN/HALT recommendations
- `cost_tracker.py` — tracks cumulative LLM cost in real time with budget enforcement and OK/WARN/CRITICAL/HALT alert levels
- `test_failure_modes.py` — reproduces all 7 failure modes with synthetic traces and validates detection accuracy

## Run

```bash
python test_failure_modes.py
```
