# Chapter 9: Error Recovery and Agent Resilience

Production code for Chapter 9 of *LLM Agent Engineering* by Kylan C. Holt.

## Modules

- `recovery_engine.py` — step- and task-level failure recovery state machine classifying failures (TRANSIENT/PERMANENT/AMBIGUOUS) and mapping them to retry/fallback/skip/escalate actions
- `checkpoint_manager.py` — content-addressed checkpoint/resume with SHA-256 deduplication, crash recovery, and pruning; guarantees zero re-execution of completed steps
- `failure_reporter.py` — fabrication-risk detector (no-tool-calls/unsupported-claims/missing-topics) that returns honest partial results as structured FailureResponse objects
- `chaos_suite.py` — resilience test suite injecting five failure categories (tool failure/latency spike/budget exhaustion/context corruption/partial tool failure) and measuring recovery rate

## Run

```bash
python chaos_suite.py
```
