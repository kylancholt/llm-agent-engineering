# Chapter 3: Designing the Agent Loop + Cost Visibility

Production code for Chapter 3 of *LLM Agent Engineering* by Kylan C. Holt.

## Modules

- `agent_loop.py` — production agent loop state machine (IDLE → RUNNING → FINAL_ANSWER/BUDGET_EXCEEDED/etc.) using native tool_use with checkpointing and signal handling
- `agent_state.py` — serializable mutable agent state with validation, checkpointing, snapshotting, and field-level diff for debugging
- `cost_guard.py` — real-time budget enforcement with per-turn cost calculation, alert level classification, and halt signaling
- `interrupt_handler.py` — human-in-the-loop interrupt management with interactive review, resume decisions, and JSONL audit log

## Run

```bash
python agent_loop.py
```
