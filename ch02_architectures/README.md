# Chapter 2: Agent Architectures That Actually Ship

Production code for Chapter 2 of *LLM Agent Engineering* by Kylan C. Holt.

## Modules

- `architecture_benchmark.py` — benchmarks Direct, ReAct, PlanExecute, and Supervisor architectures on a 10-task set, printing a results table and JSON output
- `react_agent.py` — Reason+Act loop agent using native tool_use blocks with real-time budget enforcement
- `plan_execute_agent.py` — two-phase agent: Phase 1 produces a JSON step list, Phase 2 executes each step with optional retry on failure
- `supervisor_agent.py` — supervisor + subagent delegation: Haiku routing layer decides between direct answer or specialist Sonnet subagents (RESEARCHER/ANALYZER/EXECUTOR/WRITER)

## Run

```bash
python architecture_benchmark.py
```
