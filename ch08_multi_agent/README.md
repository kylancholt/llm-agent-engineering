# Chapter 8: Multi-Agent Orchestration

Production code for Chapter 8 of *LLM Agent Engineering* by Kylan C. Holt.

## Modules

- `supervisor.py` — multi-agent supervisor: decomposes a task, dispatches waves of specialist subagents in parallel, and synthesizes results
- `parallel_runner.py` — asyncio-native parallel subagent executor with concurrency cap, dependency ordering, and partial-failure isolation
- `aggregator.py` — aggregates results from multiple agents via three strategies: MERGE (LLM fusion), CONSENSUS (agreement-based), SUPERVISOR_DECIDES (LLM arbitration)

## Run

```bash
python supervisor.py
```
