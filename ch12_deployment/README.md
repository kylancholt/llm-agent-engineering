# Chapter 12: Deploying and Operating Agents at Scale

Production code for Chapter 12 of *LLM Agent Engineering* by Kylan C. Holt.

## Modules

- `session_manager.py` — namespace-partitioned session manager with TTL, isolation verification, and a load test fixture (500 concurrent sessions, 0 isolation violations)
- `task_queue.py` — priority task queue (HIGH/NORMAL/LOW) with back-pressure hold at 80% depth, worker pool, and a spike test demonstrating 94% failure rate without back-pressure

## Run

```bash
python task_queue.py
```
