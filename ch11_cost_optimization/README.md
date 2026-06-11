# Chapter 11: Cost Optimization and Latency Budgets

Production code for Chapter 11 of *LLM Agent Engineering* by Kylan C. Holt.

## Modules

- `model_router.py` — subtask-aware model router directing 7 subtask types to the cheapest capable model: Haiku for classification/routing/formatting, Sonnet for extraction/summarization/reasoning
- `semantic_cache.py` — semantic cache for four result types (tool_result/routing/embedding/plan_step) using SHA-256 exact match and cosine similarity, with TTL and hit/miss stats

## Run

```bash
python model_router.py
```
