# Chapter 5: Long-Term Memory: Retrieval, Persistence and Lifecycle

Production code for Chapter 5 of *LLM Agent Engineering* by Kylan C. Holt.

## Modules

- `memory_store.py` — ChromaDB-backed long-term memory store with embeddings, three memory types (episodic/semantic/factual), and access tracking
- `retriever.py` — hybrid dense (cosine) + sparse (BM25) retriever with freshness re-ranking for multi-modal memory search
- `expiration_manager.py` — memory lifecycle manager combining TTL, frequency, and semantic staleness scores into composite expiration labels (fresh/aging/stale/expired)

## Run

```bash
python memory_store.py
```
