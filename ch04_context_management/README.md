# Chapter 4: Short-Term Context Management

Production code for Chapter 4 of *LLM Agent Engineering* by Kylan C. Holt.

## Modules

- `window_manager.py` — central context-window manager with three eviction strategies (FIFO/IMPORTANCE/SUMMARY) and message pinning support
- `token_counter.py` — accurate token counting via the Anthropic count_tokens API with a chars/4 fallback and in-memory caching
- `summarizer.py` — context compressor using Claude Haiku to compress agent history into structured 3-section summaries

## Run

```bash
python window_manager.py
```
