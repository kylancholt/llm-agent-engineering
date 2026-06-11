# Chapter 6: Output Quality and Agent Self-Evaluation

Production code for Chapter 6 of *LLM Agent Engineering* by Kylan C. Holt.

## Modules

- `output_validator.py` — three-level validator: Schema (Pydantic), Semantic completeness, and Evidence grounding from tool results; returns PASS/RETRY/ESCALATE
- `confidence_scorer.py` — LLM-free confidence estimator from four trace signals: tool success rate, evidence coverage, reasoning consistency, and budget pressure
- `fallback_policy.py` — decision tree for low-confidence outputs mapping scores to PASS_THROUGH/RETRY_WITH_HINT/PARTIAL_RESULT/ESCALATE_HUMAN/ABORT actions

## Run

```bash
python output_validator.py
```
