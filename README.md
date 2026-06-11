# LLM Agent Engineering

Production code for *LLM Agent Engineering: Orchestration, Memory and Tool Use for Production Systems* by Kylan C. Holt.

---

## Structure

Each folder maps to a chapter of the book. Run any module directly to reproduce the benchmark numbers cited in the chapter.

| Folder | Chapter |
|--------|---------|
| `ch01_why_agents_break` | Chapter 1 — Why Your Agent Breaks in Production |
| `ch02_architectures` | Chapter 2 — Agent Architectures That Actually Ship |
| `ch03_agent_loop` | Chapter 3 — Designing the Agent Loop + Cost Visibility |
| `ch04_context_management` | Chapter 4 — Short-Term Context Management |
| `ch05_long_term_memory` | Chapter 5 — Long-Term Memory: Retrieval, Persistence and Lifecycle |
| `ch06_output_quality` | Chapter 6 — Output Quality and Agent Self-Evaluation |
| `ch07_tool_use` | Chapter 7 — Reliable Tool Use at Scale |
| `ch08_multi_agent` | Chapter 8 — Multi-Agent Orchestration |
| `ch09_error_recovery` | Chapter 9 — Error Recovery and Agent Resilience |
| `ch10_observability` | Chapter 10 — Observability and Debugging for Agent Systems |
| `ch11_cost_optimization` | Chapter 11 — Cost Optimization and Latency Budgets |
| `ch12_deployment` | Chapter 12 — Deploying and Operating Agents at Scale |

---

## Setup

```bash
git clone https://github.com/kylancholt/llm-agent-engineering.git
cd llm-agent-engineering
pip install -r requirements.txt
cp .env.example .env
```

Add your Anthropic API key to `.env`:

---

## Running the examples

Each module runs standalone and prints output matching the Expected Output boxes in the book:

```bash
python ch01_why_agents_break/failure_modes.py
python ch02_architectures/architecture_benchmark.py
python ch03_agent_loop/agent_loop.py
python ch10_observability/dashboard.py  # requires: streamlit run
```

---

## Requirements

- Python 3.10+
- Anthropic API key (claude-sonnet-4-6 used by default)
- See `requirements.txt` for full dependency list

---

## Book

*LLM Agent Engineering: Orchestration, Memory and Tool Use for Production Systems*
Kylan C. Holt
Available on Amazon.
