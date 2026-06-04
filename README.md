# LLM Agent Engineering
Production code for LLM Agent Engineering by Kylan C. Holt.

## Structure
Each folder maps to a chapter of the book.
| Folder | Chapter |
|--------|---------|
| ch01_why_agents_break | Chapter 1 — Why Your Agent Breaks in Production |
| ch02_architectures | Chapter 2 — Agent Architectures That Actually Ship |
| ch03_agent_loop | Chapter 3 — Designing the Agent Loop + Cost Visibility |
| ch04_context_management | Chapter 4 — Short-Term Context Management |
| ch05_long_term_memory | Chapter 5 — Long-Term Memory |
| ch06_output_quality | Chapter 6 — Output Quality and Agent Self-Evaluation |
| ch07_tool_use | Chapter 7 — Reliable Tool Use at Scale |
| ch08_multi_agent | Chapter 8 — Multi-Agent Orchestration |
| ch09_error_recovery | Chapter 9 — Error Recovery and Agent Resilience |
| ch10_observability | Chapter 10 — Observability and Debugging |
| ch11_cost_optimization | Chapter 11 — Cost Optimization and Latency Budgets |
| ch12_deployment | Chapter 12 — Deploying and Operating Agents at Scale |

## Setup
pip install -r requirements.txt
cp .env.example .env

## Run any module
python ch01_why_agents_break/failure_modes.py
