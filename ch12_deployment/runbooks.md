# Agent System On-Call Runbooks

## Come usare questi runbook

- Ogni runbook segue la struttura: **Detection Signal → Immediate Mitigation → Root Cause Investigation → Fix and Verify**
- **Tempo target di mitigazione: < 15 minuti dal detection**
- I comandi Python presuppongono che il working directory sia la root del repo (`llm-agent-engineering/`)
- I path dei log sono relativi alla root del repo: `traces/`, `logs/`, `data/`

---

## Incident 1: Cost Spike

**Descrizione:** Il costo medio per task supera 2x il baseline per 10+ minuti consecutivi. Causa tipica: model router che scala a Sonnet/Opus su task classificabili come Haiku, semantic cache fredda, o prompt inflation.

### Detection Signal

```
cost_per_task > 2x baseline  per >= 10 minuti consecutivi
metric sorgente: StructuredLogger, EventType "cost_report"
alert: avg cost_usd nelle ultime 10 entry > 2 * rolling_avg_24h
```

### Immediate Mitigation

**Passo 1 — Forza il router su Haiku per i nuovi task**

Modifica temporanea di `ModelRouter` abbassando la qualità richiesta per evitare escalation a Sonnet/Opus:

```python
# ch11_cost_optimization/model_router.py
from ch11_cost_optimization.model_router import ModelRouter
r = ModelRouter(quality_threshold=0.99, allow_opus=False)
# quality_threshold=0.99 fa sì che quasi nessun task venga escalato a Opus
```

**Passo 2 — Invalida il semantic cache per "routing" e "plan_step"**

Se il cache è in stato inconsistente (molti miss che causano chiamate costose):

```python
# ch11_cost_optimization/semantic_cache.py
from ch11_cost_optimization.semantic_cache import AgentSemanticCache
cache = AgentSemanticCache()
n_routing  = cache.invalidate_by_type("routing")
n_plan     = cache.invalidate_by_type("plan_step")
print(f"Invalidate: routing={n_routing}, plan_step={n_plan}")
```

**Passo 3 — Abilita back-pressure sulla task queue per limitare il throughput**

Riduce il flusso di task fino a stabilizzare il costo:

```python
# ch12_deployment/queue/task_queue.py
from ch12_deployment.queue.task_queue import AgentTaskQueue, Priority
q = AgentTaskQueue(max_queue_depth=100, worker_pool_size=10, back_pressure_threshold=0.5)
# threshold=0.5 attiva HOLD gia a depth>=50 invece che 80
```

### Root Cause Investigation

**1. Identifica quali subtask type vengono instradati al modello sbagliato:**

```python
from ch11_cost_optimization.model_router import ModelRouter, _DEMO_TASKS
r = ModelRouter(quality_threshold=0.90, allow_opus=False)
result = r.benchmark(_DEMO_TASKS)
result.print_table()
# Cerca righe dove model_selected=sonnet/opus con avg_cost_routed alto
# e quality_score < quality_threshold (causa escalation inutile)
```

**2. Controlla il semantic cache hit rate (basso hit rate = piu' chiamate API = piu' costo):**

```python
from ch11_cost_optimization.semantic_cache import AgentSemanticCache
cache = AgentSemanticCache()
cache.print_stats()
# Se hit_rate < 20% per "routing" e "plan_step" -> cache fredda o threshold troppo alto
# Se bytes_saved basso -> pochi riutilizzi
```

**3. Cerca eventi "cost_report" anomali nei log strutturati:**

```python
from ch10_observability.structured_logger import StructuredLogger
logger = StructuredLogger(log_dir="logs/")
events = logger.search(event_type="cost_report", hours=1)
for e in events:
    if e.metadata.get("cost_usd", 0) > 0.01:
        print(e.task_id, e.metadata)
```

**4. Leggi le trace per identificare i task piu' costosi:**

```python
from ch10_observability.agent_tracer import AgentTracer
import glob, json
for path in sorted(glob.glob("traces/trace_*.jsonl"))[-10:]:
    with open(path) as f:
        data = json.loads(f.read())
    cost = sum(t.get("cost_usd", 0) for t in data.get("turns", []))
    if cost > 0.05:
        print(path, f"cost={cost:.4f}")
```

### Fix and Verify

```python
# Verifica: costo medio torna al baseline
from ch11_cost_optimization.semantic_cache import AgentSemanticCache
cache = AgentSemanticCache()
stats = cache.get_stats()
for t, s in stats.items():
    print(f"{t}: hit_rate={s.hit_rate:.1%}, cost_saved=${s.cost_saved_usd:.4f}")
# Target: hit_rate routing > 30%, cost_saved in crescita

# Verifica: benchmark mostra cost_reduction_pct positivo
from ch11_cost_optimization.model_router import ModelRouter, _DEMO_TASKS
r = ModelRouter(quality_threshold=0.90)
r.benchmark(_DEMO_TASKS).print_table()
# Target: avg_cost_routed < avg_cost_sonnet per tutti i tipi
```

**Incident chiuso quando:** `cost_per_task < 1.2x baseline` per 5 minuti consecutivi.

---

## Incident 2: Quality Regression

**Descrizione:** Il confidence score medio scende sotto 0.7, oppure si accumulano eventi `ESCALATE_HUMAN` nella queue di escalation. Causa tipica: grounding failure (output non supportato dal contesto), schema validation failures, o model degradation.

### Detection Signal

```
confidence_score < 0.70  per >= 5 task consecutivi
OPPURE
escalation_queue.jsonl cresce di > 10 entry in 5 minuti
metric sorgente: ConfidenceScorer, FallbackPolicy escalation JSONL
```

### Immediate Mitigation

**Passo 1 — Leggi la escalation queue per capire i pattern di failure:**

```python
import json
from pathlib import Path

# La coda e' scritta da FallbackPolicy._enqueue_escalation()
# Path default: configurato dal chiamante (spesso /tmp/fallback_escalations.jsonl)
queue_path = Path("/tmp/fallback_escalations.jsonl")
if queue_path.exists():
    entries = [json.loads(l) for l in queue_path.read_text().splitlines() if l.strip()]
    print(f"Escalations in coda: {len(entries)}")
    for e in entries[-5:]:
        print(e.get("task_id"), e.get("reason"), e.get("hint"))
```

**Passo 2 — Forza il FallbackPolicy in modalita' ESCALATE_HUMAN per tutti i task borderline:**

```python
# ch06_output_quality/fallback_policy.py
# Abbassa i threshold per massimizzare la deviazione verso revisione umana
from ch06_output_quality.fallback_policy import FallbackPolicy
# Usa confidence_threshold alto (0.95) -> piu' task finiscono in ESCALATE
policy = FallbackPolicy(confidence_threshold=0.95, escalation_queue_path="/tmp/escalations_incident.jsonl")
```

**Passo 3 — Disabilita il serving dei risultati LOW confidence:**

```python
# ch06_output_quality/output_validator.py
from ch06_output_quality.output_validator import OutputValidator
validator = OutputValidator()
# Passa strict_grounding=True per bloccare output non groundati
# e forzare RETRY prima di servire al cliente
```

### Root Cause Investigation

**1. Analizza la distribuzione dei confidence score:**

```python
from ch06_output_quality.confidence_scorer import ConfidenceScorer
from ch10_observability.structured_logger import StructuredLogger

logger = StructuredLogger(log_dir="logs/")
events = logger.search(event_type="tool_result", hours=2)
scorer = ConfidenceScorer()

scores = []
for e in events:
    result = scorer.score(e.metadata)
    scores.append(result.score)
    if result.score < 0.7:
        print(f"LOW: task={e.task_id}, score={result.score:.2f}, signals={result.signals}")

avg = sum(scores) / len(scores) if scores else 0
print(f"Avg confidence (2h): {avg:.3f}")
```

**2. Controlla il calibration error (ECE) dello scorer:**

```python
from ch06_output_quality.confidence_scorer import ConfidenceScorer
scorer = ConfidenceScorer()
# Esegui calibrate() su un campione di eventi recenti per verificare ECE
# ECE > 0.10 indica che il modello e' mal calibrato
```

**3. Verifica i livelli di validation (schema/semantic/grounding) falliti:**

```python
from ch06_output_quality.output_validator import OutputValidator
from ch06_output_quality.fallback_policy import FallbackPolicy, get_policy_stats

policy = FallbackPolicy(escalation_queue_path="/tmp/fallback_escalations.jsonl")
stats = policy.get_policy_stats()
print(stats)
# Cerca: ESCALATE_HUMAN count alto, PASS count basso
# Se schema_fail_count alto -> model sta cambiando output format
# Se grounding_fail_count alto -> hallucination o context insufficiente
```

**4. Replay le trace dei task falliti:**

```python
from ch10_observability.agent_tracer import AgentTracer
import glob

tracer = AgentTracer(output_dir="traces/")
# Cerca trace con status != "completed"
for path in sorted(glob.glob("traces/trace_*.jsonl"))[-20:]:
    AgentTracer.replay(path)
    # Cerca "diagnosis" field con anti-pattern: ignored_results, latency_outlier
```

### Fix and Verify

```python
# Verifica: output validator torna in stato PASS
from ch06_output_quality.output_validator import OutputValidator
from ch06_output_quality.fallback_policy import FallbackPolicy

policy = FallbackPolicy()
stats = policy.get_policy_stats()
pass_rate = stats.get("PASS", 0) / max(1, sum(stats.values()))
print(f"Pass rate: {pass_rate:.1%}")
# Target: pass_rate > 85%

# Verifica: nessun nuovo elemento nella escalation queue negli ultimi 5 minuti
import os, time
queue_path = "/tmp/fallback_escalations.jsonl"
size_before = os.path.getsize(queue_path) if os.path.exists(queue_path) else 0
time.sleep(300)
size_after = os.path.getsize(queue_path) if os.path.exists(queue_path) else 0
print(f"Queue growth: {size_after - size_before} bytes (target: 0)")
```

**Incident chiuso quando:** confidence_score medio > 0.75 per 10 task consecutivi e escalation rate < 5%.

---

## Incident 3: Stuck Queue

**Descrizione:** La task queue e' in stato HOLD (back-pressure attiva) ma `tasks_processed` non cresce, oppure i worker sono tutti idle con queue non vuota. Causa tipica: worker thread crashed, tool dependency non disponibile, o deadlock su shared lock.

### Detection Signal

```
queue.depth > 80  (back-pressure attiva)
E  queue.tasks_processed invariato per >= 5 minuti
OPPURE
queue.workers_active == 0  con  queue.depth > 0
metric sorgente: AgentTaskQueue.get_stats()
```

### Immediate Mitigation

**Passo 1 — Snapshot immediato dello stato della queue:**

```python
from ch12_deployment.queue.task_queue import AgentTaskQueue
q = AgentTaskQueue(max_queue_depth=100, worker_pool_size=10)
stats = q.get_stats()
print(f"depth={stats.depth}/{stats.capacity}")
print(f"workers: active={stats.workers_active}, idle={stats.workers_idle}")
print(f"processed={stats.tasks_processed}, rejected={stats.tasks_rejected}")
print(f"bp_triggers={stats.back_pressure_triggers}")
print(f"throughput={stats.throughput_per_sec:.1f} tasks/s")
print(f"avg_process_ms={stats.avg_process_time_ms:.1f}")
```

**Passo 2 — Drena la queue di HIGH priority con worker aggiuntivi (scale-out temporaneo):**

```python
# Crea una seconda queue temporanea con piu' worker per smaltire il backlog
from ch12_deployment.queue.task_queue import AgentTaskQueue, AgentTask, Priority

q_drain = AgentTaskQueue(
    max_queue_depth=200,
    worker_pool_size=20,          # doppi worker per il drain
    back_pressure_threshold=0.95, # soglia alta per accettare quasi tutto
)
# Sposta i task HIGH priority dalla queue originale a questa
```

**Passo 3 — Se i worker sono tutti bloccati, chiudi e ricrea la queue:**

```python
# Close() inietta sentinel items per sbloccare i worker
# e poi join() su ogni thread con timeout 5s
q.close(timeout=5.0)

# Ricrea la queue con worker puliti
from ch12_deployment.queue.task_queue import AgentTaskQueue
q = AgentTaskQueue(max_queue_depth=100, worker_pool_size=10)
```

### Root Cause Investigation

**1. Verifica se i tool usati dai worker sono healthy:**

```python
from ch07_tool_use.tool_registry import ToolRegistry
registry = ToolRegistry()
report = registry.validate_all(verbose=True)
# Cerca: health=FAIL (tool ha sollevato eccezione o e' andato in timeout)
# Cerca: health=WARN (tool ha risposto ma oltre sla_ms)
for name, r in report.tool_results.items():
    if r.health != "OK":
        print(f"PROBLEMA: {name} -> health={r.health}, latency={r.latency_ms:.0f}ms, note={r.health_note}")
```

**2. Controlla il retry handler per tool in budget esaurito:**

```python
from ch07_tool_use.retry_handler import RetryHandler
handler = RetryHandler()
# Verifica se ci sono tool con budget di retry esaurito
# (tutti i tentativi consumati -> worker si blocca senza progredire)
```

**3. Cerca errori PERMANENT nei log del recovery engine:**

```python
from ch09_error_recovery.recovery_engine import RecoveryEngine, AgentState
from ch10_observability.structured_logger import StructuredLogger

logger = StructuredLogger(log_dir="logs/")
events = logger.search(event_type="error", hours=1)
for e in events:
    if e.metadata.get("failure_class") == "PERMANENT":
        print(f"PERMANENT failure: task={e.task_id}, step={e.metadata.get('step_id')}, msg={e.metadata.get('message')}")
```

**4. Leggi i checkpoint per identificare task bloccati senza progresso:**

```python
from ch09_error_recovery.checkpoint_manager import CheckpointManager
import glob, json

for path in glob.glob("data/checkpoints/*.json"):
    with open(path) as f:
        cp = json.load(f)
    last_step = max(cp.get("completed_steps", [0]))
    total     = cp.get("total_steps", 0)
    print(f"task={cp['task_id']}, progress={last_step}/{total}, ts={cp['saved_at']}")
```

### Fix and Verify

```python
# Verifica: throughput torna positivo
from ch12_deployment.queue.task_queue import AgentTaskQueue
q = AgentTaskQueue()
stats = q.get_stats()
print(f"throughput: {stats.throughput_per_sec:.1f} tasks/s")  # target: > 0

# Verifica: tool registry torna OK
from ch07_tool_use.tool_registry import ToolRegistry
registry = ToolRegistry()
report = registry.validate_all(verbose=False)
all_ok = all(r.health == "OK" for r in report.tool_results.values())
print(f"All tools healthy: {all_ok}")
```

**Incident chiuso quando:** `throughput_per_sec > 0` e `workers_active >= workers_idle` per 2 minuti.

---

## Incident 4: Session Leak

**Descrizione:** Il numero di sessioni attive cresce monotonicamente senza stabilizzarsi, oppure `check_isolation()` riporta violazioni di namespace. Causa tipica: sessioni non chiuse alla fine del task, bug nel TTL check, o cross-contamination del metadata tra sessioni.

### Detection Signal

```
session_manager.active_count cresce senza bound per >= 15 minuti
OPPURE
check_isolation().violations > 0  (metadata cross-contaminazione)
metric sorgente: SessionManager.active_count, IsolationReport
```

### Immediate Mitigation

**Passo 1 — Esegui isolation check immediato:**

```python
from ch12_deployment.session.session_manager import SessionManager
sm = SessionManager(max_concurrent_sessions=1000, session_ttl_seconds=3600)
report = sm.check_isolation()
print(f"Sessions checked: {report.sessions_checked}")
print(f"Violations:       {report.violations}")
print(f"Is isolated:      {report.is_isolated}")
if not report.is_isolated:
    for v in report.violation_details:
        print(f"  VIOLATION: {v}")
```

**Passo 2 — Forza l'eviction delle sessioni scadute:**

```python
# _evict_expired_locked() e' chiamata internamente da create_session();
# puoi forzarla creando una sessione dummy per triggherare il ciclo di eviction
sm.create_session(user_id="__eviction_trigger__")
print(f"Active dopo eviction: {sm.active_count}")
```

**Passo 3 — Abbassa il TTL temporaneamente per accelerare l'eviction organica:**

```python
# Crea una nuova istanza con TTL ridotto (es. 300s = 5 minuti)
from ch12_deployment.session.session_manager import SessionManager
sm_short = SessionManager(max_concurrent_sessions=1000, session_ttl_seconds=300)
# Le nuove sessioni create qui scadranno dopo 5 minuti
```

### Root Cause Investigation

**1. Identifica le sessioni piu' vecchie (potenziali leak):**

```python
from ch12_deployment.session.session_manager import SessionManager
import time

sm = SessionManager()
# Snapshotta le sessioni fuori dal lock leggendo _sessions
with sm._lock:
    sessions_snapshot = list(sm._sessions.values())

now = time.time()
for sess in sorted(sessions_snapshot, key=lambda s: s.created_at):
    age_min = (now - sess.created_at) / 60
    print(f"session={sess.session_id[:12]}, user={sess.user_id}, age={age_min:.1f}m, "
          f"metadata_keys={len(sess.state.metadata)}")
```

**2. Verifica che ogni metadata key rispetti il namespace della sessione:**

```python
from ch12_deployment.session.session_manager import SessionManager

sm = SessionManager()
report = sm.check_isolation()
# Se is_isolated=False, le violation_details mostrano:
#   - la chiave violante
#   - il namespace della sessione proprietaria
#   - il namespace della sessione che ha scritto la chiave
```

**3. Stima il tasso di creazione sessioni nei log:**

```python
from ch10_observability.structured_logger import StructuredLogger

logger = StructuredLogger(log_dir="logs/")
events = logger.search(event_type="tool_call", hours=1)
# Conta quante sessioni uniche appaiono nell'ultima ora
session_ids = {e.task_id for e in events}
print(f"Sessioni uniche (1h): {len(session_ids)}")
# Se >> max_concurrent_sessions -> leak confermato
```

**4. Controlla il namespace_index per consistenza:**

```python
from ch12_deployment.session.session_manager import SessionManager
sm = SessionManager()
with sm._lock:
    n_sessions     = len(sm._sessions)
    n_ns_entries   = len(sm._namespace_index)
print(f"_sessions={n_sessions}, _namespace_index={n_ns_entries}")
# Devono essere identici; se divergono -> bug nel _remove_locked()
```

### Fix and Verify

```python
# Verifica: active_count stabile o in calo
from ch12_deployment.session.session_manager import SessionManager
import time

sm = SessionManager()
before = sm.active_count
time.sleep(60)
after = sm.active_count
print(f"active_count: {before} -> {after}  (target: stabile o in calo)")

# Verifica: isolation check pulito
report = sm.check_isolation()
print(f"is_isolated={report.is_isolated}, violations={report.violations}")
# Target: is_isolated=True, violations=0
```

**Incident chiuso quando:** `active_count` non cresce per 10 minuti e `check_isolation().violations == 0`.

---

## Incident 5: Tool Outage

**Descrizione:** Uno o piu' tool nel registry ritornano `health=FAIL` per 2+ check consecutivi. I task che dipendono dal tool falliscono o entrano in loop di retry. Causa tipica: dipendenza esterna non disponibile, schema validation failure, o timeout.

### Detection Signal

```
tool_registry.health(tool_name) == FAIL  per >= 2 check consecutivi
OPPURE
retry_handler.budget_exhausted(tool_name) == True
metric sorgente: ToolRegistry.validate_all(), RetryHandler budget windows
```

### Immediate Mitigation

**Passo 1 — Health check immediato su tutti i tool:**

```python
from ch07_tool_use.tool_registry import ToolRegistry

registry = ToolRegistry()
report = registry.validate_all(verbose=True)
# Output: tabella con name | schema | health | latency_ms | note
# Identifica quali tool sono FAIL o WARN
for name, r in report.tool_results.items():
    status = "OK" if r.health == "OK" else f"*** {r.health} ***"
    print(f"  {name:<30} schema={r.schema}  health={status}  latency={r.latency_ms:.0f}ms")
    if r.health != "OK":
        print(f"    -> {r.health_note}")
```

**Passo 2 — Bypassa temporaneamente il tool guasto con fallback:**

```python
from ch07_tool_use.retry_handler import RetryHandler

handler = RetryHandler()
# Classifica l'errore per determinare se e' retriable o permanent
# error_class = handler.classify_error(tool_name, exception)
# Se PERMANENT -> smetti di ritentare e usa fallback
# Se TRANSIENT -> rispetta il backoff esponenziale del handler
```

**Passo 3 — Attiva il failure reporter per risposta onesta ai downstream:**

```python
from ch09_error_recovery.failure_reporter import FailureReporter

reporter = FailureReporter()
# Usa reporter.report_partial() per restituire i risultati parziali
# disponibili senza aspettare il tool guasto
# FailureResponse include: completed_steps, failed_steps, partial_result,
# fabrication_detected (True se il modello ha inventato dati mancanti)
```

### Root Cause Investigation

**1. Verifica se il tool ha cambiato schema (Pydantic validation failure):**

```python
from ch07_tool_use.tool_registry import ToolRegistry

registry = ToolRegistry()
# Chiama il tool con un input minimale sintetizzato da _synthesise_input()
# per distinguere schema failure (FAIL con ValidationError) da network failure
report = registry.validate_all(verbose=True)
for name, r in report.tool_results.items():
    if r.schema == "FAIL":
        print(f"SCHEMA FAILURE: {name} -> {r.schema_note}")
```

**2. Controlla il result validator per output malformati:**

```python
from ch07_tool_use.result_validator import ResultValidator

validator = ResultValidator()
# Esegui validate() su un campione di output recenti del tool guasto
# I livelli sono: structure (JSON valido), range (valori entro bounds), plausibility
# validate() ritorna anche sanitized_result se la sanitization ha corretto l'output
```

**3. Analizza i log di retry per vedere il pattern di failure:**

```python
from ch10_observability.structured_logger import StructuredLogger

logger = StructuredLogger(log_dir="logs/")
events = logger.search(event_type="tool_error", hours=2)
for e in events:
    tool  = e.metadata.get("tool_name", "?")
    err   = e.metadata.get("error_type", "?")
    retry = e.metadata.get("retry_count", 0)
    print(f"tool={tool:<25} error={err:<30} retry={retry}")
```

**4. Verifica la failure detection tramite chaos suite:**

```python
# ch09_error_recovery/chaos_suite.py
# Esegui lo scenario "tool_failure" per testare la resilience
from ch09_error_recovery.chaos_suite import ChaosSuite

suite = ChaosSuite()
result = suite.run_scenario("tool_failure")
print(f"Resilience score: {result.resilience_score:.2f}")
# Target: >= 0.80 (sistema resiliente con fallback attivi)
```

**5. Controlla i span del tool guasto nella span hierarchy:**

```python
from ch10_observability.span_builder import SpanBuilder
import glob, json

for path in glob.glob("traces/trace_*.jsonl")[-5:]:
    with open(path) as f:
        data = json.loads(f.read())
    # Cerca span con error=True associati al tool_name guasto
    builder = SpanBuilder()
    tree = builder.build_from_trace(data)
    tree.render()  # ASCII tree: +-- tool_name [FAIL 1234ms]
```

### Fix and Verify

```python
# Verifica: il tool ritorna healthy
from ch07_tool_use.tool_registry import ToolRegistry

registry = ToolRegistry()
report = registry.validate_all(verbose=False)
all_ok = all(r.health == "OK" for r in report.tool_results.values())
print(f"All tools healthy: {all_ok}")

# Verifica: retry handler non ha piu' budget esauriti
from ch07_tool_use.retry_handler import RetryHandler
handler = RetryHandler()
# Verifica che i tentativi tornino successful (non piu' TRANSIENT o PERMANENT failures)

# Verifica: failure reporter non rileva fabrication
from ch09_error_recovery.failure_reporter import FailureReporter
reporter = FailureReporter()
# fabrication_detected=False nella prossima FailureResponse
```

**Incident chiuso quando:** `validate_all()` ritorna `health=OK` per tutti i tool per 3 check consecutivi a distanza di 1 minuto.

---

## Alerting Thresholds

| Metric | Warning | Critical | Page On-Call | Modulo sorgente |
|---|---|---|---|---|
| `cost_per_task` | > 1.5x baseline | > 2x baseline | > 3x baseline per 5 min | `ch11_cost_optimization/model_router.py` |
| `semantic_cache.hit_rate` | < 25% | < 15% | < 10% per 10 min | `ch11_cost_optimization/semantic_cache.py` |
| `confidence_score` (avg) | < 0.75 | < 0.70 | < 0.60 per 5 task | `ch06_output_quality/confidence_scorer.py` |
| `escalation_queue.growth` | > 5 entry/min | > 15 entry/min | > 30 entry/min | `ch06_output_quality/fallback_policy.py` |
| `queue.depth` | > 60% capacity | > 80% capacity (BP active) | BP attiva + throughput=0 | `ch12_deployment/queue/task_queue.py` |
| `queue.throughput_per_sec` | < 50% baseline | < 25% baseline | = 0 per 5 min | `ch12_deployment/queue/task_queue.py` |
| `session.active_count` | > 800 | > 950 | > 990 o crescita non-stop | `ch12_deployment/session/session_manager.py` |
| `session.isolation_violations` | > 0 | > 0 | Qualsiasi violazione | `ch12_deployment/session/session_manager.py` |
| `tool.health` (any tool) | WARN | FAIL singolo | FAIL per 2+ check | `ch07_tool_use/tool_registry.py` |
| `tool.retry_budget_exhausted` | 1 tool | 2+ tool | Tool critico | `ch07_tool_use/retry_handler.py` |
| `chaos_suite.resilience_score` | < 0.85 | < 0.75 | < 0.60 | `ch09_error_recovery/chaos_suite.py` |
| `trace.latency_p99_ms` | > 5000 | > 15000 | > 30000 | `ch10_observability/agent_tracer.py` |

> **Nota:** I threshold "Warning" inviano notifica al canale Slack `#agent-alerts`. "Critical" apre un ticket automatico. "Page On-Call" chiama il reperibile via PagerDuty. Tutti i valori sono configurabili nei parametri dei rispettivi moduli.
