"""
ContextSummarizer: compresses agent history segments into structured summaries.

Uses Claude Haiku by default to minimise cost. The prompt instructs the model
to preserve decisions taken, completed tool calls with key results, and critical
information supplied by the user. Output follows a fixed three-section format:

  Completato:          Actions and tool calls executed with key results.
  Informazioni chiave: Critical facts, user requirements, and constraints.
  Prossimi step:       Planned or pending activities and open questions.

If the generated summary exceeds max_summary_tokens a second, stricter
compression pass is performed automatically. Cost from both passes is
accumulated in SummaryResult.cost_usd.

Supports both plain-string message content and Anthropic tool_use /
tool_result content blocks -- the transcript formatter handles both.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ── pricing ($/1M tokens) ─────────────────────────────────────────────────────

_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":          {"input":  0.80, "output":  4.00},
    "claude-haiku-4-5-20251001": {"input":  0.80, "output":  4.00},
    "claude-sonnet-4-6":         {"input":  3.00, "output": 15.00},
    "claude-opus-4-7":           {"input": 15.00, "output": 75.00},
    "claude-opus-4-8":           {"input": 15.00, "output": 75.00},
}
_DEFAULT_MODEL = "claude-haiku-4-5"

# Fixed token overhead for the prompt template (instructions + section headers)
_PROMPT_OVERHEAD_TOKENS = 280


# ── public types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SummaryResult:
    """Result of a ContextSummarizer.summarize() call.

    Attributes:
        summary_text:      Structured summary with three labelled sections.
        original_tokens:   Estimated token count of the original messages.
        summary_tokens:    Estimated token count of the generated summary.
        compression_ratio: original_tokens / summary_tokens; higher = more compact.
        cost_usd:          Total USD cost of all API calls in this summarisation.
    """
    summary_text:      str
    original_tokens:   int
    summary_tokens:    int
    compression_ratio: float
    cost_usd:          float


# ── context summarizer ────────────────────────────────────────────────────────

class ContextSummarizer:
    """Compresses agent history segments into structured three-section summaries.

    Usage::

        summarizer = ContextSummarizer(client=anthropic_client)
        est = summarizer.estimate_cost(messages, task_context=task)
        print(f"Estimated cost: ${est:.6f}")
        result = summarizer.summarize(messages, task_context=task)
        print(result.summary_text)
        print(f"Ratio {result.compression_ratio:.1f}x  cost ${result.cost_usd:.6f}")

    Args:
        client:             Anthropic client instance (required).
        model:              Model ID used for summarisation (default haiku).
        max_summary_tokens: Maximum tokens allowed in the output summary.
                            Triggers a second compression pass if exceeded.

    Raises:
        ValueError: If client is None.
    """

    # ── prompt templates ──────────────────────────────────────────────────────

    _SUMMARISE_PROMPT: str = (
        "Sei un assistente specializzato nella compressione di storie di conversazioni "
        "tra agenti AI e utenti.\n\n"
        "CONVERSAZIONE DA COMPRIMERE:\n{transcript}\n\n"
        "{task_section}"
        "Produci un summary strutturato con esattamente queste tre sezioni. "
        "Ogni sezione ha da 1 a 5 bullet point concisi. "
        "Non aggiungere testo fuori da queste sezioni.\n\n"
        "Completato:\n"
        "- [azioni eseguite, tool call completate con i risultati piu' importanti]\n\n"
        "Informazioni chiave:\n"
        "- [fatti critici, requisiti e vincoli dichiarati dall'utente]\n\n"
        "Prossimi step:\n"
        "- [attivita' pianificate non ancora completate, domande aperte]"
    )

    _TASK_SECTION: str = "CONTESTO DEL TASK:\n{task_context}\n\n"

    _RECOMPRESS_PROMPT: str = (
        "Il summary seguente supera il limite di {max_tokens} token. "
        "Riducilo ulteriormente mantenendo SOLO le informazioni piu' critiche. "
        "Rispetta il formato a tre sezioni (Completato, Informazioni chiave, Prossimi step).\n\n"
        "SUMMARY ATTUALE:\n{summary}"
    )

    # ── init ──────────────────────────────────────────────────────────────────

    def __init__(
        self,
        client:             Any,
        model:              str = _DEFAULT_MODEL,
        max_summary_tokens: int = 500,
    ) -> None:
        if client is None:
            raise ValueError("client must be an Anthropic instance; got None.")
        self._client            = client
        self.model              = model
        self.max_summary_tokens = max_summary_tokens

    # ── public API ────────────────────────────────────────────────────────────

    def summarize(
        self,
        messages:     list[dict[str, Any]],
        task_context: str = "",
    ) -> SummaryResult:
        """Compress messages into a structured three-section summary.

        Formats the messages as a readable transcript, calls the configured
        model, and validates the output length. If the first pass produces a
        summary longer than max_summary_tokens, a second stricter pass is run
        automatically and the cost of both calls is accumulated.

        Args:
            messages:     History to compress. Handles plain-string content
                          and Anthropic tool_use / tool_result blocks.
            task_context: Optional task description injected into the prompt to
                          help the model focus on what is relevant.

        Returns:
            SummaryResult with text, token counts, compression ratio, and cost.

        Raises:
            RuntimeError: If the underlying API call fails.
        """
        original_tokens = _count_tokens(messages)

        transcript    = _format_transcript(messages)
        task_section  = (
            self._TASK_SECTION.format(task_context=task_context.strip())
            if task_context.strip() else ""
        )
        prompt = self._SUMMARISE_PROMPT.format(
            transcript=transcript,
            task_section=task_section,
        )

        summary_text, cost = self._call(prompt)
        total_cost         = cost

        # Validate; re-compress if over token budget
        summary_tokens = _estimate_tokens(summary_text)
        if summary_tokens > self.max_summary_tokens:
            recompress_prompt = self._RECOMPRESS_PROMPT.format(
                max_tokens=self.max_summary_tokens,
                summary=summary_text,
            )
            summary_text, cost2 = self._call(recompress_prompt)
            total_cost         += cost2
            summary_tokens      = _estimate_tokens(summary_text)

        ratio = round(original_tokens / max(summary_tokens, 1), 2)
        return SummaryResult(
            summary_text=      summary_text,
            original_tokens=   original_tokens,
            summary_tokens=    summary_tokens,
            compression_ratio= ratio,
            cost_usd=          round(total_cost, 8),
        )

    def estimate_cost(
        self,
        messages:     list[dict[str, Any]],
        task_context: str = "",
    ) -> float:
        """Estimate USD cost of summarising these messages before calling.

        Uses chars/4 token estimation and the model's pricing table. Adds
        _PROMPT_OVERHEAD_TOKENS for the fixed template instructions.

        Args:
            messages:     History to estimate for.
            task_context: Optional task description (adds to prompt length).

        Returns:
            Estimated cost in USD.
        """
        pricing       = _PRICING.get(self.model, _PRICING[_DEFAULT_MODEL])
        transcript    = _format_transcript(messages)
        task_tokens   = _estimate_tokens(task_context)
        input_tokens  = (
            _estimate_tokens(transcript)
            + task_tokens
            + _PROMPT_OVERHEAD_TOKENS
        )
        output_tokens = self.max_summary_tokens
        cost = (
            input_tokens  * pricing["input"]  / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )
        return round(cost, 8)

    # ── private helpers ───────────────────────────────────────────────────────

    def _call(self, prompt: str) -> tuple[str, float]:
        """Execute one API call and return (response_text, cost_usd).

        Cost is estimated from chars/4 because the actual usage is not
        available synchronously without an extra count_tokens call.
        """
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_summary_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text          = response.content[0].text.strip()
        pricing       = _PRICING.get(self.model, _PRICING[_DEFAULT_MODEL])
        input_tokens  = _estimate_tokens(prompt) + _PROMPT_OVERHEAD_TOKENS
        output_tokens = _estimate_tokens(text)
        cost = (
            input_tokens  * pricing["input"]  / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )
        return (text, round(cost, 8))


# ── module-level helpers ──────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Cheap chars/4 token estimate used throughout the module."""
    return max(1, len(text) // 4) if text else 0


def _count_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens across all messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += _estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    total += _estimate_tokens(block.get("text", ""))
                else:
                    total += _estimate_tokens(json.dumps(block))
    return total


def _format_message(msg: dict[str, Any]) -> str:
    """Render a message dict as a plain-text line for the summarisation prompt.

    Handles three content shapes:
    - str:  used as-is.
    - list: tool_use blocks become ``[Tool call: name(input)]``;
            tool_result blocks become ``[Tool result: ...]``;
            text blocks are included verbatim.
    """
    role    = msg.get("role", "unknown").upper()
    content = msg.get("content", "")

    if isinstance(content, str):
        return f"{role}: {content}"

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                name = block.get("name", "tool")
                inp  = json.dumps(block.get("input", {}), ensure_ascii=True)
                parts.append(f"[Tool call: {name}({inp[:200]})]")
            elif btype == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, list):
                    inner = " ".join(
                        b.get("text", "") for b in inner
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                parts.append(f"[Tool result: {str(inner)[:300]}]")
            else:
                parts.append(f"[{btype}: {json.dumps(block)[:80]}]")
        return f"{role}: {' '.join(parts)}"

    return f"{role}: {content!s}"


def _format_transcript(messages: list[dict[str, Any]]) -> str:
    """Convert a full message list to a numbered plain-text transcript."""
    return "\n".join(
        f"[{i + 1:02d}] {_format_message(msg)}"
        for i, msg in enumerate(messages)
    )


# ── demo conversation ─────────────────────────────────────────────────────────
# 12-turn (24-message) simulated agent session: analysing a Python service
# for security vulnerabilities and creating a refactoring plan.

_DEMO_MESSAGES: list[dict[str, Any]] = [
    # ── Turn 1 ──────────────────────────────────────────────────────────────
    {
        "role": "user",
        "content": (
            "Analizza il repository Python in /workspace/api-service. "
            "Trova tutti i bug critici di sicurezza e performance, poi crea un piano "
            "di refactoring completo. Budget massimo per questa sessione: $5. "
            "La deadline per consegnare il piano e' fine mese."
        ),
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text",
             "text": "Avvio l'analisi. Prima esploro la struttura del repository."},
            {"type": "tool_use", "id": "tu_001", "name": "list_directory",
             "input": {"path": "/workspace/api-service", "recursive": True}},
        ],
    },
    # ── Turn 2 ──────────────────────────────────────────────────────────────
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu_001",
             "content": (
                 "api/__init__.py  api/users.py  api/orders.py  api/auth.py  "
                 "models/db.py  tests/  config/settings.py  requirements.txt  main.py"
             )},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Struttura chiara. Leggo il file principale."},
            {"type": "tool_use", "id": "tu_002", "name": "read_file",
             "input": {"path": "/workspace/api-service/main.py"}},
        ],
    },
    # ── Turn 3 ──────────────────────────────────────────────────────────────
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu_002",
             "content": (
                 "def get_user(id):\n"
                 "    query = f'SELECT * FROM users WHERE id={id}'\n"
                 "    return db.execute(query)\n\n"
                 "def delete_session(token):\n"
                 "    db.execute(f'DELETE FROM sessions WHERE token={token}')"
             )},
        ],
    },
    {
        "role": "assistant",
        "content": (
            "CRITICO: SQL injection in main.py. La funzione get_user() concatena "
            "direttamente il parametro id nella query SQL senza sanitizzazione. "
            "Stessa vulnerabilita' in delete_session(). Cerco pattern simili nel modulo api/."
        ),
    },
    # ── Turn 4 ──────────────────────────────────────────────────────────────
    {
        "role": "user",
        "content": "Controlla anche api/users.py e api/orders.py per le stesse vulnerabilita'.",
    },
    {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "tu_003", "name": "search_code",
             "input": {"pattern": "f'SELECT|f\"SELECT|execute\\(f", "path": "/workspace/api-service/api/"}},
        ],
    },
    # ── Turn 5 ──────────────────────────────────────────────────────────────
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu_003",
             "content": (
                 "api/users.py:45:  db.execute(f'SELECT * FROM users WHERE email={email}')\n"
                 "api/orders.py:23: db.execute(f'DELETE FROM orders WHERE id={order_id}')\n"
                 "api/orders.py:67: db.execute(f'INSERT INTO orders VALUES ({user_id}, {amount})')"
             )},
        ],
    },
    {
        "role": "assistant",
        "content": (
            "Inventario SQL injection completato: 5 occorrenze totali in 3 file "
            "(main.py:12, main.py:31, api/users.py:45, api/orders.py:23, api/orders.py:67). "
            "Tutte usano f-string nelle query. Ora verifico la copertura dei test."
        ),
    },
    # ── Turn 6 ──────────────────────────────────────────────────────────────
    {
        "role": "user",
        "content": (
            "I test sono in pytest. Ci sono anche problemi di performance con query N+1 "
            "nel metodo get_orders_with_items() in api/orders.py."
        ),
    },
    {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "tu_004", "name": "run_command",
             "input": {"cmd": "cd /workspace/api-service && pytest --tb=short -q 2>&1"}},
        ],
    },
    # ── Turn 7 ──────────────────────────────────────────────────────────────
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu_004",
             "content": (
                 "23 passed, 4 failed, 1 error\n"
                 "FAILED tests/test_orders.py::test_delete_order - AssertionError\n"
                 "FAILED tests/test_users.py::test_update_email - TypeError\n"
                 "FAILED tests/test_auth.py::test_login - ConnectionRefusedError\n"
                 "FAILED tests/test_auth.py::test_logout - ConnectionRefusedError\n"
                 "ERROR  tests/test_auth.py - ImportError: No module named 'jwt'"
             )},
        ],
    },
    {
        "role": "assistant",
        "content": (
            "Test results: 23 pass, 4 fail, 1 error. Problemi critici: "
            "(1) 'jwt' non installato -- blocca tutti i test di autenticazione. "
            "(2) ConnectionRefusedError in test_auth -- il servizio auth non e' avviato in CI. "
            "Leggo get_orders_with_items() per analizzare il problema N+1."
        ),
    },
    # ── Turn 8 ──────────────────────────────────────────────────────────────
    {
        "role": "user",
        "content": "L'autenticazione usa PyJWT. Il servizio auth viene avviato con docker-compose in CI.",
    },
    {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "tu_005", "name": "read_file",
             "input": {"path": "/workspace/api-service/api/orders.py", "lines": "55-90"}},
        ],
    },
    # ── Turn 9 ──────────────────────────────────────────────────────────────
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu_005",
             "content": (
                 "def get_orders_with_items(user_id):\n"
                 "    orders = db.query('SELECT * FROM orders WHERE user_id=?', [user_id])\n"
                 "    for order in orders:  # N+1: una query per ogni ordine\n"
                 "        order.items = db.query('SELECT * FROM items WHERE order_id=?', [order.id])\n"
                 "    return orders"
             )},
        ],
    },
    {
        "role": "assistant",
        "content": (
            "N+1 confermata: con 100 ordini vengono eseguite 101 query invece di 1 con JOIN. "
            "Fix: SELECT orders.*, items.* FROM orders LEFT JOIN items ON items.order_id = orders.id "
            "WHERE orders.user_id = ? -- una sola query. "
            "Ho tutto il necessario per redigere il piano di refactoring."
        ),
    },
    # ── Turn 10 ─────────────────────────────────────────────────────────────
    {
        "role": "user",
        "content": (
            "Includi nel piano le priorita' e le stime di tempo. "
            "Abbiamo 2 sviluppatori disponibili per 3 settimane."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "PIANO DI REFACTORING -- 2 sviluppatori, 3 settimane:\n\n"
            "Settimana 1 (Sicurezza - CRITICO):\n"
            "  - Fix 5 SQL injection con parametrized queries: 6h\n"
            "  - Aggiungere PyJWT a requirements.txt e configurare CI: 2h\n"
            "  - Fix docker-compose per avviare auth in CI: 2h\n\n"
            "Settimana 2 (Performance & Test):\n"
            "  - Refactoring N+1 con JOIN in get_orders_with_items: 6h\n"
            "  - Fix 4 test falliti: 4h\n"
            "  - Aumentare copertura test al 80%: 12h\n\n"
            "Settimana 3 (Code quality):\n"
            "  - Estrarre logica SQL in un layer Repository: 16h\n"
            "  - Review del codice e documentazione: 8h\n\n"
            "Totale stimato: 56h (3.5 settimane con 2 sviluppatori a 8h/g)."
        ),
    },
    # ── Turn 11 ─────────────────────────────────────────────────────────────
    {
        "role": "user",
        "content": (
            "Il fix delle SQL injection deve essere retrocompatibile: "
            "non possiamo cambiare le signature delle funzioni ne' il formato delle risposte JSON."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Confermato. Il fix usera' parametrized queries come drop-in replacement: "
            "db.execute('SELECT * FROM users WHERE id = ?', [id]) al posto di "
            "db.execute(f'SELECT * FROM users WHERE id={id}'). "
            "Le signature rimangono identiche, il formato JSON di risposta non cambia. "
            "Zero breaking changes per i client dell'API."
        ),
    },
    # ── Turn 12 ─────────────────────────────────────────────────────────────
    {
        "role": "user",
        "content": "Perfetto. Inizia con il fix di main.py e tienimi aggiornato sui progressi.",
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text",
             "text": "Avvio il fix. Modifico le due funzioni vulnerabili in main.py."},
            {"type": "tool_use", "id": "tu_006", "name": "edit_file",
             "input": {
                 "path": "/workspace/api-service/main.py",
                 "replacements": [
                     {"old": "f'SELECT * FROM users WHERE id={id}'",
                      "new": "'SELECT * FROM users WHERE id = ?', [id]"},
                     {"old": "f'DELETE FROM sessions WHERE token={token}'",
                      "new": "'DELETE FROM sessions WHERE token = ?', [token]"},
                 ],
             }},
        ],
    },
]


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time

    _ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_ROOT))

    def _load_env(path: Path) -> None:
        """Load KEY=VALUE pairs from a .env file into os.environ (idempotent)."""
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    _load_env(_ROOT / ".env")

    # ── client setup (optional: dry-run works without key) ───────────────────
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client  = None
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            print("Anthropic client ready -- live API call enabled.\n")
        except ImportError:
            print("anthropic package not installed -- running in dry-run mode.\n")
    else:
        print("ANTHROPIC_API_KEY not set -- running in dry-run mode (no API calls).\n")

    MODEL        = "claude-haiku-4-5"
    MAX_TOKENS   = 500
    TASK_CONTEXT = (
        "L'agente sta analizzando un servizio API Python per trovare vulnerabilita' "
        "di sicurezza (SQL injection) e problemi di performance (query N+1), "
        "con l'obiettivo di produrre un piano di refactoring in 3 settimane."
    )

    messages     = _DEMO_MESSAGES  # 24 messages = 12 turns
    sep          = "=" * 72
    original_tok = _count_tokens(messages)

    # ── cost estimate (works without client) ──────────────────────────────────
    pricing       = _PRICING[MODEL]
    transcript    = _format_transcript(messages)
    input_tok_est = _estimate_tokens(transcript) + _estimate_tokens(TASK_CONTEXT) + _PROMPT_OVERHEAD_TOKENS
    estimated_cost = round(
        (input_tok_est * pricing["input"] + MAX_TOKENS * pricing["output"]) / 1_000_000, 8
    )

    print(sep)
    print(f"  ContextSummarizer demo  |  model={MODEL}  |  max_tokens={MAX_TOKENS}")
    print(sep)
    print(f"\n  Input conversation  : {len(messages)} messages ({original_tok:,} tokens est.)")
    print(f"  Transcript lines    : {len(transcript.splitlines())}")
    print(f"  Estimated cost      : ${estimated_cost:.6f} USD")
    print(f"  Model               : {MODEL}  (${pricing['input']}/M in, ${pricing['output']}/M out)")
    print(f"  Max summary tokens  : {MAX_TOKENS}")

    # ── show transcript sample ────────────────────────────────────────────────
    print(f"\n  Transcript sample (first 4 / last 2 lines):")
    tlines = transcript.splitlines()
    for ln in tlines[:4]:
        print(f"    {ln[:90]}")
    print("    ...")
    for ln in tlines[-2:]:
        print(f"    {ln[:90]}")

    # ── live call or dry-run ──────────────────────────────────────────────────
    if client is not None:
        summarizer = ContextSummarizer(
            client=client, model=MODEL, max_summary_tokens=MAX_TOKENS,
        )
        print(f"\n  Calling API...")
        t0      = time.perf_counter()
        result  = summarizer.summarize(messages, task_context=TASK_CONTEXT)
        elapsed = (time.perf_counter() - t0) * 1000

        print(sep)
        print("  METRICS")
        print(f"  {'-'*50}")
        print(f"  Original tokens   : {result.original_tokens:,}")
        print(f"  Summary tokens    : {result.summary_tokens:,}")
        print(f"  Compression ratio : {result.compression_ratio:.1f}x")
        print(f"  Actual cost       : ${result.cost_usd:.6f} USD")
        print(f"  Estimated cost    : ${estimated_cost:.6f} USD")
        delta_pct = (result.cost_usd - estimated_cost) / max(estimated_cost, 1e-9) * 100
        print(f"  Estimate delta    : {delta_pct:+.1f}%")
        print(f"  Latency           : {elapsed:.0f}ms")

        print(f"\n{sep}")
        print("  SUMMARY GENERATO")
        print(sep)
        for line in result.summary_text.splitlines():
            print(f"  {line}")
        print(sep)

    else:
        # Dry-run: show what the output would look like
        _DRY_SUMMARY = (
            "Completato:\n"
            "- Scansione struttura repository: trovati api/, models/, tests/, main.py\n"
            "- Identificate 5 SQL injection (main.py:12, :31; api/users.py:45; api/orders.py:23, :67)\n"
            "- Eseguiti test pytest: 23 pass, 4 fail, 1 error (jwt mancante)\n"
            "- Confermata query N+1 in get_orders_with_items() -- fix: LEFT JOIN\n"
            "- Redatto piano di refactoring 3 settimane, avviato fix main.py\n\n"
            "Informazioni chiave:\n"
            "- Budget sessione: $5 | Deadline piano: fine mese\n"
            "- 2 sviluppatori disponibili per 3 settimane\n"
            "- Fix SQL injection DEVE essere retrocompatibile (no cambi a signature o JSON)\n"
            "- Autenticazione usa PyJWT; CI avvia auth tramite docker-compose\n\n"
            "Prossimi step:\n"
            "- Completare fix SQL injection in api/users.py e api/orders.py (dopo main.py)\n"
            "- Aggiungere PyJWT a requirements.txt e correggere docker-compose in CI\n"
            "- Refactoring N+1 con JOIN -- settimana 2"
        )
        dry_tokens = _estimate_tokens(_DRY_SUMMARY)
        dry_ratio  = round(original_tok / max(dry_tokens, 1), 2)

        print(f"\n{sep}")
        print("  METRICS (dry-run -- add ANTHROPIC_API_KEY to .env for live values)")
        print(f"  {'-'*50}")
        print(f"  Original tokens   : {original_tok:,}")
        print(f"  Summary tokens    : {dry_tokens:,}  (sample output)")
        print(f"  Compression ratio : {dry_ratio:.1f}x  (sample output)")
        print(f"  Estimated cost    : ${estimated_cost:.6f} USD")

        print(f"\n{sep}")
        print("  SAMPLE SUMMARY (format atteso con API key)")
        print(sep)
        for line in _DRY_SUMMARY.splitlines():
            print(f"  {line}")
        print(sep)
