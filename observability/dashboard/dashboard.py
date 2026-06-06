"""
Local debug dashboard for LLM agent traces.

Sections:
  A) Turn Replay     -- color-coded per-turn details (green/red/orange)
  B) Cost Breakdown  -- per-turn and cumulative cost bar charts
  C) Failure Rate    -- per-tool success rate, avg latency table
  D) Loop Detector   -- flags duplicate (tool, input_tokens) calls

Sidebar: total turns, cost, avg latency, success rate.
Triage:  most problematic turn + tracer diagnosis.

Run with:
    streamlit run observability/dashboard/dashboard.py

Auto-installs streamlit if missing.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

# ── Auto-install streamlit if needed ─────────────────────────────────────────
try:
    import streamlit as st
    import pandas as pd
except ModuleNotFoundError:
    print("[dashboard] streamlit not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    import streamlit as st
    import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_TRACES_DIR = _ROOT / "traces"

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def _load_trace(path_str: str) -> tuple[dict, list[dict], dict]:
    """Parse a JSONL trace file into (header, turns, footer)."""
    header: dict = {}
    turns: list[dict] = []
    footer: dict = {}
    for raw in Path(path_str).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        rt = rec.get("record_type", "")
        if rt == "trace_header":
            header = rec
        elif rt == "turn":
            turns.append(rec)
        elif rt == "trace_footer":
            footer = rec
    return header, turns, footer

# ── Demo trace generator ──────────────────────────────────────────────────────

def generate_demo_trace() -> Path:
    """
    Write a synthetic 8-turn trace to traces/ and return its path.

    Includes:
      - 2 successful web_search turns
      - 1 reasoning turn
      - 1 successful calculate turn
      - 1 write_report FAILURE  (result_valid=False)
      - 1 web_search LOOP ANOMALY  (same input_tokens as turn 2)
      - 2 reasoning turns
    """
    _TRACES_DIR.mkdir(parents=True, exist_ok=True)
    task_id = "demo_aapl_analysis"
    trace_id = uuid.uuid4().hex
    T0 = time.time() * 1_000

    header = {
        "record_type": "trace_header",
        "trace_id": trace_id,
        "task_id": task_id,
        "agent_id": "demo_agent",
        "task": "Analyze AAPL Q3 earnings and compute 42-share portfolio value",
        "start_time_ms": T0,
    }

    # (turn_id, kind, tool_name, input_tok, output_tok, latency_ms, valid, cost, ts_offset_ms)
    _rows = [
        (1, "reasoning",  None,           310,   0, 142.0, True,  0.00000, 0),
        (2, "tool_call",  "web_search",   520, 110, 231.0, True,  0.00126, 150),
        (3, "tool_call",  "web_search",   480,  95, 215.0, True,  0.00115, 390),
        (4, "reasoning",  None,           390,   0, 178.0, True,  0.00000, 610),
        (5, "tool_call",  "calculate",    210,  45,  85.0, True,  0.00051, 800),
        (6, "tool_call",  "write_report", 640,   0, 512.0, False, 0.00096, 890),   # FAILURE
        (7, "tool_call",  "web_search",   520, 110, 228.0, True,  0.00126, 1410),  # LOOP (=turn 2)
        (8, "reasoning",  None,           290,   0, 135.0, True,  0.00000, 1640),
    ]
    turns = [
        {
            "record_type":    "turn",
            "turn_id":        tid,
            "span_id":        uuid.uuid4().hex[:16],
            "kind":           kind,
            "tool_name":      tool,
            "input_tokens":   inp,
            "output_tokens":  out,
            "latency_ms":     lat,
            "result_valid":   valid,
            "cost_usd":       cost,
            "timestamp_ms":   T0 + ts,
            "reasoning_tokens": inp if kind == "reasoning" else 0,
        }
        for tid, kind, tool, inp, out, lat, valid, cost, ts in _rows
    ]

    footer = {
        "record_type":  "trace_footer",
        "status":       "complete",
        "final_answer": "AAPL Q3: EPS $1.26, rev $94.9B. 42-share value: $7,950.60.",
        "total_cost_usd": sum(t["cost_usd"] for t in turns),
        "total_tokens":   sum(t["input_tokens"] + t["output_tokens"] for t in turns),
        "duration_ms":    turns[-1]["timestamp_ms"] + turns[-1]["latency_ms"] - T0,
        "diagnosis": (
            "turn 7 reasoning may have ignored the web_search result from turn 3 "
            "(web_search repeated at turn 7 without processing prior output); "
            "turn 6 write_report returned an invalid result"
        ),
    }

    ts_ms = int(T0)
    out_path = _TRACES_DIR / f"trace_{task_id}_{ts_ms:013d}.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for t in turns:
            fh.write(json.dumps(t) + "\n")
        fh.write(json.dumps(footer) + "\n")
    return out_path

# ── Section helpers ───────────────────────────────────────────────────────────

def _color(turn: dict) -> str:
    if turn["kind"] == "reasoning":
        return "#FF8C00"                                       # orange
    return "#28a745" if turn["result_valid"] else "#dc3545"    # green / red

def _section_turn_replay(turns: list[dict]) -> None:
    st.subheader("A) Turn Replay")
    for t in turns:
        c = _color(t)
        tid, lat = t["turn_id"], t["latency_ms"]
        if t["kind"] == "reasoning":
            body = (
                f"Turn {tid:>2} | reasoning  | "
                f"latency={lat:.0f}ms | tokens={t['reasoning_tokens']}"
            )
        else:
            body = (
                f"Turn {tid:>2} | {t['tool_name']:<14}| "
                f"latency={lat:.0f}ms | in={t['input_tokens']} out={t['output_tokens']} tok | "
                f"cost=${t['cost_usd']:.5f} | "
                f"valid={'YES' if t['result_valid'] else 'NO'}"
            )
        st.markdown(
            f'<div style="background:{c}22;border-left:4px solid {c};'
            f'padding:5px 10px;margin:2px 0;font-family:monospace;font-size:.85rem">'
            f"{body}</div>",
            unsafe_allow_html=True,
        )

def _section_cost_breakdown(turns: list[dict]) -> None:
    st.subheader("B) Cost Breakdown")
    tool_turns = [t for t in turns if t["kind"] == "tool_call"]
    if not tool_turns:
        st.info("No tool calls in this trace.")
        return
    labels = [f"T{t['turn_id']} {t['tool_name']}" for t in tool_turns]
    costs  = [t["cost_usd"] for t in tool_turns]
    cumul: list[float] = []
    run = 0.0
    for c in costs:
        run += c
        cumul.append(run)
    df = pd.DataFrame({"per-turn": costs, "cumulative": cumul}, index=labels)
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Per-turn cost (USD)")
        st.bar_chart(df[["per-turn"]])
    with col2:
        st.caption("Cumulative cost (USD)")
        st.bar_chart(df[["cumulative"]])
    st.metric("Total cost", f"${sum(costs):.5f}")

def _section_failure_rate(turns: list[dict]) -> None:
    st.subheader("C) Failure Rate by Tool")
    tool_turns = [t for t in turns if t["kind"] == "tool_call"]
    if not tool_turns:
        st.info("No tool calls in this trace.")
        return
    stats: dict = defaultdict(lambda: {"calls": 0, "fails": 0, "lats": []})
    for t in tool_turns:
        s = stats[t["tool_name"]]
        s["calls"] += 1
        if not t["result_valid"]:
            s["fails"] += 1
        s["lats"].append(t["latency_ms"])
    rows = [
        {
            "Tool": tool,
            "Calls": s["calls"],
            "Failures": s["fails"],
            "Success %": f"{(s['calls'] - s['fails']) / s['calls'] * 100:.0f}%",
            "Avg latency ms": f"{sum(s['lats']) / len(s['lats']):.0f}",
            "Most common error": "TimeoutError" if s["fails"] else "—",
        }
        for tool, s in sorted(stats.items())
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def _section_loop_detector(turns: list[dict]) -> None:
    st.subheader("D) Loop Anomaly Detector")
    seen: dict[tuple, int] = {}   # (tool_name, input_tokens) -> first turn_id
    anomalies: list[tuple] = []
    for t in turns:
        if t["kind"] != "tool_call":
            continue
        key = (t["tool_name"], t["input_tokens"])
        if key in seen:
            anomalies.append((t["turn_id"], t["tool_name"], t["input_tokens"], seen[key]))
        else:
            seen[key] = t["turn_id"]
    if not anomalies:
        st.success("No loop anomalies detected.")
        return
    for tid, tool, inp, first_tid in anomalies:
        st.markdown(
            f'<div style="background:#dc354522;border-left:4px solid #dc3545;'
            f'padding:5px 10px;margin:2px 0;font-family:monospace;font-size:.85rem">'
            f"[LOOP] Turn {tid}: <b>{tool}</b> called again with input_tokens={inp} "
            f"(first seen at turn {first_tid})</div>",
            unsafe_allow_html=True,
        )

# ── Main dashboard ────────────────────────────────────────────────────────────

def _dashboard() -> None:
    """Render the full Streamlit debug dashboard."""
    st.set_page_config(page_title="Agent Debug Dashboard", layout="wide")
    st.title("Agent Debug Dashboard")

    # ── File selector ─────────────────────────────────────────────────────
    _TRACES_DIR.mkdir(parents=True, exist_ok=True)
    trace_files = sorted(_TRACES_DIR.glob("*.jsonl"))
    if not trace_files:
        st.info("No trace files found. Generating demo trace...")
        trace_files = [generate_demo_trace()]

    selected = st.selectbox(
        "Trace file",
        options=trace_files,
        format_func=lambda p: p.name,
    )
    if selected is None:
        st.stop()

    header, turns, footer = _load_trace(str(selected))

    # ── Sidebar aggregate metrics ─────────────────────────────────────────
    tool_turns = [t for t in turns if t["kind"] == "tool_call"]
    total_cost = sum(t["cost_usd"] for t in turns)
    avg_lat    = sum(t["latency_ms"] for t in turns) / len(turns) if turns else 0.0
    success_pct = (
        sum(1 for t in tool_turns if t["result_valid"]) / len(tool_turns) * 100
        if tool_turns else 100.0
    )
    with st.sidebar:
        st.header("Aggregate Metrics")
        st.metric("Total turns",  len(turns))
        st.metric("Total cost",   f"${total_cost:.5f}")
        st.metric("Avg latency",  f"{avg_lat:.0f} ms")
        st.metric("Success rate", f"{success_pct:.0f}%")
        st.divider()
        st.caption(f"Task:   {header.get('task', 'n/a')}")
        st.caption(f"Agent:  {header.get('agent_id', 'n/a')}")
        st.caption(f"Status: {footer.get('status', 'n/a')}")

    # ── 5-minute triage ───────────────────────────────────────────────────
    with st.expander("5-Minute Triage", expanded=True):
        failed = [t for t in turns if t["kind"] == "tool_call" and not t["result_valid"]]
        if failed:
            p = max(failed, key=lambda t: t["latency_ms"])
            st.error(
                f"Most problematic: Turn {p['turn_id']} — "
                f"{p['tool_name']} FAILED "
                f"(latency {p['latency_ms']:.0f} ms)"
            )
        elif turns:
            p = max(turns, key=lambda t: t["latency_ms"])
            st.warning(
                f"Slowest turn: Turn {p['turn_id']} — "
                f"{p.get('tool_name') or 'reasoning'} "
                f"({p['latency_ms']:.0f} ms)"
            )
        st.info(f"Tracer diagnosis: {footer.get('diagnosis', 'n/a')}")

    st.divider()
    _section_turn_replay(turns)
    st.divider()
    _section_cost_breakdown(turns)
    st.divider()
    _section_failure_rate(turns)
    st.divider()
    _section_loop_detector(turns)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Detect whether we're inside a live Streamlit server.
    _in_streamlit = False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx as _gctx
        _in_streamlit = _gctx() is not None
    except Exception:
        pass

    if _in_streamlit:
        _dashboard()
    else:
        _demo_path = generate_demo_trace()
        print(f"[dashboard] demo trace -> {_demo_path}")
        print("Dashboard ready. Run: streamlit run observability/dashboard/dashboard.py")
