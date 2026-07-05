"""Evaluation page -- read-only view of the eval harness reports.

Fetches GET /eval-results and renders, for a selected run: the retrieval metrics
(recall@k / recall@1 / MRR / refusal accuracy), the Ragas generation metrics
when the run was produced on the OpenAI path, a vector-vs-hybrid A/B comparison,
a metric trend across runs (kept honest by separating provider tiers), and a
per-question pass/fail table.

The app never runs the eval itself (that re-ingests and, on OpenAI, costs money
and time); reports are generated out-of-band with `make eval` / `make
eval-compare` and simply displayed here.
"""
from __future__ import annotations

import html
import json
from datetime import datetime

import common
import pandas as pd
import requests
import streamlit as st
from common import API_URL, get_health, invalidate_cache

SEMANTIC = ("openai", "hf")   # embedding tiers for which refusal/Ragas are meaningful

common.use_wide()
common.render_header()
st.markdown("### :material/verified: Model evaluation")


def pretty_ts(stamp: str) -> str:
    """Render an eval timestamp (e.g. 20260702T172702Z) for humans."""
    try:
        return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        return str(stamp)


def fmt(v, nd: int = 3) -> str:
    """Format a metric, or an em dash when it's missing/not applicable."""
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


_REPORT_BRAND = "Knowledge Assistant"


def build_html_report(run: dict) -> str:
    """Render one eval run as a standalone, print-friendly HTML document.

    Includes the run metadata, the metric summary, and one card per question with
    the model's answer and the expected answer. Open it in a browser and print to
    PDF (Ctrl/Cmd-P) for a clean, shareable report. Everything is escaped and
    inlined, so the file is self-contained.
    """
    prov = run.get("providers", {}) or {}
    rm = run.get("retrieval_metrics", {}) or {}
    ragas = run.get("ragas_metrics") or {}
    per_q = run.get("per_question", []) or []
    esc = html.escape

    def card(q: dict) -> str:
        ok = bool(q.get("correct"))
        status, cls = ("PASS", "pass") if ok else ("FAIL", "fail")
        kind = "Refusal" if q.get("refusal_question") else "Answerable"
        exp = q.get("expected_sources", []) or []
        retr = q.get("retrieved_sources", []) or []
        parts = [
            f'<div class="q">{esc(str(q.get("question", "")))}</div>',
            f'<div class="meta"><span class="badge {cls}">{status}</span>'
            f'<span class="badge kind">{kind}</span>'
            f'<span>first rank: {q.get("first_relevant_rank") or "—"}</span>'
            f'<span>${q.get("cost_usd", 0):.6f}</span>'
            f'<span>{q.get("latency_ms", 0):.0f} ms</span></div>',
        ]
        answer = q.get("answer")
        if answer is not None:
            parts.append(f'<div class="lbl">Model answer</div><div class="ans">{esc(str(answer))}</div>')
        gt = q.get("ground_truth")
        if gt:
            parts.append(f'<div class="lbl">Expected answer</div><div class="gt">{esc(str(gt))}</div>')
        exp_h = ", ".join(f"<code>{esc(s)}</code>" for s in exp) or "<i>none (refusal expected)</i>"
        retr_h = ", ".join(f'<code>{esc(s)}</code>{" &#10003;" if s in exp else ""}'
                           for s in retr) or "<i>none</i>"
        parts.append(f'<div class="src"><b>Expected sources:</b> {exp_h}</div>')
        parts.append(f'<div class="src"><b>Retrieved sources:</b> {retr_h}</div>')
        return f'<div class="card">{"".join(parts)}</div>'

    passed = sum(1 for q in per_q if q.get("correct"))
    prov_line = " · ".join(f"{esc(str(k))}={esc(str(v))}" for k, v in prov.items()) or "—"
    summary = dict(rm)
    if ragas:
        summary.update({f"ragas.{k}": v for k, v in ragas.items()})
    rows = "".join(f"<tr><td>{esc(str(k))}</td><td>{esc(str(v))}</td></tr>" for k, v in summary.items())
    css = (
        "*{box-sizing:border-box}"
        "body{font:14px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
        "color:#1e293b;max-width:900px;margin:24px auto;padding:0 20px}"
        "h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:26px 0 10px}"
        ".sub{color:#64748b;font-size:12.5px;margin-bottom:16px}"
        "table.summary{border-collapse:collapse;font-size:13px;margin:6px 0}"
        "table.summary td{border:1px solid #e2e8f0;padding:4px 10px}"
        "table.summary td:first-child{color:#475569}"
        ".card{border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px;margin:12px 0;"
        "page-break-inside:avoid}"
        ".q{font-weight:600;font-size:15px;margin-bottom:8px}"
        ".meta{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;color:#64748b;"
        "align-items:center;margin-bottom:8px}"
        ".badge{font-weight:700;font-size:11px;border-radius:999px;padding:2px 9px}"
        ".badge.pass{background:#dcfce7;color:#166534}.badge.fail{background:#fee2e2;color:#991b1b}"
        ".badge.kind{background:#eef2ff;color:#3730a3}"
        ".lbl{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#94a3b8;"
        "margin:10px 0 3px;font-weight:600}"
        ".ans{background:#f8fafc;border-left:3px solid #6366f1;border-radius:6px;padding:8px 12px;"
        "white-space:pre-wrap}"
        ".gt{background:#f8fafc;border-left:3px solid #10b981;border-radius:6px;padding:8px 12px;"
        "white-space:pre-wrap}"
        ".src{font-size:12.5px;color:#475569;margin-top:6px}"
        "code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}"
        "@media print{body{margin:0}.card{border-color:#cbd5e1}}"
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{esc(_REPORT_BRAND)} — evaluation report</title>"
        f"<style>{css}</style></head><body>"
        f"<h1>{esc(_REPORT_BRAND)} — evaluation report</h1>"
        f"<div class='sub'>{prov_line}<br>Run: {esc(str(run.get('timestamp', run.get('_name', ''))))}"
        f" · {passed}/{len(per_q)} passed</div>"
        f"<h2>Metrics</h2><table class='summary'>{rows}</table>"
        f"<h2>Per-question detail ({len(per_q)})</h2>"
        f"{''.join(card(q) for q in per_q)}"
        "</body></html>"
    )


health = get_health()
if health is None:
    st.markdown(
        f"""
<div class="panel err">
  <h3>Can't reach the API</h3>
  <p>The Evaluation page reads reports from the FastAPI service.</p>
  <p class="mono">{API_URL}</p>
</div>
""",
        unsafe_allow_html=True)
    if st.button("Retry connection", type="primary"):
        invalidate_cache()
        st.rerun()
    st.stop()

# --- Load reports -------------------------------------------------------------
top = st.columns([1, 0.16])
with top[1]:
    if st.button(":material/refresh: Refresh", use_container_width=True):
        st.rerun()
try:
    resp = requests.get(f"{API_URL}/eval-results", timeout=15)
    resp.raise_for_status()
    payload = resp.json()
except requests.RequestException as exc:  # noqa: BLE001
    st.error(f"Couldn't load evaluation reports: {exc}")
    st.stop()

eval_runs = payload.get("eval_runs", [])
compare_runs = payload.get("compare_runs", [])

if not eval_runs:
    st.info("No evaluation reports found yet.")
    st.markdown(
        """
<div class="panel">
  <h3>Generate a report</h3>
  <p>Run the eval harness from a terminal, then click <b>↻ Refresh</b>:</p>
  <p class="mono">make eval NO_RAGAS=1&nbsp;&nbsp;# retrieval metrics — no API key</p>
  <p class="mono">make eval&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# + Ragas generation metrics (needs OPENAI_API_KEY)</p>
  <p class="mono">make eval-compare&nbsp;&nbsp;# vector vs hybrid A/B</p>
  <p>Reports are written to <span class="mono">eval/results/</span>.</p>
</div>
""",
        unsafe_allow_html=True)
    st.stop()

# --- Run selector -------------------------------------------------------------
labels = []
for r in eval_runs:
    p = r.get("providers", {}) or {}
    labels.append(f"{pretty_ts(r.get('timestamp', r.get('_name', '?')))}  ·  "
                  f"llm={p.get('llm', '?')}, emb={p.get('embedding', '?')}, "
                  f"mode={p.get('retrieval_mode', '?')}")
choice = st.selectbox("Evaluation run", range(len(eval_runs)),
                      format_func=lambda i: labels[i])
run = eval_runs[choice]
prov = run.get("providers", {}) or {}
rm = run.get("retrieval_metrics", {}) or {}
ragas = run.get("ragas_metrics") or None
semantic = prov.get("embedding") in SEMANTIC

# Provider banner (escape the report's provider strings, like every other view).
pills = " ".join(
    f'<span class="pill">{k}:{html.escape(str(prov.get(k, "?")))}</span>'
    for k in ("llm", "embedding", "vector_backend", "retrieval_mode", "top_k")
)
st.markdown(f'<div class="pills" style="margin:2px 0 6px">{pills}</div>',
            unsafe_allow_html=True)
if not semantic:
    st.caption("⚠ This run used keyword-hashing `fake` embeddings, so **refusal "
               "accuracy** and **Ragas** metrics aren't meaningful. Re-run with "
               "`EMBEDDING_PROVIDER=hf` or `openai` for real quality numbers.")

# --- Retrieval metrics --------------------------------------------------------
st.markdown("#### Retrieval quality")
c = st.columns(4)
c[0].metric("Context recall@k", fmt(rm.get("context_recall_at_k")))
c[1].metric("Recall@1", fmt(rm.get("recall_at_1")))
c[2].metric("MRR", fmt(rm.get("mrr")))
c[3].metric("Refusal accuracy",
            fmt(rm.get("refusal_accuracy")) if semantic else "n/a",
            help="Out-of-scope questions correctly refused. Needs semantic embeddings.")
c2 = st.columns(4)
c2[0].metric("Answerable Qs", rm.get("answerable_questions", "—"))
c2[1].metric("Refusal Qs", rm.get("refusal_questions", "—"))
c2[2].metric("Avg cost / query", f"${rm.get('avg_cost_usd', 0):.6f}")
c2[3].metric("Avg latency", f"{rm.get('avg_latency_ms', 0):.0f} ms")

# --- Generation metrics (Ragas) ----------------------------------------------
st.markdown("#### Generation quality (Ragas)")
if ragas:
    g = st.columns(4)
    g[0].metric("Faithfulness", fmt(ragas.get("faithfulness")),
                help="Share of answer claims supported by the retrieved context (anti-hallucination).")
    g[1].metric("Answer relevancy", fmt(ragas.get("answer_relevancy")))
    g[2].metric("Context precision", fmt(ragas.get("context_precision")))
    g[3].metric("Context recall", fmt(ragas.get("context_recall")))
else:
    st.info("No Ragas metrics in this run — they require the OpenAI path "
            "(`LLM_PROVIDER=openai`). Run `make eval` with a key to populate them.")

# --- Quality-at-a-glance bar --------------------------------------------------
bars = [("Recall@k", rm.get("context_recall_at_k")),
        ("Recall@1", rm.get("recall_at_1")),
        ("MRR", rm.get("mrr"))]
if semantic:
    bars.append(("Refusal", rm.get("refusal_accuracy")))
if ragas:
    bars += [("Faithful.", ragas.get("faithfulness")),
             ("Ans.rel.", ragas.get("answer_relevancy")),
             ("Ctx prec.", ragas.get("context_precision")),
             ("Ctx rec.", ragas.get("context_recall"))]
bar_df = pd.DataFrame([(n, float(v)) for n, v in bars if isinstance(v, (int, float))],
                      columns=["metric", "score"])
if not bar_df.empty:
    st.bar_chart(bar_df, x="metric", y="score", color="#4F46E5", height=260)
    st.caption("All scores are on a 0–1 scale (higher is better).")

# --- A/B: vector vs hybrid ----------------------------------------------------
st.markdown("#### Retrieval A/B — vector vs hybrid")
if compare_runs:
    # Prefer an A/B run from the same embedding tier as the selected run, so the
    # deltas below can't be misread as belonging to a different provider.
    sel_emb = prov.get("embedding")
    match = next((c for c in compare_runs
                  if (c.get("providers", {}) or {}).get("embedding") == sel_emb), None)
    comp = match or compare_runs[0]
    if match is None:
        st.caption(f"⚠ No A/B run for the selected `{sel_emb}` embeddings — showing the "
                   "most recent A/B, which used a different provider.")
    res = comp.get("results", {}) or {}
    vec, hyb = res.get("vector", {}) or {}, res.get("hybrid", {}) or {}
    metric_names = [("context_recall_at_k", "Recall@k"), ("recall_at_1", "Recall@1"), ("mrr", "MRR")]
    long = []
    for key, label in metric_names:
        for mode, d in (("vector", vec), ("hybrid", hyb)):
            if isinstance(d.get(key), (int, float)):
                long.append({"metric": label, "mode": mode, "score": float(d[key])})
    if long:
        st.bar_chart(pd.DataFrame(long), x="metric", y="score", color="mode",
                     stack=False, height=260)
    delta_rows = []
    for key, label in metric_names:
        v, h = vec.get(key), hyb.get(key)
        if isinstance(v, (int, float)) and isinstance(h, (int, float)):
            d = round(h - v, 3)
            delta_str = f"+{d:.3f}" if d > 0 else f"{d:.3f}"   # keep the + on gains, incl. ints
        else:
            delta_str = "—"
        delta_rows.append({"Metric": label, "vector": fmt(v), "hybrid": fmt(h),
                           "Δ (hybrid−vector)": delta_str})
    st.dataframe(pd.DataFrame(delta_rows), hide_index=True, use_container_width=True)
    st.caption(f"From `{comp.get('_name', 'compare')}` · embedding="
               f"`{(comp.get('providers', {}) or {}).get('embedding', '?')}`. "
               "On real semantic embeddings hybrid's edge is typically larger.")
else:
    st.info("No A/B comparison yet — run `make eval-compare` to generate one.")

# --- Trend across runs --------------------------------------------------------
st.markdown("#### Metric trend across runs")
metric_opts = {"Context recall@k": ("retrieval_metrics", "context_recall_at_k"),
               "Recall@1": ("retrieval_metrics", "recall_at_1"),
               "MRR": ("retrieval_metrics", "mrr"),
               "Refusal accuracy": ("retrieval_metrics", "refusal_accuracy")}
if any(r.get("ragas_metrics") for r in eval_runs):
    metric_opts["Faithfulness (Ragas)"] = ("ragas_metrics", "faithfulness")
    metric_opts["Answer relevancy (Ragas)"] = ("ragas_metrics", "answer_relevancy")
sel = st.selectbox("Metric", list(metric_opts), index=2)
section, key = metric_opts[sel]
trend = []
for r in reversed(eval_runs):   # chronological
    block = r.get(section) or {}
    val = block.get(key)
    p = r.get("providers", {}) or {}
    # Refusal accuracy is only meaningful on semantic embeddings; skip fake runs
    # here too, matching the "n/a" treatment on the cards and quality bar.
    if key == "refusal_accuracy" and p.get("embedding") not in SEMANTIC:
        continue
    if isinstance(val, (int, float)):
        trend.append({
            "run": pd.to_datetime(r.get("timestamp"), format="%Y%m%dT%H%M%SZ", errors="coerce"),
            sel: float(val),
            "providers": f"{p.get('llm', '?')}/{p.get('embedding', '?')}",
        })
trend_df = pd.DataFrame(trend)
if not trend_df.empty:   # guard: an empty list has no "run" column to dropna on
    trend_df = trend_df.dropna(subset=["run"])
if len(trend_df) >= 2:
    st.line_chart(trend_df, x="run", y=sel, color="providers", height=260)
    st.caption("Lines are split by provider tier (`llm/embedding`) so fake and real "
               "runs aren't blended into one misleading trend.")
elif len(trend_df) == 1:
    st.caption("Only one run has this metric — run the eval again to see a trend.")
else:
    st.caption("No runs carry this metric yet.")

# --- Per-question detail ------------------------------------------------------
st.markdown("#### Per-question detail")
per_q = run.get("per_question", []) or []
if per_q:
    pdf = pd.DataFrame([{
        "question": q.get("question", ""),
        "correct": bool(q.get("correct")),
        "refusal_q": bool(q.get("refusal_question")),
        "first_rank": q.get("first_relevant_rank"),
        "expected": ", ".join(q.get("expected_sources", []) or []),
        "retrieved": ", ".join(q.get("retrieved_sources", []) or []),
        "cost_usd": q.get("cost_usd", 0.0),
        "latency_ms": q.get("latency_ms", 0.0),
    } for q in per_q])

    view = st.radio("Show", ["All", "Passed", "Failed", "Refusal questions"], horizontal=True)
    if view == "Passed":
        pdf = pdf[pdf["correct"]]
    elif view == "Failed":
        pdf = pdf[~pdf["correct"]]
    elif view == "Refusal questions":
        pdf = pdf[pdf["refusal_q"]]

    st.dataframe(
        pdf, hide_index=True, use_container_width=True,
        column_config={
            "question": st.column_config.TextColumn("Question", width="medium"),
            "correct": st.column_config.CheckboxColumn("Pass"),
            "refusal_q": st.column_config.CheckboxColumn("Refusal Q"),
            "first_rank": st.column_config.NumberColumn("First rank"),
            "expected": st.column_config.TextColumn("Expected sources", width="small"),
            "retrieved": st.column_config.TextColumn("Retrieved sources", width="medium"),
            "cost_usd": st.column_config.NumberColumn("Cost", format="$%.6f"),
            "latency_ms": st.column_config.NumberColumn("Latency", format="%.0f ms"),
        },
    )
    st.caption(f"Showing {len(pdf)} of {len(per_q)} questions.")

    # Drill into one question: full text, model answer, expected answer, sources.
    st.markdown("##### Inspect a question")

    def _q_label(i: int) -> str:
        q = per_q[i]
        icon = "✅" if q.get("correct") else "❌"
        return f"{icon}  {(q.get('question') or '')[:90]}"

    idx = st.selectbox("Pick a question to read its full answer", range(len(per_q)),
                       format_func=_q_label, key="perq_inspect")
    q = per_q[idx]
    st.caption(
        f"{'✅ Passed' if q.get('correct') else '❌ Failed'} · "
        f"{'Refusal question' if q.get('refusal_question') else 'Answerable'} · "
        f"first rank: {q.get('first_relevant_rank') or '—'} · "
        f"${q.get('cost_usd', 0):.6f} · {q.get('latency_ms', 0):.0f} ms"
    )
    st.markdown(f"**Question**\n\n> {q.get('question', '')}")

    st.markdown("**Model answer**")
    if q.get("answer") is not None:
        with st.container(border=True):
            st.markdown(q.get("answer") or "_(empty answer)_")
    else:
        st.info("This report predates answer capture. Re-run the eval "
                "(`python -m eval.run_eval`, updated harness) to record the model's "
                "answer for each question.")

    if q.get("ground_truth"):
        st.markdown("**Expected answer (ground truth)**")
        with st.container(border=True):
            st.markdown(q.get("ground_truth"))

    exp = q.get("expected_sources", []) or []
    retr = q.get("retrieved_sources", []) or []
    sc = st.columns(2)
    sc[0].markdown("**Expected sources**\n\n"
                   + ("\n".join(f"- `{s}`" for s in exp) or "_none (refusal expected)_"))
    sc[1].markdown("**Retrieved sources**\n\n"
                   + ("\n".join(f"- `{s}`" + ("  ✓" if s in exp else "") for s in retr) or "_none_"))

st.divider()
_dl = st.columns(2)
_base = (run.get("_name", "eval-report.json") or "eval-report").rsplit(".", 1)[0]
_dl[0].download_button("⬇  Report (JSON)", data=json.dumps(run, indent=2),
                       file_name=f"{_base}.json", mime="application/json",
                       use_container_width=True)
_dl[1].download_button("⬇  Report (HTML — open & print to PDF)", data=build_html_report(run),
                       file_name=f"{_base}.html", mime="text/html",
                       use_container_width=True)
st.caption("Tip: open the HTML report and print to PDF (Ctrl/Cmd-P → Save as PDF) for a "
           "clean, shareable document with every question and answer.")
