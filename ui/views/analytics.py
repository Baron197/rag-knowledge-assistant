"""Analytics page -- filterable charts over the per-query trace log.

Pulls the raw per-query records from GET /analytics and turns them into a small
dashboard: KPIs, time-series (cost, latency, tokens), a latency distribution, a
retrieved-document frequency chart, a scatter, and a filterable/exportable table.

Chart choices match the data:
  * time-series (cost/latency/tokens over time) -> line / area charts
  * a distribution (latency)                    -> histogram (binned bar chart)
  * category frequency (documents)              -> bar chart
  * a relationship (tokens vs latency)          -> scatter chart

Note: cache hits are intentionally NOT traced, so this reflects real (uncached)
retrieval + generation work only.
"""
from __future__ import annotations

import math
from collections import Counter

import common
import pandas as pd
import requests
import streamlit as st
from common import API_URL, get_health, invalidate_cache

REFUSAL_MARKER = "don't have enough information"

common.use_wide()   # data-dense dashboard -> widen past the chat reading column
common.render_header()
st.markdown("### :material/monitoring: Query analytics")

health = get_health()
if health is None:
    st.markdown(
        f"""
<div class="panel err">
  <h3>Can't reach the API</h3>
  <p>Analytics reads the query traces from the FastAPI service.</p>
  <p class="mono">{API_URL}</p>
</div>
""",
        unsafe_allow_html=True)
    if st.button("Retry connection", type="primary"):
        invalidate_cache()
        st.rerun()
    st.stop()

# --- Load the trace records ---------------------------------------------------
try:
    resp = requests.get(f"{API_URL}/analytics", params={"limit": 5000}, timeout=15)
    resp.raise_for_status()
    rows = resp.json().get("queries", [])
except requests.RequestException as exc:  # noqa: BLE001
    st.error(f"Couldn't load analytics: {exc}")
    st.stop()

if not rows:
    st.info("No queries have been traced yet. Ask a few questions on the **Ask** "
            "page (cache hits aren't traced), then come back.")
    st.stop()


@st.cache_data(show_spinner=False)
def to_frame(records: list[dict]) -> pd.DataFrame:
    """Flatten the trace records into a tidy, typed DataFrame."""
    out = []
    for x in records:
        t = x.get("timings_ms", {}) or {}
        tok = x.get("tokens", {}) or {}
        retr = float(t.get("retrieval", 0.0))
        gen = float(t.get("generation", 0.0))
        preview = (x.get("answer_preview", "") or "")
        out.append({
            "ts": pd.to_datetime(x.get("ts", 0), unit="s"),
            "question": x.get("question", ""),
            "answer_preview": preview,
            "retrieval_ms": retr,
            "generation_ms": gen,
            "latency_ms": sum(float(v) for v in t.values()),
            "embedding_tok": int(tok.get("embedding", 0)),
            "prompt_tok": int(tok.get("prompt", 0)),
            "completion_tok": int(tok.get("completion", 0)),
            "tokens": int(tok.get("prompt", 0)) + int(tok.get("completion", 0)),
            "cost_usd": float(x.get("cost_usd", 0.0)),
            "n_contexts": int(x.get("n_contexts", 0)),
            "refused": REFUSAL_MARKER in preview.lower(),
            "sources": tuple(x.get("sources", []) or ()),
        })
    return pd.DataFrame(out).sort_values("ts").reset_index(drop=True)


df = to_frame(rows)


def range_filter(label: str, series: pd.Series, is_int: bool = False):
    """A slider that degrades gracefully when the column has no spread."""
    lo, hi = float(series.min()), float(series.max())
    if hi <= lo:
        st.caption(f"{label}: single value ({lo:g}) — no range to filter.")
        return None
    if is_int:
        return st.slider(label, int(lo), int(hi), (int(lo), int(hi)))
    return st.slider(label, lo, hi, (lo, hi))


# --- Filters (sidebar) --------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sgroup">Filters</div>', unsafe_allow_html=True)

    kind = st.radio("Answer type", ["All", "Answered", "Refused"], horizontal=True)

    # Time range (guarded against a single-timestamp dataset).
    time_range = None
    tmin, tmax = df["ts"].min(), df["ts"].max()
    if tmax > tmin:
        time_range = st.slider(
            "Time range",
            min_value=tmin.to_pydatetime(), max_value=tmax.to_pydatetime(),
            value=(tmin.to_pydatetime(), tmax.to_pydatetime()), format="MM/DD HH:mm",
        )

    lat_range = range_filter("Latency (ms)", df["latency_ms"])
    cost_range = range_filter("Cost (USD)", df["cost_usd"])
    ctx_range = range_filter("Contexts", df["n_contexts"], is_int=True)
    search = st.text_input("Search question", placeholder="e.g. rate limit")
    st.caption("Cache hits aren't traced, so charts reflect real uncached queries.")

# --- Apply filters ------------------------------------------------------------
f = df.copy()
if kind == "Answered":
    f = f[~f["refused"]]
elif kind == "Refused":
    f = f[f["refused"]]
if time_range is not None:
    f = f[(f["ts"] >= pd.Timestamp(time_range[0])) & (f["ts"] <= pd.Timestamp(time_range[1]))]
for col, rng in (("latency_ms", lat_range), ("cost_usd", cost_range), ("n_contexts", ctx_range)):
    if rng is not None:
        f = f[(f[col] >= rng[0]) & (f[col] <= rng[1])]
if search.strip():
    # regex=False: treat the query as a literal so "(" / "[" don't raise.
    f = f[f["question"].str.contains(search.strip(), case=False, na=False, regex=False)]

st.caption(f"Showing **{len(f)}** of {len(df)} traced queries.")
if f.empty:
    st.warning("No queries match the current filters.")
    st.stop()

# --- KPI row ------------------------------------------------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Queries", len(f))
k2.metric("Total cost", f"${f['cost_usd'].sum():.4f}")
k3.metric("Avg latency", f"{f['latency_ms'].mean():.0f} ms")
k4.metric("p95 latency", f"{f['latency_ms'].quantile(0.95):.0f} ms")
k5.metric("Avg tokens", f"{f['tokens'].mean():.0f}")
k6.metric("Refused", f"{f['refused'].mean() * 100:.0f}%")

# --- Time-series --------------------------------------------------------------
st.markdown("#### Cost over time")
cost_ts = f[["ts", "cost_usd"]].copy()
cost_ts["cumulative_cost_usd"] = cost_ts["cost_usd"].cumsum()
st.area_chart(cost_ts, x="ts", y="cumulative_cost_usd", color="#4F46E5", height=240)
if f["cost_usd"].sum() == 0:
    st.caption("Cost is $0 on the fake / local providers — switch to OpenAI to see real spend.")

c_lat, c_tok = st.columns(2)
with c_lat:
    st.markdown("#### Latency over time")
    st.line_chart(f, x="ts", y=["retrieval_ms", "generation_ms"],
                  color=["#9CB4FF", "#4F46E5"], height=240)
with c_tok:
    st.markdown("#### Token usage over time")
    st.area_chart(f, x="ts", y=["prompt_tok", "completion_tok"],
                  color=["#9CB4FF", "#4F46E5"], height=240)

# --- Distribution + document frequency ----------------------------------------
c_dist, c_src = st.columns(2)
with c_dist:
    st.markdown("#### Latency distribution")
    lo, hi = f["latency_ms"].min(), f["latency_ms"].max()
    if hi > lo:
        binned = pd.cut(f["latency_ms"], bins=10)
        hist = binned.value_counts().sort_index()
        # Adapt label precision to the bin width so narrow ranges (e.g. fake-mode
        # sub-millisecond latencies) don't round to duplicate "0–0" buckets.
        bin_width = (hi - lo) / 10
        dec = max(0, min(4, math.ceil(-math.log10(bin_width)) + 1)) if bin_width > 0 else 0
        # Use the numeric bin lower-edge as x so bars stay in latency order — string
        # range labels would sort alphabetically ("1200–…" before "800–…").
        hist_df = pd.DataFrame({
            "latency (ms, ≥)": [round(float(iv.left), dec) for iv in hist.index],
            "queries": hist.to_numpy(),
        })
        st.bar_chart(hist_df, x="latency (ms, ≥)", y="queries", color="#4F46E5", height=240)
    else:
        st.caption(f"All queries share ~{lo:.0f} ms latency — no spread to bin.")
with c_src:
    st.markdown("#### Most retrieved documents")
    counter: Counter[str] = Counter()
    for srcs in f["sources"]:
        counter.update(set(srcs))   # count each document once per query
    if counter:
        src_df = (pd.DataFrame(counter.items(), columns=["document", "queries"])
                  .sort_values("queries", ascending=False).head(15))
        st.bar_chart(src_df, x="document", y="queries", color="#059669", height=240)
    else:
        st.caption("No sources recorded for the filtered queries.")

# --- Relationship -------------------------------------------------------------
st.markdown("#### Tokens vs latency")
st.caption("Each point is a query; colour marks whether it was refused.")
st.scatter_chart(f, x="tokens", y="latency_ms", color="refused", height=260)

# --- Query log ----------------------------------------------------------------
st.markdown("#### Query log")
table = f[["ts", "question", "answer_preview", "latency_ms", "retrieval_ms",
           "generation_ms", "tokens", "cost_usd", "n_contexts", "refused"]]
st.dataframe(
    table.iloc[::-1],   # newest first
    use_container_width=True,
    hide_index=True,
    column_config={
        "ts": st.column_config.DatetimeColumn("Time", format="MMM D, HH:mm:ss"),
        "question": st.column_config.TextColumn("Question", width="medium"),
        "answer_preview": st.column_config.TextColumn("Answer preview", width="medium"),
        "latency_ms": st.column_config.NumberColumn("Latency", format="%.0f ms"),
        "retrieval_ms": st.column_config.NumberColumn("Retrieval", format="%.0f ms"),
        "generation_ms": st.column_config.NumberColumn("Generation", format="%.0f ms"),
        "tokens": st.column_config.NumberColumn("Tokens"),
        "cost_usd": st.column_config.NumberColumn("Cost", format="$%.6f"),
        "n_contexts": st.column_config.NumberColumn("Contexts"),
        "refused": st.column_config.CheckboxColumn("Refused"),
    },
)
st.download_button(
    "Download filtered data (CSV)",
    data=f.drop(columns=["sources"]).to_csv(index=False),
    file_name="knowledge-assistant-analytics.csv",
    mime="text/csv",
)
