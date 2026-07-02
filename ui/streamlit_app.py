"""Streamlit UI -- a thin client over the FastAPI service.

Deliberately does no RAG itself: it calls the same HTTP API as everything else,
which keeps the architecture honest (UI -> API -> pipeline) and mirrors how
you'd build a real product. Shows the answer, its citations, and the per-query
cost/latency so the "production" story is visible, not just claimed.

Run:  streamlit run ui/streamlit_app.py   (with the API already running)
"""
from __future__ import annotations

import os

import requests
import streamlit as st

# Where the FastAPI service is reachable (overridable via env for Docker/remote).
API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚", layout="centered")
st.title("📚 RAG Knowledge Assistant")
st.caption("Grounded Q&A over your documentation — with citations, cost and latency.")

# --- Sidebar: live API status, a re-ingest button, uploads, and metrics --------
with st.sidebar:
    st.header("Status")
    try:
        h = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success("API connected")
        st.write(f"**Backend:** {h['vector_backend']}")
        st.write(f"**LLM:** {h['llm_provider']}")
        st.write(f"**Retrieval:** {h.get('retrieval_mode', 'vector')}")
        st.write(f"**Indexed chunks:** {h['indexed_chunks']}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"API not reachable at {API_URL}\n\n{exc}")
        st.info("Start it with:  `make api`")

    # Trigger a re-index of data/docs without leaving the UI.
    if st.button("Re-ingest documents"):
        with st.spinner("Ingesting..."):
            r = requests.post(f"{API_URL}/ingest", timeout=600).json()
        st.success(f"Indexed {r['ingested_chunks']} chunks")

    # Upload new documents from the browser: saved into data/docs on the server,
    # then indexed immediately via the /upload endpoint (UI -> API -> pipeline).
    st.divider()
    st.subheader("Add documents")
    uploaded = st.file_uploader(
        "Upload files to index",
        type=["md", "txt", "html", "htm", "pdf"],
        accept_multiple_files=True,
    )
    if st.button("Upload & ingest") and uploaded:
        payload = [("files", (f.name, f.getvalue())) for f in uploaded]
        with st.spinner("Uploading and indexing..."):
            r = requests.post(f"{API_URL}/upload", files=payload, timeout=600).json()
        st.success(f"Indexed {r['indexed_chunks']} chunks from {len(r['saved'])} file(s)")
        if r.get("skipped"):
            st.warning("Skipped (unsupported): " + ", ".join(r["skipped"]))

    st.divider()
    # Lifetime metrics pulled from /metrics (aggregated across all past queries).
    m = {}
    try:
        m = requests.get(f"{API_URL}/metrics", timeout=5).json()
    except Exception:  # noqa: BLE001
        pass
    if m.get("queries"):
        st.header("Lifetime metrics")
        c1, c2 = st.columns(2)
        c1.metric("Queries", m["queries"])
        c2.metric("Avg contexts", m.get("avg_contexts", 0))
        c3, c4 = st.columns(2)
        c3.metric("Avg latency", f"{m['avg_latency_ms']:.0f} ms")
        c4.metric("p95 latency", f"{m.get('p95_latency_ms', 0):.0f} ms")
        c5, c6 = st.columns(2)
        c5.metric("Total cost", f"${m['total_cost_usd']:.4f}")
        c6.metric("Avg cost / query", f"${m.get('avg_cost_usd', 0):.6f}")

# --- Main panel: ask a question and render the grounded answer ------------------
question = st.text_input("Ask a question about the documentation:", placeholder="How do I reset my API key?")
if st.button("Ask", type="primary") and question:
    with st.spinner("Thinking..."):
        resp = requests.post(f"{API_URL}/query", json={"question": question}, timeout=120).json()

    st.subheader("Answer")
    st.write(resp["answer"])

    # Small badge showing retrieval mode and whether this was a cache hit.
    badge = f"mode: {resp.get('retrieval_mode', 'vector')}"
    if resp.get("cached"):
        badge += " · ⚡ cached (no LLM cost)"
    st.caption(badge)

    # Per-query latency / cost / number of context passages.
    c1, c2, c3 = st.columns(3)
    total_ms = sum(resp["timings_ms"].values())
    c1.metric("Latency", f"{total_ms:.0f} ms")
    c2.metric("Cost", f"${resp['cost_usd']:.6f}")
    c3.metric("Contexts", resp["n_contexts"])

    # The cited sources, each with a short snippet so the answer is verifiable.
    with st.expander(f"Sources ({len(resp['citations'])})"):
        for cit in resp["citations"]:
            st.markdown(f"**[{cit['n']}] {cit['source']}**")
            st.caption(cit["snippet"] + "…")
