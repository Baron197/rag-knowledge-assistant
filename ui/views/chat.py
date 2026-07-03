"""Ask page -- chat-first grounded Q&A.

A polished thin client: persistent conversation in session state, answers
rendered with tinted [n] citation markers, a per-answer telemetry strip
(latency split, cost, tokens, contexts), the cache-hit indicator, and a sidebar
workbench (config, top-k, document upload/re-index, export, lifetime metrics).
"""
from __future__ import annotations

import html

import common
import requests
import streamlit as st
from common import (
    ALLOWED_TYPES,
    API_URL,
    MAX_FILE_BYTES,
    MAX_REQUEST_BYTES,
    REQUEST_TIMEOUT,
    SERVER_DEFAULT_K,
    error_detail,
    get_health,
    get_metrics,
    invalidate_cache,
)

# Demo starter questions for the bundled "Nimbus" corpus. Framed as examples so
# they read as swappable demo content, not core UI.
DEMO_QUESTIONS = [
    "How do I authenticate with an API key?",
    "What are the rate limits per plan?",
    "How do I verify a webhook signature?",
    "Why am I getting a 429 error?",
]

# Chat-specific session state.
if "messages" not in st.session_state:
    st.session_state.messages = []        # list[{role, content, meta?}]
    st.session_state.k = SERVER_DEFAULT_K
    st.session_state.pending = None        # a question awaiting processing


def tint_citations(answer: str, citations: list[dict]) -> str:
    """XSS-safe: escape the answer, then wrap literal [n] markers in a tint span.

    Only &<> are escaped, so Markdown (bold, lists) still renders downstream.
    """
    safe = html.escape(answer, quote=False)
    for c in citations:
        marker = f"[{c['n']}]"
        safe = safe.replace(marker, f'<span class="cite">{marker}</span>')
    return safe


def run_query(question: str) -> dict:
    """POST /query and return the answer payload or an {"error": ...} dict."""
    body: dict = {"question": question}
    if st.session_state.k != SERVER_DEFAULT_K:
        body["k"] = st.session_state.k
    try:
        with st.status("Retrieving passages and generating a grounded answer…",
                       expanded=False) as status:
            resp = requests.post(f"{API_URL}/query", json=body, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                status.update(label="Query failed", state="error")
                return {"error": error_detail(resp)}
            status.update(label="Answer ready", state="complete")
            return resp.json()
    except requests.exceptions.Timeout:
        return {"error": f"The request timed out ({REQUEST_TIMEOUT}s). "
                         "Try a smaller k or a simpler question."}
    except requests.RequestException as exc:  # noqa: BLE001
        return {"error": str(exc)}


def render_assistant(meta: dict, idx: int) -> None:
    """Render one assistant turn from stored meta (never re-calls the API)."""
    if "error" in meta:
        st.markdown('<div class="badges"><span class="badge err">⚠ Query error'
                    '</span></div>', unsafe_allow_html=True)
        st.code(meta["error"], language=None)
        if st.button("Ask again", key=f"retry_{idx}"):
            st.session_state.pending = meta.get("question", "")
            st.rerun()
        return

    citations = meta.get("citations", []) or []
    grounded = len(citations) > 0

    # (1) Answer, rendered as Markdown with the literal [n] markers tinted.
    if not grounded:
        st.markdown('<div class="badges"><span class="badge refuse">⚠ Not grounded '
                    'in the corpus</span></div>', unsafe_allow_html=True)
    st.markdown(tint_citations(meta.get("answer", ""), citations), unsafe_allow_html=True)

    # (2) Badge row: retrieval mode + cache-hit.
    badges = ('<span class="badge mode">mode: '
              f'{html.escape(str(meta.get("retrieval_mode", "?")))}</span>')
    if meta.get("cached"):
        badges += '<span class="badge cache">⚡ Cache hit</span>'
    st.markdown(f'<div class="badges">{badges}</div>', unsafe_allow_html=True)

    # (3) Telemetry strip.
    timings = meta.get("timings_ms", {}) or {}
    total_ms = sum(timings.values())
    tokens = meta.get("tokens", {}) or {}
    tok_total = int(tokens.get("prompt", 0)) + int(tokens.get("completion", 0))
    tok_tip = (f"embedding {tokens.get('embedding', 0)} · prompt {tokens.get('prompt', 0)}"
               f" · completion {tokens.get('completion', 0)}")
    cached = bool(meta.get("cached"))
    cost = float(meta.get("cost_usd", 0.0))
    if cached:
        cost_sub = "Free (cached)"
    elif cost == 0.0:
        cost_sub = "local · $0"
    else:
        cost_sub = ""

    lat_bar = ""
    if not cached and total_ms > 0 and "retrieval" in timings and "generation" in timings:
        r_pct = timings["retrieval"] / total_ms * 100
        g_pct = timings["generation"] / total_ms * 100
        lat_bar = (
            f'<div class="latbar"><div class="r" style="width:{r_pct:.0f}%"></div>'
            f'<div class="g" style="width:{g_pct:.0f}%"></div></div>'
            f'<div class="sub">{timings["retrieval"]:.0f} retr / '
            f'{timings["generation"]:.0f} gen</div>'
        )

    tiles = (
        f'<div class="tile"><div class="lab">Latency</div>'
        f'<div class="val">{total_ms:.0f} ms</div>{lat_bar}</div>'
        f'<div class="tile"><div class="lab">Cost</div>'
        f'<div class="val">${cost:.6f}</div>'
        + (f'<div class="sub">{cost_sub}</div>' if cost_sub else "") + "</div>"
        f'<div class="tile"><div class="lab">Contexts</div>'
        f'<div class="val">{int(meta.get("n_contexts", 0))}</div></div>'
        f'<div class="tile" title="{html.escape(tok_tip)}"><div class="lab">Tokens</div>'
        f'<div class="val">{tok_total}</div><div class="sub">prompt+completion</div></div>'
    )
    st.markdown(f'<div class="telemetry">{tiles}</div>', unsafe_allow_html=True)

    # (4) Sources.
    if grounded:
        with st.expander(f"📎 Sources ({len(citations)})", expanded=False):
            for c in citations:
                st.markdown(
                    f'<div class="srccard"><div class="hd">[{int(c["n"])}] '
                    f'{html.escape(str(c["source"]))}</div>'
                    f'<div class="snip">{html.escape(str(c["snippet"]))}…</div></div>',
                    unsafe_allow_html=True)
    else:
        st.caption("No sources — the answer was not grounded in the corpus.")


def render_message(m: dict, idx: int) -> None:
    """Render one stored message (user or assistant) from session state."""
    if m["role"] == "user":
        with st.chat_message("user"):
            st.markdown(m["content"])
    else:
        with st.chat_message("assistant", avatar="📘"):
            render_assistant(m["meta"], idx)


def build_export() -> str:
    """Serialise the conversation to Markdown for the download button."""
    lines = ["# Nimbus Console — conversation", ""]
    for m in st.session_state.messages:
        if m["role"] == "user":
            lines += [f"## Q: {m['content']}", ""]
            continue
        meta = m["meta"]
        if "error" in meta:
            lines += [f"**Error:** {meta['error']}", ""]
            continue
        lines += [meta.get("answer", ""), ""]
        cites = meta.get("citations", []) or []
        if cites:
            lines.append("**Sources:**")
            lines += [f"- [{c['n']}] {c['source']}" for c in cites]
        total_ms = sum((meta.get("timings_ms", {}) or {}).values())
        tok = meta.get("tokens", {}) or {}
        flags = "cached" if meta.get("cached") else meta.get("retrieval_mode", "")
        lines += [
            "",
            f"`{total_ms:.0f} ms · ${float(meta.get('cost_usd', 0)):.6f} · "
            f"{int(tok.get('prompt', 0)) + int(tok.get('completion', 0))} tokens · "
            f"{int(meta.get('n_contexts', 0))} contexts · {flags}`",
            "",
        ]
    return "\n".join(lines)


# ==============================================================================
common.render_header()
health = get_health()

# --- Sidebar ------------------------------------------------------------------
with st.sidebar:
    with st.container(border=True):
        st.markdown('<div class="sgroup">Configuration</div>', unsafe_allow_html=True)
        if health is not None:
            st.markdown(
                f'Vector backend &nbsp;<span class="pill">{html.escape(str(health.get("vector_backend","?")))}</span>',
                unsafe_allow_html=True)
            st.markdown(
                f'LLM provider &nbsp;<span class="pill">{html.escape(str(health.get("llm_provider","?")))}</span>',
                unsafe_allow_html=True)
            st.markdown(
                f'Retrieval mode &nbsp;<span class="pill">{html.escape(str(health.get("retrieval_mode","?")))}</span>',
                unsafe_allow_html=True)
            st.markdown(
                f'Indexed chunks &nbsp;<span class="pill">{int(health.get("indexed_chunks",0))}</span>',
                unsafe_allow_html=True)
            st.caption("Retrieval mode is set server-side — not switchable here.")
        else:
            st.markdown('<span class="pill"><span class="dot off"></span>API offline</span>',
                        unsafe_allow_html=True)
            st.caption("Start it with:  `uvicorn src.rag.api:app --port 8000`")
        if st.button("Recheck connection", use_container_width=True):
            invalidate_cache()
            st.rerun()

    with st.container(border=True):
        st.markdown('<div class="sgroup">Retrieval</div>', unsafe_allow_html=True)
        st.session_state.k = st.slider(
            "Passages retrieved (k)", 1, 20, st.session_state.k,
            help="Sent as `k` on your next question. Higher k = more context and "
                 f"cost. Server default is top_k={SERVER_DEFAULT_K}.",
        )
        if st.session_state.k == SERVER_DEFAULT_K:
            st.caption(f"Using the server default (k={SERVER_DEFAULT_K}).")
        else:
            st.caption(f"Next question retrieves k={st.session_state.k}.")

    with st.container(border=True):
        st.markdown('<div class="sgroup">Documents</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Add files to the knowledge base",
            type=ALLOWED_TYPES,
            accept_multiple_files=True,
            help=".md .txt .html .pdf · 10 MB/file · 50 MB/request",
        )
        if st.button("Upload & index", type="primary", use_container_width=True,
                     disabled=health is None or not uploaded):
            files = uploaded or []
            oversize = [f.name for f in files if f.size > MAX_FILE_BYTES]
            total = sum(f.size for f in files)
            if oversize:
                st.error("Over the 10 MB/file limit: " + ", ".join(oversize))
            elif total > MAX_REQUEST_BYTES:
                st.error("Total upload exceeds the 50 MB/request limit.")
            else:
                payload = [("files", (f.name, f.getvalue())) for f in files]
                try:
                    with st.status("Uploading & indexing…", expanded=False) as status:
                        r = requests.post(f"{API_URL}/upload", files=payload, timeout=600)
                        if r.status_code == 413:
                            status.update(label="Upload rejected", state="error")
                            st.error("Upload too large — 50 MB per request max.")
                        elif r.status_code == 400:
                            status.update(label="Upload rejected", state="error")
                            st.error(error_detail(r))
                        elif r.status_code != 200:
                            status.update(label="Upload failed", state="error")
                            st.error(f"Upload failed (HTTP {r.status_code}).")
                        else:
                            data = r.json()
                            status.update(label="Indexed", state="complete")
                            st.toast(f"Indexed {data['indexed_chunks']} chunks · "
                                     f"{len(data['saved'])} file(s)", icon="✅")
                            if data.get("skipped"):
                                st.warning("Skipped: " + ", ".join(data["skipped"]))
                            invalidate_cache()
                            st.rerun()
                except requests.RequestException as exc:  # noqa: BLE001
                    st.error(f"Upload failed: {exc}")

        if st.button("Re-index corpus", use_container_width=True, disabled=health is None):
            try:
                with st.status("Rebuilding the index…", expanded=False) as status:
                    r = requests.post(f"{API_URL}/ingest", params={"reset": "true"}, timeout=600)
                    r.raise_for_status()
                    status.update(label="Index rebuilt", state="complete")
                st.toast(f"Indexed {r.json()['ingested_chunks']} chunks", icon="✅")
                invalidate_cache()
                st.rerun()
            except requests.RequestException as exc:  # noqa: BLE001
                st.error(f"Re-index failed: {exc}")
        st.caption("Rebuilds the index from the server's `data/docs` — also how "
                   "you load the bundled demo (Nimbus) corpus.")

    with st.container(border=True):
        st.markdown('<div class="sgroup">Session</div>', unsafe_allow_html=True)
        st.download_button(
            "Export conversation",
            data=build_export(),
            file_name="nimbus-conversation.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=not st.session_state.messages,
        )
        if st.button("Clear chat", use_container_width=True,
                     disabled=not st.session_state.messages):
            st.session_state.messages = []
            st.toast("Conversation cleared", icon="🧹")
            st.rerun()
        st.caption("Tip: ask the same question twice to see a cache hit — $0 cost.")

    with st.container(border=True):
        st.markdown('<div class="sgroup">Lifetime metrics</div>', unsafe_allow_html=True)
        m = get_metrics()
        if m.get("queries"):
            c1, c2 = st.columns(2)
            c1.metric("Queries", m.get("queries", 0))
            c2.metric("Avg contexts", m.get("avg_contexts", 0))
            c3, c4 = st.columns(2)
            c3.metric("Avg latency", f"{m.get('avg_latency_ms', 0):.0f} ms")
            c4.metric("p95 latency", f"{m.get('p95_latency_ms', 0):.0f} ms")
            c5, c6 = st.columns(2)
            c5.metric("Total cost", f"${m.get('total_cost_usd', 0):.4f}")
            c6.metric("Avg cost / query", f"${m.get('avg_cost_usd', 0):.6f}")
            st.caption("Full charts on the **Analytics** page →")
        else:
            st.caption("No queries yet — ask something to populate metrics.")

    st.caption("Thin client · UI → FastAPI → RAG pipeline")


# --- Main conversation area (state machine) -----------------------------------
if health is None:
    st.markdown(
        f"""
<div class="panel err">
  <h3>Can't reach the API</h3>
  <p>The UI is a thin client and needs the FastAPI service running.</p>
  <p class="mono">{html.escape(API_URL)}</p>
</div>
""",
        unsafe_allow_html=True)
    st.code(st.session_state.get("health_err", "connection error"), language=None)
    if st.button("Retry connection", type="primary"):
        invalidate_cache()
        st.rerun()

elif int(health.get("indexed_chunks", 0)) == 0:
    st.markdown(
        """
<div class="panel">
  <h3>Your knowledge base is empty</h3>
  <ol>
    <li>Upload documents in the sidebar (<b>Documents → Upload &amp; index</b>), or</li>
    <li>click <b>Re-index corpus</b> to load the bundled demo (Nimbus) corpus,</li>
    <li>then ask grounded questions here.</li>
  </ol>
</div>
""",
        unsafe_allow_html=True)
    st.chat_input("Add documents to start asking…", disabled=True)

else:
    if not st.session_state.messages and not st.session_state.pending:
        st.markdown(
            """
<div class="hero">
  <h2>Ask anything about your indexed docs</h2>
  <p>Answers are grounded in the corpus, with visible citations and per-query
     cost &amp; latency. Ask the same question twice to see a cache hit.</p>
  <div class="lbl">Example questions · demo Nimbus corpus</div>
</div>
""",
            unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate(DEMO_QUESTIONS):
            if cols[i % 2].button(q, key=f"chip_{i}", use_container_width=True):
                st.session_state.pending = q
                st.rerun()

    for idx, msg in enumerate(st.session_state.messages):
        render_message(msg, idx)

    if st.session_state.pending:
        q = st.session_state.pending
        st.session_state.pending = None
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="📘"):
            meta = run_query(q)
            meta.setdefault("question", q)
            render_assistant(meta, len(st.session_state.messages) + 1)
        st.session_state.messages.append({"role": "user", "content": q})
        st.session_state.messages.append({"role": "assistant", "content": q, "meta": meta})
        st.session_state.metrics = None   # refresh lifetime metrics after a real query
        st.rerun()

    prompt = st.chat_input("Ask about your documentation…",
                           disabled=bool(st.session_state.pending))
    if prompt:
        st.session_state.pending = prompt
        st.rerun()
