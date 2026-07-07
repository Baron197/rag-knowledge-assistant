"""Corpus page -- browse the source documents the vector index was built from.

Read-only view over GET /documents (+ /documents/{name}): lists every file in the
corpus with its type, size and chunk count, and for a selected document shows the
full text the ingester extracted (what got chunked + embedded) plus the individual
chunks the vector store actually holds. The corpus itself is changed on the Ask
page (upload / re-index).
"""
from __future__ import annotations

from urllib.parse import quote

import common
import pandas as pd
import requests
import streamlit as st
from common import API_URL, error_detail, get_health, invalidate_cache

common.require_auth()  # gate behind APP_PASSWORD (no-op when unset)
common.use_wide()      # data-dense page -> widen past the chat reading column
common.render_header()
st.markdown("### :material/folder_open: Corpus")
st.caption("The source documents indexed in the vector database. Add or re-index "
           "documents from the **Ask** page (sidebar → Upload documents).")


def _human_size(n: int) -> str:
    """Human-readable byte size (e.g. '1.2 KB')."""
    size = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


health = get_health()
if health is None:
    st.markdown(
        f"""
<div class="panel err">
  <h3>Can't reach the API</h3>
  <p>The corpus view reads the indexed documents from the FastAPI service.</p>
  <p class="mono">{API_URL}</p>
</div>
""",
        unsafe_allow_html=True)
    if st.button("Retry connection", type="primary"):
        invalidate_cache()
        st.rerun()
    st.stop()

# --- Load the document list ---------------------------------------------------
try:
    resp = requests.get(f"{API_URL}/documents", timeout=15)
    resp.raise_for_status()
    data = resp.json()
except requests.RequestException as exc:  # noqa: BLE001
    st.error(f"Couldn't load the corpus: {exc}")
    st.stop()

docs = data.get("documents", [])
if not docs:
    st.info("No documents in the corpus yet. Upload some on the **Ask** page "
            "(sidebar → Upload documents), then come back.")
    st.stop()

# --- Summary ------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Documents", len(docs))
c2.metric("Chunks indexed", data.get("total_chunks", 0))
c3.metric("Vector backend", health.get("vector_backend", "?"))

# --- Document table -----------------------------------------------------------
df = pd.DataFrame(docs)
df["size"] = df["size_bytes"].apply(_human_size)
table = df[["name", "suffix", "size", "chunks"]].rename(
    columns={"name": "Document", "suffix": "Type", "size": "Size", "chunks": "Chunks"}
)
st.dataframe(table, hide_index=True, use_container_width=True)

# --- Document viewer ----------------------------------------------------------
st.markdown("#### :material/description: View a document")
name = st.selectbox("Document", [d["name"] for d in docs], label_visibility="collapsed")
if name:
    try:
        r = requests.get(f"{API_URL}/documents/{quote(name)}", timeout=15)
        if r.status_code != 200:
            st.error(error_detail(r))
            st.stop()
        doc = r.json()
    except requests.RequestException as exc:  # noqa: BLE001
        st.error(f"Couldn't load {name}: {exc}")
        st.stop()

    st.caption(f"{doc['chars']:,} characters · {doc['n_chunks']} chunk(s) · "
               f"{_human_size(doc['size_bytes'])} · {doc['suffix']}")

    tab_render, tab_raw, tab_chunks = st.tabs(
        ["Rendered", "Raw text", f"Chunks ({doc['n_chunks']})"]
    )
    with tab_render:
        with st.container(height=460, border=True):
            st.markdown(doc["text"])   # renders .md; raw HTML is shown as text, not executed
    with tab_raw:
        with st.container(height=460, border=True):
            st.text(doc["text"])       # the exact extracted text the ingester chunked
    with tab_chunks:
        st.caption("The passages the vector store actually holds for this document — "
                   "each is embedded and retrieved independently.")
        if not doc.get("chunks"):
            st.info("This document has no chunks in the store yet (not indexed).")
        for ch in doc.get("chunks", []):
            with st.expander(f"Chunk {ch['index']}  ·  `{ch['id']}`"):
                st.text(ch["text"])
