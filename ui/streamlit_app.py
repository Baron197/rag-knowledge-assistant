"""Streamlit thin client over the FastAPI RAG service -- multipage router.

Five pages share the same session and API:
  * Ask        (views/chat.py)       -- chat-first grounded Q&A with citations.
  * Corpus     (views/corpus.py)     -- browse the indexed source documents.
  * Analytics  (views/analytics.py)  -- filterable charts over the query traces.
  * Evaluation (views/evaluation.py) -- read-only view of the eval reports.
  * Guide      (views/guide.py)      -- in-app tutorial: usage + metric meanings.

The UI holds no RAG logic; every action is an HTTP call (UI -> API -> pipeline).

Run:  streamlit run ui/streamlit_app.py     (with the API already running)
Env:  RAG_API_URL  -> where the FastAPI service lives (default localhost:8000)
"""
from __future__ import annotations

import common
import streamlit as st

st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon=":material/forum:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme: a session-state flag drives light/dark. The toggle (below the nav) is
# bound to the same key, so flipping it reruns and re-injects the right palette.
st.session_state.setdefault("dark_mode", False)
common.inject_theme("dark" if st.session_state.dark_mode else "light")

# Guide is the public landing page; Ask/Analytics/Evaluation sit behind a password
# gate (see common.require_auth, enabled by APP_PASSWORD).
pages = [
    st.Page("views/guide.py", title="Guide", icon=":material/menu_book:", default=True),
    st.Page("views/chat.py", title="Ask", icon=":material/forum:"),
    st.Page("views/corpus.py", title="Corpus", icon=":material/folder_open:"),
    st.Page("views/analytics.py", title="Analytics", icon=":material/monitoring:"),
    st.Page("views/evaluation.py", title="Evaluation", icon=":material/verified:"),
]
nav = st.navigation(pages)
with st.sidebar:
    st.toggle(":material/dark_mode: Dark mode", key="dark_mode",
              help="Switch between light and dark themes.")
    common.logout_button()   # only shown once signed in
nav.run()
