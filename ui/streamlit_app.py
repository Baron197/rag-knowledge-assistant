"""Streamlit thin client over the FastAPI RAG service -- multipage router.

Two pages share the same session and API:
  * Ask       (views/chat.py)      -- chat-first grounded Q&A with citations.
  * Analytics (views/analytics.py) -- filterable charts over the query traces.

The UI holds no RAG logic; every action is an HTTP call (UI -> API -> pipeline).

Run:  streamlit run ui/streamlit_app.py     (with the API already running)
Env:  RAG_API_URL  -> where the FastAPI service lives (default localhost:8000)
"""
from __future__ import annotations

import common
import streamlit as st

st.set_page_config(
    page_title="Nimbus Console",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)
common.inject_css()

pages = [
    st.Page("views/chat.py", title="Ask", icon="💬", default=True),
    st.Page("views/analytics.py", title="Analytics", icon="📊"),
    st.Page("views/evaluation.py", title="Evaluation", icon="🎯"),
]
st.navigation(pages).run()
