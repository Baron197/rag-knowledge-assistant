"""Shared bits for the Streamlit thin client: config, styling, and small API
helpers used by both the Ask (chat) and Analytics pages.

Kept deliberately thin -- the UI holds no RAG logic; every call is HTTP to the
FastAPI service (UI -> API -> pipeline).
"""
from __future__ import annotations

import html
import os

import requests
import streamlit as st

# Where the FastAPI service lives (overridable for Docker/remote).
API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000").rstrip("/")
SERVER_DEFAULT_K = 4            # matches Settings.top_k; k is only sent when overridden
ALLOWED_TYPES = ["md", "txt", "html", "htm", "pdf"]
MAX_FILE_BYTES = 10 * 1024 * 1024      # per-file cap (mirrors the API)
MAX_REQUEST_BYTES = 50 * 1024 * 1024   # whole-request cap (mirrors the API)
REQUEST_TIMEOUT = 120                  # seconds for a /query call

CSS = """
<style>
:root{
  --bg:#F7F8FA; --surface:#FFFFFF; --surface-2:#F8FAFC; --border:#E2E8F0;
  --ink:#0F172A; --body:#475569; --muted:#94A3B8; --primary:#4F46E5;
  --cite-bg:#EEF0FF; --success:#059669; --success-bg:#ECFDF5;
  --warn:#B45309; --warn-bg:#FFFBEB; --cache:#7A5CFF; --cache-bg:#F3F0FF;
  --danger:#DC2626; --danger-bg:#FEF2F2; --seg-retr:#9CB4FF; --seg-gen:#4F46E5;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
/* Keep the reading column tight even on a wide layout. */
.block-container{ max-width:900px; padding-top:3rem; padding-bottom:5rem; }
[data-testid="stChatInput"]{ max-width:840px; margin:0 auto; }
/* Let the header band show through Streamlit's top toolbar. */
[data-testid="stHeader"]{ background:transparent; }
/* Answer prose (Streamlit renders the markdown; we only tint [n] markers). */
[data-testid="stChatMessage"] .stMarkdown p{ font-size:14.5px; line-height:1.62; }
/* Cap heading sizes inside answers so messy content (e.g. a raw chunk that
   starts with "#") can't blow up the layout; real bold/lists/code still render. */
[data-testid="stChatMessage"] .stMarkdown :is(h1,h2,h3,h4){
  font-size:15px; font-weight:700; margin:.5em 0 .25em; line-height:1.4; }

/* Header */
.apphead{ display:flex; align-items:center; justify-content:space-between;
  flex-wrap:wrap; gap:10px; }
.brand{ font-size:22px; font-weight:700; color:var(--ink); letter-spacing:-.01em; }
.tagline{ color:var(--body); font-size:13px; margin:2px 0 12px; }
.pills{ display:flex; flex-wrap:wrap; gap:6px; }
.pill{ font:600 12px var(--mono); background:var(--surface); border:1px solid var(--border);
  border-radius:999px; padding:3px 10px; color:var(--body); }
.dot{ height:8px; width:8px; border-radius:50%; display:inline-block; margin-right:6px;
  vertical-align:middle; }
.dot.ok{ background:var(--success); } .dot.off{ background:var(--danger); }

/* Inline citation markers */
.cite{ background:var(--cite-bg); color:var(--primary); font-weight:600;
  border-radius:6px; padding:0 5px; font-size:.82em; }

/* Badges */
.badges{ margin:8px 0 2px; }
.badge{ display:inline-flex; align-items:center; gap:4px; font:600 12px var(--mono);
  border-radius:999px; padding:2px 10px; margin-right:6px; }
.badge.mode{ background:var(--surface-2); color:var(--body); border:1px solid var(--border); }
.badge.cache{ background:var(--cache-bg); color:var(--cache); }
.badge.refuse{ background:var(--warn-bg); color:var(--warn); }
.badge.err{ background:var(--danger-bg); color:var(--danger); }

/* Telemetry strip */
.telemetry{ display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 2px; }
.tile{ background:var(--surface-2); border:1px solid var(--border); border-radius:10px;
  padding:6px 11px; min-width:82px; }
.tile .lab{ font:600 11px/1.4 var(--mono); text-transform:uppercase; letter-spacing:.04em;
  color:var(--muted); }
.tile .val{ font:16px/1.4 var(--mono); color:var(--ink); }
.tile .sub{ font:11px/1.3 var(--mono); color:var(--muted); }
.latbar{ height:6px; border-radius:999px; overflow:hidden; display:flex;
  background:var(--border); width:100%; margin-top:6px; }
.latbar .r{ background:var(--seg-retr); } .latbar .g{ background:var(--seg-gen); }

/* Source cards */
.srccard{ border-left:3px solid var(--primary); background:var(--surface-2);
  border-radius:8px; padding:8px 12px; margin:6px 0; }
.srccard .hd{ font:600 13px var(--mono); color:var(--ink); }
.srccard .snip{ font-size:12.5px; color:var(--body); margin-top:3px; }

/* Panels (empty index / API down) */
.panel{ background:var(--surface); border:1px solid var(--border); border-radius:14px;
  padding:22px 24px; margin-top:8px; }
.panel.err{ border-color:#F3C7C7; background:var(--danger-bg); }
.panel h3{ margin:0 0 6px; font-size:17px; color:var(--ink); }
.panel p{ color:var(--body); font-size:13.5px; margin:4px 0; }
.panel ol{ color:var(--body); font-size:13.5px; margin:8px 0 0 18px; }
.panel .mono{ font-family:var(--mono); font-size:12.5px; color:var(--body); }

/* Hero (empty conversation) */
.hero{ text-align:center; padding:26px 8px 10px; }
.hero h2{ font-size:20px; color:var(--ink); margin:0 0 6px; }
.hero p{ color:var(--body); font-size:13.5px; margin:0 auto; max-width:520px; }
.hero .lbl{ font:600 11px var(--mono); text-transform:uppercase; letter-spacing:.06em;
  color:var(--muted); margin:18px 0 2px; }

/* Sidebar group headings + section labels */
.sgroup{ font:700 11px var(--mono); text-transform:uppercase; letter-spacing:.06em;
  color:var(--muted); margin:2px 0 4px; }
</style>
"""


def inject_css() -> None:
    """Inject the shared stylesheet (called once per run by the router)."""
    st.markdown(CSS, unsafe_allow_html=True)


def use_wide(max_px: int = 1280) -> None:
    """Widen the reading column for data-dense dashboard pages.

    Overrides the chat page's tighter column; injected from the page body so it
    renders after the router's global CSS and therefore wins, and only affects
    the page that calls it.
    """
    st.markdown(f"<style>.block-container{{max-width:{max_px}px !important;}}</style>",
                unsafe_allow_html=True)


def invalidate_cache() -> None:
    """Force /health and /metrics to refetch on the next run."""
    st.session_state.health = None
    st.session_state.metrics = None


def get_health() -> dict | None:
    """Fetch /health once per run cycle; None means the API is unreachable."""
    if st.session_state.get("health") is None:
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            r.raise_for_status()
            st.session_state.health = r.json()
        except requests.RequestException as exc:  # noqa: BLE001
            st.session_state.health = "__error__"
            st.session_state.health_err = str(exc)
    h = st.session_state.get("health")
    return None if h == "__error__" else h


def get_metrics() -> dict:
    """Fetch /metrics once per run cycle (best-effort)."""
    if st.session_state.get("metrics") is None:
        try:
            r = requests.get(f"{API_URL}/metrics", timeout=5)
            r.raise_for_status()
            st.session_state.metrics = r.json()
        except requests.RequestException:  # noqa: BLE001
            st.session_state.metrics = {}
    return st.session_state.get("metrics") or {}


def error_detail(resp: requests.Response) -> str:
    """Human-readable error text from an API response.

    Handles FastAPI's two `detail` shapes: a string (HTTPException, e.g. 400/413/
    500) and a list of error objects (422 request-validation) -- the latter is
    joined into a sentence instead of dumped as a raw Python repr.
    """
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text or f"HTTP {resp.status_code}"
    if isinstance(detail, list):
        return "; ".join(str(e.get("msg", e)) if isinstance(e, dict) else str(e)
                         for e in detail)
    return str(detail)


def render_header() -> None:
    """Brand lockup + live status pills (shared across pages)."""
    health = get_health()
    if health is not None:
        pills = (
            '<span class="pill"><span class="dot ok"></span>Connected</span>'
            f'<span class="pill">backend:{html.escape(str(health.get("vector_backend", "?")))}</span>'
            f'<span class="pill">llm:{html.escape(str(health.get("llm_provider", "?")))}</span>'
            f'<span class="pill">mode:{html.escape(str(health.get("retrieval_mode", "?")))}</span>'
            f'<span class="pill">{int(health.get("indexed_chunks", 0))} chunks</span>'
        )
    else:
        pills = '<span class="pill"><span class="dot off"></span>Offline</span>'
    st.markdown(
        f"""
<div class="apphead">
  <div class="brand">📘 Nimbus Console</div>
  <div class="pills">{pills}</div>
</div>
<div class="tagline">Grounded answers over your indexed docs — with visible
citations and per-query cost, latency &amp; token observability.</div>
""",
        unsafe_allow_html=True,
    )
    st.divider()
