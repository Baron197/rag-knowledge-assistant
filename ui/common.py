"""Shared bits for the Streamlit thin client: config, theming, and small API
helpers used by every page. No RAG logic -- every call is HTTP to the FastAPI
service (UI -> API -> pipeline).

Theming: the UI is driven by CSS variables, so light/dark is a matter of swapping
one variable block (plus a few overrides for Streamlit's own chrome in dark mode).
`inject_theme(mode)` is called once per run by the router.
"""
from __future__ import annotations

import html
import os

import requests
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000").rstrip("/")
# Shared-password gate for the Ask/Analytics/Evaluation pages. If unset, the gate
# is DISABLED (local dev just works); set APP_PASSWORD in the deployment env to
# turn it on. The Guide page is always public.
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
SERVER_DEFAULT_K = 4            # matches Settings.top_k; k is only sent when overridden
ALLOWED_TYPES = ["md", "txt", "html", "htm", "pdf"]
MAX_FILE_BYTES = 10 * 1024 * 1024      # per-file cap (mirrors the API)
MAX_REQUEST_BYTES = 50 * 1024 * 1024   # whole-request cap (mirrors the API)
REQUEST_TIMEOUT = 120                  # seconds for a /query call

# Monochrome brand logomark: an "indexed cloud" (a cloud carrying two
# retrieved-doc index lines). currentColor -> inherits the wrapper's --body tone
# so it stays neutral and adapts to light/dark.
BRAND_MARK = (
    '<span style="color:var(--body)"><svg width="20" height="20" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" '
    'stroke-linejoin="round" style="flex:none;vertical-align:-3px;margin-right:8px">'
    '<path d="M7 17.5h9a3.6 3.6 0 0 0 .5-7.16 4.8 4.8 0 0 0-9.2-1.2A3.35 3.35 0 0 0 7 17.5Z"/>'
    '<path d="M9 13h6" stroke-width="1.4"/><path d="M9 15.2h4" stroke-width="1.4"/></svg></span>'
)

# Tiny status-badge glyphs (12px). fill/stroke=currentColor so each inherits its
# badge's own tint. Distinct shapes so amber-vs-red isn't the only signal.
CACHE_ICON = (
    '<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor" style="flex:none">'
    '<path d="M9 1 3.5 9H7l-1 6 6.5-8.5H9L9 1Z"/></svg>'
)
WARN_ICON = (
    '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" '
    'stroke-width="1.5" stroke-linejoin="round" style="flex:none">'
    '<path d="M8 2.5 14.5 13.5H1.5L8 2.5Z"/><path d="M8 6.6v3.1" stroke-linecap="round"/>'
    '<circle cx="8" cy="11.7" r=".55" fill="currentColor" stroke="none"/></svg>'
)
ERR_ICON = (
    '<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor" style="flex:none">'
    '<path d="M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Zm2.3 8.3-1 1L8 8.9l-1.3 1.4-1-1'
    'L7.1 8 5.7 6.6l1-1L8 7.1l1.3-1.4 1 1L9 8Z"/></svg>'
)

# --- Theme palettes (only these change between light and dark) ----------------
_LIGHT_VARS = """
:root{
  --bg:#F7F8FA; --surface:#FFFFFF; --surface-2:#F8FAFC; --sidebar-bg:#F1F4F9; --border:#E2E8F0;
  --ink:#0F172A; --body:#475569; --muted:#94A3B8; --primary:#4F46E5;
  --cite-bg:#EEF0FF; --success:#059669; --success-bg:#ECFDF5;
  --warn:#B45309; --warn-bg:#FFFBEB; --cache:#7A5CFF; --cache-bg:#F3F0FF;
  --danger:#DC2626; --danger-bg:#FEF2F2; --seg-retr:#9CB4FF; --seg-gen:#4F46E5;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
"""
_DARK_VARS = """
:root{
  --bg:#0E1420; --surface:#182234; --surface-2:#131C2B; --sidebar-bg:#0B111C; --border:#2A3852;
  --ink:#EAF0F9; --body:#A9B6CA; --muted:#8B98AE; --primary:#8E97FF;
  --cite-bg:#2A2F5C; --success:#34D399; --success-bg:#0F2A20;
  --warn:#FBBF24; --warn-bg:#2C2410; --cache:#B9A6FF; --cache-bg:#241E3F;
  --danger:#F87171; --danger-bg:#2C1616; --seg-retr:#4E5C93; --seg-gen:#8E97FF;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
"""

# --- Structural rules (theme-agnostic; read from the variables) ---------------
_STRUCT = """
html, body{ background:var(--bg); }
.stApp, [data-testid="stAppViewContainer"]{ background:var(--bg); }
[data-testid="stSidebar"]{ background:var(--sidebar-bg); }
[data-testid="stHeader"]{ background:transparent; }
.block-container{ max-width:840px; padding-top:3rem; padding-bottom:5rem; }
[data-testid="stChatInput"]{ max-width:840px; margin:0 auto; }
[data-testid="stChatMessage"] .stMarkdown p{ font-size:14.5px; line-height:1.62; }
[data-testid="stChatMessage"] .stMarkdown :is(h1,h2,h3,h4){
  font-size:15px; font-weight:700; margin:.5em 0 .25em; line-height:1.4; }
/* Sidebar lifetime metrics: shrink st.metric so long values ("$0.000012",
   "853 ms") fit the narrow sidebar column instead of truncating with an ellipsis. */
[data-testid="stSidebar"] [data-testid="stMetricValue"]{ font-size:1.05rem; line-height:1.3; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p{ font-size:11.5px; }
"""

# --- Component styles (all colours via the variables above) -------------------
_COMPONENTS = """
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

.cite{ background:var(--cite-bg); color:var(--primary); font-weight:600;
  border-radius:6px; padding:0 5px; font-size:.82em; }

.badges{ margin:8px 0 2px; }
.badge{ display:inline-flex; align-items:center; gap:4px; font:600 12px var(--mono);
  border-radius:999px; padding:2px 10px; margin-right:6px; }
.badge.mode{ background:var(--surface-2); color:var(--body); border:1px solid var(--border); }
.badge.cache{ background:var(--cache-bg); color:var(--cache); }
.badge.refuse{ background:var(--warn-bg); color:var(--warn); }
.badge.err{ background:var(--danger-bg); color:var(--danger); }

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

.srccard{ border-left:3px solid var(--primary); background:var(--surface-2);
  border-radius:8px; padding:8px 12px; margin:6px 0; }
.srccard .hd{ font:600 13px var(--mono); color:var(--ink); }
.srccard .snip{ font-size:12.5px; color:var(--body); margin-top:3px; }

.panel{ background:var(--surface); border:1px solid var(--border); border-radius:14px;
  padding:22px 24px; margin-top:8px; }
.panel.err{ border-color:var(--danger); background:var(--danger-bg); }
.panel h3{ margin:0 0 6px; font-size:17px; color:var(--ink); }
.panel p{ color:var(--body); font-size:13.5px; margin:4px 0; }
.panel ol{ color:var(--body); font-size:13.5px; margin:8px 0 0 18px; }
.panel .mono{ font-family:var(--mono); font-size:12.5px; color:var(--body); }

.hero{ text-align:center; padding:26px 8px 10px; }
.hero h2{ font-size:20px; color:var(--ink); margin:0 0 6px; }
.hero p{ color:var(--body); font-size:13.5px; margin:0 auto; max-width:520px; }
.hero .lbl{ font:600 11px var(--mono); text-transform:uppercase; letter-spacing:.06em;
  color:var(--muted); margin:18px 0 2px; }

.sgroup{ font:700 11px var(--mono); text-transform:uppercase; letter-spacing:.06em;
  color:var(--muted); margin:2px 0 4px; }
"""

# --- Dark-mode overrides for Streamlit's own chrome (only in dark) ------------
# Streamlit widgets/charts/dataframes follow the config theme (light), so in dark
# mode we override the chrome here. Charts + dataframes can't be CSS-themed, so
# they're rendered as clean light "cards" on the dark page (a common pattern).
_DARK_CHROME = """
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div,
[data-testid="stMarkdownContainer"], .stMarkdown{ color:var(--ink); }
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5{ color:var(--ink) !important; }
[data-testid="stCaptionContainer"]{ color:var(--muted) !important; }
[data-testid="stMetricValue"]{ color:var(--ink) !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *{ color:var(--body) !important; }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *{ color:var(--body) !important; }

/* Inputs, selects, textareas, chat input */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"],
[data-baseweb="select"] > div{ background:var(--surface) !important; border-color:var(--border) !important; }
input, textarea{ background:var(--surface) !important; color:var(--ink) !important;
  caret-color:var(--ink) !important; }
[data-testid="stChatInput"], [data-testid="stChatInput"] textarea{
  background:var(--surface) !important; color:var(--ink) !important; }
[data-baseweb="select"] *{ color:var(--ink) !important; }
/* Placeholder text: Streamlit leaves it at the light-mode ink colour (dark navy),
   which is invisible on the dark surface. -webkit-text-fill-color also needed. */
input::placeholder, textarea::placeholder{
  color:var(--muted) !important; -webkit-text-fill-color:var(--muted) !important; opacity:1 !important; }

/* Buttons (secondary) + download button */
.stButton > button, .stDownloadButton > button,
[data-testid="stFileUploaderDropzone"] button, [data-testid="stBaseButton-secondary"]{
  background:var(--surface) !important; color:var(--ink) !important; border-color:var(--border) !important; }
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFileUploaderDropzone"] button:hover{
  border-color:var(--primary) !important; color:var(--primary) !important; }

/* The pinned bottom bar + the chat-input box (Streamlit renders these on the
   light secondary surface; theme them to the dark page/surface). */
[data-testid="stBottom"] > div{ background:var(--bg) !important; }
[data-testid="stChatInput"], [data-testid="stChatInput"] div{ background:var(--surface) !important; }
[data-testid="stChatInput"] textarea{ background:var(--surface) !important; color:var(--ink) !important; }

/* Inline `code` in markdown/captions (was on a light chip). */
.stMarkdown code, [data-testid="stCaptionContainer"] code{
  background:var(--surface-2) !important; color:var(--body) !important; }

/* Bordered containers, expanders, chat bubbles, uploader */
[data-testid="stVerticalBlockBorderWrapper"]{ border-color:var(--border) !important; }
[data-testid="stExpander"], [data-testid="stExpander"] details, [data-testid="stExpander"] summary{
  background:var(--surface) !important; border-color:var(--border) !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary *{ color:var(--ink) !important; }
[data-testid="stChatMessage"]{ background:var(--surface) !important; }
/* Chat avatars: the icon sat on a white box (testid is stChatMessageAvatarCustom
   for :material/...: avatars, so match the prefix). Theme it to the surface. */
[data-testid^="stChatMessageAvatar"]{ background:var(--surface) !important;
  border:1px solid var(--border) !important; }
[data-testid^="stChatMessageAvatar"], [data-testid^="stChatMessageAvatar"] *{
  color:var(--body) !important; fill:var(--body) !important; }
[data-testid="stForm"]{ background:var(--surface) !important; border-color:var(--border) !important; }
[data-testid="stFileUploaderDropzone"]{ background:var(--surface-2) !important; }
hr{ border-color:var(--border) !important; }

/* Tabs + sidebar nav links */
[data-baseweb="tab"]{ color:var(--body) !important; }
[data-baseweb="tab"][aria-selected="true"]{ color:var(--primary) !important; }
[data-testid="stSidebarNav"] a span{ color:var(--body) !important; }

/* Charts + dataframes: keep them as readable light cards on the dark page. */
[data-testid="stVegaLiteChart"], [data-testid="stDataFrame"], [data-testid="stTable"]{
  background:#FFFFFF; border:1px solid var(--border); border-radius:10px; padding:6px; }

/* ---- Portal overlays + light-theme leftovers (all verified in dark mode) ---- */

/* Selectbox dropdown menu: baseweb portals it OUTSIDE .stApp, where it keeps
   Streamlit's light base theme (renders as a #F7F8FA panel). Target unscoped. */
[data-testid="stSelectboxVirtualDropdown"],
[data-baseweb="popover"] [role="listbox"]{
  background:var(--surface) !important; border:1px solid var(--border) !important; }
[data-baseweb="popover"] [role="option"]{
  background:var(--surface) !important; color:var(--ink) !important; }
[data-baseweb="popover"] [role="option"] *{ color:var(--ink) !important; }
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [role="option"][aria-selected="true"]{
  background:var(--surface-2) !important; }

/* Help tooltips: also portalled outside .stApp on the light base theme. */
[data-testid="stTooltipContent"]{
  background:var(--surface) !important; border:1px solid var(--border) !important; }
[data-testid="stTooltipContent"], [data-testid="stTooltipContent"] *{ color:var(--ink) !important; }

/* st.toast pop-ups: light box on the light base theme -> dark surface. */
[data-testid="stToast"]{ background:var(--surface) !important; border:1px solid var(--border) !important; }
[data-testid="stToast"], [data-testid="stToast"] *{ color:var(--ink) !important; }

/* st.code(...) blocks (used here for plain error text): light chip -> dark surface. */
[data-testid="stCode"]{ background:var(--surface-2) !important; border:1px solid var(--border) !important;
  border-radius:8px !important; }
[data-testid="stCode"] pre, [data-testid="stCode"] code, [data-testid="stCode"] span{
  background:transparent !important; color:var(--ink) !important; }

/* file_uploader per-file chip (shown after a file is chosen). */
[data-testid="stFileUploaderFile"]{ background:var(--surface) !important;
  border:1px solid var(--border) !important; border-radius:8px !important; }

/* Chrome autofill repaints the field near-white with dark text; force it back. */
input:-webkit-autofill, input:-webkit-autofill:hover, input:-webkit-autofill:focus{
  -webkit-text-fill-color:var(--ink) !important; caret-color:var(--ink) !important;
  -webkit-box-shadow:0 0 0 1000px var(--surface) inset !important; }

/* Dark scrollbars (Chromium): the light base theme keeps a light default bar. */
::-webkit-scrollbar{ width:11px; height:11px; }
::-webkit-scrollbar-track{ background:var(--bg); }
::-webkit-scrollbar-thumb{ background:var(--border); border-radius:999px; border:2px solid var(--bg); }
::-webkit-scrollbar-thumb:hover{ background:var(--muted); }
[data-testid="stSidebar"] ::-webkit-scrollbar-track{ background:var(--sidebar-bg); }
"""


def inject_theme(mode: str) -> None:
    """Inject the full stylesheet for the given theme ('light' | 'dark')."""
    vars_css = _DARK_VARS if mode == "dark" else _LIGHT_VARS
    chrome = _DARK_CHROME if mode == "dark" else ""
    st.markdown(f"<style>{vars_css}\n{_STRUCT}\n{_COMPONENTS}\n{chrome}</style>",
                unsafe_allow_html=True)


def use_wide(max_px: int = 1280) -> None:
    """Widen the reading column for data-dense dashboard pages."""
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
    """Human-readable error text from an API response (string or 422 list `detail`)."""
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
  <div class="brand">{BRAND_MARK}Knowledge Assistant</div>
  <div class="pills">{pills}</div>
</div>
<div class="tagline">Grounded answers over your indexed docs — with visible
citations and per-query cost, latency &amp; token observability.</div>
""",
        unsafe_allow_html=True,
    )
    st.divider()


def require_auth() -> None:
    """Gate a page behind the shared APP_PASSWORD.

    No-op when APP_PASSWORD is unset (local dev). Otherwise an unauthenticated
    visitor gets a centered password prompt and the page halts; one success
    unlocks every gated page for the browser session. Call this at the very top
    of the pages that should be protected (not the public Guide page).
    """
    if not APP_PASSWORD or st.session_state.get("authed"):
        return
    st.markdown(
        f'<div class="apphead"><div class="brand">{BRAND_MARK}Knowledge Assistant</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### :material/lock: Sign in")
    st.caption("This area is password-protected. Enter the access password to continue — "
               "or open the **Guide** page, which is public.")
    with st.form("login_form", clear_on_submit=False):
        pw = st.text_input("Password", type="password", placeholder="Access password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        if pw == APP_PASSWORD:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Incorrect password. Try again.")
    st.stop()


def logout_button() -> None:
    """Render a Log out button (call inside a sidebar context) when signed in."""
    if APP_PASSWORD and st.session_state.get("authed"):
        if st.button(":material/logout: Log out", use_container_width=True):
            st.session_state.authed = False
            st.rerun()
