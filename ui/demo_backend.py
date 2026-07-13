"""Single-process demo backend for hosts that run only ONE process
(e.g. Streamlit Community Cloud).

Normally this project is a thin Streamlit client over a SEPARATE FastAPI service
(UI -> API -> pipeline). Some free hosts run a single process, so here we boot
that FastAPI service in-process on first load -- using the keyless `fake`
providers -- and let the unchanged UI talk to it on localhost.

It is a NO-OP anywhere a real API is already reachable (local dev, Docker, a cloud
VM), so the clean two-process architecture is left intact; this only activates on
a single-process host where nothing is listening yet.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlparse

import streamlit as st

_ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root (ui/ -> ..)


def _reachable(base: str) -> bool:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=1.5):
            return True
    except Exception:  # noqa: BLE001 -- any failure means "not up yet"
        return False


@st.cache_resource(show_spinner="Starting the demo backend (first load only)...")
def ensure_local_backend():
    """Boot an in-process FastAPI backend when no external API is reachable.

    Cached, so it runs once per app (not per session). Returns the uvicorn
    subprocess handle (kept alive by the cache) or None when an external API is
    already serving.
    """
    base = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000")
    if _reachable(base):
        return None  # a real API is already up -> stay a pure thin client

    # Keyless demo defaults (only fills what a deployer hasn't already set).
    os.environ.setdefault("LLM_PROVIDER", "fake")
    os.environ.setdefault("EMBEDDING_PROVIDER", "fake")
    os.environ.setdefault("VECTOR_BACKEND", "numpy")
    os.environ.setdefault("RETRIEVAL_MODE", "hybrid")

    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    # Build the index once from the bundled corpus (fast on the fake path); the
    # API subprocess then reads the same on-disk index.
    from src.rag.ingest import ingest
    ingest(reset=True)

    parsed = urlparse(base)
    host, port = (parsed.hostname or "127.0.0.1"), (parsed.port or 8000)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.rag.api:app",
         "--host", host, "--port", str(port), "--log-level", "warning"],
        cwd=str(_ROOT),
    )
    for _ in range(90):  # wait up to ~45s for the API to answer /health
        if _reachable(f"http://{host}:{port}"):
            break
        time.sleep(0.5)
    return proc
