"""Tests for the /upload endpoint.

Uploaded files are saved into DOCS_DIR, the index is rebuilt, and the new content
becomes searchable -- all on the keyless `fake` path. Env vars point DOCS_DIR /
INDEX_DIR / TRACE_DIR at a temp directory so the real `data/docs` corpus is never
touched, and the api module is reloaded so its singleton picks up that config.
"""
from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def _client(tmp_path, monkeypatch) -> TestClient:
    """A TestClient whose pipeline reads/writes temp docs + index dirs (keyless)."""
    monkeypatch.setenv("DOCS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")  # hermetic: ignore a dev .env
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    from src.rag import api
    importlib.reload(api)  # rebuild module-level app + singleton with the new env
    return TestClient(api.app)


def test_upload_saves_indexes_and_is_searchable(tmp_path, monkeypatch):
    """A .md upload is stored, indexed, and then retrievable via /query."""
    client = _client(tmp_path, monkeypatch)
    body = b"# Refunds\n\nRefunds are processed within 14 business days via the billing portal."
    r = client.post("/upload", files=[("files", ("refunds.md", body, "text/markdown"))])
    assert r.status_code == 200, r.text
    data = r.json()
    assert "refunds.md" in data["saved"]
    assert data["indexed_chunks"] > 0
    # The uploaded document is now searchable.
    q = client.post("/query", json={"question": "How long do refunds take?"})
    assert q.status_code == 200
    assert "refunds.md" in {c["source"] for c in q.json()["citations"]}


def test_upload_rejects_unsupported_type(tmp_path, monkeypatch):
    """An unsupported extension is rejected with HTTP 400 (nothing indexed)."""
    client = _client(tmp_path, monkeypatch)
    r = client.post("/upload", files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 400
