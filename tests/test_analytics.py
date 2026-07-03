"""Tests for the /analytics endpoint and the tracer's record reader.

Runs on the keyless `fake` path through a FastAPI test client: a query writes a
trace, and /analytics returns it in the documented per-query shape. Env vars
point DOCS_DIR / INDEX_DIR / TRACE_DIR at a temp directory so the real corpus
and trace file are never touched.
"""
from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("DOCS_DIR", "data/docs")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")  # hermetic: ignore a dev .env
    from src.rag import api
    importlib.reload(api)
    return TestClient(api.app)


def test_analytics_empty_before_any_query(tmp_path, monkeypatch):
    """With no traces yet, /analytics returns an empty list."""
    client = _client(tmp_path, monkeypatch)
    client.post("/ingest")
    r = client.get("/analytics")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0
    assert body["queries"] == []


def test_analytics_records_a_query(tmp_path, monkeypatch):
    """A query is traced and surfaces in /analytics with the documented shape."""
    client = _client(tmp_path, monkeypatch)
    client.post("/ingest")
    client.post("/query", json={"question": "What are the rate limits?"})
    r = client.get("/analytics")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    row = body["queries"][0]
    for key in ("ts", "question", "answer_preview", "timings_ms", "tokens",
                "cost_usd", "n_contexts", "sources"):
        assert key in row
    assert isinstance(row["timings_ms"], dict)
    assert isinstance(row["sources"], list)


def test_analytics_respects_limit(tmp_path, monkeypatch):
    """The `limit` param bounds the number of returned rows to the most recent."""
    client = _client(tmp_path, monkeypatch)
    client.post("/ingest")
    for q in ("What are the rate limits?", "How do I rotate a key?", "What plans exist?"):
        client.post("/query", json={"question": q})
    r = client.get("/analytics", params={"limit": 2})
    assert r.status_code == 200
    assert r.json()["count"] == 2
