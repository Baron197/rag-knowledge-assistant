"""Tests for the /eval-results endpoint.

Points EVAL_RESULTS_DIR at a temp dir with crafted reports (including an
OpenAI-mode run that carries Ragas metrics) and checks the endpoint surfaces
them in the documented shape, newest first, without touching the real corpus.
"""
from __future__ import annotations

import importlib
import json

from fastapi.testclient import TestClient


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("EVAL_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")  # hermetic: ignore a dev .env
    from src.rag import api
    importlib.reload(api)
    return TestClient(api.app)


def _write_openai_eval(results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "eval-20260101T000000Z.json").write_text(json.dumps({
        "timestamp": "20260101T000000Z",
        "providers": {"llm": "openai", "embedding": "openai",
                      "vector_backend": "numpy", "retrieval_mode": "hybrid", "top_k": 4},
        "retrieval_metrics": {"context_recall_at_k": 1.0, "recall_at_1": 0.83, "mrr": 0.9,
                              "refusal_accuracy": 1.0, "answerable_questions": 23,
                              "refusal_questions": 5, "avg_cost_usd": 0.0003, "avg_latency_ms": 900.0},
        "ragas_metrics": {"faithfulness": 0.95, "answer_relevancy": 0.9,
                          "context_precision": 0.85, "context_recall": 0.88},
        "per_question": [{"question": "q", "expected_sources": ["a.md"],
                          "retrieved_sources": ["a.md"], "refusal_question": False,
                          "correct": True, "first_relevant_rank": 1,
                          "cost_usd": 0.0003, "latency_ms": 900.0}],
    }))
    (results_dir / "compare-20260101T000000Z.json").write_text(json.dumps({
        "timestamp": "20260101T000000Z",
        "providers": {"embedding": "openai", "top_k": 4},
        "results": {"vector": {"context_recall_at_k": 1.0, "recall_at_1": 0.7, "mrr": 0.8},
                    "hybrid": {"context_recall_at_k": 1.0, "recall_at_1": 0.83, "mrr": 0.88}},
    }))


def test_eval_results_surfaces_openai_run_with_ragas(tmp_path, monkeypatch):
    """An OpenAI-mode report (with Ragas) is returned in both lists."""
    _write_openai_eval(tmp_path / "results")
    client = _client(tmp_path, monkeypatch)
    r = client.get("/eval-results")
    assert r.status_code == 200
    body = r.json()
    assert len(body["eval_runs"]) == 1
    assert len(body["compare_runs"]) == 1
    run = body["eval_runs"][0]
    assert run["providers"]["llm"] == "openai"
    assert run["ragas_metrics"]["faithfulness"] == 0.95
    assert run["_name"] == "eval-20260101T000000Z.json"


def test_eval_results_empty_when_no_reports(tmp_path, monkeypatch):
    """A missing/empty results dir yields empty lists, not an error."""
    client = _client(tmp_path, monkeypatch)
    r = client.get("/eval-results")
    assert r.status_code == 200
    body = r.json()
    assert body["eval_runs"] == []
    assert body["compare_runs"] == []


def test_eval_results_skips_corrupt_json(tmp_path, monkeypatch):
    """A torn/corrupt report file is skipped rather than failing the endpoint."""
    results = tmp_path / "results"
    _write_openai_eval(results)
    (results / "eval-20260102T000000Z.json").write_text("{ not valid json")
    client = _client(tmp_path, monkeypatch)
    r = client.get("/eval-results")
    assert r.status_code == 200
    assert len(r.json()["eval_runs"]) == 1  # only the valid one
