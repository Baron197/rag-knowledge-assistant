"""End-to-end tests on the keyless `fake` path.

These run in CI with no API key and no network: they prove the full
ingest -> retrieve -> generate -> cite path works, that retrieval finds the
right document, and that out-of-scope questions are refused rather than
hallucinated.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.config import Settings
from src.rag.ingest import ingest
from src.rag.pipeline import RAGPipeline


@pytest.fixture()
def pipeline(tmp_path) -> RAGPipeline:
    """A fresh keyless pipeline with the demo corpus ingested into a temp index."""
    settings = Settings(
        llm_provider="fake",
        embedding_provider="fake",
        vector_backend="numpy",
        index_dir=tmp_path / "index",
        docs_dir=Path("data/docs"),
        trace_dir=tmp_path / "traces",
        top_k=4,
    )
    n = ingest(settings=settings, reset=True)
    assert n > 0, "expected to ingest at least one chunk"
    return RAGPipeline(settings=settings)


def test_ingest_indexes_chunks(pipeline: RAGPipeline):
    """Ingestion populated the vector store."""
    assert pipeline.store.count() > 0


def test_answer_returns_citations(pipeline: RAGPipeline):
    """A normal question returns a non-empty answer with at least one citation."""
    ans = pipeline.answer("What does the Free plan include?")
    assert ans.n_contexts > 0
    assert ans.citations, "expected at least one citation"
    assert ans.answer.strip() != ""


def test_retrieval_finds_relevant_source(pipeline: RAGPipeline):
    """Retrieval surfaces the document that actually contains the answer."""
    ans = pipeline.answer("What is the per-second rate limit on the Growth plan?")
    sources = {c.source for c in ans.citations}
    assert "03-rate-limits.md" in sources


def test_out_of_scope_question_is_refused(tmp_path):
    """With the relevance gate on, an out-of-scope question retrieves nothing and
    the answer is the exact refusal sentence -- not a hallucinated response."""
    settings = Settings(
        llm_provider="fake",
        embedding_provider="fake",
        vector_backend="numpy",
        index_dir=tmp_path / "index",
        docs_dir=Path("data/docs"),
        trace_dir=tmp_path / "traces",
        top_k=4,
        # The relevance gate only applies in vector mode, so pin the mode
        # rather than inherit whatever a local .env sets.
        retrieval_mode="vector",
        # No real chunk can score this high against a nonsense query, so every
        # hit is gated out and the pipeline takes the no-context refusal path.
        min_relevance_score=0.99,
    )
    ingest(settings=settings, reset=True)
    ans = RAGPipeline(settings=settings).answer("zzzqqq nonexistent topic about quantum giraffes")
    assert ans.n_contexts == 0
    assert "don't have enough information" in ans.answer.lower()


def test_cost_is_zero_on_fake_path(pipeline: RAGPipeline):
    """The keyless fake providers incur no cost."""
    ans = pipeline.answer("How do I rotate a leaked API key?")
    assert ans.cost_usd == 0.0
