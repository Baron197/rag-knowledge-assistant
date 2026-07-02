"""Tests for the added features: BM25 lexical search, hybrid retrieval, the answer
cache, the dimension-mismatch guard, and chunk hard-splitting. All run on the
keyless `fake` path."""
from __future__ import annotations

from pathlib import Path

from src.rag.config import Settings
from src.rag.ingest import ingest
from src.rag.lexical import BM25, reciprocal_rank_fusion
from src.rag.pipeline import RAGPipeline
from src.rag.vectorstore import Doc, Hit


def _settings(tmp_path, **overrides) -> Settings:
    """Build keyless Settings pointing at a temp index, with optional overrides."""
    base = dict(
        llm_provider="fake",
        embedding_provider="fake",
        vector_backend="numpy",
        index_dir=tmp_path / "index",
        docs_dir=Path("data/docs"),
        trace_dir=tmp_path / "traces",
        top_k=4,
    )
    base.update(overrides)
    return Settings(**base)


def test_bm25_ranks_keyword_match_first():
    """BM25 ranks the document containing the query's keywords first."""
    docs = [
        Doc(id="a", text="Rate limits use a token bucket algorithm.", source="a", chunk_index=0),
        Doc(id="b", text="Webhooks deliver delivery events to your endpoint.", source="b", chunk_index=0),
    ]
    bm25 = BM25(docs)
    hits = bm25.search("token bucket rate limit", k=2)
    assert hits, "BM25 should return matches"
    assert hits[0].doc.id == "a"


def test_rrf_rewards_agreement():
    """Reciprocal Rank Fusion ranks a doc that both lists agree on first."""
    d1 = Doc(id="1", text="x", source="1", chunk_index=0)
    d2 = Doc(id="2", text="y", source="2", chunk_index=0)
    list_a = [Hit(d1, 0.9), Hit(d2, 0.1)]
    list_b = [Hit(d1, 5.0), Hit(d2, 1.0)]
    fused = reciprocal_rank_fusion([list_a, list_b])
    assert fused[0].doc.id == "1"  # agreed top by both -> ranked first


def test_hybrid_retrieval_finds_relevant_source(tmp_path):
    """Hybrid (vector + BM25) retrieval finds the right source and reports its mode."""
    s = _settings(tmp_path, retrieval_mode="hybrid")
    ingest(settings=s, reset=True)
    pipe = RAGPipeline(settings=s)
    ans = pipe.answer("What is the per-second rate limit on the Growth plan?")
    assert "03-rate-limits.md" in {c.source for c in ans.citations}
    assert ans.retrieval_mode == "hybrid"


def test_cache_hit_is_marked_and_free(tmp_path):
    """A repeated question is served from cache: marked cached and costing zero."""
    s = _settings(tmp_path, enable_cache=True)
    ingest(settings=s, reset=True)
    pipe = RAGPipeline(settings=s)
    q = "How do I rotate a leaked API key?"
    first = pipe.answer(q)
    second = pipe.answer(q)
    assert first.cached is False
    assert second.cached is True
    assert second.cost_usd == 0.0
    assert second.answer == first.answer


def test_cache_can_be_disabled(tmp_path):
    """With caching off, repeats are recomputed (never marked cached)."""
    s = _settings(tmp_path, enable_cache=False)
    ingest(settings=s, reset=True)
    pipe = RAGPipeline(settings=s)
    q = "What webhook events does Nimbus emit?"
    pipe.answer(q)
    assert pipe.answer(q).cached is False


def test_dimension_mismatch_raises(tmp_path):
    """Adding vectors of a different dimension raises DimensionMismatchError."""
    import numpy as np
    import pytest

    from src.rag.vectorstore import DimensionMismatchError, Doc, NumpyVectorStore

    store = NumpyVectorStore(tmp_path / "idx")
    store.add([Doc(id="a", text="x", source="a", chunk_index=0)], np.zeros((1, 8), dtype="float32"))
    with pytest.raises(DimensionMismatchError):
        store.add([Doc(id="b", text="y", source="b", chunk_index=0)], np.zeros((1, 16), dtype="float32"))


def test_cache_key_normalizes_k(tmp_path):
    """k=None and k=<default top_k> map to the same cache entry."""
    s = _settings(tmp_path, enable_cache=True, top_k=4)
    ingest(settings=s, reset=True)
    pipe = RAGPipeline(settings=s)
    q = "What does the Free plan include?"
    first = pipe.answer(q, k=None)       # uses default top_k internally
    second = pipe.answer(q, k=4)          # explicit default -> must hit the same cache entry
    assert first.cached is False
    assert second.cached is True


def test_chunk_hard_split_caps_chunk_size():
    """A single over-long segment is hard-split so no chunk blows the size budget."""
    from src.rag.chunking import _ntokens, chunk_text

    long_line = " ".join(f"word{i}" for i in range(4000))  # one segment, no sentence breaks
    chunks = chunk_text(long_line, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    # allow a small overshoot from the last word pushing over the boundary
    assert all(_ntokens(c.text) <= 200 * 1.5 for c in chunks)
