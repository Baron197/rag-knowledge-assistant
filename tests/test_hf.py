"""Optional tests for the local Hugging Face providers.

These are SKIPPED entirely unless `sentence-transformers` is installed (it is not
part of the default deps or CI -- run `pip install -r requirements-hf.txt` first). When run, they
download a small model once and verify the HF embedder integrates correctly and
can power real retrieval. The generation model is not exercised here because it
is large; the embedder is the meaningful, lightweight check.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.config import Settings
from src.rag.embeddings import get_embedder


def test_hf_embedder_produces_unit_vectors():
    """The HF embedder returns (n, dim) L2-normalised vectors (cosine == dot)."""
    pytest.importorskip("sentence_transformers")
    import numpy as np

    emb = get_embedder(Settings(embedding_provider="hf"))
    vecs = emb.embed(["rate limits", "webhook events"])
    assert vecs.shape == (2, emb.dim)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_hf_embedder_powers_retrieval(tmp_path):
    """Real HF embeddings + the fake LLM still find the right source document."""
    pytest.importorskip("sentence_transformers")
    from src.rag.ingest import ingest
    from src.rag.pipeline import RAGPipeline

    s = Settings(
        llm_provider="fake",
        embedding_provider="hf",
        vector_backend="numpy",
        index_dir=tmp_path / "idx",
        docs_dir=Path("data/docs"),
        trace_dir=tmp_path / "tr",
        top_k=4,
    )
    ingest(settings=s, reset=True)
    ans = RAGPipeline(settings=s).answer("What is the per-second rate limit on the Growth plan?")
    assert "03-rate-limits.md" in {c.source for c in ans.citations}
