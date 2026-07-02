"""Retrieval layer: turn a question into the most relevant passages.

Two modes, selected by config:

* "vector"  -- pure semantic search over the vector store.
* "hybrid"  -- semantic search AND BM25 keyword search, combined with
               Reciprocal Rank Fusion. Hybrid reliably improves ranking,
               especially for exact tokens (error codes, API names) that
               embeddings tend to blur.

Retrieval is kept separate from generation so it can be evaluated on its own --
retrieval quality is the usual root cause of bad RAG answers.
"""
from __future__ import annotations

from .embeddings import Embedder
from .lexical import BM25, reciprocal_rank_fusion
from .vectorstore import Hit, VectorStore


class Retriever:
    """Embeds a query and returns the top passages, in either vector or hybrid mode."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int = 4,
        min_score: float = 0.0,
        mode: str = "vector",
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k          # default number of passages to return
        self.min_score = min_score  # vector-mode relevance gate (0 = off)
        self.mode = mode            # "vector" | "hybrid"
        self.rrf_k = rrf_k          # fusion constant for hybrid mode
        self._bm25: BM25 | None = None  # lazily built keyword index (hybrid only)

    def _bm25_index(self) -> BM25:
        """Build (once) and cache the BM25 index from the store's chunks."""
        if self._bm25 is None:
            self._bm25 = BM25(self.store.all_docs())
        return self._bm25

    def retrieve(self, query: str, k: int | None = None) -> tuple[list[Hit], int]:
        """Return (hits, embedding_tokens_used) for the query.

        In hybrid mode we pull a wider candidate pool from both the vector store
        and BM25, then fuse with RRF and keep the top k. In vector mode we apply
        the optional cosine relevance gate (hybrid fuses incomparable scores, so
        the gate doesn't apply there).
        """
        k = k or self.top_k
        vec = self.embedder.embed([query])[0]
        tokens = self.embedder.last_token_count()  # for cost accounting upstream

        if self.mode == "hybrid":
            pool = max(k * 5, 20)  # over-fetch from each retriever before fusing
            vector_hits = self.store.search(vec, pool)
            lexical_hits = self._bm25_index().search(query, pool)
            hits = reciprocal_rank_fusion([vector_hits, lexical_hits], self.rrf_k)[:k]
        else:
            hits = self.store.search(vec, k)
            if self.min_score > 0.0:
                hits = [h for h in hits if h.score >= self.min_score]

        return hits, tokens
