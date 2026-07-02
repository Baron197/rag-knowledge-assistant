"""RAG orchestration: retrieve -> build grounded prompt -> generate -> trace.

This is the seam everything else plugs into. `answer()` returns not just text
but the citations, the retrieved contexts, token/cost accounting and per-stage
latency -- the structured result an API, a UI, and an eval harness all need.

It also adds a small, thread-safe LRU answer cache: repeated questions are served
from memory at zero LLM cost, a simple real cost-optimisation lever.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, replace

from .config import Settings, get_settings
from .embeddings import get_embedder
from .llm import get_llm
from .observability import Trace, Tracer, cost_usd
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retriever import Retriever
from .vectorstore import Hit, get_vector_store

# Fallback embedding dim only if an embedder somehow doesn't expose `.dim`.
_FALLBACK_DIM = 256


@dataclass
class Citation:
    """A numbered source reference shown with an answer: number, source file, snippet."""

    n: int
    source: str
    snippet: str


@dataclass
class Answer:
    """The full structured result of a query, returned to the API/UI/eval."""

    question: str
    answer: str
    citations: list[Citation]
    cost_usd: float
    timings_ms: dict
    tokens: dict
    n_contexts: int
    retrieval_mode: str = "vector"
    cached: bool = False  # True when served from the LRU cache (cost 0)


class RAGPipeline:
    """Wires the components together and runs the query lifecycle.

    Built once and reused. Holds the embedder, vector store, retriever, LLM,
    tracer, and the answer cache.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder = get_embedder(self.settings)
        # Vector dimension comes from the embedder itself (correct for any model).
        dim = getattr(self.embedder, "dim", _FALLBACK_DIM)
        self.store = get_vector_store(self.settings, dim)
        self.retriever = Retriever(
            self.embedder,
            self.store,
            self.settings.top_k,
            self.settings.min_relevance_score,
            self.settings.retrieval_mode,
            self.settings.rrf_k,
        )
        self.llm = get_llm(self.settings)
        self.tracer = Tracer(self.settings.trace_dir)
        # LRU cache of recent answers, guarded by a lock for thread-safety.
        self._cache: OrderedDict[tuple, Answer] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _cache_get(self, key: tuple) -> Answer | None:
        """Return a cached answer for `key` (marking it most-recently-used)."""
        with self._cache_lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def _cache_put(self, key: tuple, ans: Answer) -> None:
        """Insert an answer and evict the least-recently-used over capacity."""
        with self._cache_lock:
            self._cache[key] = ans
            while len(self._cache) > self.settings.cache_size:
                self._cache.popitem(last=False)

    def answer(self, question: str, k: int | None = None) -> Answer:
        """Run the full query lifecycle and return a structured `Answer`.

        Steps: cache check -> (timed) retrieval -> grounded prompt -> (timed)
        generation -> cost accounting -> trace -> cache store.
        """
        # Normalise k so an explicit default and an omitted value share a cache key.
        k = k or self.settings.top_k

        if self.settings.enable_cache:
            hit = self._cache_get((question, k))
            if hit is not None:
                # Served from cache: free and instant. Not traced, so /metrics
                # reflects real (uncached) query cost and latency.
                return replace(hit, cost_usd=0.0, cached=True, timings_ms={"cache_hit": 0.0})

        trace = Trace(question=question)

        # 1) Retrieval (timed): question -> top passages.
        with self.tracer.span(trace, "retrieval"):
            hits, embed_tokens = self.retriever.retrieve(question, k)

        # 2) Generation (timed): grounded prompt -> answer with citations.
        user_prompt = build_user_prompt(question, hits)
        with self.tracer.span(trace, "generation"):
            result = self.llm.complete(SYSTEM_PROMPT, user_prompt)

        # 3) Cost = embedding tokens + prompt/completion tokens, via the price table.
        if self.settings.embedding_provider == "openai":
            embed_model = self.settings.openai_embedding_model
        elif self.settings.embedding_provider == "hf":
            embed_model = self.settings.hf_embedding_model  # local -> $0 (unknown to price table)
        else:
            embed_model = "fake-embed"
        total_cost = cost_usd(embed_model, embed_tokens) + cost_usd(
            result.model, result.prompt_tokens, result.completion_tokens
        )

        # 4) Fill in and persist the trace.
        trace.tokens = {
            "embedding": embed_tokens,
            "prompt": result.prompt_tokens,
            "completion": result.completion_tokens,
        }
        trace.cost_usd = total_cost
        trace.n_contexts = len(hits)
        trace.answer = result.text
        trace.sources = [h.doc.source for h in hits]
        self.tracer.record(trace)

        # 5) Assemble the structured answer and cache it.
        ans = Answer(
            question=question,
            answer=result.text,
            citations=self._citations(hits),
            cost_usd=total_cost,
            timings_ms=trace.timings_ms,
            tokens=trace.tokens,
            n_contexts=len(hits),
            retrieval_mode=self.settings.retrieval_mode,
            cached=False,
        )

        if self.settings.enable_cache:
            self._cache_put((question, k), ans)
        return ans

    @staticmethod
    def _citations(hits: list[Hit]) -> list[Citation]:
        """Turn retrieved hits into numbered citation records with short snippets."""
        out = []
        for i, h in enumerate(hits, start=1):
            snippet = h.doc.text.strip().replace("\n", " ")
            out.append(Citation(n=i, source=h.doc.source, snippet=snippet[:200]))
        return out
