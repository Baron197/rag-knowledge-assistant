"""Lexical (keyword) retrieval with a small, dependency-free BM25 index.

Vector search captures *meaning* but can miss exact tokens (error codes, API
names, rare identifiers). BM25 is the classic keyword ranker and is strong
exactly where embeddings are weak. Running both and fusing the results ("hybrid
retrieval") is a standard production technique that reliably beats either alone.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .vectorstore import Doc, Hit

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase the text and split it into alphanumeric word tokens."""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 keyword ranker over a fixed set of documents.

    Precomputes, per document, the term frequencies and length, plus a
    corpus-wide inverse-document-frequency (IDF) so rarer query terms count for
    more. `k1` controls term-frequency saturation; `b` controls length
    normalisation.
    """

    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(d.text) for d in docs]   # tokens per doc
        self._freqs = [Counter(toks) for toks in self._tokens]  # term freq per doc
        self._len = [len(toks) for toks in self._tokens]  # length per doc
        self.n = len(docs)
        self.avgdl = (sum(self._len) / self.n) if self.n else 0.0  # average doc length

        # Document frequency: in how many docs each term appears.
        df: Counter[str] = Counter()
        for toks in self._tokens:
            df.update(set(toks))
        # BM25 idf with +1 smoothing inside the log to keep it non-negative.
        self._idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, k: int) -> list[Hit]:
        """Score every document against the query and return the top-k (score>0)."""
        if self.n == 0:
            return []
        q_terms = tokenize(query)
        scores = [0.0] * self.n
        for term in q_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue  # query term not in the corpus -> contributes nothing
            for i in range(self.n):
                f = self._freqs[i].get(term, 0)
                if f == 0:
                    continue
                # Okapi BM25 term contribution with length normalisation.
                denom = f + self.k1 * (1 - self.b + self.b * self._len[i] / (self.avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        ranked = sorted(range(self.n), key=lambda i: -scores[i])[:k]
        return [Hit(doc=self.docs[i], score=float(scores[i])) for i in ranked if scores[i] > 0]


def reciprocal_rank_fusion(rank_lists: list[list[Hit]], k0: int = 60) -> list[Hit]:
    """Combine several ranked lists into one via Reciprocal Rank Fusion (RRF).

    RRF rewards documents that rank highly across *multiple* retrievers without
    needing to compare their incomparable score scales: each document's fused
    score is the sum over lists of ``1 / (k0 + rank)``. A document ranked highly
    in two lists outranks one ranked highly in only one.
    """
    scores: dict[str, float] = {}
    doc_by_id: dict[str, Doc] = {}
    for hits in rank_lists:
        for rank, h in enumerate(hits):
            scores[h.doc.id] = scores.get(h.doc.id, 0.0) + 1.0 / (k0 + rank + 1)
            doc_by_id[h.doc.id] = h.doc
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    return [Hit(doc=doc_by_id[doc_id], score=score) for doc_id, score in ordered]
