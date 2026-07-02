"""Embedding providers behind one small interface.

An *embedding* turns a piece of text into a fixed-length vector of numbers such
that texts with similar meaning get vectors pointing in similar directions. This
module hides the choice of provider behind the `Embedder` protocol:

- `OpenAIEmbedder` calls the real OpenAI embedding API (paid, needs a key).
- `HuggingFaceEmbedder` runs a Hugging Face sentence-transformers model **locally**
  -- free, no API key, downloads the model once and then works offline.
- `FakeEmbedder` is a deterministic hashing vectoriser -- no network, no key, no
  cost -- so the whole pipeline (and CI) can run offline. Its vectors only encode
  word overlap, not deep meaning, so retrieval still behaves sensibly in tests.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from .config import Settings, resolve_hf_device

# Splits text into lowercase alphanumeric word tokens (used by the fake embedder).
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Structural interface every embedding provider must satisfy.

    Any class exposing these members "is an" Embedder (duck typing) -- no base
    class needed.
    """

    dim: int  # dimensionality of the vectors this embedder produces

    def embed(self, texts: list[str]) -> np.ndarray:  # (n, dim) float32, L2-normalized
        """Embed a batch of texts into an (n, dim) array of unit vectors."""
        ...

    def last_token_count(self) -> int:
        """Tokens consumed by the most recent embed() call (for cost accounting)."""
        ...


def _normalize(mat: np.ndarray) -> np.ndarray:
    """L2-normalise each row to length 1, so cosine similarity == dot product.

    The zero-vector guard avoids dividing by zero for empty/all-out-of-vocab text.
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class FakeEmbedder:
    """Deterministic, dependency-free embedder (a hashed bag-of-words).

    Each word is hashed to a bucket in a fixed-size vector and counted. This
    captures keyword overlap (enough for tests) but not true semantics -- which is
    why semantic-only metrics like refusal accuracy are only meaningful on a real
    (OpenAI or Hugging Face) path.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self._tokens = 0  # remembered for last_token_count()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Hash each text's words into a count vector, then L2-normalise."""
        self._tokens = sum(len(t.split()) for t in texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _TOKEN_RE.findall(text.lower()):
                # md5 -> stable integer -> bucket index in [0, dim)
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _normalize(out)

    def last_token_count(self) -> int:
        return self._tokens


class HuggingFaceEmbedder:
    """Local, free embeddings via sentence-transformers -- no API key.

    Downloads the model the first time (then cached/offline). `encode(...,
    normalize_embeddings=True)` returns unit vectors, matching the rest of the
    app's "dot product == cosine" assumption. Cost is always $0 (runs on your
    machine), so `last_token_count()` is only a rough word count.
    """

    def __init__(self, settings: Settings) -> None:
        # Imported lazily so the keyless path never needs torch/sentence-transformers.
        from sentence_transformers import SentenceTransformer

        self.model_name = settings.hf_embedding_model
        self._model = SentenceTransformer(
            self.model_name, device=resolve_hf_device(settings.hf_device)
        )
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self._tokens = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self._tokens = sum(len(t.split()) for t in texts)
        mat = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return mat.astype(np.float32)

    def last_token_count(self) -> int:
        return self._tokens


# Output dimensions of known OpenAI embedding models, used to size the vector
# store up front. For other models (or the API's dimension-reduction feature),
# set OPENAI_EMBEDDING_DIM explicitly.
_OPENAI_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    """Real embeddings via the OpenAI API (paid)."""

    def __init__(self, settings: Settings) -> None:
        # Import lazily so the keyless path never needs the openai package/key.
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for embedding_provider=openai. "
                "Set it in .env, or use EMBEDDING_PROVIDER=fake (offline) or "
                "EMBEDDING_PROVIDER=hf (free local Hugging Face) instead."
            )
        # Timeout + retries so a transient API failure doesn't fail the query.
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_s,
            max_retries=settings.openai_max_retries,
        )
        self.model = settings.openai_embedding_model
        # An explicit dim is forwarded to the API as the `dimensions` parameter
        # (text-embedding-3-* models support shortening their output vectors).
        self._requested_dim = settings.openai_embedding_dim
        self.dim = self._requested_dim or _OPENAI_DIMS.get(self.model, 1536)
        self._tokens = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        """Call the API for a batch, record token usage, and L2-normalise."""
        kwargs: dict = {"model": self.model, "input": texts}
        if self._requested_dim and self.model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self._requested_dim
        resp = self._client.embeddings.create(**kwargs)
        self._tokens = resp.usage.total_tokens
        mat = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # Trust the API response over the lookup table for unknown models; the
        # store's dimension-mismatch guard reports any inconsistency clearly.
        self.dim = int(mat.shape[1])
        return _normalize(mat)

    def last_token_count(self) -> int:
        return self._tokens


def get_embedder(settings: Settings) -> Embedder:
    """Factory: pick the embedding provider named in settings."""
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings)
    if settings.embedding_provider == "hf":
        return HuggingFaceEmbedder(settings)
    return FakeEmbedder()
