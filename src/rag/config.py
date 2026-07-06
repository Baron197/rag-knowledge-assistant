"""Central configuration for the whole app.

Every tunable value lives here in one typed `Settings` object instead of being
read with scattered `os.getenv` calls. pydantic-settings loads the values from
environment variables and an optional `.env` file, validates their types once,
and exposes documented defaults. The defaults are chosen so the project runs
end-to-end with **no API key** (the keyless `fake` providers + the local `numpy`
vector store), which is what the tests and CI rely on.

Provider tiers (set via LLM_PROVIDER / EMBEDDING_PROVIDER):
  * "fake"   -- deterministic, offline, zero-cost (default; used by tests/CI).
  * "hf"     -- real open-source models from Hugging Face, run LOCALLY, free,
                NO API KEY (needs the optional `requirements-hf.txt` deps).
  * "openai" -- the paid OpenAI API (needs an API key).
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, populated from env / `.env`.

    Field values are resolved in this order: explicit constructor argument >
    environment variable (case-insensitive name) > `.env` file entry > the
    default defined here. `extra="ignore"` means unknown keys in `.env` are
    ignored rather than raising.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Provider selection -------------------------------------------------
    # "fake" (default, offline) | "hf" (local Hugging Face, free, no key) | "openai".
    llm_provider: str = "fake"
    embedding_provider: str = "fake"

    # --- OpenAI settings (used only when a provider above is "openai") -------
    openai_api_key: str | None = None
    openai_llm_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    # Override the embedding dimension. For text-embedding-3-* models this is
    # forwarded to the API as `dimensions` (shortened output vectors); for other
    # models it just declares the true size up front. None = auto.
    openai_embedding_dim: int | None = None
    # Client resilience: per-request timeout and SDK retries (with backoff) so a
    # transient API failure or rate limit doesn't fail the query outright.
    openai_timeout_s: float = 60.0
    openai_max_retries: int = 3

    # --- Hugging Face settings (used when a provider above is "hf") ----------
    # Free, local, no API key. Defaults are small, ungated, CPU-friendly models.
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
    hf_llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"                    # Apache-2.0, ungated
    hf_device: str = "cpu"          # "cpu" or a GPU index like "0"
    hf_max_new_tokens: int = 512    # generation length cap for the local LLM

    # --- Vector store -------------------------------------------------------
    vector_backend: str = "numpy"         # "numpy" (default) | "pgvector"
    index_dir: Path = Path("data/index")  # where the numpy index is persisted
    pg_dsn: str = "postgresql://rag:rag@localhost:5432/rag"  # used for pgvector

    # --- Retrieval / chunking knobs -----------------------------------------
    chunk_size: int = 800                 # target tokens per chunk
    chunk_overlap: int = 120              # tokens shared between adjacent chunks
    top_k: int = 4                        # how many passages to retrieve
    # "vector" = semantic search only; "hybrid" = semantic + BM25 fused (RRF).
    retrieval_mode: str = "vector"
    rrf_k: int = 60                       # Reciprocal Rank Fusion constant
    # Drop retrieved chunks below this cosine similarity (0.0 = disabled). A guard
    # against feeding weak/irrelevant context to the model. ~0.25 is a reasonable
    # start for text-embedding-3-small. Applied in vector mode only (hybrid fuses
    # incomparable score scales).
    min_relevance_score: float = 0.0

    # --- Caching (cost optimisation): cache answers for repeated questions ---
    enable_cache: bool = True
    cache_size: int = 256                 # max distinct questions kept in the LRU

    # --- API auth (optional) ------------------------------------------------
    # If set, the cost/mutation endpoints (/query, /ingest, /upload) require a
    # matching `X-API-Key` header. Empty (default) leaves the API open, so
    # keyless/local use is unchanged. The Streamlit UI forwards it automatically.
    api_key: str = ""

    # --- Paths --------------------------------------------------------------
    docs_dir: Path = Path("data/docs")    # source documents to ingest
    trace_dir: Path = Path("traces")      # where per-query traces are written
    # Where the eval harness writes reports and the /eval-results endpoint reads
    # them. Anchored to the repo root (not the process CWD) so the reader (API,
    # which may be launched from anywhere) and the writer (eval.run_eval) always
    # agree on one absolute directory. Override with EVAL_RESULTS_DIR.
    eval_results_dir: Path = Path(__file__).resolve().parents[2] / "eval" / "results"


def get_settings() -> Settings:
    """Return a fresh `Settings` instance (re-reads env/.env each call)."""
    return Settings()

def resolve_hf_device(value: str) -> str:
    """Map the HF_DEVICE setting to a torch device string usable by BOTH
    sentence-transformers and transformers (e.g. "cpu", "cuda:0", "mps").

    Accepts "cpu"/""/"auto" (-> "cpu"), a bare GPU index like "0" (-> "cuda:0"),
    or an explicit torch device string (passed through unchanged).
    """
    v = (value or "").strip().lower()
    if v in ("", "cpu", "auto"):
        return "cpu"
    if v.isdigit():
        return f"cuda:{v}"
    return value
