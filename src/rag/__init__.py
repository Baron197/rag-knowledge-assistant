"""Top-level package for the RAG Knowledge Assistant.

This package implements a small but production-shaped Retrieval-Augmented
Generation (RAG) service. The public sub-modules are:

- ``config``        : typed settings loaded from environment / .env
- ``embeddings``    : text -> vector providers (OpenAI + keyless fake)
- ``chunking``      : split documents into overlapping, token-bounded passages
- ``vectorstore``   : store + nearest-neighbour search (NumPy + pgvector)
- ``lexical``       : BM25 keyword search + Reciprocal Rank Fusion (hybrid)
- ``retriever``     : turn a question into the most relevant passages
- ``prompts``       : the grounded / cited / refusal prompt contract
- ``llm``           : answer generation providers (OpenAI + keyless fake)
- ``observability`` : per-query traces (latency / tokens / cost) + aggregation
- ``pipeline``      : orchestrates retrieve -> prompt -> generate -> trace (+cache)
- ``ingest``        : load -> chunk -> embed -> store (CLI entry point)
- ``api``           : FastAPI service exposing the pipeline

``__version__`` is the package version surfaced by the API and reports.
"""

__version__ = "0.1.0"
