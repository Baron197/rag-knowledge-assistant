"""FastAPI service exposing the RAG pipeline.

Endpoints:
  GET  /health    -> liveness + store size + active config
  POST /ingest    -> (re)build the index from DOCS_DIR
  POST /upload    -> save uploaded file(s) into DOCS_DIR, then re-index them
  POST /query     -> grounded answer with citations, cost and latency
  GET  /metrics   -> aggregate cost/latency/throughput from traces
  GET  /analytics -> per-query trace records for the analytics dashboard
  GET  /eval-results -> evaluation reports (retrieval/Ragas/A-B) for the eval dashboard

The pipeline is built once and reused, so the API is the single source of truth
that both the Streamlit UI and the eval harness can call. Request bodies are
validated by pydantic models, and handlers convert unexpected errors into clean
HTTP 500s instead of leaking stack traces.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .ingest import ingest
from .pipeline import RAGPipeline

logger = logging.getLogger("rag.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="RAG Knowledge Assistant", version="0.3.0")
# Lazily-built singleton pipeline, shared across requests (reset on /ingest &
# /upload). Guarded by a lock: FastAPI runs sync handlers in a threadpool, so
# construction and reset could otherwise race.
_pipeline: RAGPipeline | None = None
_pipeline_lock = threading.Lock()

# Document types the ingestion pipeline knows how to read (see ingest._read_file).
ALLOWED_SUFFIXES = {".md", ".txt", ".html", ".htm", ".pdf"}
# Per-file cap for /upload so a huge upload can't exhaust memory or disk.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Whole-request cap (multiple files allowed, each <= MAX_UPLOAD_BYTES).
MAX_REQUEST_BYTES = 50 * 1024 * 1024


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Reject oversized requests up front (HTTP 413) before the body is parsed.

    Starlette spools the full multipart body to disk before a handler runs, so
    the per-file check inside /upload alone wouldn't stop a huge request from
    consuming bandwidth and temp disk. Checks Content-Length, which every
    normal client sends; chunked uploads without it fall through to the
    per-file cap.
    """
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_REQUEST_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request too large (max {MAX_REQUEST_BYTES // (1024 * 1024)} MB)."},
        )
    return await call_next(request)


def get_pipeline() -> RAGPipeline:
    """Return the shared pipeline, constructing it on first use (thread-safe)."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = RAGPipeline()
        return _pipeline


def _reset_pipeline() -> None:
    """Drop the cached pipeline so the next request rebuilds it against the
    refreshed store (and starts with an empty answer cache)."""
    global _pipeline
    with _pipeline_lock:
        _pipeline = None


class QueryRequest(BaseModel):
    """Validated body for POST /query."""

    question: str = Field(..., min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)  # optional override of top_k


class CitationModel(BaseModel):
    """A citation as returned in the API response."""

    n: int
    source: str
    snippet: str


class QueryResponse(BaseModel):
    """The JSON shape returned by POST /query."""

    question: str
    answer: str
    citations: list[CitationModel]
    cost_usd: float
    timings_ms: dict
    tokens: dict
    n_contexts: int
    retrieval_mode: str
    cached: bool


@app.get("/health")
def health() -> dict:
    """Liveness probe: returns status plus the active backend/provider/mode + size."""
    p = get_pipeline()
    return {
        "status": "ok",
        "vector_backend": p.settings.vector_backend,
        "llm_provider": p.settings.llm_provider,
        "retrieval_mode": p.settings.retrieval_mode,
        "indexed_chunks": p.store.count(),
    }


@app.post("/ingest")
def run_ingest(reset: bool = True) -> dict:
    """(Re)build the index from DOCS_DIR, then drop the cached pipeline so the next
    request sees the refreshed store."""
    try:
        n = ingest(reset=reset)
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    _reset_pipeline()
    return {"ingested_chunks": n}


@app.post("/upload")
async def upload(files: Annotated[list[UploadFile], File(...)]) -> dict:
    """Save uploaded document(s) into DOCS_DIR and rebuild the index.

    Accepts one or more files as multipart/form-data. Each filename is reduced to
    a bare name (stripping any directory components) to prevent path traversal,
    and files that are oversized or whose extension the loader can't read are
    skipped. After saving, the index is rebuilt so the new content is immediately
    searchable.
    """
    docs_dir = Path(get_settings().docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    skipped: list[str] = []
    for f in files:
        name = Path(f.filename or "").name  # strip any path -> just the filename
        if not name or Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
            skipped.append(f.filename or "(unnamed)")
            continue
        # Read at most cap+1 bytes: enough to detect an oversized file without
        # pulling it fully into memory (the request itself is already bounded
        # by the Content-Length middleware above).
        content = await f.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            skipped.append(f"{name} (over {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit)")
            continue
        (docs_dir / name).write_bytes(content)
        saved.append(name)

    if not saved:
        raise HTTPException(
            status_code=400,
            detail=f"No supported files uploaded. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )
    try:
        # This handler is async (for `await f.read`), so the synchronous,
        # potentially slow ingest must run in the threadpool -- running it
        # inline would stall the event loop and freeze every other endpoint.
        n = await run_in_threadpool(ingest, reset=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingestion after upload failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    # Threadpool too: the lock may be held by a request mid-pipeline-build.
    await run_in_threadpool(_reset_pipeline)
    return {"saved": saved, "skipped": skipped, "indexed_chunks": n}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Answer a question with grounded citations, cost and latency."""
    try:
        ans = get_pipeline().answer(req.question, req.k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
    return QueryResponse(
        question=ans.question,
        answer=ans.answer,
        citations=[CitationModel(n=c.n, source=c.source, snippet=c.snippet) for c in ans.citations],
        cost_usd=ans.cost_usd,
        timings_ms=ans.timings_ms,
        tokens=ans.tokens,
        n_contexts=ans.n_contexts,
        retrieval_mode=ans.retrieval_mode,
        cached=ans.cached,
    )


@app.get("/metrics")
def metrics() -> dict:
    """Aggregate cost/latency/throughput across all recorded query traces."""
    return get_pipeline().tracer.aggregate()


@app.get("/analytics")
def analytics(limit: int = 2000) -> dict:
    """Per-query trace records (most recent first bounded by `limit`) for the
    analytics dashboard: timings, tokens, cost, contexts and sources per query.

    Only uncached queries are traced (cache hits are intentionally not recorded),
    so this reflects real retrieval/generation work — the same rows `/metrics`
    aggregates, but unrolled so the client can filter and chart them.
    """
    limit = max(1, min(limit, 5000))
    rows = get_pipeline().tracer.records(limit)
    queries = [
        {
            "ts": r.get("ts"),
            "question": r.get("question", ""),
            "answer_preview": (r.get("answer", "") or "")[:240],
            "timings_ms": r.get("timings_ms", {}),
            "tokens": r.get("tokens", {}),
            "cost_usd": r.get("cost_usd", 0.0),
            "n_contexts": r.get("n_contexts", 0),
            "sources": r.get("sources", []),
        }
        for r in rows
    ]
    return {"count": len(queries), "queries": queries}


@app.get("/eval-results")
def eval_results(limit: int = 50) -> dict:
    """Evaluation reports written by the eval harness (from `eval_results_dir`),
    for the Evaluation dashboard. Returns eval runs (retrieval metrics, optional
    Ragas generation metrics on the OpenAI path, and per-question detail) and
    vector-vs-hybrid A/B comparisons, newest first.

    Read-only: reports are generated out-of-band by `python -m eval.run_eval`
    (see the Makefile), not triggered from here.
    """
    limit = max(1, min(limit, 200))
    results_dir = Path(get_settings().eval_results_dir)
    evals: list[dict] = []
    compares: list[dict] = []
    if results_dir.exists():
        # Filenames are timestamped (eval-YYYYMMDDT…), so a reverse name sort is
        # newest-first. Bad/partial files are skipped rather than failing the call.
        for path in sorted(results_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if not isinstance(data, dict):
                continue
            data["_name"] = path.name
            if path.name.startswith("compare-"):
                compares.append(data)
            elif path.name.startswith("eval-"):
                evals.append(data)
    return {
        "results_dir": str(results_dir),
        "eval_runs": evals[:limit],
        "compare_runs": compares[:limit],
    }
