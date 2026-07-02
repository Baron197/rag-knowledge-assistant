# Lean image for the FastAPI service (UI and eval deps intentionally excluded).
FROM python:3.11-slim

WORKDIR /app

# Runtime dependencies (incl. python-multipart for /upload and the pgvector
# driver so VECTOR_BACKEND=pgvector works in the container). Eval-only
# libraries (ragas, datasets) and the UI are excluded -- see requirements-api.txt.
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Note: the free local Hugging Face providers (LLM_PROVIDER=hf) need the
# heavy deps in requirements-hf.txt; install them in a derived image if used.
COPY src/ ./src/
COPY data/ ./data/

ENV LLM_PROVIDER=fake \
    EMBEDDING_PROVIDER=fake \
    VECTOR_BACKEND=numpy

EXPOSE 8000

# Build the index at start, then serve. (For real deployments, ingest as a
# separate job and bake/mount the index instead.)
CMD ["sh", "-c", "python -m src.rag.ingest --reset && uvicorn src.rag.api:app --host 0.0.0.0 --port 8000"]
