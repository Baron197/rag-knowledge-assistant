#!/usr/bin/env sh
# Boot both processes in one container for a Hugging Face Space:
#   1) build the index from the bundled corpus
#   2) run the FastAPI service on 127.0.0.1:8000 (internal only)
#   3) run the Streamlit UI on 0.0.0.0:7860 (the Space's public port)
# Runs from the app root (WORKDIR), so the paths below are repo-relative.
set -e

echo ">> Ingesting corpus (embeddings=${EMBEDDING_PROVIDER:-fake}, backend=${VECTOR_BACKEND:-numpy})..."
python -m src.rag.ingest --reset

echo ">> Starting API on 127.0.0.1:8000..."
uvicorn src.rag.api:app --host 127.0.0.1 --port 8000 &

echo ">> Waiting for the API to become healthy..."
i=0
while [ "$i" -lt 60 ]; do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" >/dev/null 2>&1; then
    echo ">> API is up."
    break
  fi
  i=$((i + 1))
  sleep 1
done

echo ">> Starting Streamlit UI on 0.0.0.0:7860..."
exec streamlit run ui/streamlit_app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
