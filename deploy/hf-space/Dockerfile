# ─────────────────────────────────────────────────────────────────────────────
# Hugging Face Space (Docker SDK) — RAG Knowledge Assistant, keyless live demo.
#
# One container runs BOTH the FastAPI service (internal, :8000) and the Streamlit
# UI (public, :7860). Defaults to the zero-cost `fake` providers, so it needs no
# API key and costs nothing. To run real local models instead, see the
# "hf mode" block below.
#
# This file is the Space's build definition — copy it to the ROOT of your Space
# repo as `Dockerfile`. It clones the application from GitHub at build time, so
# the Space itself only needs this Dockerfile + a README.md (see DEPLOY_GUIDE.md).
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# git clones the app at build time.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run the container as uid 1000 — create that user and work in its home
# so the index, traces and model cache are all writable at runtime.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# App code — clone the public repo (tracks main). Pin a tag/SHA for a frozen demo.
RUN git clone --depth 1 https://github.com/Baron197/rag-knowledge-assistant.git .

# Python dependencies (the keyless `fake` path needs only these).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── hf mode (OPTIONAL): real, local, open-source models ──────────────────────
# Uncomment to bundle PyTorch (CPU-only wheel) + transformers, then set
# LLM_PROVIDER=hf and EMBEDDING_PROVIDER=hf in the Space's Settings → Variables.
# HF_LLM_MODEL below is the small 0.5B model so CPU answers stay reasonably quick.
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
#  && pip install --no-cache-dir "sentence-transformers>=3.0" "transformers>=4.44" "accelerate>=0.33"

# Runtime configuration — override any of these in the Space's Variables.
ENV LLM_PROVIDER=fake \
    EMBEDDING_PROVIDER=fake \
    VECTOR_BACKEND=numpy \
    RETRIEVAL_MODE=hybrid \
    RAG_API_URL=http://127.0.0.1:8000 \
    HF_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
    HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    HF_DEVICE=cpu \
    HF_HOME=/home/user/.cache/huggingface

EXPOSE 7860
# start.sh comes from the cloned repo (deploy/hf-space/start.sh).
CMD ["sh", "deploy/hf-space/start.sh"]
