# RAG Knowledge Assistant

> A production-style **Retrieval-Augmented Generation** service that answers questions over a document corpus with **grounded, cited answers**, **hybrid retrieval**, a **built-in evaluation pipeline**, and **cost / latency observability**.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI-412991?logo=openai&logoColor=white)
![Vector DB](https://img.shields.io/badge/Vector%20DB-pgvector%20%2F%20NumPy-4169E1?logo=postgresql&logoColor=white)
![Retrieval](https://img.shields.io/badge/Retrieval-Hybrid%20(BM25%2BVector)-8A2BE2)
![Evaluation](https://img.shields.io/badge/Eval-Ragas%20%2B%20CI%20gate-orange)
![Lint](https://img.shields.io/badge/Lint-ruff-000000)
![License](https://img.shields.io/badge/License-MIT-green)

<!-- After pushing to GitHub, enable the live CI badge (replace OWNER/REPO):
![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)
-->

This is not a notebook demo. It is the part of RAG that is actually hard in
production: knowing whether the answers are correct, what they cost, and being
able to change the system without breaking it. Those concerns — grounding,
hybrid retrieval, evaluation, observability, and clean swappable components — are
first-class here.

> **Try it in 2 minutes with no API key.** A keyless `fake` provider mode runs
> the entire app, tests, and CI offline at zero cost. Flip two environment
> variables to switch to real OpenAI models.

> **Two implementations, by design.** This repo builds the RAG service **from
> scratch** -- hand-written retrieval, BM25, Reciprocal Rank Fusion and vector
> store, no framework. A companion repo, **`rag-langchain`**, builds the *same*
> service idiomatically with **LangChain (LCEL)**. Doing it both ways shows
> first-principles understanding **and** fluency with the industry-standard
> framework.

---

## What this project demonstrates

For reviewers, here is the applied-AI engineering signal at a glance:

- **End-to-end RAG** — ingestion (load → chunk → embed → store) and query
  (retrieve → ground → generate → cite) built from first principles.
- **Hybrid retrieval** — semantic (vector) **+** keyword (BM25) search combined
  with Reciprocal Rank Fusion; a hand-written BM25 (no black box).
- **Anti-hallucination by design** — answers are grounded, cite their sources,
  and **refuse** when the context is insufficient.
- **Evaluation, with numbers** — a golden Q/A set scored for recall@k, recall@1
  and MRR, plus an **A/B harness** comparing retrieval strategies; Ragas
  faithfulness/relevancy on the real path.
- **Eval-as-CI-gate** — a pull request that regresses retrieval quality **fails
  the build** before it can merge.
- **Observability & cost control** — per-query latency, token usage and USD cost
  traces; a `/metrics` rollup; and an **LRU answer cache** that serves repeats
  for free.
- **Clean architecture** — embedding provider, LLM provider, retrieval mode and
  vector backend each sit behind a small interface and are swapped via config.
- **Three provider tiers** — keyless `fake` (offline, $0, for tests/CI), `hf`
  (real open-source models running locally, free, **no API key**), and `openai`
  (paid API). Swap with one env var.
- **Runs anywhere** — zero-config NumPy store for instant local runs; Postgres +
  `pgvector` for production; a **Dockerfile** to containerise the API.
- **Engineered like a product** — typed config, ruff linting, and a
  fast deterministic test suite, all enforced in CI.

> Demo corpus: docs for *Nimbus*, a fictional messaging-API platform
> (`data/docs/`). Point it at any folder of `.md` / `.txt` / `.html` / `.pdf` and
> it works unchanged.

## Example interaction

*Illustrative answer on the OpenAI path:*

```
Q: How do I rotate a leaked API key?

A: Create a new key, deploy it, then delete the old key from
   Settings → API Keys. Deletion takes effect immediately, so any request
   using the old key afterwards returns 401. Rotation always produces a new
   key value — there is no reset that preserves the old string. [1][2]

Sources:
   [1] 02-authentication.md
   [2] 06-troubleshooting.md

mode: hybrid · ~1.2 s · ~$0.0004 · 4 context passages
```

Ask something outside the docs and it refuses instead of inventing an answer:

```
Q: What is the CEO's phone number?
A: I don't have enough information in the documentation to answer that.
```

<!-- Tip: add a screenshot or a 60–90s Loom of the Streamlit UI here. -->

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        D[Docs: md/txt/html/pdf] --> C[Chunk] --> E1[Embed] --> V[(Vector store)]
    end
    subgraph Query
        Q[Question] --> E2[Embed] --> R[Retrieve: vector + BM25 -> RRF]
        V --> R
        R --> P[Grounded prompt + citations] --> L[LLM] --> A[Answer + sources]
    end
    A --> CACHE[(LRU answer cache)]
    A --> T[(Traces: latency / tokens / cost)]
    UI[Streamlit UI] --> API[FastAPI] --> Query
    EVAL[Eval harness + Ragas] --> API
```

**Request lifecycle:** `UI → FastAPI → pipeline (retrieve → prompt → generate) →
trace`. The UI and the evaluation harness both go through the same API, so there
is a single source of truth. Each query is timed and costed; repeated questions
are served from the cache.

## Design decisions & trade-offs

The interesting engineering is in the choices, not the line count:

| Decision | Why | Trade-off |
|---|---|---|
| Hybrid retrieval (vector + BM25, RRF) | Keyword search catches exact tokens (error codes, API names) that embeddings blur; fusion beats either alone | Two retrievers to run; BM25 index built from the corpus |
| Hand-written NumPy vector store as default | Zero setup; transparent cosine search; shows what a vector DB does | Not for large corpora → `pgvector` for production |
| Provider abstractions + keyless `fake` mode | App, tests and CI run with no key and no cost | Fake embeddings are keyword-based, not semantic → quality metrics need a real model |
| Evaluation wired into CI as a gate | Quality can't silently regress between changes | Requires maintaining a golden set |
| LRU answer cache | Repeated questions cost nothing and return instantly | In-memory (per-process); a shared cache (Redis) would be the next step |
| FastAPI service with a thin Streamlit UI | Clean separation; the API is the single source of truth | Two processes to run locally |
| Local JSONL tracer (Langfuse-shaped) | Dependency-free observability out of the box | Swap to Langfuse / OpenTelemetry at scale |
| `temperature=0` for generation | Reproducible answers, stable evaluation | Less varied phrasing |

## Results

**Latest keyless run** (`fake` providers — validates retrieval, cost & latency
plumbing end-to-end):

| Metric | Value |
|---|---|
| Context recall@k (answerable) | **1.0** |
| Recall@1 | 0.78 |
| MRR | 0.87 |
| Avg cost / query | $0.00 |
| Tests | 21 / 21 passing |

**Retrieval A/B — `make eval-compare`** (vector vs hybrid, keyless run):

| Metric | vector | hybrid |
|---|---|---|
| Context recall@k | 1.0 | 1.0 |
| Recall@1 | 0.78 | 0.78 |
| MRR | 0.873 | 0.880 |

> On this small demo corpus with the keyword-hashing `fake` embedder, the two
> strategies are close by construction. With real **semantic** embeddings,
> hybrid's advantage on exact-token queries is larger — run `make eval-compare`
> with OpenAI configured to measure it on your data. Refusal accuracy and the
> Ragas generation metrics also require the real path; results land in
> `eval/results/`.

## Quickstart — no API key, ~2 minutes

```bash
pip install -r requirements.txt
cp .env.example .env            # defaults to the keyless 'fake' providers

python -m src.rag.ingest --reset   # build the index
make api                        # http://localhost:8000  (interactive docs at /docs)
make ui                         # http://localhost:8501  (second terminal)
make test                       # run the test suite
make lint                       # ruff
make eval-compare               # vector vs hybrid A/B
```

The `fake` providers return deterministic, grounded-looking output so you can
click through the whole app — citations, latency, (zero) cost — entirely offline.

Add your own documents two ways: drop files into `data/docs/` and re-ingest, or
upload them straight from the Streamlit sidebar (**Add documents → Upload &
ingest**), which saves them server-side and rebuilds the index via `POST /upload`
(accepts `.md`, `.txt`, `.html`, `.pdf`).

## Run free & locally with Hugging Face (no API key)

Prefer real open-source models but **no API key and no cost**? Use the `hf`
provider tier, which runs models on your own machine:

```bash
make install-hf                 # one-time: installs transformers + torch (heavy)
# in .env
LLM_PROVIDER=hf
EMBEDDING_PROVIDER=hf
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2   # 384-dim, fast on CPU
HF_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct                    # Apache-2.0, ungated
HF_DEVICE=cpu                                              # or a GPU index like 0
```

Then `python -m src.rag.ingest --reset` and run as usual. Models download once
and then run offline; cost is always `$0`. The defaults are small, ungated
models (no login/token needed) that run on CPU — pick a bigger `HF_LLM_MODEL`
if you have a GPU. `make eval` then gives real **refusal / retrieval-quality**
numbers for free (Ragas faithfulness still needs an OpenAI judge).

## Run with real OpenAI models

```bash
# .env
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
RETRIEVAL_MODE=hybrid           # vector | hybrid
MIN_RELEVANCE_SCORE=0.25        # optional hallucination guard (vector mode)
```

Then `python -m src.rag.ingest --reset` and run as above. `make eval` now
produces real faithfulness / refusal numbers.

## Run in Docker

```bash
docker compose up --build api   # API at http://localhost:8000 (keyless by default)
# or just the production database:
make db-up                      # Postgres + pgvector; set VECTOR_BACKEND=pgvector
```

The application code is unchanged across backends — `numpy` and `pgvector`
implement the same `VectorStore` interface (Chroma / Qdrant would slot in
identically).

## Evaluation

```bash
make eval                       # retrieval metrics (+ Ragas if OpenAI is configured)
make eval NO_RAGAS=1            # retrieval metrics only (no key)
make eval-compare               # A/B: vector vs hybrid retrieval
python -m eval.run_eval --min-recall 0.8   # the CI regression gate
```

The golden set (`eval/golden_set.jsonl`, 28 Q/A pairs including out-of-scope
questions that *should* be refused) is scored and saved to `eval/results/`.

## How to review this repo

If you have five minutes and want the signal quickly, read these in order:

1. `src/rag/pipeline.py` — orchestration; the whole query lifecycle + cache.
2. `src/rag/retriever.py` + `src/rag/lexical.py` — hybrid retrieval and BM25/RRF.
3. `src/rag/vectorstore.py` — the two backends behind one interface.
4. `eval/run_eval.py` — how quality is measured, A/B-compared, and gated.
5. `src/rag/observability.py` — how latency, tokens and cost are tracked.

## Project structure

```
src/rag/
  config.py         typed settings (pydantic-settings)
  embeddings.py     OpenAI + Hugging Face (local) + keyless fake embedder
  chunking.py       token-aware chunking with overlap
  vectorstore.py    numpy (default) + pgvector backends
  lexical.py        BM25 keyword search + Reciprocal Rank Fusion
  retriever.py      vector or hybrid retrieval (+ relevance gate)
  prompts.py        grounded, citeable system/user prompts
  llm.py            OpenAI + Hugging Face (local) + keyless fake LLM
  observability.py  per-query traces, token cost, latency aggregation
  pipeline.py       retrieve -> prompt -> generate -> trace (+ LRU cache)
  ingest.py         load -> chunk -> embed -> store (CLI)
  api.py            FastAPI: /health /ingest /upload /query /metrics /analytics /eval-results
ui/
  streamlit_app.py  multipage router (thin client over the API)
  common.py         shared config, styling and API helpers
  views/chat.py     grounded chat with citations + telemetry
  views/analytics.py filterable charts over the query traces
  views/evaluation.py read-only dashboard of the eval reports (retrieval + Ragas + A/B)
  views/guide.py      in-app tutorial: how to use the app + what every metric means
eval/               golden set + A/B + Ragas harness + results
tests/              end-to-end tests on the keyless path
Dockerfile          containerised API
.github/workflows/  CI: lint + tests + retrieval regression gate
```

## Testing & CI

- **Tests** (`pytest`, 21) run end-to-end on the keyless path — fast,
  deterministic, no network or API key.
- **Lint** (`ruff`) enforces style and import hygiene.
- **CI** (GitHub Actions) lints, runs the tests, then runs the evaluation as a
  **regression gate** (`--min-recall 0.8`): a change that breaks retrieval
  quality fails the build.

## Tech stack

Python · FastAPI · Streamlit · OpenAI **or** local Hugging Face models (transformers + sentence-transformers) · pgvector / NumPy ·
BM25 · Ragas · pydantic-settings · Docker · GitHub Actions · ruff · pytest.

## Roadmap

- Cross-encoder re-ranking on top of hybrid candidates.
- API hardening for multi-user deployments: authentication and per-client rate
  limiting (uploads are already size- and type-restricted).
- Multi-provider routing (GPT vs Claude vs open-weight) with a cost / quality
  comparison dashboard.
- Langfuse / OpenTelemetry tracing in place of the local JSONL tracer.
- Shared cache (Redis) and multi-tenant isolation with per-tenant cost tracking.

## About

**Baron Purwa Hartono** — AI / Applied AI Engineer (RAG, agentic systems,
production LLM applications).

- LinkedIn: https://www.linkedin.com/in/baronpurwahartono/
- Email: baronhartono@gmail.com

## License

MIT — see [`LICENSE`](LICENSE).
