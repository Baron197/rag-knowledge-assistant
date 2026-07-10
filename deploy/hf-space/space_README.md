---
title: RAG Knowledge Assistant
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Grounded, cited RAG over docs — keyless demo (FastAPI + Streamlit)
---

# RAG Knowledge Assistant — live demo

A keyless (`fake`-provider) demo of a **from-scratch** Retrieval-Augmented
Generation service: grounded answers with inline citations, hybrid (vector + BM25)
retrieval, and per-query cost / latency / token telemetry — served by a FastAPI
backend behind a multipage Streamlit UI.

On this keyless path the **retrieval, citations and telemetry are real**; the
answer *text* is a deterministic stand-in (the fake model echoes the top passage).
Real generated answers and the Ragas evaluation numbers are on the OpenAI path —
see the full write-up and reproducible eval reports in the source repo.

**Source & docs:** https://github.com/Baron197/rag-knowledge-assistant

---

*Want real generated answers here?* Uncomment the `hf` block in the `Dockerfile`,
set `LLM_PROVIDER=hf` and `EMBEDDING_PROVIDER=hf` in **Settings → Variables**, and
restart. It runs a small local open-source model (Qwen2.5-0.5B) — real answers, no
API key, but slower on the free CPU tier.
