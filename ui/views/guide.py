"""Guide page -- an in-app tutorial for using the app and reading every number.

Explains the workflow across the three working pages, then defines each
operational metric (per-answer + Analytics) and each evaluation metric so a
first-time user (or a portfolio reviewer) can interpret the dashboards without
outside context. Static content -- no API calls of its own beyond the shared
header's health check.
"""
from __future__ import annotations

import common
import streamlit as st
from common import CACHE_ICON, WARN_ICON

# Monochrome guide-card marks (16px, currentColor at --body) that echo the five
# nav Material Symbols, for the "The pages" cards.
_GC_ASK = ('<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
           'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" style="color:var(--body);flex:none">'
           '<path d="M14 9a3 3 0 0 1-3 3H7l-3 2V6a3 3 0 0 1 3-3h4a3 3 0 0 1 3 3Z"/>'
           '<path d="M17 8h1a3 3 0 0 1 3 3v7l-3-2h-4a3 3 0 0 1-3-3"/></svg>')
_GC_CORPUS = ('<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
              'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" style="color:var(--body);flex:none">'
              '<path d="M6 19a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h3l2 2h5a2 2 0 0 1 2 2v1"/>'
              '<path d="M3.5 10h16.4a1.5 1.5 0 0 1 1.44 1.93l-1.6 5.4A2 2 0 0 1 18 19Z"/></svg>')
_GC_ANALYTICS = ('<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                 'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" style="color:var(--body);flex:none">'
                 '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="m7 14 3-4 3 3 5-6"/></svg>')
_GC_EVAL = ('<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" style="color:var(--body);flex:none">'
            '<path d="M12 3 5 6v5c0 4.2 3 7.5 7 9 4-1.5 7-4.8 7-9V6Z"/><path d="m9 12 2 2 4-4"/></svg>')
_GC_GUIDE = ('<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
             'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" style="color:var(--body);flex:none">'
             '<path d="M12 6v14"/><path d="M3 5h6a3 3 0 0 1 3 3 3 3 0 0 1 3-3h6"/>'
             '<path d="M3 5v13h6a3 3 0 0 1 3 3 3 3 0 0 1 3-3h6V5"/></svg>')

common.render_header()

# Guide-specific styling (page-scoped; reuses the shared design tokens).
st.markdown(
    """
<style>
.gcard{ background:var(--surface); border:1px solid var(--border); border-radius:10px;
  padding:11px 14px; margin:8px 0; }
.gcard-h{ font-weight:700; color:var(--ink); font-size:14px; display:flex; align-items:center;
  gap:8px; flex-wrap:wrap; }
.gcard-b{ color:var(--body); font-size:13px; margin-top:4px; line-height:1.55; }
.gcard-b code{ font-family:var(--mono); font-size:12px; background:var(--surface-2);
  padding:1px 5px; border-radius:5px; }
.gtag{ font:600 10px var(--mono); text-transform:uppercase; letter-spacing:.03em;
  padding:2px 8px; border-radius:999px; white-space:nowrap; }
.gtag.up{ background:var(--success-bg); color:var(--success); }
.gtag.down{ background:var(--cite-bg); color:var(--primary); }
.gtag.info{ background:var(--surface-2); color:var(--body); border:1px solid var(--border); }
.step{ display:flex; gap:12px; margin:12px 0; align-items:flex-start; }
.step .n{ flex:none; width:26px; height:26px; border-radius:50%; background:var(--primary);
  color:#fff; font:700 12px var(--mono); display:flex; align-items:center; justify-content:center; }
.step .t{ color:var(--body); font-size:13.5px; line-height:1.5; padding-top:2px; }
.step .t b{ color:var(--ink); }
.ann{ color:var(--muted); font-size:12px; margin:2px 0 14px; padding-left:2px; }
.ann b{ color:var(--body); }
</style>
""",
    unsafe_allow_html=True)

st.markdown("### :material/menu_book: Guide")
st.caption("How the app works, and what every metric and evaluation number means. "
           "New here? Start with **Getting started**, then dip into the metric tabs "
           "whenever a number needs explaining.")


def card(title: str, body: str, tag: str | None = None, kind: str = "info") -> None:
    """Render one definition card (title + optional 'good direction' tag + body)."""
    tag_html = f'<span class="gtag {kind}">{tag}</span>' if tag else ""
    st.markdown(
        f'<div class="gcard"><div class="gcard-h">{title}{tag_html}</div>'
        f'<div class="gcard-b">{body}</div></div>',
        unsafe_allow_html=True)


tabs = st.tabs([":material/rocket_launch: Getting started",
                ":material/forum: Reading an answer",
                ":material/monitoring: Analytics metrics",
                ":material/verified: Evaluation metrics",
                ":material/book_2: Concepts"])

# ── Getting started ───────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("#### What this app does")
    st.markdown(
        "It answers questions about **your documents**. Instead of guessing, it "
        "**retrieves** the most relevant passages from an indexed corpus and asks a "
        "language model to answer using *only* those passages — with **citations**, and "
        "an honest **refusal** when the answer isn't in the docs. That's *Retrieval-"
        "Augmented Generation* (RAG).")

    st.markdown("#### Your first question in four steps")
    steps = [
        ("Check the corpus is indexed — the header shows <code>N chunks</code>. If it's "
         "<b>0</b>, open <b>Ask → Documents</b> and click <b>Re-index corpus</b> to load "
         "the bundled demo docs (or upload your own). Browse what's indexed anytime on the "
         "<b>Corpus</b> page."),
        ("Go to the <b>Ask</b> page and type a question, or click one of the example "
         "chips. Each answer appears in a chat you can keep scrolling."),
        ("Read the answer: the <b>[1] [2]</b> markers are citations, and the "
         "<b>Sources</b> expander shows the passages they came from."),
        ("Explore the numbers: <b>Analytics</b> shows how the system is <i>running</i> "
         "(cost, latency, usage); <b>Evaluation</b> shows how <i>good</i> the answers are "
         "(measured against a labelled test set)."),
    ]
    for i, s in enumerate(steps, 1):
        st.markdown(f'<div class="step"><div class="n">{i}</div><div class="t">{s}</div></div>',
                    unsafe_allow_html=True)

    st.markdown("#### The pages")
    card(f"{_GC_ASK}Ask", "The chat. Ask grounded questions, adjust how many passages are "
                          "retrieved (<code>k</code>), upload documents, and re-index the corpus.")
    card(f"{_GC_CORPUS}Corpus", "Browse exactly what the vector database indexed — every source "
                          "file with its type, size and chunk count, plus a viewer showing the "
                          "extracted text and the individual chunks stored for retrieval.")
    card(f"{_GC_ANALYTICS}Analytics", "Operational dashboard over past queries — cost, latency, "
                          "tokens, and which documents get retrieved most. <i>How is it running?</i>")
    card(f"{_GC_EVAL}Evaluation", "Quality dashboard — retrieval and answer-quality scores against "
                          "a fixed test set, plus a vector-vs-hybrid comparison. <i>How good is it?</i>")
    card(f"{_GC_GUIDE}Guide", "You are here.")

    st.markdown("#### Bringing your own documents")
    st.markdown(
        "On the **Ask** page, open **Documents → Upload & index** and drop in "
        "`.md`, `.txt`, `.html`, or `.pdf` files (10 MB each). They're saved server-side "
        "and indexed immediately — then just ask about them. **Re-index corpus** rebuilds "
        "from scratch.")

    st.markdown("#### Provider tiers (how answers are generated)")
    st.markdown(
        "The header shows the active `llm` and `backend`. There are three tiers, set "
        "server-side:")
    card("<code>fake</code>", "Offline, zero-cost stand-in used for demos and tests. "
         "Deterministic but not truly \"smart\" — great for clicking around at $0.", "$0 · offline", "info")
    card("<code>hf</code>", "Real open-source models running locally (Hugging Face). Free, "
         "no API key, genuinely semantic.", "$0 · local", "info")
    card("<code>openai</code>", "The paid OpenAI API — best quality, real cost per query "
         "(shown on every answer).", "paid", "info")

# ── Reading an answer ─────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("#### Anatomy of an answer")
    st.markdown("Every answer on the **Ask** page has the same parts. Here's a sample, annotated:")

    st.markdown(
        'To rotate a key, create a new one, deploy it, then delete the old key '
        '<span class="cite">[1]</span>. Deletion is immediate <span class="cite">[2]</span>.',
        unsafe_allow_html=True)
    st.markdown('<div class="ann"><b>Citations</b> — the <b>[n]</b> markers point to the '
                'numbered sources the model used. Open the <b>Sources</b> expander to read '
                'the exact passage behind each one and verify the claim.</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="badges"><span class="badge mode">mode: hybrid</span>'
        f'<span class="badge cache">{CACHE_ICON} Cache hit</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="ann"><b>Badges</b> — <b>mode</b> is the retrieval strategy '
                '(<code>vector</code> or <code>hybrid</code>). <b>Cache hit</b> means this '
                'exact question was asked before and served instantly from memory at $0.</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="telemetry">'
        '<div class="tile"><div class="lab">Latency</div><div class="val">1180 ms</div>'
        '<div class="latbar"><div class="r" style="width:18%"></div>'
        '<div class="g" style="width:82%"></div></div>'
        '<div class="sub">210 retr / 970 gen</div></div>'
        '<div class="tile"><div class="lab">Cost</div><div class="val">$0.000270</div></div>'
        '<div class="tile"><div class="lab">Contexts</div><div class="val">4</div></div>'
        '<div class="tile"><div class="lab">Tokens</div><div class="val">1150</div>'
        '<div class="sub">prompt+completion</div></div>'
        '</div>', unsafe_allow_html=True)
    st.markdown('<div class="ann"><b>Telemetry strip</b> — the four tiles are explained on '
                'the <b>Analytics metrics</b> tab. The little bar under <b>Latency</b> splits '
                'the time into <span style="color:var(--seg-retr)">retrieval</span> (finding '
                'passages) vs <span style="color:var(--seg-gen)">generation</span> (the model '
                'writing).</div>', unsafe_allow_html=True)

    st.markdown("#### When the docs don't have the answer")
    st.markdown(
        f'<div class="badges"><span class="badge refuse">{WARN_ICON} Not grounded in the corpus</span></div>',
        unsafe_allow_html=True)
    st.markdown(
        "Ask something outside the documents and a well-configured system **refuses** — "
        "*\"I don't have enough information in the documentation to answer that.\"* — instead "
        "of inventing an answer. This is the core anti-hallucination behaviour. (On the "
        "`fake` tier it can't always tell; real embeddings make refusal reliable.)")

# ── Analytics metrics ─────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("#### The per-query numbers")
    st.caption("Shown on each answer and aggregated on the Analytics page.")
    card("Latency", "Total time to answer, in milliseconds — split into <b>retrieval</b> "
         "(finding passages) and <b>generation</b> (the model writing). "
         "<b>p95 latency</b> is the value 95% of queries come in under (a worst-case feel).",
         "lower is better", "down")
    card("Cost (USD)", "Dollar cost of the query — embedding + prompt + completion tokens "
         "priced per model. <b>$0</b> on the <code>fake</code>/<code>hf</code> tiers; real "
         "cents on <code>openai</code>. A cache hit is always $0.", "lower is better", "down")
    card("Tokens", "Units of text the model processed: <b>prompt</b> (the context + your "
         "question sent in) plus <b>completion</b> (the answer). Tokens drive cost.", "info")
    card("Contexts", "How many passages were retrieved and fed to the model — the "
         "<code>k</code> value. More context can help, but adds tokens and noise.", "info")
    card("Cache hit", "The question matched a previous one exactly, so the stored answer "
         "was returned instantly at $0. Cache hits are <i>not</i> traced, so Analytics "
         "reflects real (uncached) work only.", "info")

    st.markdown("#### The charts")
    card("Cost / latency / tokens over time", "Trends across queries — watch cost accumulate "
         "and latency move as you change settings or providers.", "info")
    card("Latency distribution", "A histogram: how many queries fall into each latency band. "
         "A long right tail means occasional slow queries.", "info")
    card("Most retrieved documents", "Which source files get pulled into answers most often — "
         "shows what your corpus is actually being used for.", "info")
    card("Tokens vs latency", "Each dot is a query; look for whether bigger prompts (more "
         "tokens) track with slower answers.", "info")

# ── Evaluation metrics ────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("#### Why evaluation is separate from analytics")
    st.markdown(
        "Analytics tells you how the system *runs*; **evaluation** tells you how *good the "
        "answers are*. It scores the system against a fixed **golden set** — labelled "
        "questions with known correct sources, including out-of-scope questions that "
        "*should* be refused.")

    st.markdown("#### Retrieval quality (did it find the right passages?)")
    card("Context recall@k", "Of the answerable questions, how often did the top-<code>k</code> "
         "retrieved passages include a correct source? If retrieval misses, no prompting can "
         "fix the answer — so this is measured first. <b>1.0</b> = always found.",
         "higher is better", "up")
    card("Recall@1", "How often the <i>very top</i> result was correct — a stricter test of "
         "ranking, not just presence.", "higher is better", "up")
    card("MRR (Mean Reciprocal Rank)", "Averages <code>1 / rank</code> of the first correct "
         "source. Rewards ranking the right passage <i>higher</i> (rank 1 → 1.0, rank 2 → 0.5). "
         "Sensitive to ranking improvements like hybrid retrieval.", "higher is better", "up")
    card("Refusal accuracy", "Of the out-of-scope questions, how often the system correctly "
         "<b>refused</b> instead of inventing an answer. Only meaningful with real semantic "
         "embeddings — shown as <code>n/a</code> on the <code>fake</code> tier.",
         "higher is better", "up")

    st.markdown("#### Generation quality — Ragas (needs the OpenAI path)")
    st.caption("These use an LLM judge, so they appear only for runs made with LLM_PROVIDER=openai.")
    card("Faithfulness", "Is every claim in the answer supported by the retrieved context? "
         "This is the direct <b>anti-hallucination</b> measure.", "higher is better", "up")
    card("Answer relevancy", "Does the answer actually address the question that was asked?",
         "higher is better", "up")
    card("Context precision", "Were the retrieved passages relevant (signal), or padded with "
         "noise?", "higher is better", "up")
    card("Context recall", "Did retrieval bring back <i>all</i> the information the answer "
         "needed?", "higher is better", "up")

    st.markdown("#### Comparisons & detail")
    card("A/B — vector vs hybrid", "Runs the same golden set under both retrieval strategies "
         "and shows the per-metric delta. This is how you justify a retrieval change with "
         "data instead of a hunch.", "info")
    card("Per-question table", "The pass/fail breakdown for every golden question — which "
         "ones retrieval got right, at what rank, and which were refused. Filter to "
         "<b>Failed</b> to see exactly where the system struggles.", "info")

    st.info("**Generate reports** from a terminal, then open the Evaluation page:\n\n"
            "```\nmake eval NO_RAGAS=1   # retrieval metrics, no API key\n"
            "make eval             # + Ragas generation metrics (needs OPENAI_API_KEY)\n"
            "make eval-compare     # vector vs hybrid A/B\n```")

# ── Concepts ──────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("#### Core ideas, in plain English")
    left, right = st.columns(2)
    with left:
        card("RAG", "Retrieval-Augmented Generation: search your docs first, then let the "
             "model answer using what search found — open-book, not from memory.")
        card("Grounding", "Answering <i>only</i> from the retrieved passages, never from the "
             "model's training memory. Enforced by the system prompt.")
        card("Citation", "The <b>[n]</b> reference tying a claim to the exact source passage, "
             "so a human can verify it.")
        card("Refusal", "Declining with a fixed sentence when the passages don't contain the "
             "answer — the alternative to hallucinating.")
        card("Embedding", "A text turned into a vector of numbers so that similar meanings "
             "land near each other — how semantic search works.")
        card("Chunk", "A document is split into passages (~800 tokens) so retrieval returns "
             "precise pieces, not whole files.")
    with right:
        card("Hybrid retrieval", "Running <b>semantic</b> (vector) search and <b>keyword</b> "
             "(BM25) search together and fusing the results — catches both meaning and exact "
             "tokens like error codes.")
        card("BM25", "The classic keyword ranking algorithm; strong exactly where embeddings "
             "blur — rare identifiers, codes, API names.")
        card("RRF", "Reciprocal Rank Fusion: merges two ranked lists by rank position, so a "
             "passage both retrievers like rises to the top.")
        card("Top-k", "How many passages retrieval returns (the <code>k</code> slider on the "
             "Ask page). Default is 4.")
        card("Vector store", "Where the embeddings live and get searched — a NumPy matrix by "
             "default, or Postgres + pgvector for scale.")
        card("Cache", "An in-memory store of recent answers; repeat questions return instantly "
             "at $0 (flagged as a cache hit).")

st.divider()
st.caption("Want the deep-dive? The repo ships a full written study guide at "
           "`docs/RAG_Study_Guide.pdf`.")
