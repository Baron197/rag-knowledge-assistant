"""Prompt construction for grounded, citeable answers.

The text here is where most of the system's *safety* behaviour lives. The system
prompt enforces the two rules that separate a real RAG system from a chatbot:
(1) answers must be grounded only in the retrieved passages, and (2) the model
must refuse instead of hallucinating when those passages don't contain the
answer. Each passage is numbered so the model can cite it as [1], [2], ...
"""
from __future__ import annotations

from .vectorstore import Hit

# The "rulebook" sent as the system message on every query. The exact refusal
# sentence is matched by the evaluation harness, so keep it stable.
SYSTEM_PROMPT = (
    "You are a precise documentation assistant. Answer the user's question using "
    "ONLY the numbered context passages provided. Rules:\n"
    "1. Ground every claim in the context. Cite the passages you used inline, e.g. [1], [2].\n"
    "2. If the context does not contain the answer, reply exactly: "
    "\"I don't have enough information in the documentation to answer that.\" "
    "Do not use outside knowledge.\n"
    "3. Be concise and specific. Prefer steps and concrete details over fluff."
)


def build_context_block(hits: list[Hit]) -> str:
    """Format retrieved passages into a numbered, citeable context block.

    Each hit becomes ``[i] (source: <file>)\\n<text>`` so the model can both read
    the passage and reference it by its number when citing.
    """
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"[{i}] (source: {h.doc.source})\n{h.doc.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    """Assemble the user message: the numbered context plus the question.

    If no passages were retrieved we insert an explicit "(no relevant context
    found)" marker, which is the model's cue to return the refusal sentence.
    """
    context = build_context_block(hits) if hits else "(no relevant context found)"
    return f"Context passages:\n\n{context}\n\nQuestion: {question}\n\nAnswer (with citations):"
