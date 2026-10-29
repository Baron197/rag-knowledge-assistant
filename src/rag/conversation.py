"""Conversational follow-ups: turn a context-dependent question into a standalone one.

Retrieval and generation see ONE question. To support follow-ups such as "does
deleting it take effect immediately?", the client sends the recent chat history
and the pipeline *condenses* history + follow-up into a standalone question
before retrieval ("Does deleting an API key take effect immediately?").

History is used only to resolve references -- never as evidence. The answer is
still grounded solely in the retrieved passages, so a wrong earlier answer can't
be "cited" by a later one, and forged history can at most change what gets
searched (the same power the question field already gives a caller).

A cheap deterministic gate (`needs_rewrite`) decides whether a question leans on
earlier turns at all. Only those are rewritten; a standalone question ignores the
history, so it costs no rewrite call and still hits the single-turn answer cache.

Two rewrite strategies:
  * a real LLM (openai / hf) rewrites with `prompts.CONDENSE_SYSTEM_PROMPT`;
  * the keyless `fake` path can't rewrite, so a deterministic heuristic prepends
    the earlier user question(s) the follow-up depends on.
"""
from __future__ import annotations

import logging
from typing import Any

from .lexical import tokenize
from .llm import LLM, LLMResult
from .prompts import CONDENSE_SYSTEM_PROMPT, build_condense_prompt

logger = logging.getLogger("rag.conversation")

MAX_TURN_CHARS = 1000      # per-message cap: long answers would bloat the rewrite prompt
MAX_REWRITE_CHARS = 1000   # a "standalone question" longer than this is a bad rewrite
_MAX_CHAIN = 3             # heuristic: how many earlier user turns a follow-up may pull in

# Words that usually point back at something said earlier ("is IT free?").
_REFERENCES = {
    "it", "its", "itself", "this", "that", "these", "those", "they", "them",
    "their", "theirs", "one", "ones", "same", "else", "former", "latter",
}
_LEADING_CONNECTIVES = {"and", "also", "but", "so", "then"}


def normalize_history(history: list[Any] | None, window: int) -> list[dict[str, str]]:
    """Keep the last `window` well-formed messages as `{"role", "content"}` dicts.

    Drops anything that isn't a user/assistant message with non-blank text and
    trims each message to MAX_TURN_CHARS. `window <= 0` disables history.
    """
    if not history or window <= 0:
        return []
    turns: list[dict[str, str]] = []
    for t in history:
        if not isinstance(t, dict):
            continue
        role, content = t.get("role"), t.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            turns.append({"role": role, "content": content[:MAX_TURN_CHARS]})
    return turns[-window:]


def history_key(turns: list[dict[str, str]]) -> tuple:
    """A hashable fingerprint of the (normalized) history, for the answer cache."""
    return tuple((t["role"], t["content"]) for t in turns)


def looks_like_follow_up(question: str) -> bool:
    """Heuristic: does the question lean on earlier context?

    True when it contains a referring word (it, they, that, ...), opens with a
    connective ("and ...", "what about ..."), or is too short to stand alone ("why?").
    """
    tokens = tokenize(question)
    if len(tokens) <= 2:
        return True
    if tokens[0] in _LEADING_CONNECTIVES or tokens[:2] in (["what", "about"], ["how", "about"]):
        return True
    return any(t in _REFERENCES for t in tokens)


def needs_rewrite(question: str, history: list[dict[str, str]]) -> bool:
    """True when there is history to resolve against AND the question needs it.

    The gate is deliberately cheap and deterministic: it runs before the cache
    lookup, so a standalone question asked mid-conversation behaves exactly like a
    single-turn one. Tradeoff: an implicit follow-up with no referring word
    ("How long does approval take?") is not rewritten.
    """
    return bool(history) and looks_like_follow_up(question)


def heuristic_rewrite(question: str, history: list[dict[str, str]]) -> str:
    """Keyless rewrite: prepend the earlier user question(s) a follow-up depends on.

    Walks back over user turns until one that stands on its own (at most
    _MAX_CHAIN), so a chain like "webhook events?" -> "how do I verify them?" ->
    "and if that fails?" still carries the original topic. Standalone questions
    are returned unchanged, so a change of topic isn't polluted by old turns.
    """
    if not looks_like_follow_up(question):
        return question
    context: list[str] = []
    for turn in reversed(history):
        if turn["role"] != "user":
            continue
        if turn["content"].strip().lower() == question.strip().lower():
            continue  # a retried question adds nothing new
        context.append(turn["content"])
        if not looks_like_follow_up(turn["content"]) or len(context) == _MAX_CHAIN:
            break
    if not context:
        return question
    return " ".join([*reversed(context), question])


def _clean_rewrite(text: str) -> str:
    """First non-empty line of the model's reply, minus quotes and an echoed label."""
    for line in text.splitlines():
        line = line.strip().strip("\"'`").strip()
        if line.lower().startswith("standalone question:"):
            line = line[len("standalone question:"):].strip().strip("\"'`").strip()
        if line:
            return line
    return ""


def rewrite_question(
    question: str,
    history: list[dict[str, str]],
    llm: LLM,
    use_llm: bool,
) -> tuple[str, LLMResult | None]:
    """Return (standalone_question, llm_result_or_None) for a possibly-follow-up question.

    No history -> the question unchanged and no LLM call. `use_llm=False` (the
    keyless path) -> the heuristic. Otherwise one LLM call; an empty/overlong
    reply or an API error falls back to the heuristic, so a bad rewrite never
    fails the query. The LLMResult is returned even on fallback, because those
    tokens were still paid for.
    """
    if not history:
        return question, None
    if not use_llm:
        return heuristic_rewrite(question, history), None
    try:
        result = llm.complete(CONDENSE_SYSTEM_PROMPT, build_condense_prompt(question, history))
    except Exception:  # noqa: BLE001 -- a failed rewrite must not fail the whole query
        logger.warning("follow-up rewrite failed; using the heuristic", exc_info=True)
        return heuristic_rewrite(question, history), None
    standalone = _clean_rewrite(result.text)
    if not standalone or len(standalone) > MAX_REWRITE_CHARS:
        return heuristic_rewrite(question, history), result
    return standalone, result
