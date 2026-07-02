"""Token-aware text chunking with overlap.

Chunking is one of the highest-leverage RAG decisions: chunks that are too big
make retrieval imprecise, while chunks that are too small lose surrounding
context. The strategy here: split on paragraph/sentence boundaries first, then
pack those segments into ~`chunk_size`-token windows with a small `overlap` so a
fact sitting on a boundary still lands whole in at least one chunk. A single
segment that is itself larger than `chunk_size` (e.g. one very long unbroken
line) is hard-split so no chunk can exceed the budget and blow the embedding
model's input limit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Token counting: use the real tokenizer when available, else a char-based
# approximation so the module still works fully offline.
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def _ntokens(text: str) -> int:
        """Exact token count via tiktoken."""
        return len(_ENC.encode(text))
except Exception:  # pragma: no cover - tiktoken optional fallback
    def _ntokens(text: str) -> int:
        """Approximate token count (~4 characters per token)."""
        return max(1, len(text) // 4)


# Split points: a blank line (paragraph break) or whitespace after . ! ? (sentence end).
_SPLIT_RE = re.compile(r"(\n\s*\n|(?<=[.!?])\s+)")


@dataclass
class Chunk:
    """A single passage produced by chunking: its text and its ordinal index."""

    text: str
    index: int


def _hard_split(segment: str, max_tokens: int) -> list[str]:
    """Split an over-long segment into word-bounded pieces under `max_tokens`.

    Used for pathological inputs (a long line with no sentence/paragraph breaks)
    so a single segment can never produce an oversized chunk.
    """
    if _ntokens(segment) <= max_tokens:
        return [segment]
    words = segment.split()
    pieces: list[str] = []
    cur: list[str] = []
    for w in words:
        cur.append(w)
        if _ntokens(" ".join(cur)) >= max_tokens:
            pieces.append(" ".join(cur))
            cur = []
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _segments(text: str, max_tokens: int) -> list[str]:
    """Split text into natural segments, hard-splitting any that are too long."""
    parts = _SPLIT_RE.split(text)
    # Drop the separator captures and empty strings, keep the real fragments.
    segs = [p.strip() for p in parts if p and not _SPLIT_RE.fullmatch(p)]
    out: list[str] = []
    for s in segs:
        if s:
            out.extend(_hard_split(s, max_tokens))
    return out


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Pack a document's segments into overlapping, token-bounded chunks.

    Greedily appends segments until the next one would exceed `chunk_size`, emits
    the chunk, then carries the tail (up to `overlap` tokens) into the next chunk
    so boundary-spanning facts aren't lost.
    """
    segs = _segments(text, chunk_size)
    chunks: list[Chunk] = []
    cur: list[str] = []        # segments accumulated for the current chunk
    cur_tokens = 0
    idx = 0

    def flush() -> None:
        """Emit the accumulated segments as one Chunk (if any)."""
        nonlocal cur, cur_tokens, idx
        if cur:
            chunks.append(Chunk(text=" ".join(cur).strip(), index=idx))
            idx += 1

    for seg in segs:
        seg_tokens = _ntokens(seg)
        if cur_tokens + seg_tokens > chunk_size and cur:
            flush()
            # Carry overlap from the tail of the previous chunk into the next one.
            carry: list[str] = []
            carry_tokens = 0
            for s in reversed(cur):
                t = _ntokens(s)
                if carry_tokens + t > overlap:
                    break
                carry.insert(0, s)
                carry_tokens += t
            cur = carry
            cur_tokens = carry_tokens
        cur.append(seg)
        cur_tokens += seg_tokens

    flush()
    return chunks
