"""Lightweight observability: per-query traces, token usage, cost, and latency.

Every query writes a structured trace (retrieval timing, generation timing,
tokens, USD cost) to a JSONL file, and `aggregate()` rolls those up for a
metrics endpoint / dashboard. In a larger deployment you'd swap this for
Langfuse or OpenTelemetry -- the seams are intentionally in the same shape.
"""
from __future__ import annotations

import json
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Price table in USD per 1,000,000 tokens. Source: OpenAI public pricing.
# The "fake-*" entries are zero so the keyless path reports $0.00.
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "fake-llm": {"input": 0.0, "output": 0.0},
    "fake-embed": {"input": 0.0, "output": 0.0},
}


def cost_usd(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Convert token counts to a USD cost using the price table (0 if unknown)."""
    p = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


@dataclass
class Trace:
    """The structured record of a single query: timings, tokens, cost, the sources
    used and the answer. Serialised to one JSON line per query."""

    question: str
    timings_ms: dict[str, float] = field(default_factory=dict)  # per-stage durations
    tokens: dict[str, int] = field(default_factory=dict)        # embedding/prompt/completion
    cost_usd: float = 0.0
    n_contexts: int = 0
    answer: str = ""
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (with a wall-clock timestamp) for logging."""
        return {
            "ts": time.time(),
            "question": self.question,
            "timings_ms": self.timings_ms,
            "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 8),
            "n_contexts": self.n_contexts,
            "sources": self.sources,
            "answer": self.answer,
        }


class Tracer:
    """Writes traces to `<trace_dir>/queries.jsonl` and aggregates them for /metrics."""

    def __init__(self, trace_dir: Path) -> None:
        self.dir = Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "queries.jsonl"
        self._lock = threading.Lock()  # serialise concurrent writes

    @contextmanager
    def span(self, trace: Trace, name: str):
        """Time the wrapped block and record it under `name` in `trace.timings_ms`.

        Used as ``with tracer.span(trace, "retrieval"): ...``. The timing is taken
        in a `finally`, so it's recorded even if the block raises.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            trace.timings_ms[name] = round((time.perf_counter() - start) * 1000, 2)

    def record(self, trace: Trace) -> None:
        """Append one trace as a JSON line (lock-guarded against interleaving)."""
        line = json.dumps(trace.to_dict()) + "\n"
        # Lock so concurrent requests can't interleave half-written lines.
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)

    def _read_rows(self) -> list[dict[str, Any]]:
        """Parse every trace line, skipping torn/partial lines rather than raising."""
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn/partial line shouldn't 500 the endpoint
        return rows

    def records(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return recorded traces (most recent `limit`, or all) for the analytics API."""
        rows = self._read_rows()
        if limit is not None and limit >= 0:
            # Last `limit` traces (or ALL when limit >= len); limit=0 -> []. A
            # len-relative slice was wrong for len < limit < 2*len (it dropped the
            # oldest rows instead of returning all of them).
            rows = rows[-limit:] if limit else []
        return rows

    def aggregate(self) -> dict[str, Any]:
        """Roll up all recorded traces into summary metrics for the /metrics endpoint."""
        rows = self._read_rows()
        if not rows:
            return {"queries": 0}
        n = len(rows)
        # Read every field defensively (`.get`) so a legacy/hand-edited trace line
        # that parses but is missing a key can't 500 the endpoint -- matching the
        # tolerance of records()/`/analytics`.
        total_cost = sum(r.get("cost_usd", 0.0) for r in rows)
        lat = sorted(sum((r.get("timings_ms") or {}).values()) for r in rows)
        # Nearest-rank p95 (clamped), correct even for very small sample counts.
        p95_idx = min(n - 1, max(0, math.ceil(0.95 * n) - 1))
        return {
            "queries": n,
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_usd": round(total_cost / n, 6),
            "avg_latency_ms": round(sum(lat) / n, 1),
            "p95_latency_ms": round(lat[p95_idx], 1),
            "avg_contexts": round(sum(r.get("n_contexts", 0) for r in rows) / n, 2),
        }
