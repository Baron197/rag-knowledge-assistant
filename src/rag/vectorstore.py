"""Vector store behind a swappable interface.

This module stores one embedding vector per chunk and answers the question
"which stored vectors are most similar to this query vector?". Two
interchangeable backends implement the same `VectorStore` protocol:

- `NumpyVectorStore` (default): a transparent cosine-similarity store persisted
  to disk. Zero external services -- clone and run. Search is implemented
  directly, which keeps the mechanics visible: at its core a vector DB is
  similarity search over a matrix.
- `PgVectorStore` (production): Postgres + the pgvector extension. Brought up
  with one `docker compose up -d db`.

Because both expose identical methods, the rest of the app never changes when you
switch `VECTOR_BACKEND`. Both also guard against embedding **dimension
mismatches** -- the classic footgun when you switch embedding models without
rebuilding the index.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Settings


@dataclass
class Doc:
    """A stored chunk: a stable `id`, its `text`, the `source` filename, and the
    `chunk_index` (its ordinal position within that source document)."""

    id: str
    text: str
    source: str
    chunk_index: int


@dataclass
class Hit:
    """A search result: the matched `doc` and its similarity `score`."""

    doc: Doc
    score: float


class DimensionMismatchError(RuntimeError):
    """Raised when an index built with one embedding size is used with another."""


class VectorStore(Protocol):
    """Structural interface both backends implement (reset/add/search/all_docs/count)."""

    def reset(self) -> None:
        """Clear all stored vectors and metadata."""
        ...

    def add(self, docs: list[Doc], embeddings: np.ndarray) -> None:
        """Insert chunks and their embeddings (row i of `embeddings` <-> docs[i])."""
        ...

    def search(self, query_embedding: np.ndarray, k: int) -> list[Hit]:
        """Return the k most similar chunks to the query vector, best first."""
        ...

    def all_docs(self) -> list[Doc]:
        """Return every stored chunk (used to build the BM25 index for hybrid mode)."""
        ...

    def count(self) -> int:
        """Number of chunks currently stored."""
        ...


class NumpyVectorStore:
    """Default backend: all vectors in one NumPy matrix on disk; cosine search via
    a single matrix multiply. Persists to `vectors.npy` + `meta.json`."""

    def __init__(self, index_dir: Path) -> None:
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._vec_path = self.dir / "vectors.npy"     # the (n, dim) float32 matrix
        self._meta_path = self.dir / "meta.json"      # {"dim", "docs": [...]}
        self._vectors: np.ndarray | None = None
        self._docs: list[Doc] = []
        self._dim: int | None = None                  # embedding size this index holds
        self._load()

    def _load(self) -> None:
        """Load a previously persisted index from disk, if present."""
        if not self._meta_path.exists():
            return
        meta = json.loads(self._meta_path.read_text())
        # meta is {"dim": int, "docs": [...]}; tolerate the older bare-list format.
        docs = meta.get("docs", []) if isinstance(meta, dict) else meta
        self._dim = meta.get("dim") if isinstance(meta, dict) else None
        if docs and self._vec_path.exists():
            self._vectors = np.load(self._vec_path)
            self._docs = [Doc(**d) for d in docs]
            if self._dim is None and self._vectors is not None:
                self._dim = int(self._vectors.shape[1])

    def _persist(self) -> None:
        """Write the current vectors + metadata back to disk."""
        if self._vectors is not None:
            np.save(self._vec_path, self._vectors)
        self._meta_path.write_text(
            json.dumps({"dim": self._dim, "docs": [asdict(d) for d in self._docs]})
        )

    def reset(self) -> None:
        """Clear in-memory state and on-disk index (used by ingest --reset)."""
        self._vectors = None
        self._docs = []
        self._dim = None
        # Write empty metadata so a stale vectors file is ignored on reload;
        # the vectors file itself is overwritten on the next add(). unlink is
        # best-effort because some mounted filesystems disallow it.
        self._meta_path.write_text(json.dumps({"dim": None, "docs": []}))
        try:
            if self._vec_path.exists():
                self._vec_path.unlink()
        except OSError:
            pass

    def add(self, docs: list[Doc], embeddings: np.ndarray) -> None:
        """Append chunks + vectors, guarding against an embedding-size change."""
        if len(docs) == 0:
            return
        embeddings = embeddings.astype(np.float32)
        new_dim = int(embeddings.shape[1])
        if self._dim is None:
            self._dim = new_dim
        elif new_dim != self._dim:
            # Mixing 256-dim and 1536-dim vectors would corrupt search; fail loudly.
            raise DimensionMismatchError(
                f"Index holds {self._dim}-dim vectors but got {new_dim}-dim. "
                "You likely changed embedding model/provider; re-run ingestion "
                "with --reset."
            )
        self._vectors = (
            embeddings if self._vectors is None else np.vstack([self._vectors, embeddings])
        )
        self._docs.extend(docs)
        self._persist()

    def search(self, query_embedding: np.ndarray, k: int) -> list[Hit]:
        """Cosine-rank all chunks against the query and return the top k."""
        if self._vectors is None or len(self._docs) == 0:
            return []
        q = query_embedding.reshape(-1).astype(np.float32)
        if self._dim is not None and q.shape[0] != self._dim:
            raise DimensionMismatchError(
                f"Index is {self._dim}-dim but the query embedding is {q.shape[0]}-dim. "
                "The query embedder must match the one used to build the index; "
                "re-run ingestion with --reset after changing models."
            )
        # Vectors are L2-normalised at embedding time, so dot product == cosine.
        # One matrix-vector product scores every chunk at once.
        scores = self._vectors @ q
        top = np.argsort(-scores)[:k]  # indices of the k highest scores
        return [Hit(doc=self._docs[i], score=float(scores[i])) for i in top]

    def all_docs(self) -> list[Doc]:
        """Return a copy of all stored chunks."""
        return list(self._docs)

    def count(self) -> int:
        """Number of stored chunks."""
        return len(self._docs)


class PgVectorStore:
    """Production backend: Postgres + pgvector. Same interface as the NumPy store,
    but search and storage are handled by the database and scale far further."""

    def __init__(self, dsn: str, dim: int) -> None:
        # Imported lazily so the default (numpy) path needs no DB driver installed.
        import psycopg
        from pgvector.psycopg import register_vector

        self._psycopg = psycopg
        self._dsn = dsn
        self.dim = dim
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._ensure_table(dim)

    def _existing_dim(self) -> int | None:
        """Return the declared vector dimension of the existing `chunks` table, if any."""
        # atttypmod for a pgvector column encodes the declared dimension.
        row = self._conn.execute(
            "SELECT a.atttypmod FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "WHERE c.relname = 'chunks' AND a.attname = 'embedding'"
        ).fetchone()
        if not row or row[0] is None or row[0] < 0:
            return None
        return int(row[0])

    def _ensure_table(self, dim: int) -> None:
        """Create the `chunks` table, rebuilding it if its dimension changed."""
        existing = self._existing_dim()
        if existing is not None and existing != dim:
            # Index was built for a different embedding size; rebuild cleanly.
            self._conn.execute("DROP TABLE IF EXISTS chunks")
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id           TEXT PRIMARY KEY,
                text         TEXT NOT NULL,
                source       TEXT NOT NULL,
                chunk_index  INT  NOT NULL,
                embedding    VECTOR({dim})
            )
            """
        )

    def reset(self) -> None:
        """Remove all rows from the chunks table."""
        self._conn.execute("TRUNCATE chunks")

    def add(self, docs: list[Doc], embeddings: np.ndarray) -> None:
        """Upsert chunks + vectors (re-ingesting an id updates it, not duplicates)."""
        with self._conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (id, text, source, chunk_index, embedding) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE "
                "SET text=EXCLUDED.text, embedding=EXCLUDED.embedding",
                [
                    (d.id, d.text, d.source, d.chunk_index, np.asarray(e, dtype=np.float32))
                    for d, e in zip(docs, embeddings, strict=False)
                ],
            )

    def search(self, query_embedding: np.ndarray, k: int) -> list[Hit]:
        """Nearest-neighbour search via pgvector's `<=>` cosine-distance operator."""
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        rows = self._conn.execute(
            # 1 - cosine_distance = cosine_similarity, so higher score = closer.
            "SELECT id, text, source, chunk_index, 1 - (embedding <=> %s) AS score "
            "FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (q, q, k),
        ).fetchall()
        return [
            Hit(Doc(id=r[0], text=r[1], source=r[2], chunk_index=r[3]), score=float(r[4]))
            for r in rows
        ]

    def all_docs(self) -> list[Doc]:
        """Return every stored chunk (ordered by id) for building the BM25 index."""
        rows = self._conn.execute(
            "SELECT id, text, source, chunk_index FROM chunks ORDER BY id"
        ).fetchall()
        return [Doc(id=r[0], text=r[1], source=r[2], chunk_index=r[3]) for r in rows]

    def count(self) -> int:
        """Number of rows in the chunks table."""
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def get_vector_store(settings: Settings, dim: int) -> VectorStore:
    """Factory: build the vector store named in settings (`dim` = embedding size)."""
    if settings.vector_backend == "pgvector":
        return PgVectorStore(settings.pg_dsn, dim)
    return NumpyVectorStore(settings.index_dir)
