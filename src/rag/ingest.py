"""Ingestion pipeline: load -> chunk -> embed -> store.

Builds the searchable index. It loads .md/.txt/.html/.pdf files from a directory,
chunks each document, embeds the chunks in batches, and writes them to the
configured vector store. Run as a module:

    python -m src.rag.ingest --reset
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .chunking import chunk_text
from .config import Settings, get_settings
from .embeddings import get_embedder
from .vectorstore import Doc, get_vector_store


def _read_file(path: Path) -> str:
    """Extract plain text from one file based on its extension.

    .md/.txt are read directly; .html is stripped of tags (BeautifulSoup);
    .pdf text is extracted page by page (pypdf). Unknown types return "".
    """
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        for tag in soup(["script", "style"]):  # drop non-content tags
            tag.decompose()
        return soup.get_text(separator="\n")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return ""


def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    """Walk `docs_dir` recursively and return [(source_name, text), ...] for every
    supported, non-empty file."""
    docs: list[tuple[str, str]] = []
    for path in sorted(Path(docs_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".html", ".htm", ".pdf"}:
            text = _read_file(path).strip()
            if text:
                docs.append((path.name, text))
    return docs


def ingest(settings: Settings | None = None, reset: bool = False, batch: int = 64) -> int:
    """Build the index: load -> chunk -> embed (in batches of `batch`) -> store.

    With `reset=True` the store is cleared first (avoids duplicates). Returns the
    number of chunks indexed.
    """
    settings = settings or get_settings()
    embedder = get_embedder(settings)
    dim = getattr(embedder, "dim", 256)  # embedding size, used to init pgvector
    store = get_vector_store(settings, dim)
    if reset:
        store.reset()

    documents = load_documents(settings.docs_dir)
    if not documents:
        print(f"No documents found in {settings.docs_dir}")
        return 0

    # Accumulate chunks and embed them in batches for efficiency.
    pending: list[Doc] = []
    texts: list[str] = []
    n_chunks = 0

    def flush() -> None:
        """Embed and store the currently accumulated batch, then clear it."""
        nonlocal pending, texts, n_chunks
        if not pending:
            return
        embeddings = embedder.embed(texts)
        store.add(pending, embeddings)
        n_chunks += len(pending)
        pending, texts = [], []

    for source, text in documents:
        for ch in chunk_text(text, settings.chunk_size, settings.chunk_overlap):
            # Each chunk gets a stable id like "02-authentication.md::3".
            pending.append(
                Doc(id=f"{source}::{ch.index}", text=ch.text, source=source, chunk_index=ch.index)
            )
            texts.append(ch.text)
            if len(pending) >= batch:
                flush()
    flush()  # store any remaining partial batch

    print(f"Ingested {n_chunks} chunks from {len(documents)} documents into '{settings.vector_backend}'.")
    return n_chunks


def main() -> None:
    """CLI entry point: `python -m src.rag.ingest [--reset]`."""
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--reset", action="store_true", help="Clear the store before ingesting.")
    args = parser.parse_args()
    ingest(reset=args.reset)


if __name__ == "__main__":
    main()
