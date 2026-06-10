"""pipeline.py — ingestion orchestrator: parse → chunk → embed → store.

Ties the ingestion steps together:

    PDFs on disk
      → load_pdf()    (parser.py: PyPDFLoader → page-level Documents)
      → split_text()  (parser.py: RecursiveCharacterTextSplitter → chunks)
      → Chroma        (embeds each chunk with BGE-M3 and persists it)

Chroma embeds the chunk text itself using the model from embedding.py, so we
never call embed_documents() here — we hand Chroma the model plus the texts,
and it stores (vector, text, metadata) together for retrieval + citations.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Allow running this file directly (python app/ingestion/pipeline.py) by making
# the project root importable for the absolute `app.*` imports below.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_chroma import Chroma

from app.ingestion.embedding import get_embedding_model
from app.ingestion.parser import load_pdf, split_text

# Where Chroma persists the index on disk, plus the collection name.
PERSIST_DIR = str(_PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "research_docs"


def get_vector_store() -> Chroma:
    """Return a persistent Chroma store wired to the BGE-M3 embedding model.

    Shared by both ingestion (here) and the retriever, so they index and search
    with the same embedding function and the same on-disk collection.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=PERSIST_DIR,
    )


def _chunk_id(metadata: dict, text: str) -> str:
    """A stable, content-based ID for a chunk.

    Derived from the source filename + the chunk text, so re-ingesting the same
    document produces the SAME ids. Chroma then upserts (overwrites) instead of
    creating duplicates — making ingestion safe to re-run when you add a new PDF.
    """
    key = f"{metadata.get('source', 'unknown')}::{text}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def ingest() -> Chroma:
    """Run the full ingestion pipeline and return the populated vector store.

    Safe to re-run: chunks get deterministic ids, so adding a new PDF and
    re-running re-indexes only the new content and leaves existing chunks intact
    (no duplicates).
    """
    docs = load_pdf()
    print(f"Loaded {len(docs)} pages from PDFs.")

    chunks = split_text(docs)
    print(f"Split into {len(chunks)} chunks.")

    # split_text returns dicts {"text": ..., "metadata": {...}}; Chroma takes the
    # texts (which it embeds) and the parallel metadatas (carried alongside).
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [_chunk_id(chunk["metadata"], chunk["text"]) for chunk in chunks]

    store = get_vector_store()
    # Passing ids makes add idempotent: same id -> upsert, not a duplicate row.
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    print(f"Stored {len(texts)} chunks in Chroma at {PERSIST_DIR!r}.")

    return store


if __name__ == "__main__":
    ingest()