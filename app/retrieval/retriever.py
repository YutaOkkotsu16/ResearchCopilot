"""retriever.py — query the Chroma store for relevant chunks.

A standalone building block (no LangGraph): given a question, embed it with
BGE-M3 and return the top-k most similar chunks from the persisted Chroma store.
The Retriever node in the QA graph will simply call retrieve() — keeping the
actual vector-search logic here, separate from graph orchestration.

Reuses get_vector_store() from the ingestion pipeline so retrieval searches the
exact same collection, embedding model, and on-disk directory that was indexed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly (python app/retrieval/retriever.py "...") by making the
# project root importable for the absolute `app.*` imports below.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.documents import Document

from app.ingestion.pipeline import get_vector_store

DEFAULT_K = 5


def _source_filter(source: str | None) -> dict | None:
    """Build a Chroma metadata filter restricting results to one document.

    Returns None when no source is given (search across all documents).
    """
    return {"source": source} if source else None


def retrieve(query: str, k: int = DEFAULT_K, source: str | None = None) -> list[Document]:
    """Return the top-k chunks most relevant to ``query``.

    Args:
        query: The retrieval query (in the QA graph, this is the Planner's output).
        k: How many chunks to return.
        source: If given (a filename), restrict the search to that one document.
            If None, search across all indexed documents.

    Returns:
        A list of LangChain ``Document`` objects, each with ``page_content`` (the
        chunk text) and ``metadata`` (page + source) for citations.
    """
    store = get_vector_store()
    return store.similarity_search(query, k=k, filter=_source_filter(source))


def retrieve_with_scores(
    query: str, k: int = DEFAULT_K, source: str | None = None
) -> list[tuple[Document, float]]:
    """Like retrieve(), but also return a relevance score per chunk.

    Useful for the Context Grader: a low top score is a signal that retrieval
    was weak and the query may need rewriting.
    """
    store = get_vector_store()
    return store.similarity_search_with_relevance_scores(
        query, k=k, filter=_source_filter(source)
    )


if __name__ == "__main__":
    # Windows consoles default to cp1252 and crash on characters like "ˇ" that
    # appear in scientific PDFs. Switch stdout to UTF-8 and replace anything it
    # still can't render, so a display quirk never masks a working retrieval.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Quick smoke test. Pass a question as an argument, or use the default.
    query = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"
    print(f"Query: {query!r}\n")

    results = retrieve_with_scores(query)
    if not results:
        print("No chunks returned — is the Chroma store populated?")
    for i, (doc, score) in enumerate(results, start=1):
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"[{i}] score={score:.3f} | page={page}")
        print(f"    {preview}...\n")
