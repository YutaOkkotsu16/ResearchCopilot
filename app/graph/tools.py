"""tools.py — the capabilities the agent can call.

Each function is a LangChain ``@tool``. The agent (Gemini) reads each tool's
NAME and DOCSTRING to decide when to call it, so the docstrings are written for
the model, not just for humans — they describe *when* to use the tool and what
it returns.

Tools return plain strings: the agent reads that text back and uses it to
compose its reply to the user.

Starting set:
  - retrieve_doc    : RAG over the user's uploaded document (Chroma)
  - similar_papers  : related work from arXiv (external)

(find_errors / future_directions are added next; they build on retrieve_doc.)
"""

from __future__ import annotations

import arxiv
from langchain_core.tools import tool

from app.ingestion.pipeline import get_vector_store
from app.llm.client import get_chat_model
from app.retrieval.retriever import retrieve

# Excerpts are retrieved fragments, so they often start/end mid-sentence. Tell
# the LLM not to mistake that truncation for a flaw in the document.
_FRAGMENT_NOTE = (
    "The excerpts below are retrieved fragments and may be cut off at the start "
    "or end. Do NOT treat truncated sentences, mid-word cuts, or apparent "
    "missing surrounding context as problems — judge only the actual content."
)


def _retrieve_context(query: str, k: int = 8, source: str | None = None) -> str:
    """Pull relevant chunks from the doc and format them with page + source.

    Shared by the analysis tools (find_errors, future_directions) so their LLM
    reasoning is grounded in actual excerpts, each labelled for citation. Uses a
    larger k than a plain lookup to give the model more of the document to
    reason over. If ``source`` is given, only that document is searched.
    """
    docs = retrieve(query, k=k, source=source)
    if not docs:
        return ""
    blocks = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        src = doc.metadata.get("source", "?")
        blocks.append(f"[{src}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _list_sources() -> list[str]:
    """Return the distinct source filenames currently indexed in Chroma."""
    data = get_vector_store().get(include=["metadatas"])
    sources = {m.get("source") for m in data["metadatas"] if m.get("source")}
    return sorted(sources)


@tool
def list_documents() -> str:
    """List the documents the user has uploaded and indexed.

    Use this when the user asks what documents are available, or when they refer
    to a specific paper and you need its exact filename to pass as the `source`
    argument of the other document tools.
    """
    sources = _list_sources()
    if not sources:
        return "No documents are currently indexed."
    return "Indexed documents:\n" + "\n".join(f"- {s}" for s in sources)


@tool
def retrieve_doc(query: str, source: str = "") -> str:
    """Search the user's uploaded documents for passages relevant to a query.

    Use this whenever the user asks something about THEIR document(s) — content,
    methods, results, claims, etc. Returns the most relevant excerpts labelled
    with their source filename and page number so you can cite them.

    If the user is asking about ONE specific paper, pass its filename as
    `source` to restrict the search to that document (call list_documents first
    if you don't know the exact filename). Leave `source` empty to search across
    all indexed documents.

    Args:
        query: What to look for, in a few keywords or a short phrase.
        source: Optional filename to restrict the search to one document.
    """
    docs = retrieve(query, k=5, source=source or None)
    if not docs:
        return "No relevant passages found in the indexed documents."

    blocks = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        src = doc.metadata.get("source", "?")
        blocks.append(f"[{src}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


@tool
def similar_papers(topic: str, max_results: int = 5) -> str:
    """Find related research papers from arXiv on a topic.

    Use this when the user wants related work, similar papers, or to see what
    else exists in the literature. This searches the public arXiv repository —
    it does NOT use the user's uploaded document. Returns a short list of papers
    with title, authors, link, and a snippet of the abstract.

    Args:
        topic: The research topic or keywords to search arXiv for.
        max_results: How many papers to return (default 5).
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    for i, result in enumerate(client.results(search), start=1):
        authors = ", ".join(a.name for a in result.authors[:4])
        if len(result.authors) > 4:
            authors += ", et al."
        abstract = result.summary.replace("\n", " ").strip()
        papers.append(
            f"{i}. {result.title}\n"
            f"   Authors: {authors}\n"
            f"   Published: {result.published.date()}\n"
            f"   Link: {result.entry_id}\n"
            f"   Abstract: {abstract[:300]}..."
        )

    if not papers:
        return f"No arXiv papers found for: {topic!r}"
    return "\n\n".join(papers)


@tool
def find_errors(focus: str = "", source: str = "") -> str:
    """Find possible problems in the user's uploaded document.

    Use this when the user asks you to review, critique, check, or find
    mistakes/weaknesses in THEIR document — e.g. logical gaps, unsupported
    claims, methodological flaws, inconsistencies, or missing details. Grounds
    its critique in the actual text and cites page numbers. Only flags issues
    supported by the retrieved passages; it does not invent problems.

    Pass `source` (a filename) to review one specific paper; leave it empty to
    review across all indexed documents.

    Args:
        focus: Optional aspect to concentrate on (e.g. "methodology",
            "statistical analysis"). Leave empty to review broadly.
        source: Optional filename to restrict the review to one document.
    """
    query = focus or "methodology, results, claims, assumptions, limitations, conclusions"
    context = _retrieve_context(query, source=source or None)
    if not context:
        return "No content found in the indexed documents to review."

    prompt = (
        "You are a meticulous peer reviewer. Using ONLY the excerpts below from "
        "the user's document, identify possible problems: logical gaps, "
        "unsupported claims, methodological weaknesses, inconsistencies, or "
        "missing details. For each issue, cite the source and page (e.g. "
        "'paper.pdf, p. 4') and quote or paraphrase the relevant text. If the "
        "excerpts contain no clear problems, say so honestly. Do NOT invent "
        "issues not supported by the text.\n\n"
        f"{_FRAGMENT_NOTE}\n\n"
        f"Focus: {focus or 'general review'}\n\n"
        f"Excerpts:\n{context}"
    )
    return get_chat_model().invoke(prompt).content


@tool
def future_directions(focus: str = "", source: str = "") -> str:
    """Suggest future research directions based on the user's document.

    Use this when the user asks where the research could go next, what to do
    after this work, follow-up experiments, extensions, or open questions.
    Grounds its suggestions in the document's contributions and limitations and
    cites page numbers where relevant.

    Pass `source` (a filename) to base directions on one specific paper; leave
    empty to draw on all indexed documents.

    Args:
        focus: Optional area to focus the suggestions on. Leave empty for
            broad directions.
        source: Optional filename to restrict to one document.
    """
    query = focus or "contributions, limitations, future work, open questions, conclusions"
    context = _retrieve_context(query, source=source or None)
    if not context:
        return "No content found in the indexed documents to build on."

    prompt = (
        "You are a research advisor. Using ONLY the excerpts below from the "
        "user's document, propose concrete, actionable future research "
        "directions: extensions, follow-up experiments, unaddressed questions, "
        "or ways to overcome stated limitations. Ground each suggestion in the "
        "document (cite source and page, e.g. 'paper.pdf, p. 4') and explain why "
        "it follows from the work. Avoid generic advice not tied to this document.\n\n"
        f"{_FRAGMENT_NOTE}\n\n"
        f"Focus: {focus or 'general future directions'}\n\n"
        f"Excerpts:\n{context}"
    )
    return get_chat_model().invoke(prompt).content


# Convenience: the list the agent binds to.
TOOLS = [list_documents, retrieve_doc, similar_papers, find_errors, future_directions]


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=== retrieve_doc ===")
    print(retrieve_doc.invoke({"query": "how were the calculations performed"})[:400])

    print("\n=== similar_papers ===")
    print(similar_papers.invoke({"topic": "machine learning interatomic potentials", "max_results": 2}))

    print("\n=== find_errors ===")
    print(find_errors.invoke({"focus": "methodology"})[:600])

    print("\n=== future_directions ===")
    print(future_directions.invoke({"focus": ""})[:600])