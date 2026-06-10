# Research Copilot

A source-grounded research assistant. Upload one or more research papers (PDF),
then **chat** with an agent that grounds its answers in your documents and can
reach out to arXiv to situate the work in the wider literature.

It doesn't just answer questions — it helps you make progress: surfacing flaws,
proposing future directions, and finding related work, all cited back to the
source filename and page.

---

## How it works

Research Copilot is a **multi-turn ReAct agent** built on LangGraph. Instead of a
fixed pipeline, the LLM is given a set of tools and decides — turn by turn —
which to call based on your natural-language request.

```
user msg ─► AGENT (Gemini + tools) ─► picks tool(s)
              ├─ list_documents      (what's indexed / resolve a filename)
              ├─ retrieve_doc        (RAG over your uploaded docs)
              ├─ find_errors         (critique the doc; grounded via retrieval)
              ├─ future_directions   (propose next steps; grounded via retrieval)
              └─ similar_papers      (arXiv search — the only external tool)
                         │ results return to agent
                         ▼
              AGENT composes a grounded, cited reply ─► user ─► (next turn)
```

Conversation history persists across turns via a LangGraph checkpointer keyed by
`thread_id`, so the agent retains context within a session.

See [`docs/architecture.md`](docs/architecture.md) for the full design.

### Ingestion pipeline

```
PDFs on disk
  → load_pdf()    (PyPDFLoader → page-level Documents)
  → split_text()  (RecursiveCharacterTextSplitter → 1000-char chunks, 200 overlap)
  → Chroma        (embeds each chunk and persists it on disk)
```

Each chunk carries its **source filename** and **page number** as metadata, so
retrieval can be scoped to a single paper and every answer can be cited.
Re-ingestion is idempotent: chunks get content-based ids, so adding a new PDF and
re-running only indexes the new content (no duplicates).

---

## Tech stack

| Layer            | Choice                                                       |
|------------------|--------------------------------------------------------------|
| Orchestration    | LangGraph (`create_react_agent`) + LangChain                 |
| LLM              | Google Gemini (`gemini-2.5-flash`, temperature 0)            |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim)    |
| Vector store     | Chroma (persisted to `chroma_db/`)                           |
| PDF ingestion    | PyPDFLoader + RecursiveCharacterTextSplitter                 |
| External search  | arXiv API (free, no key)                                     |
| Memory           | LangGraph `MemorySaver` checkpointer (in-RAM)                |

The embedding model runs **locally** — no API key, no per-call cost. Weights
(~90 MB) are downloaded and cached on first use. The only network calls are to
Gemini and to arXiv.

---

## Project layout

```
research-copilot/
├── app/
│   ├── llm/
│   │   └── client.py          # construct the Gemini chat model (bind tools here)
│   ├── ingestion/
│   │   ├── parser.py          # PDF → page-level text → chunks (+ page/source metadata)
│   │   ├── embedding.py       # local all-MiniLM-L6-v2 embedding model
│   │   └── pipeline.py        # parse → chunk → embed → store (orchestrator)
│   ├── retrieval/
│   │   └── retriever.py       # top-k vector search over the Chroma store
│   └── graph/
│       ├── tools.py           # the 5 agent tools as @tool functions
│       ├── state.py           # ResearchState (TypedDict)
│       └── builder.py         # build the ReAct agent + run an interactive chat
├── docs/
│   └── architecture.md        # the agentic design (v2)
├── frontend/
│   └── stitch-export/         # static UI mockups (desktop/mobile, light/dark)
├── pdfs/                       # drop your source PDFs here
├── chroma_db/                  # persisted vector store (gitignored)
├── requirements.txt
└── .env                        # GOOGLE_API_KEY, PDF_PATH (gitignored)
```

---

## Setup

**Requirements:** Python 3.13, a Google (Gemini) API key.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell)
# source .venv/bin/activate      # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment — create a .env file in the project root (see below)
```

`.env`:

```ini
GOOGLE_API_KEY=your_gemini_api_key_here
PDF_PATH=pdfs
```

- `GOOGLE_API_KEY` — your Gemini key ([get one here](https://aistudio.google.com/app/apikey)).
- `PDF_PATH` — directory the ingestion pipeline scans for `.pdf` files.

---

## Usage

### 1. Add your papers

Drop one or more PDFs into the `pdfs/` directory (or wherever `PDF_PATH` points).

### 2. Ingest them into the vector store

```bash
python -m app.ingestion.pipeline
```

This loads, chunks, embeds, and persists every PDF into `chroma_db/`. Re-run it
any time you add new papers — it only indexes what's new.

### 3. Chat with your documents

```bash
python -m app.graph.builder
```

Then talk to it in plain English — the agent routes each request to the right
tool:

```
You: what does this paper claim as its main contribution?
You: are there any methodological weaknesses?
You: where could this research go next?
You: find me similar papers on arXiv
You: what documents do you have indexed?
```

Type `exit` or `quit` to leave. Tool usage is printed inline for visibility.

---

## Verifying individual components

Each module has a small smoke test you can run directly:

```bash
python -m app.llm.client            # confirm the Gemini key works ("pong")
python -m app.ingestion.embedding   # embed two texts, report vector dimension
python -m app.retrieval.retriever "your query here"   # test vector search
python -m app.graph.tools           # exercise each of the 5 tools
```

---

## The tools

| Tool                | What it does                                                                 |
|---------------------|------------------------------------------------------------------------------|
| `list_documents`    | List the indexed papers (and resolve exact filenames for scoping).           |
| `retrieve_doc`      | RAG search over your uploaded docs; returns excerpts cited by source + page. |
| `find_errors`       | Peer-review critique — logical gaps, unsupported claims, methodological flaws. Grounded in retrieved text. |
| `future_directions` | Concrete, actionable next steps grounded in the doc's contributions/limitations. |
| `similar_papers`    | Related work from arXiv (title, authors, date, link, abstract snippet).      |

The document tools accept an optional `source` (a filename) to scope analysis to
a single paper; leave it empty to work across everything indexed.

---

## Out of scope (for now)

- Multi-user collaboration.
- Editing or writing back to the user's document.
- Persistent cross-session memory beyond the in-RAM checkpointer (a
  `SqliteSaver` swap is the planned upgrade).
- Re-ranking / hybrid retrieval (vector-only top-k for now).
