# Research Copilot — Agentic Workflow (v2)

This supersedes the linear QA-graph design in `architechture.md.txt`. v1 answered a
single question per run. v2 is a **multi-turn chat agent**: the user uploads a paper,
then converses with an agent that decides — turn by turn — which capability to invoke.

## Vision

A source-grounded research assistant. The user submits a document, then chats. The
agent grounds its answers in that document and can also reach outward (arXiv) to
situate the work in the literature. It doesn't just answer — it helps the user make
progress: spotting flaws, proposing directions, finding related work.

---

## Core shift from v1: tools, not fixed nodes

v1 wired a fixed path (Planner → Retriever → Grader → …). v2 gives the LLM a set of
**tools** and lets it route based on the user's natural-language request.

- **Node** = a step *we* route to (deterministic edges we author).
- **Tool** = a capability the *LLM* chooses to call at runtime.

Because the user phrases intent freely ("find mistakes", "show related work",
"where could this go?"), the LLM is the right router. We use the **ReAct agent**
pattern: one agent node + a tool-executor node, looping until the agent replies.

---

## Multi-turn chat

State persists across turns via a **checkpointer** keyed by `thread_id`:

- `messages` accumulates the whole conversation (reducer: `add_messages`).
- Each user message re-enters the graph with full history.
- "Loop until the user is satisfied" is the **conversation itself** — the human
  drives it across turns; there is no explicit satisfaction-checker node.

Dev: `MemorySaver` (in-RAM). Later: `SqliteSaver` for persistence across restarts.

---

## High-level flow

```
START → agent → did it call a tool?
                  ├─ yes → tools → back to agent     (ReAct loop, one turn)
                  └─ no  → END (reply to user)
                                   │
                       user replies → re-invoke on same thread_id (next turn)
```

```
user msg ─► AGENT (Gemini + tools) ─► picks tool(s)
              ├─ retrieve_doc        (search the uploaded doc — shared grounding)
              ├─ find_errors         (analyze the doc for flaws; grounded via retriever)
              ├─ future_directions   (propose research directions; grounded via retriever)
              └─ similar_papers      (arXiv search — external)
                         │ results return to agent
                         ▼
              AGENT composes reply ─► user ─► (next turn)
```

---

## Tools

### retrieve_doc
- **Purpose:** fetch the most relevant chunks from the user's uploaded document.
- **Backed by:** the existing `app/retrieval/retriever.py` over the Chroma store.
- **Role:** the shared grounding primitive. `find_errors` and `future_directions`
  rely on it so their analysis is anchored in the actual document, not invented.

### find_errors
- **Purpose:** surface possible problems — logical gaps, unsupported claims,
  methodological weaknesses, inconsistencies.
- **Grounding:** retrieves relevant sections first, then has the LLM critique them.
- **Output:** a list of issues, each tied to a location (page) for citation.

### future_directions
- **Purpose:** suggest where the research could go next.
- **Grounding:** retrieves the doc's contributions/limitations, then extrapolates.
- **Output:** concrete, actionable directions (the v1 "Next-Step Generator" idea,
  promoted to a first-class tool).

### similar_papers
- **Purpose:** find related work from the literature.
- **Source:** **arXiv API** (free, no key, research-focused).
- **Output:** a short list of papers (title, authors, link, abstract snippet).
- **Note:** this is the only tool that leaves the local store and hits the internet.

---

## State (TypedDict)

```python
class ResearchState(TypedDict):
    # Conversation history — grows across turns.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Per-turn working fields (overwritten each turn) may be added as needed,
    # e.g. retrieved_chunks, retry_count. The ReAct agent itself only requires
    # `messages`; extra fields are for custom nodes if we add them.
```

---

## Components / files

| File | Role |
|------|------|
| `app/llm/client.py` | construct the Gemini chat model (bind tools here) |
| `app/retrieval/retriever.py` | vector search over the doc (done) |
| `app/graph/tools.py` | the 4 tools above, as `@tool` functions |
| `app/graph/state.py` | `ResearchState` |
| `app/graph/builder.py` | build the ReAct agent graph + `MemorySaver` |
| `app/ingestion/*` | parse → chunk → embed → store (done) |

---

## Out of scope (for now)

- Multi-user collaboration.
- Editing/writing back to the user's document.
- Persistent cross-session memory beyond the checkpointer.
- Re-ranking / hybrid retrieval (vector-only top-k for now).

---

## Success criteria

A user can:
1. Upload a paper and have it indexed.
2. Chat across multiple turns, with context retained.
3. Ask — in natural language — to find errors, get future directions, or find
   similar papers, and have the agent route to the right capability.
4. Receive grounded, cited answers for doc-based tools.
5. Receive real arXiv results for literature search.
