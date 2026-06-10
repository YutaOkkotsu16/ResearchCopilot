"""builder.py — assemble the multi-turn ReAct agent (the QA graph).

Wires everything together:
  - the Gemini chat model        (app/llm/client.py)
  - the four tools               (app/graph/tools.py)
  - a system prompt              (how to behave / when to use which tool)
  - a MemorySaver checkpointer   (multi-turn memory, keyed by thread_id)

We use LangGraph's prebuilt create_react_agent, which builds the agent<->tools
loop for us:

    START -> agent -> (called a tool?) -> tools -> agent ...  -> END

Run this file directly for an interactive chat:
    python -m app.graph.builder
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly by making the project root importable for `app.*`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.graph.tools import TOOLS
from app.llm.client import get_chat_model

# Behaviour + routing guidance. The agent also reads each tool's docstring, but
# this gives it the overall role and a quick map of when to reach for what.
SYSTEM_PROMPT = (
    "You are Research Copilot, a source-grounded research assistant. The user "
    "has uploaded one or more research documents; help them understand them, "
    "review them, and move their research forward.\n\n"
    "Choosing a tool:\n"
    "- What documents are available, or to resolve which paper the user means "
    "-> list_documents.\n"
    "- Questions about the user's OWN document (content, methods, results, "
    "claims) -> retrieve_doc.\n"
    "- Review/critique a document for problems -> find_errors.\n"
    "- Next steps, extensions, or open questions for the work -> future_directions.\n"
    "- Related work from the wider literature -> similar_papers (arXiv).\n\n"
    "Multiple documents: the doc tools accept an optional `source` (a filename). "
    "If the user asks about ONE specific paper, pass its filename as `source` to "
    "scope to it; if you're unsure of the exact filename, call list_documents "
    "first. Leave `source` empty to work across all documents.\n\n"
    "Ground every claim about a document in retrieved text and cite the source "
    "filename and page. If the tools don't return enough to answer, say so "
    "honestly instead of inventing facts."
)


def build_agent():
    """Build and compile the ReAct agent with multi-turn memory.

    Returns a compiled LangGraph app. Invoke it with a messages payload and a
    config carrying a thread_id; the checkpointer keeps that thread's history.
    """
    return create_react_agent(
        get_chat_model(),
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )


def _message_text(content) -> str:
    """Flatten an AIMessage's content down to plain reply text.

    langchain-core 1.x (and Gemini via langchain-google-genai) may return
    `content` as a list of content blocks — e.g. [{"type": "text", "text": ...}]
    — rather than a plain string. Printing that list shows dict metadata, so we
    pull out and join just the text parts here. A plain string passes through.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def chat(agent, message: str, thread_id: str = "cli-session", debug: bool = False) -> str:
    """Send one user message to the agent and return its text reply.

    The thread_id selects the conversation; reusing it preserves history across
    calls (multi-turn). A new thread_id starts a fresh conversation.

    When debug=True, prints just the name of each tool the agent calls as it
    happens, by streaming the agent's steps instead of just invoking.
    """
    config = {"configurable": {"thread_id": thread_id}}
    payload = {"messages": [{"role": "user", "content": message}]}

    if not debug:
        result = agent.invoke(payload, config)
        return _message_text(result["messages"][-1].content)

    # Debug path: stream step-by-step updates so we can see the tool calls.
    # stream_mode="updates" yields one dict per node that ran this turn, e.g.
    # {"agent": {"messages": [AIMessage(...)]}} or {"tools": {"messages": [...]}}.
    final_reply = ""
    for chunk in agent.stream(payload, config, stream_mode="updates"):
        for update in chunk.values():
            for msg in update.get("messages", []):
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    # The agent decided to call one or more tools — show only
                    # the tool name, not its args or returned content.
                    for call in tool_calls:
                        print(f"  [→ {call['name']}]")
                elif getattr(msg, "content", "") and not getattr(msg, "name", None):
                    # An AIMessage with text and no tool calls = the final answer.
                    # (ToolMessages also have content but carry a `name`; skip them.)
                    final_reply = _message_text(msg.content)
    return final_reply


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    agent = build_agent()
    print("Research Copilot — chat with your document. Type 'exit' to quit.")
    print("(tool usage is shown for debugging)\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue
        reply = chat(agent, user, debug=True)
        print(f"\nCopilot: {reply}\n")
