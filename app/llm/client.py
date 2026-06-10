"""client.py — the Gemini chat model for the agent and tools.

Single source of truth for the LLM. Its only job is to construct and hand back a
ready-to-use chat model, so the agent/tools never repeat the setup.

Embeddings are NOT here — those live in app/ingestion/embedding.py. This file is
the chat model only (answering, reasoning, tool-calling).
"""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load GOOGLE_API_KEY (and other vars) from .env so the key lives in one place
# and is never hardcoded. langchain-google-genai reads GOOGLE_API_KEY itself.
load_dotenv()

MODEL_NAME = "gemini-2.5-flash"


@lru_cache(maxsize=1)
def get_chat_model():
    """Return a cached Gemini chat model.

    temperature=0 keeps responses grounded and deterministic — important for a
    RAG agent that shows and route tools reliably.

    Cached so the client is built once and reused by the agent and every tool.
    """
    return init_chat_model(
        MODEL_NAME,
        model_provider="google_genai",
        temperature=0,
    )


if __name__ == "__main__":
    # Quick smoke test: confirm the key works and Gemini responds.
    reply = get_chat_model().invoke("Reply with exactly: pong")
    print("model:", MODEL_NAME)
    print("reply:", reply.content)