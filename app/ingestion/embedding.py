"""embedding.py — the embedding model for the ingestion + retrieval pipeline.

Wraps sentence-transformers/all-MiniLM-L6-v2 (a small, fast English embedding
model) via sentence-transformers, exposed as a LangChain ``Embeddings`` object so
it plugs directly into Chroma and the retriever. The same model must be used for
BOTH ingesting chunks and embedding queries, so everything imports it from here.

The model runs locally — no API key, no per-call cost. The weights (~90 MB) are
downloaded and cached on first use under the HuggingFace cache directory. It is
deliberately lightweight to keep load time and memory low on modest machines.
"""

from __future__ import annotations

import os

# Load BGE-M3 from the local HuggingFace cache without hitting the network.
# Must be set BEFORE importing huggingface_hub / langchain_huggingface. This
# avoids a closed-client bug in the huggingface_hub 1.x httpx downloader and
# means startup never depends on connectivity once the model is cached.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

# all-MiniLM-L6-v2: 384-dim embeddings, 256-token window, English. Small & fast.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _select_device() -> str:
    """Use the GPU if torch can see one, otherwise fall back to CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a cached BGE-M3 embedding model.

    Cached so the (heavy) weights load into memory only once per process and
    the same instance is reused by both ingestion and retrieval.

    all-MiniLM-L6-v2 is trained for cosine similarity, so embeddings are
    L2-normalized (``normalize_embeddings=True``) to match how Chroma compares
    vectors.
    """
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": _select_device()},
        encode_kwargs={"normalize_embeddings": True},
    )


if __name__ == "__main__":
    # Quick smoke test: embed two short texts and report the vector size.
    model = get_embedding_model()
    vectors = model.embed_documents(["hello world", "research copilot"])
    print(f"model: {MODEL_NAME}")
    print(f"embedded {len(vectors)} texts, dimension = {len(vectors[0])}")