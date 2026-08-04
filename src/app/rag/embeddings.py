"""Embeddings for the RAG pipeline.

Single implementation: OpenAI's text-embedding-3-small. No local/offline
fallback (no TF-IDF, no other "dev mode") -- the same embedding call runs
locally and in production, so there's exactly one retrieval code path to
understand and defend, not two.
"""
from __future__ import annotations

from app.config import Settings


class Embedder:
    def __init__(self, settings: Settings, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required (used for both the agent LLM and embeddings).")
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


def get_embedder(settings: Settings) -> Embedder:
    return Embedder(settings)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    import numpy as np

    va, vb = np.asarray(a), np.asarray(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-9
    return float(np.dot(va, vb) / denom)
