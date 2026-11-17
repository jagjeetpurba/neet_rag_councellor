"""Local embeddings via Ollama (nomic-embed-text). No PyTorch dependency."""

from __future__ import annotations

import numpy as np
import ollama

from . import config


def _client() -> ollama.Client:
    return ollama.Client(host=config.OLLAMA_HOST)


def _installed_models(client) -> set[str]:
    resp = client.list()
    models = resp.models if hasattr(resp, "models") else resp.get("models", [])
    names = {getattr(m, "model", "") for m in models} | {getattr(m, "name", "") for m in models}
    return {n.split(":")[0] for n in names if n}


def ensure_embed_model() -> None:
    """Pull the embedding model if it is not present locally."""
    client = _client()
    installed = _installed_models(client)
    base = config.EMBED_MODEL.split(":")[0]
    if base not in installed:
        print(f"[embed] pulling {config.EMBED_MODEL} ...")
        client.pull(config.EMBED_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings -> (N, D) float32 array."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    resp = _client().embed(model=config.EMBED_MODEL, input=texts)
    embeds = resp.get("embeddings", [])
    return np.asarray(embeds, dtype=np.float32)


def embed_one(text: str) -> np.ndarray:
    return embed_texts([text])[0]