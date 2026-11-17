"""A tiny, fully local vector index: numpy cosine similarity over stored
embeddings. Deliberately simple so it is easy to explain and audit. For
production scale one swaps this for FAISS/Qdrant behind the same interface;
the retrieval code above it does not change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Passage:
    source_id: str        # stable citation id, e.g. neet_cutoffs#17
    text: str             # human-readable passage
    meta: dict[str, Any] = field(default_factory=dict)


class VectorIndex:
    def __init__(self, passages: list[Passage], embeddings: np.ndarray) -> None:
        if len(passages) != embeddings.shape[0]:
            raise ValueError("passages and embeddings length mismatch")
        self.passages = passages
        self.embeddings = embeddings.astype(np.float32)
        self._norm = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

    @property
    def size(self) -> int:
        return len(self.passages)

    def search(self, query_vec: np.ndarray, top_k: int = 6) -> list[tuple[Passage, float]]:
        """Return top-k (passage, cosine-similarity) pairs, best first."""
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = (self._norm @ q.T).ravel()
        order = np.argsort(-sims)[:top_k]
        return [(self.passages[i], float(sims[i])) for i in order]

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dir_path / "embeddings.npz", embeddings=self.embeddings)
        with open(dir_path / "passages.json", "w", encoding="utf-8") as f:
            json.dump(
                [{"source_id": p.source_id, "text": p.text, "meta": p.meta} for p in self.passages],
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, dir_path: Path) -> "VectorIndex":
        data = np.load(dir_path / "embeddings.npz")
        embeddings = data["embeddings"]
        with open(dir_path / "passages.json", encoding="utf-8") as f:
            raw = json.load(f)
        passages = [Passage(r["source_id"], r["text"], r["meta"]) for r in raw]
        return cls(passages, embeddings)