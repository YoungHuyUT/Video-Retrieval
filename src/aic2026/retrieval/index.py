from __future__ import annotations

from pathlib import Path
import numpy as np


class VectorIndex:
    """Cosine-similarity index with optional FAISS acceleration."""
    def __init__(self, vectors: np.ndarray):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.ndim != 2 or not len(self.vectors):
            raise ValueError("vectors must be a non-empty 2D array")
        self.vectors /= np.maximum(np.linalg.norm(self.vectors, axis=1, keepdims=True), 1e-12)
        try:
            import faiss  # type: ignore
            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)
        except ImportError:
            self._faiss = None

    @classmethod
    def from_npy(cls, path: Path) -> "VectorIndex":
        return cls(np.load(path))

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query, dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        k = min(k, len(self.vectors))
        if k <= 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        if self._faiss is not None:
            scores, ids = self._faiss.search(query[None, :], k)
            return ids[0], scores[0]
        scores = self.vectors @ query
        k = min(k, len(scores))
        ids = np.argpartition(-scores, k - 1)[:k]
        ids = ids[np.argsort(-scores[ids])]
        return ids, scores[ids]
