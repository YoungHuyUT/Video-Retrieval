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
        self.manifest_video_ids: list[str] = []
        try:
            import faiss  # type: ignore
            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)
        except ImportError:
            self._faiss = None

    @classmethod
    def from_npy(cls, path: Path) -> VectorIndex:
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

    def search_filtered(
        self,
        query: np.ndarray,
        k: int,
        video_ids: set[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vector search restricted to *video_ids* (FAISS/NumPy backend).

        FAISS/NumPy cannot filter on metadata at query time, so we mask the
        returned candidates by manifest video_id after the nearest-neighbour
        scan. Pass ``None`` for no filtering.
        """
        if not video_ids:
            return self.search(query, k)
        if len(self.manifest_video_ids) != len(self.vectors):
            raise ValueError("manifest_video_ids must align with the vector index")
        query = np.asarray(query, dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        allowed = np.fromiter(
            (video_id in video_ids for video_id in self.manifest_video_ids),
            dtype=bool,
            count=len(self.manifest_video_ids),
        )
        ids = np.flatnonzero(allowed)
        if len(ids) == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        scores = self.vectors[ids] @ query
        k = min(k, len(ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return ids[top], scores[top]
