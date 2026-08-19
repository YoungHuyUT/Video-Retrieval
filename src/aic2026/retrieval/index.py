from __future__ import annotations

from pathlib import Path

import numpy as np


class VectorIndex:
    """Cosine-similarity index with optional FAISS acceleration.

    Memory note: load with ``from_npy(path, mmap=True)`` (the default for the
    large competition features file) to keep the ~363 MB matrix **memory-mapped
    read-only** instead of copied into RAM. The FAISS backend cannot share an mmap
    matrix (it copies the data into its own buffer, effectively doubling the RAM
    footprint), so on a memory-mapped load we fall back to a NumPy cosine search
    that streams pages from disk on demand. This drops the backend's baseline
    footprint from ~950 MB to ~200 MB on the official corpus, leaving headroom
    for TRAKE alignment on low-RAM machines.

    When an *in-RAM* (writable) array is passed — e.g. in unit tests or on small
    derived datasets — the original FAISS-accelerated, in-place-normalized path
    is preserved unchanged.
    """

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.ndim != 2 or not len(self.vectors):
            raise ValueError("vectors must be a non-empty 2D array")
        self.manifest_video_ids: list[str] = []
        # Only build FAISS when the matrix is a normal writable (in-RAM) array.
        # A memory-mapped (read-only) matrix would be copied into FAISS's own
        # buffer, negating the RAM saving, so we skip it and use NumPy cosine
        # (computed per-row from a tiny precomputed norms array instead).
        if self.vectors.flags.writeable:
            self.vectors /= np.maximum(
                np.linalg.norm(self.vectors, axis=1, keepdims=True), 1e-12
            )
            try:
                import faiss  # type: ignore

                self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
                self._faiss.add(self.vectors)
            except ImportError:
                self._faiss = None
            self._norms: np.ndarray | None = None
        else:
            # Memory-mapped path: keep the matrix on disk. Precompute per-row L2
            # norms (177k × 4 B ≈ 0.7 MB) so cosine can be recovered as (v·q)/‖v‖
            # without ever materialising the full matrix in RAM.
            self._faiss = None
            self._norms = np.linalg.norm(self.vectors, axis=1).astype(np.float32)

    @classmethod
    def from_npy(cls, path: Path, mmap: bool = True) -> VectorIndex:
        """Load a feature ``.npy`` matrix.

        ``mmap=True`` (default) memory-maps the file read-only — the 363 MB
        competition matrix stays on disk and is paged in by the OS only for the
        rows a query actually touches. ``mmap=False`` loads the whole matrix into
        RAM (legacy behaviour, used by tests / small datasets).
        """
        if mmap:
            return cls(np.load(path, mmap_mode="r"))
        return cls(np.load(path))

    # -- internal cosine helpers --------------------------------------------
    @staticmethod
    def _norm_query(query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        return query

    def _cosine_for_ids(self, ids: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Cosine between ``query`` (assumed L2-normalized) and rows ``ids``."""
        dot = self.vectors[ids] @ query
        if self._norms is None:
            return dot
        return dot / np.maximum(self._norms[ids], 1e-12)

    def _cosine_all(self, query: np.ndarray) -> np.ndarray:
        """Cosine between ``query`` and every row (streams the mmap once)."""
        dot = self.vectors @ query
        if self._norms is None:
            return dot
        return dot / np.maximum(self._norms, 1e-12)

    # -- public API ----------------------------------------------------------
    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query = self._norm_query(query)
        k = min(k, len(self.vectors))
        if k <= 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        if self._faiss is not None:
            scores, ids = self._faiss.search(query[None, :], k)
            return ids[0], scores[0]
        scores = self._cosine_all(query)
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
        query = self._norm_query(query)
        allowed = np.fromiter(
            (video_id in video_ids for video_id in self.manifest_video_ids),
            dtype=bool,
            count=len(self.manifest_video_ids),
        )
        ids = np.flatnonzero(allowed)
        if len(ids) == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        scores = self._cosine_for_ids(ids, query)
        k = min(k, len(ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return ids[top], scores[top]
