from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np


_NORM_CHUNK_ROWS = 8192
_logger = logging.getLogger(__name__)


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

    When an *in-RAM* (writable) array is passed — the normal ``backend=faiss``
    path — FAISS is built.  Use ``backend=numpy`` when memory pressure is more
    important than query latency.
    """

    def __init__(self, vectors: np.ndarray, *, assume_normalized: bool = False):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.ndim != 2 or not len(self.vectors):
            raise ValueError("vectors must be a non-empty 2D array")
        self.manifest_video_ids: list[str] = []
        self._source_path: Path | None = None
        self._faiss_approx = False
        self._ann_candidate_factor = 5
        # Precompute video_id → row indices mapping for O(1) filtered search.
        # Avoids rebuilding a boolean mask from 177k Python string lookups per query.
        self._video_id_to_indices: dict[str, np.ndarray] = {}
        # Only build FAISS when the matrix is a normal writable (in-RAM) array.
        # A memory-mapped (read-only) matrix would be copied into FAISS's own
        # buffer, negating the RAM saving, so we skip it and use NumPy cosine
        # (computed per-row from a tiny precomputed norms array instead).
        if self.vectors.flags.writeable:
            if not assume_normalized:
                # Keep the normalization temporary bounded instead of allocating
                # another full-size matrix (1.14 GiB for the merged corpus).
                for start in range(0, len(self.vectors), _NORM_CHUNK_ROWS):
                    block = self.vectors[start : start + _NORM_CHUNK_ROWS]
                    norms = np.linalg.norm(block, axis=1, keepdims=True)
                    block /= np.maximum(norms, 1e-12)
            try:
                import faiss  # type: ignore

                self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
                self._faiss.add(self.vectors)
            except ImportError:
                self._faiss = None
            self._norms: np.ndarray | None = None
        else:
            self._faiss = None
            if assume_normalized:
                self._norms = None
            else:
                # Compute per-row norms in bounded chunks; never materialize
                # a full-size temporary copy of the memory-mapped matrix.
                self._norms = np.empty(len(self.vectors), dtype=np.float32)
                for start in range(0, len(self.vectors), _NORM_CHUNK_ROWS):
                    end = min(start + _NORM_CHUNK_ROWS, len(self.vectors))
                    self._norms[start:end] = np.linalg.norm(
                        self.vectors[start:end], axis=1
                    )

    @classmethod
    def from_npy(
        cls, path: Path, mmap: bool = True, *, assume_normalized: bool = False
    ) -> VectorIndex:
        """Load a feature ``.npy`` matrix.

        ``mmap=True`` (default) memory-maps the file read-only — the 363 MB
        competition matrix stays on disk and is paged in by the OS only for the
        rows a query actually touches. ``mmap=False`` loads the whole matrix into
        RAM (legacy behaviour, used by tests / small datasets).
        """
        if mmap:
            index = cls(np.load(path, mmap_mode="r"), assume_normalized=assume_normalized)
            index._source_path = Path(path).resolve()
            return index
        return cls(np.load(path), assume_normalized=assume_normalized)

    def enable_ivfpq(
        self,
        cache_path: Path,
        *,
        nlist: int = 512,
        m: int = 64,
        nbits: int = 8,
        nprobe: int = 48,
        train_rows: int = 32768,
        add_batch_rows: int = 16384,
    ) -> bool:
        """Attach a cached, compressed FAISS IVF-PQ index for fast mmap search.

        The PQ index is only about 25 MB for the current 397k x 768 corpus. It
        avoids copying the full 1.2 GB feature matrix into RAM. Search returns
        an oversampled approximate pool, then re-scores that pool against the
        original mmap vectors for accurate ordering.
        """
        if len(self.vectors) < 20000:
            return False
        if self.vectors.shape[1] % m:
            raise ValueError(f"Embedding dimension {self.vectors.shape[1]} is not divisible by PQ m={m}")

        try:
            import faiss  # type: ignore
        except ImportError:
            _logger.warning("FAISS is unavailable; retaining exact mmap scan")
            return False

        cache_path = Path(cache_path)
        metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        nlist = min(nlist, max(32, int(np.sqrt(len(self.vectors)))))
        source_path = self._source_path
        source_stat = source_path.stat() if source_path is not None else None
        signature = {
            "format": "ivfpq-v1",
            "shape": list(self.vectors.shape),
            "source": str(source_path.resolve()) if source_path else None,
            "source_size": source_stat.st_size if source_stat else None,
            "source_mtime_ns": source_stat.st_mtime_ns if source_stat else None,
            "nlist": nlist,
            "m": m,
            "nbits": nbits,
            "nprobe": nprobe,
        }

        if cache_path.exists() and metadata_path.exists():
            try:
                if json.loads(metadata_path.read_text(encoding="utf-8")) == signature:
                    self._faiss = faiss.read_index(str(cache_path))
                    self._faiss.nprobe = min(nprobe, nlist)
                    self._faiss_approx = True
                    _logger.info("Loaded cached FAISS IVF-PQ index from %s", cache_path)
                    return True
            except Exception:
                _logger.warning("Ignoring invalid cached IVF-PQ index %s", cache_path, exc_info=True)

        import time

        started = time.perf_counter()
        faiss.omp_set_num_threads(min(8, os.cpu_count() or 4))
        quantizer = faiss.IndexFlatIP(self.vectors.shape[1])
        index = faiss.IndexIVFPQ(
            quantizer,
            self.vectors.shape[1],
            nlist,
            m,
            nbits,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.nprobe = min(nprobe, nlist)

        rng = np.random.default_rng(20260925)
        sample_ids = rng.choice(
            len(self.vectors), size=min(train_rows, len(self.vectors)), replace=False
        )
        training = np.array(self.vectors[sample_ids], dtype=np.float32, copy=True)
        if self._norms is not None:
            training /= np.maximum(self._norms[sample_ids, None], 1e-12)
        _logger.info(
            "Training compressed FAISS index (rows=%d, dim=%d, nlist=%d, PQ=%d x %d-bit)",
            len(self.vectors), self.vectors.shape[1], nlist, m, nbits,
        )
        index.train(training)
        del training, sample_ids

        for start in range(0, len(self.vectors), add_batch_rows):
            end = min(start + add_batch_rows, len(self.vectors))
            block = np.array(self.vectors[start:end], dtype=np.float32, copy=True)
            if self._norms is not None:
                block /= np.maximum(self._norms[start:end, None], 1e-12)
            index.add(block)
            del block

        temp_index = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        temp_meta = metadata_path.with_name(f"{metadata_path.name}.{os.getpid()}.tmp")
        try:
            faiss.write_index(index, str(temp_index))
            temp_meta.write_text(json.dumps(signature, sort_keys=True), encoding="utf-8")
            os.replace(temp_index, cache_path)
            os.replace(temp_meta, metadata_path)
        finally:
            for temp in (temp_index, temp_meta):
                if temp.exists():
                    temp.unlink()

        self._faiss = index
        self._faiss_approx = True
        _logger.info(
            "FAISS IVF-PQ index ready: %d rows, %.1f MiB, built in %.1fs",
            index.ntotal,
            cache_path.stat().st_size / (1024 * 1024),
            time.perf_counter() - started,
        )
        return True

    def _search_ivfpq(self, queries: np.ndarray, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """ANN candidate generation followed by exact scoring from the mmap."""
        candidate_k = min(len(self.vectors), max(k, k * self._ann_candidate_factor, 1000))
        _approx_scores, ann_ids = self._faiss.search(queries, candidate_k)
        results: list[tuple[np.ndarray, np.ndarray]] = []
        for query, ids in zip(queries, ann_ids):
            ids = ids[ids >= 0]
            if not len(ids):
                results.append((np.array([], dtype=np.int64), np.array([], dtype=np.float32)))
                continue
            scores = self._cosine_for_ids(ids, query)
            keep = min(k, len(ids))
            top = np.argpartition(-scores, keep - 1)[:keep]
            top = top[np.argsort(-scores[top])]
            results.append((ids[top], scores[top].astype(np.float32, copy=False)))
        return results

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
        if self._faiss_approx:
            return self._search_ivfpq(query[None, :], k)[0]
        if self._faiss is not None:
            scores, ids = self._faiss.search(query[None, :], k)
            return ids[0], scores[0]
        scores = self._cosine_all(query)
        k = min(k, len(scores))
        ids = np.argpartition(-scores, k - 1)[:k]
        ids = ids[np.argsort(-scores[ids])]
        return ids, scores[ids]

    def search_many(
        self, queries: np.ndarray, k: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Batch nearest-neighbour search for parallel query variants.

        KIS action chains use several related text embeddings.  Calling FAISS
        once with a ``B × D`` matrix removes Python/FAISS call overhead and lets
        its BLAS kernel reuse the corpus cache.  The NumPy fallback likewise
        performs one matrix multiplication instead of B separate scans.
        """
        matrix = np.array(queries, dtype=np.float32, copy=True)
        if matrix.ndim != 2 or matrix.shape[1] != self.vectors.shape[1]:
            raise ValueError("queries must be a 2D matrix matching embedding dimension")
        if len(matrix) == 0 or k <= 0:
            return []
        matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        k = min(int(k), len(self.vectors))
        if self._faiss_approx:
            return self._search_ivfpq(matrix, k)
        if self._faiss is not None:
            scores, ids = self._faiss.search(matrix, k)
            return [(ids[i], scores[i]) for i in range(len(matrix))]
        scores = self.vectors @ matrix.T
        if self._norms is not None:
            scores = scores / np.maximum(self._norms[:, None], 1e-12)
        result: list[tuple[np.ndarray, np.ndarray]] = []
        for col in range(matrix.shape[0]):
            column = scores[:, col]
            ids = np.argpartition(-column, k - 1)[:k]
            ids = ids[np.argsort(-column[ids])]
            result.append((ids, column[ids]))
        return result

    def _ensure_video_index(self) -> None:
        """Build video_id → row-indices mapping (once, on first filtered search)."""
        if self._video_id_to_indices or not self.manifest_video_ids:
            return
        from collections import defaultdict
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, vid in enumerate(self.manifest_video_ids):
            groups[vid].append(idx)
        self._video_id_to_indices = {
            vid: np.asarray(indices, dtype=np.intp)
            for vid, indices in groups.items()
        }

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
        self._ensure_video_index()
        query = self._norm_query(query)
        # O(V) union of precomputed index arrays — avoids 177k Python string lookups.
        allowed_indices = np.concatenate([
            self._video_id_to_indices[vid]
            for vid in video_ids
            if vid in self._video_id_to_indices
        ]) if self._video_id_to_indices else np.array([], dtype=np.intp)
        if len(allowed_indices) == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        scores = self._cosine_for_ids(allowed_indices, query)
        k = min(k, len(allowed_indices))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return allowed_indices[top], scores[top]

    def search_many_filtered(
        self,
        queries: np.ndarray,
        k: int,
        video_ids: set[str] | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Batch vector search restricted to *video_ids*.

        Uses ``search_many`` for a single matrix multiply (FAISS batch or
        NumPy matmul), then masks each result set to ``allowed_indices``.
        Falls back to per-query ``search_filtered`` when no filter is needed.
        """
        if not video_ids:
            return self.search_many(queries, k)
        if len(self.manifest_video_ids) != len(self.vectors):
            raise ValueError("manifest_video_ids must align with the vector index")
        self._ensure_video_index()
        matrix = np.asarray(queries, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.vectors.shape[1]:
            raise ValueError("queries must be a 2D matrix matching embedding dimension")
        if len(matrix) == 0 or k <= 0:
            return []
        # Build allowed index set once for all queries.
        allowed_indices = np.concatenate([
            self._video_id_to_indices[vid]
            for vid in video_ids
            if vid in self._video_id_to_indices
        ]) if self._video_id_to_indices else np.array([], dtype=np.intp)
        if len(allowed_indices) == 0:
            return [(np.array([], dtype=int), np.array([], dtype=np.float32))] * len(matrix)
        # Score all queries against allowed vectors in one matrix multiply.
        matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        allowed_vectors = self.vectors[allowed_indices]
        if self._norms is not None:
            allowed_norms = self._norms[allowed_indices]
            scores_all = (allowed_vectors @ matrix.T) / np.maximum(allowed_norms[:, None], 1e-12)
        else:
            scores_all = allowed_vectors @ matrix.T  # [N_allowed, B]
        result: list[tuple[np.ndarray, np.ndarray]] = []
        k_eff = min(k, len(allowed_indices))
        for col in range(matrix.shape[0]):
            col_scores = scores_all[:, col]
            top = np.argpartition(-col_scores, k_eff - 1)[:k_eff]
            top = top[np.argsort(-col_scores[top])]
            result.append((allowed_indices[top], col_scores[top]))
        return result

    def search_filtered_indices(
        self,
        query: np.ndarray,
        k: int,
        row_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact cosine top-k within a prefiltered set of frame rows."""
        ids = np.asarray(row_indices, dtype=np.intp)
        if ids.size == 0 or k <= 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)
        query = self._norm_query(query)
        scores = self._cosine_for_ids(ids, query)
        take = min(int(k), len(ids))
        top = np.argpartition(-scores, take - 1)[:take]
        top = top[np.argsort(-scores[top])]
        return ids[top], scores[top].astype(np.float32, copy=False)

    def search_many_filtered_indices(
        self,
        queries: np.ndarray,
        k: int,
        row_indices: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Batch exact cosine search within object-filtered frame rows."""
        matrix = np.array(queries, dtype=np.float32, copy=True)
        if matrix.ndim != 2 or matrix.shape[1] != self.vectors.shape[1]:
            raise ValueError("queries must be a 2D matrix matching embedding dimension")
        if len(matrix) == 0 or k <= 0:
            return []
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        return [
            self.search_filtered_indices(query, k, row_indices)
            for query in matrix
        ]
