from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np

from aic2026.models import FrameRecord

from .index import VectorIndex
from .vectordb import ChromaVectorStore

logger = logging.getLogger(__name__)

# The competition corpus is static: a local FAISS index over the supplied CLIP
# matrix is both faster and simpler than a database-backed vector store.
Backend = Literal["auto", "chroma", "faiss", "numpy"]


def _resolve_backend(
    backend: str,
    chroma_dir: str | Path,
    collection_name: str,
    manifest: list[FrameRecord],
) -> str:
    """Resolve ``auto`` to the lightweight local index.

    Chroma remains available when explicitly requested, but automatic selection
    must not open or validate a persistent collection on every process start.
    """
    if backend == "auto":
        return "faiss"
    if backend in ("chroma", "faiss", "numpy"):
        return backend
    raise ValueError(
        f"Unknown index backend '{backend}'. "
        "Expected one of: auto, chroma, faiss, numpy."
    )


def build_vector_index(
    features: Path,
    manifest: list[FrameRecord],
    backend: str = "faiss",
    chroma_dir: str | Path = "data/indexes/chroma",
    collection_name: str = "aic2026_frames",
    space: str = "cosine",
) -> VectorIndex | ChromaVectorStore:
    """Build an index backend from a feature ``.npy`` + manifest.

    ``backend="faiss"`` loads the matrix into RAM so ``VectorIndex`` can build
    an actual ``IndexFlatIP``.  ``backend="numpy"`` is the explicit low-memory
    mmap/full-scan mode.  Previously both paths used mmap, so the advertised
    FAISS backend silently performed an O(N) NumPy scan for every query.
    """
    resolved = _resolve_backend(backend, chroma_dir, collection_name, manifest)
    if resolved == "chroma":
        return ChromaVectorStore.from_npy(
            features=features,
            manifest=manifest,
            persist_dir=chroma_dir,
            collection_name=collection_name,
            space=space,
        )
    return VectorIndex.from_npy(features, mmap=(resolved == "numpy"))


def load_index_for_query(
    features: Path,
    manifest: list[FrameRecord],
    backend: str = "faiss",
    chroma_dir: str | Path = "data/indexes/chroma",
    collection_name: str = "aic2026_frames",
    features_normalized: bool = False,
    ann_index_path: Path | None = None,
) -> VectorIndex | ChromaVectorStore:
    """Query-time loader for the supplied CLIP ``.npy`` matrix.

    Fails fast if the feature matrix row count does not match the manifest
    length. ``vector_id`` is the manifest line index, so a stale ``.npy``
    (regenerated without updating the manifest) would silently map every
    candidate to the wrong frame — a silent, hard-to-debug retrieval error.
    """
    resolved = _resolve_backend(backend, chroma_dir, collection_name, manifest)
    if resolved == "chroma":
        return ChromaVectorStore.from_manifest(
            manifest=manifest,
            persist_dir=chroma_dir,
            collection_name=collection_name,
        )
    # Verify alignment BEFORE building the index so a mismatch surfaces here
    # rather than as wrong results at query time.
    try:
        n_rows = int(np.load(features, mmap_mode="r").shape[0])
    except Exception as exc:
        raise ValueError(f"Cannot read feature matrix {features}: {exc}") from exc
    if n_rows != len(manifest):
        raise ValueError(
            f"Feature matrix has {n_rows} rows but manifest has {len(manifest)} "
            f"records — they must align 1:1 (vector_id == row index). A stale "
            f"'.npy' or a regenerated manifest usually causes this; rebuild both "
            f"from the same source with `prepare-official`."
        )
    index = VectorIndex.from_npy(
        features,
        mmap=(resolved == "numpy"),
        assume_normalized=features_normalized,
    )
    if ann_index_path is not None and isinstance(index, VectorIndex):
        try:
            index.enable_ivfpq(ann_index_path)
        except Exception:
            logger.exception(
                "Could not build/load compressed ANN index; falling back to exact vector scan"
            )
    return index
