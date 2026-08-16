from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

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

    ``backend="faiss"`` (the default) uses the supplied CLIP matrix directly.
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
    return VectorIndex.from_npy(features)


def load_index_for_query(
    features: Path,
    manifest: list[FrameRecord],
    backend: str = "faiss",
    chroma_dir: str | Path = "data/indexes/chroma",
    collection_name: str = "aic2026_frames",
) -> VectorIndex | ChromaVectorStore:
    """Query-time loader for the supplied CLIP ``.npy`` matrix."""
    resolved = _resolve_backend(backend, chroma_dir, collection_name, manifest)
    if resolved == "chroma":
        return ChromaVectorStore.from_manifest(
            manifest=manifest,
            persist_dir=chroma_dir,
            collection_name=collection_name,
        )
    return VectorIndex.from_npy(features)
