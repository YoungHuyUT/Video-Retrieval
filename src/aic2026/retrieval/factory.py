from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from aic2026.models import FrameRecord

from .index import VectorIndex
from .vectordb import ChromaVectorStore

logger = logging.getLogger(__name__)

Backend = Literal["auto", "chroma", "faiss", "numpy"]


def _resolve_backend(
    backend: str,
    chroma_dir: str | Path,
    collection_name: str,
    manifest: list[FrameRecord],
) -> str:
    """Resolve ``auto`` to an explicit backend based on what is available."""
    if backend == "auto":
        try:
            store = ChromaVectorStore.from_manifest(
                manifest=manifest,
                persist_dir=chroma_dir,
                collection_name=collection_name,
            )
            if store.collection_count > 0 and store.collection_count == len(manifest):
                return "chroma"
            logger.warning(
                "Chroma collection size %d does not match manifest (%d); "
                "falling back to FAISS/numpy",
                store.collection_count,
                len(manifest),
            )
        except Exception as exc:  # noqa: BLE001 — auto mode must fall back safely
            logger.debug("Chroma unavailable for auto backend: %s", exc)
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
    backend: str = "auto",
    chroma_dir: str | Path = "data/indexes/chroma",
    collection_name: str = "aic2026_frames",
    space: str = "cosine",
) -> VectorIndex | ChromaVectorStore:
    """Build an index backend from a feature ``.npy`` + manifest.

    ``backend="auto"`` prefers an existing Chroma collection and otherwise falls
    back to FAISS/numpy via ``VectorIndex``.
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
    backend: str = "auto",
    chroma_dir: str | Path = "data/indexes/chroma",
    collection_name: str = "aic2026_frames",
) -> VectorIndex | ChromaVectorStore:
    """Query-time loader: reuse an existing Chroma collection when present,
    else load the FAISS/numpy index from the ``.npy`` file."""
    resolved = _resolve_backend(backend, chroma_dir, collection_name, manifest)
    if resolved == "chroma":
        return ChromaVectorStore.from_manifest(
            manifest=manifest,
            persist_dir=chroma_dir,
            collection_name=collection_name,
        )
    return VectorIndex.from_npy(features)
