from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aic2026.models import FrameRecord
from aic2026.retrieval import ChromaVectorStore, VectorIndex
from aic2026.retrieval.factory import (
    build_vector_index,
    load_index_for_query,
)


def _sample_manifest(n: int = 6) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for i in range(n):
        records.append(
            FrameRecord(
                vector_id=i,
                video_id=f"V{i % 2}",
                frame_id=i,
                keyframe_path=f"data/raw/Keyframes/V{i % 2}/{i:03d}.jpg",
            )
        )
    return records


def _write_features(tmp_path: Path, n: int = 6, dim: int = 4) -> Path:
    rng = np.random.default_rng(1)
    path = tmp_path / "features.npy"
    np.save(path, rng.normal(size=(n, dim)).astype(np.float32))
    return path


def test_auto_falls_back_to_faiss_without_chromadb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    features = _write_features(tmp_path)
    manifest = _sample_manifest()
    monkeypatch.setattr(ChromaVectorStore, "available", staticmethod(lambda: False))
    index = load_index_for_query(features, manifest, backend="auto", chroma_dir=tmp_path / "chroma")
    assert isinstance(index, VectorIndex)


def test_auto_prefers_chroma_when_collection_exists(tmp_path: Path) -> None:
    features = _write_features(tmp_path)
    manifest = _sample_manifest()
    # Build the collection first.
    build_vector_index(
        features,
        manifest,
        backend="chroma",
        chroma_dir=tmp_path / "chroma",
    )
    index = load_index_for_query(features, manifest, backend="auto", chroma_dir=tmp_path / "chroma")
    assert isinstance(index, ChromaVectorStore)


def test_build_vector_index_raises_for_unknown_backend(tmp_path: Path) -> None:
    features = _write_features(tmp_path)
    manifest = _sample_manifest()
    with pytest.raises(ValueError):
        build_vector_index(features, manifest, backend="bogus")


def test_explicit_faiss_backend(tmp_path: Path) -> None:
    features = _write_features(tmp_path)
    manifest = _sample_manifest()
    index = load_index_for_query(features, manifest, backend="faiss", chroma_dir=tmp_path / "chroma")
    assert isinstance(index, VectorIndex)
