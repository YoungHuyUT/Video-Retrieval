from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aic2026.models import FrameRecord
from aic2026.retrieval import ChromaVectorStore, RetrievalPipeline, VectorIndex


def _sample_manifest(n: int = 6, dim: int = 4) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for i in range(n):
        records.append(
            FrameRecord(
                vector_id=i,
                video_id=f"V{i % 2}",
                frame_id=i,
                keyframe_path=f"data/raw/Keyframes/V{i % 2}/{i:03d}.jpg",
                object_labels=["người", "bàn"] if i % 2 else ["người"],
                title=f"Video {i}",
                description=None,
            )
        )
    return records


def _sample_vectors(n: int = 6, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, dim)).astype(np.float32)


def test_search_returns_manifest_indices_matching_faiss(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    vectors = _sample_vectors()
    faiss_index = VectorIndex(vectors)
    chroma = ChromaVectorStore(vectors, manifest, persist_dir=tmp_path)

    query = _sample_vectors(1, 4)[0]
    faiss_ids, faiss_scores = faiss_index.search(query, 3)
    chroma_ids, chroma_scores = chroma.search(query, 3)

    assert set(chroma_ids.tolist()) == set(faiss_ids.tolist())
    # Chroma cosine similarity matches FAISS cosine within tolerance.
    assert np.allclose(chroma_scores, faiss_scores, atol=1e-3)


def test_chroma_id_maps_to_vector_id(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    chroma = ChromaVectorStore(_sample_vectors(), manifest, persist_dir=tmp_path)
    assert chroma.id_to_manifest_index["3"] == 3
    assert chroma.manifest_index_to_id[3] == "3"


def test_metadata_stored_json_safe(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    store = ChromaVectorStore(_sample_vectors(), manifest, persist_dir=tmp_path)
    result = store._collection.get(include=["metadatas"])
    meta = result["metadatas"][0]
    # object_labels is a JSON string, not a list (Chroma rejects lists).
    assert isinstance(meta["object_labels"], str)
    assert isinstance(meta["frame_id"], int)


def test_persistence_roundtrip(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    vectors = _sample_vectors()
    ChromaVectorStore(vectors, manifest, persist_dir=tmp_path)
    reopened = ChromaVectorStore.from_manifest(manifest, persist_dir=tmp_path)
    assert len(reopened) == len(manifest)
    query = _sample_vectors(1, 4)[0]
    assert len(reopened.search(query, 3)[0]) == 3


def test_available_false_without_chromadb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "chromadb" or name.startswith("chromadb."):
            raise ImportError("no chromadb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert ChromaVectorStore.available() is False
    with pytest.raises(ImportError):
        ChromaVectorStore(_sample_vectors(), _sample_manifest(), persist_dir=tmp_path)


def test_pipeline_with_chroma_store(tmp_path: Path) -> None:
    manifest = _sample_manifest(n=8)
    vectors = _sample_vectors(8)
    faiss_pipe = RetrievalPipeline(VectorIndex(vectors), manifest)
    chroma_pipe = RetrievalPipeline(ChromaVectorStore(vectors, manifest, persist_dir=tmp_path), manifest)

    query = _sample_vectors(1, 4)[0]
    faiss_raw = faiss_pipe.retrieve_raw(query, top_frames=4)
    chroma_raw = chroma_pipe.retrieve_raw(query, top_frames=4)
    assert [c.vector_id for c in chroma_raw] == [c.vector_id for c in faiss_raw]

    # TRAKE exercises fancy-index on index.vectors — must not raise with Chroma.
    trake_queries = np.vstack([query, _sample_vectors(1, 4)[0]])
    candidates = chroma_pipe.retrieve_trake(trake_queries, top_videos=2)
    assert len(candidates) >= 0


def test_size_mismatch_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ChromaVectorStore(_sample_vectors(5), _sample_manifest(6), persist_dir=tmp_path)
