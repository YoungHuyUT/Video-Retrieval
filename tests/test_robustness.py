from __future__ import annotations

from pathlib import Path

import numpy as np

from aic2026.models import FrameRecord
from aic2026.retrieval import BM25Index


def _empty_text_manifest(n: int = 4) -> list[FrameRecord]:
    """Manifest with NO text metadata — every record tokenizes to []."""
    return [
        FrameRecord(
            vector_id=i,
            video_id=f"V{i}",
            frame_id=i,
            keyframe_path=f"V{i}/{i:03d}.jpg",
        )
        for i in range(n)
    ]


def test_bm25_empty_corpus_does_not_crash() -> None:
    """BM25Okapi raises ZeroDivisionError when every document is empty;
    our degenerate index must instead return no lexical evidence."""
    index = BM25Index(_empty_text_manifest())
    ids, scores = index.search("người nói chuyện", k=3)
    assert ids.size == 0
    assert scores.size == 0


def test_bm25_omits_documents_without_a_lexical_match() -> None:
    """Zero-score rows must not enter RRF as arbitrary lexical results."""
    manifest = [
        FrameRecord(vector_id=0, video_id="V1", frame_id=1, keyframe_path="1.jpg", object_labels=["fish"]),
        FrameRecord(vector_id=1, video_id="V2", frame_id=1, keyframe_path="2.jpg", object_labels=["turtle"]),
    ]
    index = BM25Index(manifest)

    ids, scores = index.search("sea turtle", k=10)

    assert ids.tolist() == [1]
    assert scores.size == 1


def test_bm25_skips_corrupt_archive_in_build(
    tmp_path: Path,
) -> None:
    """build_derived_artifacts must skip a 0-byte .npz instead of crashing."""
    from aic2026.data_platform.video_frames import build_derived_artifacts

    features_root = tmp_path / "features"
    keyframes_root = tmp_path / "keyframes"
    features_root.mkdir(parents=True)
    (keyframes_root / "L01_V001").mkdir(parents=True)

    # Valid archive for one video.
    np.savez_compressed(
        features_root / "L01_V001.npz",
        frame_ids=np.asarray([1, 2], dtype=np.int64),
        features=np.ones((2, 4), dtype=np.float32),
    )
    # Corrupt 0-byte leftover (a crashed run used to write this).
    (features_root / "feature.npz").write_bytes(b"")
    for i in (1, 2):
        (keyframes_root / "L01_V001" / f"{i:09d}.jpg").write_bytes(b"x")

    count = build_derived_artifacts(
        keyframes_root,
        features_root,
        tmp_path / "manifest.jsonl",
        tmp_path / "features.npy",
    )
    assert count == 2
