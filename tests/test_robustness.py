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
    """Zero-score rows must not enter RRF as arbitrary lexical results.

    rank_bm25's BM25Okapi floors negative idf values to ``epsilon * average_idf``
    (default epsilon=0.25), and terms appearing in >= half the corpus get idf=0
    → score 0 on every document. On a tiny 2-doc corpus ("sea turtle" matching
    only "turtle") each term's df equals N/2 so idf collapses to 0 and the search
    returns nothing — that is the *documented* rank_bm25 behavior, not a bug.

    To exercise the "omit zero-score docs" contract we instead use a 4-doc corpus
    where "fish" appears in only 1 document (idf>0, non-zero score) and add a
    5th document whose labels do NOT contain "fish" at all — that doc must be
    absent from the results, proving zero-score rows are filtered out of RRF.
    """
    manifest = [
        FrameRecord(vector_id=0, video_id="V1", frame_id=1, keyframe_path="1.jpg", object_labels=["fish"]),
        FrameRecord(vector_id=1, video_id="V2", frame_id=1, keyframe_path="2.jpg", object_labels=["turtle"]),
        FrameRecord(vector_id=2, video_id="V3", frame_id=1, keyframe_path="3.jpg", object_labels=["turtle"]),
        FrameRecord(vector_id=3, video_id="V4", frame_id=1, keyframe_path="4.jpg", object_labels=["dog"]),
        # V5 has a lexical term but never "fish" -> score 0 -> omitted.
        FrameRecord(vector_id=4, video_id="V5", frame_id=1, keyframe_path="5.jpg", object_labels=["bird"]),
    ]
    index = BM25Index(manifest)

    ids, scores = index.search("fish", k=10)

    # Only V1 (which literally contains "fish") is returned; the zero-score docs
    # (including the distractor V5) are omitted before RRF fusion.
    assert ids.tolist() == [0]
    assert scores.size == 1
    assert scores[0] > 0


def test_bm25_single_term_tiny_corpus_returns_empty_due_to_epsilon() -> None:
    """Guard for rank_bm25's epsilon floor on a small corpus.

    When every query term appears in exactly N/2 documents, rank_bm25 computes
    idf=0 for each term, so every score is 0 and search() correctly returns an
    empty result — the terms are not omitted from the *index*, they simply have
    no discriminative IDF weight on a corpus this small.  This is expected
    behavior; callers must rely on the vector tier for such tiny pools.

    We need >= 4 documents so that BOTH query terms land at idf=0:
    "sea" appears in docs 0,1 (N/2 of 4) and "turtle" in docs 2,3 (also N/2).
    Each term must NOT co-occur so both have df == N/2 and idf == 0.
    """
    manifest = [
        FrameRecord(vector_id=0, video_id="V1", frame_id=1, keyframe_path="1.jpg", object_labels=["sea fish"]),
        FrameRecord(vector_id=1, video_id="V2", frame_id=1, keyframe_path="2.jpg", object_labels=["sea bird"]),
        FrameRecord(vector_id=2, video_id="V3", frame_id=1, keyframe_path="3.jpg", object_labels=["turtle shell"]),
        FrameRecord(vector_id=3, video_id="V4", frame_id=1, keyframe_path="4.jpg", object_labels=["turtle land"]),
    ]
    index = BM25Index(manifest)
    ids, scores = index.search("sea turtle", k=10)
    assert ids.size == 0
    assert scores.size == 0


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
