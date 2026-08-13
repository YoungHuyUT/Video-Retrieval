from __future__ import annotations

from aic2026.models import Candidate, FrameRecord
from aic2026.reranking import rerank_with_metadata


def _records() -> dict[int, FrameRecord]:
    return {
        0: FrameRecord(
            vector_id=0,
            video_id="V1",
            frame_id=1,
            keyframe_path="V1/1.jpg",
            object_labels=["cửa hàng", "bàn"],
        ),
        1: FrameRecord(
            vector_id=1,
            video_id="V2",
            frame_id=2,
            keyframe_path="V2/2.jpg",
            object_labels=["đường phố"],
        ),
    }


def test_rerank_boosts_metadata_matched_candidate() -> None:
    records = _records()
    candidates = [
        Candidate(video_id="V1", frame_id=1, score=0.5, vector_id=0),
        Candidate(video_id="V2", frame_id=2, score=0.55, vector_id=1),
    ]
    reranked = rerank_with_metadata(
        query="cửa hàng",
        candidates=candidates,
        records=records,
    )
    # V1 object_labels contain "cửa hàng" (query term) -> boosted above V2.
    assert reranked[0].vector_id == 0
    assert reranked[0].score > 0.5


def test_rerank_leaves_unmatched_scores_unchanged() -> None:
    records = _records()
    candidates = [
        Candidate(video_id="V2", frame_id=2, score=0.55, vector_id=1),
    ]
    reranked = rerank_with_metadata(
        query="cửa hàng",
        candidates=candidates,
        records=records,
    )
    # V2 has no "cửa hàng" term -> score untouched.
    assert reranked[0].score == 0.55


def test_rerank_no_metadata_stays_stable() -> None:
    candidates = [
        Candidate(video_id="V3", frame_id=3, score=0.4, vector_id=9),
    ]
    reranked = rerank_with_metadata(
        query="anything resident",
        candidates=candidates,
        records={},
    )
    assert reranked[0].score == 0.4