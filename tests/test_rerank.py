from __future__ import annotations

from aic2026.models import Candidate, FrameRecord
from aic2026.reranking import (
    normalize_scores,
    rerank_with_metadata,
    rerank_with_object_evidence,
)


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


def test_normalize_scores_rescales_to_unit_interval() -> None:
    candidates = [
        Candidate(video_id="V1", frame_id=1, score=0.016, vector_id=0),
        Candidate(video_id="V2", frame_id=2, score=0.006, vector_id=1),
    ]
    norm = normalize_scores(candidates)
    # max -> 1.0, others scaled proportionally (max-division, not min-max).
    assert norm[0] == 1.0
    assert abs(norm[1] - (0.006 / 0.016)) < 1e-9


def test_normalize_scores_degenerate_pool_is_zero() -> None:
    candidates = [Candidate(video_id="V3", frame_id=3, score=0.0, vector_id=9)]
    assert normalize_scores(candidates) == [0.0]


def test_rerank_boosts_metadata_matched_candidate() -> None:
    records = _records()
    candidates = [
        # V1 (0.5) would beat V2 (0.55) on raw RRF; metadata match flips it back up.
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
    assert reranked[0].score > reranked[1].score


def test_rerank_metadata_bonus_is_bounded_not_dominant() -> None:
    records = _records()
    # RRF scores (~0.0018–0.016) used to be overwhelmed by a 0.05 fixed bonus;
    # now the bonus is a bounded match ratio on a normalized [0,1] scale.
    candidates = [
        Candidate(video_id="V2", frame_id=2, score=0.55, vector_id=1),  # NO match
        Candidate(video_id="V1", frame_id=1, score=0.50, vector_id=0),  # match "cửa hàng"
    ]
    reranked = rerank_with_metadata(
        query="cửa hàng",
        candidates=candidates,
        records=records,
    )
    # Matched candidate wins despite a slightly lower raw retrieval score.
    assert reranked[0].vector_id == 0
    # Scores stay bounded near [0,1.2] — the bonus (ratio*weight) does not reach
    # the old ~3x-RRF magnitude; the unmatched frame keeps a high (normalized) score.
    assert 0.0 <= reranked[0].score <= 1.5
    unmatched = next(c for c in reranked if c.vector_id == 1)
    assert unmatched.score == 0.55  # raw retrieval score preserved without distortion


def test_rerank_no_metadata_stays_stable_order() -> None:
    candidates = [
        Candidate(video_id="V2", frame_id=2, score=0.55, vector_id=1),
        Candidate(video_id="V3", frame_id=3, score=0.40, vector_id=9),
    ]
    reranked = rerank_with_metadata(
        query="anything resident",
        candidates=candidates,
        records={},
    )
    # No metadata at all: no bonus applied, relative order preserved by normalized score.
    assert reranked[0].vector_id == 1
    assert reranked[1].vector_id == 9


def test_rerank_match_ratio_scales_with_coverage() -> None:
    records = {
        0: FrameRecord(
            vector_id=0, video_id="V1", frame_id=1, keyframe_path="V1/1.jpg", object_labels=["cat", "dog"]
        ),
        1: FrameRecord(
            vector_id=1, video_id="V2", frame_id=2, keyframe_path="V2/2.jpg", object_labels=["cat"]
        ),
    }
    candidates = [
        Candidate(video_id="V1", frame_id=1, score=0.5, vector_id=0),
        Candidate(video_id="V2", frame_id=2, score=0.5, vector_id=1),
    ]
    # Query with 2 terms; V1 matches both (ratio 1.0), V2 matches one (ratio 0.5).
    reranked = rerank_with_metadata(
        query="cat dog",
        candidates=candidates,
        records=records,
    )
    assert reranked[0].vector_id == 0
    assert reranked[1].vector_id == 1


def test_object_evidence_promotes_turtle_and_penalizes_fish() -> None:
    records = {
        0: FrameRecord(vector_id=0, video_id="V1", frame_id=1, keyframe_path="1.jpg", object_labels=["Fish"]),
        1: FrameRecord(vector_id=1, video_id="V2", frame_id=1, keyframe_path="2.jpg", object_labels=["Sea turtle", "Tortoise"]),
    }
    candidates = [
        Candidate(video_id="V1", frame_id=1, score=0.51, vector_id=0),
        Candidate(video_id="V2", frame_id=1, score=0.50, vector_id=1),
    ]

    reranked = rerank_with_object_evidence("sea turtle", candidates, records)

    assert [candidate.vector_id for candidate in reranked] == [1, 0]
    assert reranked[0].score > 0.50
    assert reranked[1].score < 0.51


def test_object_evidence_accepts_people_as_human() -> None:
    records = {
        0: FrameRecord(vector_id=0, video_id="V1", frame_id=1, keyframe_path="1.jpg", object_labels=["Human"]),
        1: FrameRecord(vector_id=1, video_id="V2", frame_id=1, keyframe_path="2.jpg", object_labels=["Fish"]),
    }
    candidates = [
        Candidate(video_id="V1", frame_id=1, score=0.50, vector_id=0),
        Candidate(video_id="V2", frame_id=1, score=0.50, vector_id=1),
    ]

    reranked = rerank_with_object_evidence("people walking", candidates, records)

    assert [candidate.vector_id for candidate in reranked] == [0, 1]
