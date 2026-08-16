from __future__ import annotations

from aic2026.models import Candidate
from aic2026.reranking.adaptive import (
    count_competing_videos,
    resolve_mode,
    video_rank_order,
)


def _c(video_id: str, score: float, frame_id: int = 1, vector_id: int = 0) -> Candidate:
    return Candidate(
        video_id=video_id,
        frame_id=frame_id,
        score=score,
        vector_id=vector_id,
    )


def test_count_competing_videos_counts_top_k_only() -> None:
    # 3 videos appear, but only 1 in the top-2 (k=2) -> competing == 1.
    cands = [
        _c("V1", 0.9, vector_id=0),
        _c("V1", 0.8, vector_id=1),
        _c("V2", 0.1, vector_id=2),
        _c("V3", 0.05, vector_id=3),
    ]
    assert count_competing_videos(cands, k=2) == 1
    assert count_competing_videos(cands, k=50) == 3


def test_resolve_mode_flat_for_single_video() -> None:
    cands = [_c("V1", 0.9, vector_id=i) for i in range(5)]
    assert resolve_mode(cands, k=50) == "flat"


def test_resolve_mode_soft_for_2to3_videos() -> None:
    soft = [
        _c("V1", 0.9, vector_id=0),
        _c("V2", 0.8, vector_id=1),
    ]
    soft3 = [
        _c("V1", 0.9, vector_id=0),
        _c("V2", 0.8, vector_id=1),
        _c("V3", 0.7, vector_id=2),
    ]
    assert resolve_mode(soft, k=50) == "soft"
    assert resolve_mode(soft3, k=50) == "soft"


def test_resolve_mode_full_for_4plus_videos() -> None:
    cands = [_c(f"V{i}", 1.0 - i * 0.1, vector_id=i) for i in range(4)]
    assert resolve_mode(cands, k=50) == "full"


def test_video_rank_order_logaddexp_aggregation() -> None:
    import numpy as np

    cands = [
        _c("V1", 0.9, vector_id=0),
        _c("V1", 0.5, vector_id=1),
        _c("V2", 0.8, vector_id=2),
    ]
    scores = video_rank_order(cands, aggregation_top_k=2)
    assert scores["V1"] == float(np.logaddexp.reduce([0.9, 0.5]))
    assert scores["V2"] == float(np.logaddexp.reduce([0.8]))
