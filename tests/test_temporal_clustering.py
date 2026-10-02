"""Unit tests for Frame→Moment clustering (spec §9, §10, §11)."""

from __future__ import annotations

import numpy as np
import pytest

from aic2026.models import Candidate, FrameRecord
from aic2026.retrieval.temporal_clustering import (
    Moment,
    best_frame_from_moment,
    cluster_frames_to_moments,
    rerank_moments_to_frames,
)


def _rec(ts: float, shot: str | None = None) -> FrameRecord:
    return FrameRecord(
        vector_id=0,
        video_id="V",
        frame_id=0,
        keyframe_path="",
        object_labels=[],
        metadata_keywords=[],
        timestamp=ts,
        shot_id=shot,
    )


def _cand(video_id: str, frame_id: int, score: float, ts: float) -> Candidate:
    return Candidate(
        video_id=video_id, frame_id=frame_id, score=score,
        vector_id=frame_id, keyframe_path="", timestamp=ts,
    )


def test_clusters_by_time_gap():
    cands = [
        _cand("V", 0, 0.9, 1.0),
        _cand("V", 1, 0.8, 2.0),   # within gap → same moment
        _cand("V", 2, 0.7, 10.0),  # far → new moment
    ]
    records = {c.vector_id: _rec(c.timestamp) for c in cands}
    moments = cluster_frames_to_moments(cands, records=records, gap_seconds=3.0)
    assert len(moments) == 2
    # First (earlier) moment should score higher (max aggregation).
    assert moments[0].score == pytest.approx(0.9)
    assert moments[0].start_time == 1.0
    assert moments[0].end_time == 2.0


def test_clusters_by_shot_id():
    cands = [
        _cand("V", 0, 0.9, 1.0),
        _cand("V", 1, 0.8, 2.0),   # same shot → same moment
        _cand("V", 2, 0.7, 6.0),   # new shot + gap > 3s → new moment
    ]
    # shot_id lives on the FrameRecord (passed via records), not the Candidate.
    # It is an int field on FrameRecord, so supply ints.
    records = {
        c.vector_id: _rec(c.timestamp, shot=(1 if c.frame_id < 2 else 2))
        for c in cands
    }
    moments = cluster_frames_to_moments(
        cands, records=records, gap_seconds=3.0, use_shot_id=True
    )
    assert len(moments) == 2


def test_separates_videos():
    cands = [
        _cand("V1", 0, 0.9, 1.0),
        _cand("V1", 1, 0.8, 2.0),
        _cand("V2", 2, 0.7, 1.5),
    ]
    records = {c.vector_id: _rec(c.timestamp) for c in cands}
    moments = cluster_frames_to_moments(cands, records=records, gap_seconds=3.0)
    vids = {m.video_id for m in moments}
    assert vids == {"V1", "V2"}


def test_best_frame_from_moment():
    m = Moment(
        video_id="V",
        start_time=1.0,
        end_time=3.0,
        representative_frames=[0, 1, 2],
        frame_scores=[0.1, 0.9, 0.3],
        vector_ids=[0, 1, 2],
        # best_frame_from_moment returns the aggregated moment score.
        score=0.95,
    )
    best = best_frame_from_moment(m)
    assert best.frame_id == 1               # highest-scoring member frame
    assert best.score == pytest.approx(0.95)  # aggregated moment score


def test_rerank_moments_to_frames_returns_one_per_moment():
    cands = [
        _cand("V1", 0, 0.9, 1.0),
        _cand("V1", 1, 0.8, 2.0),
        _cand("V2", 2, 0.7, 1.5),
    ]
    records = {c.vector_id: _rec(c.timestamp) for c in cands}
    frames = rerank_moments_to_frames(cands, records=records, gap_seconds=3.0)
    # Two moments → two representative frames (best per moment).
    assert len(frames) == 2
    # Sorted by moment score (best moment first).
    assert frames[0].score >= frames[1].score


def test_empty_input():
    assert cluster_frames_to_moments([]) == []
