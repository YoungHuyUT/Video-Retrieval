"""Tests for per-modality score helpers used by the Adaptive Fusion final tier.

These run WITHOUT a retrieval index — they exercise the score math on small
synthetic candidate / record / transcript fixtures.
"""

from __future__ import annotations

from aic2026.ingestion.asr import ASRSegment, VideoTranscript
from aic2026.models import Candidate, FrameRecord
from aic2026.reranking.modality_scores import (
    asr_term_scores,
    object_coverage_scores,
)


def _rec(vector_id: int, video_id: str, labels: list[str], ts: float = 0.0) -> FrameRecord:
    return FrameRecord(
        video_id=video_id,
        frame_id=vector_id,
        vector_id=vector_id,
        keyframe_path=f"{video_id}/{vector_id}.jpg",
        timestamp=ts,
        object_labels=labels,
    )


def _cand(vector_id: int, video_id: str, score: float = 0.0) -> Candidate:
    return Candidate(
        video_id=video_id,
        frame_id=vector_id,
        score=score,
        vector_id=vector_id,
        keyframe_path=f"{video_id}/{vector_id}.jpg",
    )


def test_object_coverage_scores_matches_requested() -> None:
    # Query asks for a "bottle" (has VN alias "chai nước"); only frame 2 has it.
    records = {
        1: _rec(1, "V1", ["person", "car"]),
        2: _rec(2, "V1", ["bottle", "table"]),
        3: _rec(3, "V1", ["chair"]),
    }
    cands = [_cand(1, "V1"), _cand(2, "V1"), _cand(3, "V1")]
    scores = object_coverage_scores("chai nước", cands, records)
    # Only frame 2 mentions a requested object -> only it gets a positive score.
    assert set(scores.keys()) == {2}
    assert scores[2] == 1.0  # full coverage of the single requested concept


def test_object_coverage_empty_when_no_requested_object() -> None:
    records = {1: _rec(1, "V1", ["person"])}
    cands = [_cand(1, "V1")]
    # A pure scene query (no object concept) yields no object signal.
    assert object_coverage_scores("cảnh đường phố", cands, records) == {}


def test_asr_term_scores_overlap() -> None:
    transcript = VideoTranscript(
        video_id="V1",
        segments=[ASRSegment(text="welcome to the show", start=0.0, end=1.0)],
    )
    transcripts = {"V1": transcript}
    records = {1: _rec(1, "V1", []), 2: _rec(2, "V1", [])}
    cands = [_cand(1, "V1"), _cand(2, "V1")]
    # Query terms "the" and "show" appear in the transcript ("begins" does not)
    # -> 2 of 3 query terms matched -> ratio 2/3 for both frames (same video).
    scores = asr_term_scores("the show begins", cands, transcripts)
    assert set(scores.keys()) == {1, 2}
    assert abs(scores[1] - 2 / 3) < 1e-9


def test_asr_term_scores_empty_without_sidecar() -> None:
    records = {1: _rec(1, "V1", [])}
    cands = [_cand(1, "V1")]
    # No transcripts loaded -> asr modality unavailable, yields empty map.
    assert asr_term_scores("anything", cands, None) == {}
