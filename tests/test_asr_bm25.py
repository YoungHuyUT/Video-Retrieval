"""Tests for Phase 4 ASR ingestion + BM25 ASR-sidecar fusion (ASR-first per user).

ASR transcription itself requires faster-whisper (offline ingestion) and is NOT
exercised here — these tests cover the data model, sidecar round-trip, and the
BM25 merge path, all of which run without any heavy model.
"""

from __future__ import annotations

from pathlib import Path

from aic2026.ingestion.asr import (
    ASRSegment,
    VideoTranscript,
    load_transcripts_sidecar,
    save_transcripts_sidecar,
)
from aic2026.models import FrameRecord
from aic2026.retrieval import BM25Index


def _tmp_sidecar(tmp_path: Path) -> Path:
    trs = [
        VideoTranscript(
            video_id="L21_V001",
            segments=[
                ASRSegment(text="a woman says remember", start=1.0, end=2.5),
                ASRSegment(text="the dog runs", start=3.0, end=4.0),
            ],
        ),
        VideoTranscript(
            video_id="L21_V002",
            segments=[ASRSegment(text="the cat sleeps", start=0.5, end=1.5)],
        ),
    ]
    sidecar = tmp_path / "asr.jsonl"
    save_transcripts_sidecar(trs, sidecar)
    return sidecar


def test_sidecar_roundtrip(tmp_path: Path):
    sidecar = _tmp_sidecar(tmp_path)
    loaded = load_transcripts_sidecar(sidecar)
    assert set(loaded) == {"L21_V001", "L21_V002"}
    assert "remember" in loaded["L21_V001"].full_text
    assert loaded["L21_V002"].segments[0].text == "the cat sleeps"


def test_load_missing_sidecar_is_empty(tmp_path: Path):
    assert load_transcripts_sidecar(tmp_path / "nope.jsonl") == {}


def _multi_manifest() -> list[FrameRecord]:
    # Multiple videos so BM25 IDF is positive (single-doc corpus yields negative
    # IDF, which is an edge case handled by the >0 score filter at runtime).
    return [
        FrameRecord(video_id="L21_V001", frame_id=0, vector_id=0,
                    keyframe_path="data/keyframes/L21_V001/0.jpg"),
        FrameRecord(video_id="L21_V002", frame_id=0, vector_id=1,
                    keyframe_path="data/keyframes/L21_V002/0.jpg"),
        FrameRecord(video_id="L21_V003", frame_id=0, vector_id=2,
                    keyframe_path="data/keyframes/L21_V003/0.jpg"),
    ]


def test_bm25_from_asr_sidecar_matches_spoken_keyword(tmp_path: Path):
    sidecar = _tmp_sidecar(tmp_path)
    manifest = _multi_manifest()
    idx = BM25Index.from_asr_sidecar(manifest, sidecar)
    assert not idx.is_empty
    ids, scores = idx.search("remember", k=5)
    assert len(ids) >= 1
    # L21_V001 contains "remember"; its manifest position is 0.
    assert int(ids[0]) == 0


def test_bm25_asr_distinct_from_objects_only(tmp_path: Path):
    sidecar = _tmp_sidecar(tmp_path)
    manifest = [
        FrameRecord(video_id="L21_V001", frame_id=0, vector_id=0,
                    keyframe_path="x", object_labels=["dog", "beach"]),
        FrameRecord(video_id="L21_V002", frame_id=0, vector_id=1,
                    keyframe_path="y", object_labels=["cat"]),
        FrameRecord(video_id="L21_V003", frame_id=0, vector_id=2,
                    keyframe_path="z", object_labels=["car"]),
    ]
    idx = BM25Index.from_asr_sidecar(manifest, sidecar)
    # Spoken keyword "remember" should still match V001 (ASR branch), not just
    # the object_labels branch.
    ids, scores = idx.search("remember", k=5)
    assert int(ids[0]) == 0
    # Object branch also intact (V001 has "beach" among its objects).
    ids2, _ = idx.search("beach", k=5)
    assert int(ids2[0]) == 0
