from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aic2026.data_platform.asr import ASRExtractor
from aic2026.ingestion.asr_manifest import enrich_manifest_with_asr
from aic2026.models import Candidate, FrameRecord
from aic2026.reranking.lexical import rerank_with_metadata
from aic2026.retrieval.bm25_index import BM25Index


def test_frame_record_asr_fields() -> None:
    rec = FrameRecord(
        vector_id=0,
        video_id="L21_V001",
        frame_id=100,
        keyframe_path="L21_V001/100.jpg",
    )
    assert rec.asr_text == []
    assert rec.asr_done is False

    rec2 = FrameRecord(
        vector_id=1,
        video_id="L21_V001",
        frame_id=200,
        keyframe_path="L21_V001/200.jpg",
        asr_text=["bản tin 60 giây", "tai nạn giao thông"],
        asr_done=True,
    )
    assert len(rec2.asr_text) == 2
    assert rec2.asr_done is True


def test_asr_extractor_fallback_when_unavailable() -> None:
    with patch.dict("sys.modules", {"faster_whisper": None}):
        extractor = ASRExtractor(model_size="tiny")
        assert extractor.available is False
        assert extractor.transcribe_video("fake.mp4") == []


def test_asr_extractor_transcribe_directory(tmp_path: Path) -> None:
    extractor = ASRExtractor(model_size="tiny")
    extractor._model = MagicMock()

    mock_seg1 = MagicMock(start=0.5, end=3.2, text=" Xin chào quý vị khán giả ")
    mock_seg2 = MagicMock(start=3.5, end=6.0, text=" Bản tin hôm nay ")
    extractor._model.transcribe.return_value = ([mock_seg1, mock_seg2], MagicMock())

    v1 = tmp_path / "videos" / "L21_V001.mp4"
    v1.parent.mkdir(parents=True, exist_ok=True)
    v1.write_bytes(b"dummy mp4")

    out_dir = tmp_path / "asr_out"
    written = extractor.transcribe_directory(
        videos_dir=tmp_path / "videos",
        output_dir=out_dir,
    )
    assert len(written) == 1
    assert written[0].exists()

    data = json.loads(written[0].read_text(encoding="utf-8"))
    assert len(data) == 2
    assert data[0]["start"] == 0.5
    assert data[0]["text"] == "Xin chào quý vị khán giả"


def test_enrich_manifest_with_asr(tmp_path: Path) -> None:
    asr_dir = tmp_path / "asr"
    asr_dir.mkdir(parents=True, exist_ok=True)
    (asr_dir / "L21_V001.json").write_text(
        json.dumps([
            {"start": 1.0, "end": 4.0, "text": "người dẫn chương trình nói về giao thông"},
            {"start": 10.0, "end": 15.0, "text": "hình ảnh hiện trường vụ tai nạn"},
        ], ensure_ascii=False),
        encoding="utf-8",
    )

    manifest_in = tmp_path / "manifest_in.jsonl"
    manifest_out = tmp_path / "manifest_out.jsonl"

    # Frame 0: t = 0 / 25 = 0.0s (within 1.0 - 1.5s = -0.5s -> matches seg 1)
    # Frame 50: t = 50 / 25 = 2.0s (within 1.0..4.0 -> matches seg 1)
    # Frame 300: t = 300 / 25 = 12.0s (within 10.0..15.0 -> matches seg 2)
    # Frame 1000: t = 1000 / 25 = 40.0s (no match)
    records = [
        FrameRecord(vector_id=0, video_id="L21_V001", frame_id=0, keyframe_path="0.jpg"),
        FrameRecord(vector_id=1, video_id="L21_V001", frame_id=50, keyframe_path="50.jpg"),
        FrameRecord(vector_id=2, video_id="L21_V001", frame_id=300, keyframe_path="300.jpg"),
        FrameRecord(vector_id=3, video_id="L21_V001", frame_id=1000, keyframe_path="1000.jpg"),
    ]
    manifest_in.write_text(
        "\n".join(r.model_dump_json() for r in records) + "\n",
        encoding="utf-8",
    )

    written, enriched = enrich_manifest_with_asr(
        manifest_path=manifest_in,
        asr_dir=asr_dir,
        output_path=manifest_out,
        default_fps=25.0,
        time_window=1.5,
    )

    assert written == 4
    assert enriched == 3  # frames 0, 50, 300 matched

    res_records = [FrameRecord.model_validate_json(line) for line in manifest_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(res_records) == 4
    assert res_records[0].asr_text == ["người dẫn chương trình nói về giao thông"]
    assert res_records[1].asr_text == ["người dẫn chương trình nói về giao thông"]
    assert res_records[2].asr_text == ["hình ảnh hiện trường vụ tai nạn"]
    assert res_records[3].asr_text == []
    assert all(r.asr_done for r in res_records)


def test_bm25_search_with_asr() -> None:
    records = [
        FrameRecord(
            vector_id=0,
            video_id="L21_V001",
            frame_id=10,
            keyframe_path="10.jpg",
            object_labels=["person"],
            asr_text=["phát thanh viên thông báo về dự báo thời tiết mưa bão"],
        ),
        FrameRecord(
            vector_id=1,
            video_id="L21_V002",
            frame_id=20,
            keyframe_path="20.jpg",
            object_labels=["car"],
            asr_text=["kết quả trận đấu bóng đá"],
        ),
        FrameRecord(
            vector_id=2,
            video_id="L21_V003",
            frame_id=30,
            keyframe_path="30.jpg",
            object_labels=["dog"],
            asr_text=["nấu ăn ẩm thực món ngon"],
        ),
    ]
    bm25 = BM25Index(records)
    ids, scores = bm25.search("mưa bão", k=5)
    assert len(ids) == 1
    assert ids[0] == 0
    assert scores[0] > 0

    ids2, scores2 = bm25.search("bóng đá", k=5)
    assert len(ids2) == 1
    assert ids2[0] == 1
    assert scores2[0] > 0


def test_rerank_with_metadata_asr_boost() -> None:
    records = [
        FrameRecord(
            vector_id=0,
            video_id="L21_V001",
            frame_id=10,
            keyframe_path="10.jpg",
            asr_text=["bản tin sáng ngày hôm nay"],
        ),
        FrameRecord(
            vector_id=1,
            video_id="L21_V002",
            frame_id=20,
            keyframe_path="20.jpg",
            asr_text=["thời trang mùa hè"],
        ),
    ]
    candidates = [
        Candidate(video_id="L21_V001", frame_id=10, score=0.5, vector_id=0),
        Candidate(video_id="L21_V002", frame_id=20, score=0.5, vector_id=1),
    ]
    reranked = rerank_with_metadata(
        query="bản tin",
        candidates=candidates,
        records={r.vector_id: r for r in records},
        weight=0.2,
    )
    # L21_V001 should receive bonus score for matching "bản tin" in asr_text
    assert reranked[0].video_id == "L21_V001"
    assert reranked[0].score > reranked[1].score
