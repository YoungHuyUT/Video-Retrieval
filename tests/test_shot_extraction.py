"""Unit tests for the shot-adaptive extraction pipeline (spec §1-§13).

Covers the pieces that do NOT need the heavy model weights or scenedetect:
* AdaptiveFrameSampler count curve + temporal coverage (endpoints included).
* merge_timestamps dedupe/clip.
* ShotCatalog CRUD + query-time window lookup (spec §6, §10).
* FrameRecord new fields (shot_id/timestamp/model_version/embedding_version) load.
* PySceneDetectDetector / TransNetV2Detector contract + factory.
* validate_artifacts row-alignment check (with a tiny synthetic index).
"""

from __future__ import annotations

import numpy as np
import pytest

from aic2026.extraction import (
    AdaptiveFrameSampler,
    MotionChangeDetector,
    PySceneDetectDetector,
    ShotCatalog,
    ShotConfig,
    ShotDetector,
    TransNetV2Detector,
    get_detector,
    merge_timestamps,
)
from aic2026.extraction.detector import DEFAULT_DETECTOR
from aic2026.extraction.validate import validate_artifacts
from aic2026.models import FrameRecord


# --- sampler ----------------------------------------------------------------
def test_sampler_count_curve():
    s = AdaptiveFrameSampler()
    assert s.count_for(2.0) == 3
    assert s.count_for(4.9) == 3
    assert s.count_for(5.0) == 5  # boundary: 5s..15s -> 5
    assert s.count_for(9.0) == 5
    assert s.count_for(20.0) == 7
    assert s.count_for(60.0) == 7


def test_sampler_fixed_override():
    s = AdaptiveFrameSampler(fixed_per_shot=5)
    assert s.count_for(2.0) == 5
    assert s.count_for(999.0) == 5


def test_sampler_covers_endpoints_and_middle():
    s = AdaptiveFrameSampler()
    ts = s.sample((0.0, 10.0))
    assert ts[0] == pytest.approx(0.0)
    assert ts[-1] == pytest.approx(10.0)
    # Evenly spaced: midpoint near 5.0
    assert 4.9 < ts[len(ts) // 2] < 5.1
    assert len(ts) == 5


def test_sampler_degenerate_shot():
    s = AdaptiveFrameSampler()
    ts = s.sample((3.0, 3.0))
    assert len(ts) == 1
    assert ts[0] == pytest.approx(3.0)


def test_merge_timestamps_dedupes_and_clips():
    base = [0.0, 5.0, 10.0]
    extra = [5.0, 11.0]  # 5.0 dup, 11.0 out of [0,10]
    merged = merge_timestamps(base, extra, 0.0, 10.0, dedupe_gap_s=0.1)
    assert merged[0] == pytest.approx(0.0)
    assert merged[-1] == pytest.approx(10.0)
    assert 11.0 not in merged
    assert len(merged) == 3


# --- detector contract ------------------------------------------------------
def test_detector_factory_default():
    d = get_detector()
    assert isinstance(d, PySceneDetectDetector)
    assert d.name == DEFAULT_DETECTOR


def test_detector_factory_unknown():
    with pytest.raises(ValueError):
        get_detector("bogus")


def test_transnetv2_is_stub():
    det = TransNetV2Detector()
    assert isinstance(det, ShotDetector) or hasattr(det, "detect")
    with pytest.raises(NotImplementedError):
        det.detect("x.mp4")


# --- catalog ----------------------------------------------------------------
def test_catalog_roundtrip_and_window(tmp_path):
    db = tmp_path / "catalog.db"
    cat = ShotCatalog(db)
    cat.upsert_video("V1", 30.0, 900)
    shot_id = cat.add_shot("V1", 0.0, 10.0, n_frames=3)
    cat.add_frame("V1", 0, shot_id, 0.0, "k/0.jpg", model_version="m", embedding_version="v")
    cat.add_frame("V1", 1, shot_id, 5.0, "k/1.jpg", model_version="m", embedding_version="v")
    cat.add_frame("V1", 2, shot_id, 10.0, "k/2.jpg", model_version="m", embedding_version="v")
    # frame far outside window should not appear
    shot2 = cat.add_shot("V1", 100.0, 110.0, n_frames=1)
    cat.add_frame("V1", 3, shot2, 105.0, "k/3.jpg")
    cat.commit_frames()

    rows = cat.frames_in_window("V1", 5.0, 5.5)
    ts = [r["timestamp"] for r in rows]
    assert 0.0 in ts and 10.0 in ts and 5.0 in ts
    assert 105.0 not in ts
    # sorted by distance to center
    assert rows[0]["timestamp"] == pytest.approx(5.0)
    assert cat.frame_count() == 4
    assert cat.shot_count() == 2
    cat.close()


def test_catalog_is_video_done(tmp_path):
    cat = ShotCatalog(tmp_path / "c.db")
    assert cat.is_video_done("V9") is False
    cat.add_shot("V9", 0.0, 10.0, n_frames=1)
    cat.add_frame("V9", 0, 1, 0.0, str(tmp_path / "p.jpg"))
    cat.commit_frames()
    assert cat.is_video_done("V9") is True
    cat.close()


# --- framerecord new fields -------------------------------------------------
def test_framerecord_new_fields_optional():
    rec = FrameRecord(vector_id=0, video_id="V", frame_id=0, keyframe_path="p.jpg")
    assert rec.shot_id is None
    assert rec.timestamp is None
    assert rec.model_version is None
    assert rec.embedding_version is None


def test_framerecord_new_fields_roundtrip():
    rec = FrameRecord(
        vector_id=1, video_id="V", frame_id=1, keyframe_path="p.jpg",
        shot_id=2, timestamp=3.5, model_version="m", embedding_version="v",
    )
    obj = rec.model_dump()
    assert obj["shot_id"] == 2 and obj["timestamp"] == 3.5
    rec2 = FrameRecord.model_validate(obj)
    assert rec2 == rec


# --- validate ---------------------------------------------------------------
def test_validate_catches_misalignment(tmp_path):
    cfg = ShotConfig(root=tmp_path)
    cfg.ensure_dirs()
    # write manifest with 2 rows
    k0, k1 = str(tmp_path / "k0.jpg"), str(tmp_path / "k1.jpg")
    manifest_rows = [
        FrameRecord(vector_id=0, video_id="V", frame_id=0, keyframe_path=k0,
                    shot_id=1, timestamp=0.0).model_dump_json(),
        FrameRecord(vector_id=1, video_id="V", frame_id=1, keyframe_path=k1,
                    shot_id=1, timestamp=5.0).model_dump_json(),
    ]
    cfg.manifest_path.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    np.save(cfg.clip_feats_path, np.zeros((1, 16), dtype=np.float32))  # wrong row count → 1
    np.save(cfg.siglip2_feats_path, np.zeros((2, 16), dtype=np.float32))
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    cat = ShotCatalog(cfg.catalog_path)
    cat.add_shot("V", 0.0, 10.0, n_frames=2)
    cat.add_frame("V", 0, 1, 0.0, k0)
    cat.add_frame("V", 1, 1, 5.0, k1)
    cat.commit_frames()
    cat.close()

    report = validate_artifacts(cfg, strict=False)
    # clip rows (1) != manifest (2) must be flagged
    assert any("clip features rows" in e for e in report.errors)


def test_validate_ok_when_aligned(tmp_path):
    cfg = ShotConfig(root=tmp_path)
    cfg.ensure_dirs()
    k0, k1 = str(tmp_path / "k0.jpg"), str(tmp_path / "k1.jpg")
    manifest_rows = [
        FrameRecord(vector_id=0, video_id="V", frame_id=0, keyframe_path=k0,
                    shot_id=1, timestamp=0.0).model_dump_json(),
        FrameRecord(vector_id=1, video_id="V", frame_id=1, keyframe_path=k1,
                    shot_id=1, timestamp=5.0).model_dump_json(),
    ]
    cfg.manifest_path.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    np.save(cfg.clip_feats_path, np.zeros((2, 16), dtype=np.float32))
    np.save(cfg.siglip2_feats_path, np.zeros((2, 16), dtype=np.float32))
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    k0, k1 = str(tmp_path / "k0.jpg"), str(tmp_path / "k1.jpg")
    cat = ShotCatalog(cfg.catalog_path)
    cat.add_shot("V", 0.0, 10.0, n_frames=2)
    cat.add_frame("V", 0, 1, 0.0, k0)
    cat.add_frame("V", 1, 1, 5.0, k1)
    cat.commit_frames()
    cat.close()
    (tmp_path / "k0.jpg").write_text("x")
    (tmp_path / "k1.jpg").write_text("x")

    report = validate_artifacts(cfg, strict=True)
    assert report.ok
