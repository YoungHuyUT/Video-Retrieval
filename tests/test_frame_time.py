from __future__ import annotations

from aic2026 import frame_time
from aic2026.models import Candidate


def test_batch2_map_reports_video_fps_and_candidate_keeps_it(tmp_path, monkeypatch):
    maps = tmp_path / "map-keyframes"
    maps.mkdir()
    (maps / "M01_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n2,3.6,25.0,90\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(frame_time, "_KEYFRAME_MAP_DIRS", (maps,))
    frame_time.video_fps_info.cache_clear()

    assert frame_time.video_fps_info("M01_V001") == (25.0, "map-keyframes")
    candidate = Candidate(video_id="M01_V001", frame_id=90, score=0.8, fps=25.0)
    assert candidate.model_dump()["fps"] == 25.0

    frame_time.video_fps_info.cache_clear()


def test_batch2_vfr_timestamp_resolution_uses_pts_map(tmp_path, monkeypatch):
    maps = tmp_path / "map-keyframes"
    maps.mkdir()
    (maps / "N001-V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n2,2.0,25.0,50\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(frame_time, "_KEYFRAME_MAP_DIRS", (maps,))

    info = frame_time.resolve_frame_at_timestamp("N001-V001", 2.0)
    assert info.frame_number == 50
    assert info.actual_frame_timestamp == 2.0
    assert info.fps == 25.0
    assert info.vfr is True


def test_result_timestamp_uses_exact_pts_or_milliseconds(tmp_path, monkeypatch):
    maps = tmp_path / "map-keyframes"
    maps.mkdir()
    (maps / "M01_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n2,3.6,25.0,90\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(frame_time, "_KEYFRAME_MAP_DIRS", (maps,))
    frame_time._keyframe_pts_by_frame.cache_clear()

    assert frame_time.frame_timestamp_seconds("M01_V001", 90) == 3.6
    assert frame_time.frame_timestamp_seconds(
        "N001-V001", 12345, frame_unit="milliseconds"
    ) == 12.345

    frame_time._keyframe_pts_by_frame.cache_clear()
