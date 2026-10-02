from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aic2026.ingestion.official_index import build_official_index, probe_official_features
from aic2026.models import FrameRecord
from aic2026.retrieval import RetrievalPipeline, VectorIndex


def _make_dataset(root: Path) -> None:
    keyframes = root / "Keyframes"
    objects = root / "Objects"
    metadata = root / "Metadata"
    for folder in (keyframes, objects, metadata):
        folder.mkdir(parents=True, exist_ok=True)

    # Two videos, each with two keyframes. Filenames are ordinals (0, 1).
    for video in ("L01_V001", "L02_V003"):
        vk = keyframes / video
        vk.mkdir(parents=True, exist_ok=True)
        for i in (0, 1):
            (vk / f"{i:03d}.jpg").write_bytes(b"jpg")
        # object labels per keyframe in BTC dict format
        # ({"detection_class_entities":[...], "detection_scores":[...]}).
        # The official_index._load_object_labels parser requires this shape and
        # score >= 0.4 for the video to be kept; a list-of-dicts would be skipped.
        # build_official_index matches object JSON by 1-based keyframe ordinal:
        #   ordinal 1 -> "001.json" -> frame_id 100, ordinal 2 -> "002.json" -> frame_id 101.
        # So frame 100 ("người") is in 001.json and frame 101 ("cửa hàng") in 002.json.
        (objects / video).mkdir(parents=True, exist_ok=True)
        (objects / video / "001.json").write_text(
            json.dumps({"detection_class_entities": ["người"], "detection_scores": ["0.9"]}),
            encoding="utf-8",
        )
        if video == "L01_V001":
            (objects / video / "002.json").write_text(
                json.dumps({"detection_class_entities": ["cửa hàng"], "detection_scores": ["0.9"]}),
                encoding="utf-8",
            )
        else:
            (objects / video / "002.json").write_text(
                json.dumps({"detection_class_entities": ["đường phố"], "detection_scores": ["0.9"]}),
                encoding="utf-8",
            )
        # metadata with ground-truth frame indices (mapped ordinals -> true frames)
        (metadata / f"{video}.json").write_text(
            json.dumps({"title": "video demo", "frame_indices": [100, 101] if video == "L01_V001" else [500, 501]}),
            encoding="utf-8",
        )

    # Official CLIP features: one .npy file PER video (the multi-file layout),
    # each with 2 rows (ascending keyframe order).  `build_official_index` derives
    # `video_id` from the .npy file STEM (e.g. "L01_V001"), so we MUST write a
    # per-video file — a single shared features.npy would collapse both videos
    # into one bogus video_id "features".  See `build_official_index` docstring.
    np.save(root / "L01_V001.npy", np.arange(8, dtype=np.float32).reshape(2, 4))
    np.save(root / "L02_V003.npy", np.arange(8, 16, dtype=np.float32).reshape(2, 4))


def test_probe_official_features(tmp_path) -> None:
    _make_dataset(tmp_path)
    # probe_official_features takes a single .npy (the legacy single-file layout).
    # Build a combined matrix so count/dim reflect both videos.
    combined = tmp_path.parent / "probe_features.npy"
    arr = np.concatenate(
        [np.load(tmp_path / "L01_V001.npy"), np.load(tmp_path / "L02_V003.npy")]
    )
    np.save(combined, arr)
    report = probe_official_features(
        combined,
        raw_dir=tmp_path,
        metadata_dir=tmp_path / "Metadata",
    )
    assert report["count"] == 4
    assert report["dim"] == 4
    assert report["video_ids_in_order"] == ["L01_V001", "L02_V003"]
    assert report["metadata_frame_indices_sample"]["L01_V001"]["first"] == 100


def test_build_official_index_maps_true_frame_ids(tmp_path) -> None:
    _make_dataset(tmp_path)
    out_manifest = tmp_path / "official_manifest.jsonl"
    out_features = tmp_path / "official_features.npy"
    count = build_official_index(tmp_path, tmp_path, out_manifest, out_features)

    assert count == 4
    records = [FrameRecord.model_validate_json(line) for line in out_manifest.read_text(encoding="utf-8").splitlines()]
    by_video: dict[str, list[FrameRecord]] = {}
    for record in records:
        by_video.setdefault(record.video_id, []).append(record)
    # Frame ids must be the true video frames (100/101, 500/501), not ordinals 0/1.
    assert sorted(r.frame_id for r in by_video["L01_V001"]) == [100, 101]
    assert sorted(r.frame_id for r in by_video["L02_V003"]) == [500, 501]
    # Object labels must be attached per keyframe.
    l01 = {r.frame_id: r for r in by_video["L01_V001"]}
    assert "người" in l01[100].object_labels
    assert "cửa hàng" in l01[101].object_labels

    vectors = np.load(out_features)
    assert vectors.shape == (4, 4)


def test_official_index_count_mismatch_raises(tmp_path, caplog) -> None:
    """Image-vs-feature count mismatch logs a WARNING and does NOT crash.

    `build_official_index` derives the manifest from the CLIP feature rows (one
    record per .npy row), so the manifest count always equals the feature count
    by construction — the record-vs-feature guard cannot fire for the
    feature-driven loop. The *image-vs-feature* mismatch (keyframes downloaded
    for only 29/873 videos) is intentionally a WARNING, not a ValueError: frames
    missing an image keep an empty `keyframe_path` and retrieval still runs on
    the CLIP vectors. (Per the build docstring: image count != feature count is
    normal.)
    """
    _make_dataset(tmp_path)
    # "bad" video: 3 feature rows but only 1 keyframe image -> mismatch WARNING.
    (tmp_path / "Keyframes" / "bad").mkdir(parents=True, exist_ok=True)
    (tmp_path / "Keyframes" / "bad" / "001.jpg").write_bytes(b"jpg")
    (tmp_path / "Objects" / "bad").mkdir(parents=True, exist_ok=True)
    np.save(tmp_path / "bad.npy", np.zeros((3, 4), dtype=np.float32))
    import logging
    with caplog.at_level(logging.WARNING, logger="aic2026.ingestion.official_index"):
        count = build_official_index(
            tmp_path,
            tmp_path / "bad.npy",
            tmp_path / "m.jsonl",
            tmp_path / "f.npy",
        )
    # No crash; build returns the feature row count.
    assert count == 3
    assert (tmp_path / "f.npy").exists()
    # The mismatch is surfaced as a WARNING (Vietnamese) instead of ValueError.
    assert any("khác" in r.message for r in caplog.records)


def _mini_pipeline(tmp_path) -> tuple[RetrievalPipeline, list[FrameRecord]]:
    _make_dataset(tmp_path)
    manifest = tmp_path / "official_manifest.jsonl"
    features = tmp_path / "official_features.npy"
    build_official_index(tmp_path, tmp_path, manifest, features)
    records = [FrameRecord.model_validate_json(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    pipeline = RetrievalPipeline(VectorIndex.from_npy(features), records)
    return pipeline, records


def test_filter_videos_by_metadata_narrows(tmp_path) -> None:
    pipeline, _records = _mini_pipeline(tmp_path)
    # Term "cửa hàng" appears only in L01_V001/001 object labels.
    matched = pipeline.filter_videos_by_metadata(["cửa hàng"])
    assert matched == ["L01_V001"]
    # Term present in no video -> fallback returns all (no recall loss).
    assert pipeline.filter_videos_by_metadata(["xyz"]) == ["L01_V001", "L02_V003"]
    # Accent-insensitive match.
    assert pipeline.filter_videos_by_metadata(["cua hang"]) == ["L01_V001"]
