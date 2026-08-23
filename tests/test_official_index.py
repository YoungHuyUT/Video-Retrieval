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
    clip_features = root / "CLIP features"
    for folder in (keyframes, objects, metadata, clip_features):
        folder.mkdir(parents=True, exist_ok=True)

    # Two videos, each with two keyframes. Filenames are ordinals (0, 1).
    for video in ("L01_V001", "L02_V003"):
        vk = keyframes / video
        vk.mkdir(parents=True, exist_ok=True)
        for i in (1, 2):
            (vk / f"{i:03d}.jpg").write_bytes(b"jpg")
        # object labels per keyframe: only L01_V001 has "cửa hàng"
        (objects / video).mkdir(parents=True, exist_ok=True)
        (objects / video / "001.json").write_text(
            json.dumps({"detection_class_entities": ["người"], "detection_scores": ["0.85"]}),
            encoding="utf-8",
        )
        if video == "L01_V001":
            (objects / video / "002.json").write_text(
                json.dumps({"detection_class_entities": ["cửa hàng"], "detection_scores": ["0.90"]}),
                encoding="utf-8",
            )
            np.save(clip_features / f"{video}.npy", np.arange(8, dtype=np.float32).reshape(2, 4))
        else:
            (objects / video / "002.json").write_text(
                json.dumps({"detection_class_entities": ["đường phố"], "detection_scores": ["0.75"]}),
                encoding="utf-8",
            )
            np.save(clip_features / f"{video}.npy", np.arange(8, 16, dtype=np.float32).reshape(2, 4))
        # metadata with ground-truth frame indices (mapped ordinals -> true frames)
        (metadata / f"{video}.json").write_text(
            json.dumps({"title": "video demo", "frame_indices": [100, 101] if video == "L01_V001" else [500, 501]}),
            encoding="utf-8",
        )


def test_probe_official_features(tmp_path) -> None:
    _make_dataset(tmp_path)
    report = probe_official_features(
        tmp_path / "CLIP features",
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
    count = build_official_index(tmp_path, tmp_path / "CLIP features", out_manifest, out_features)

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


def test_official_index_count_mismatch_raises(tmp_path) -> None:
    _make_dataset(tmp_path)
    bad_dir = tmp_path / "bad_clip"
    bad_dir.mkdir(parents=True, exist_ok=True)
    # Empty dir
    try:
        build_official_index(tmp_path, bad_dir, tmp_path / "m.jsonl", tmp_path / "f.npy")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError for empty dir")


def _mini_pipeline(tmp_path) -> tuple[RetrievalPipeline, list[FrameRecord]]:
    _make_dataset(tmp_path)
    manifest = tmp_path / "official_manifest.jsonl"
    features = tmp_path / "official_features.npy"
    build_official_index(tmp_path, tmp_path / "CLIP features", manifest, features)
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
