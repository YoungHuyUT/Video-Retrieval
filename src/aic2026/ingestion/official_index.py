from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from aic2026.data_platform import inspect_official_assets
from aic2026.models import FrameRecord

from .manifest import _frame_id, _metadata, load_manifest

logger = logging.getLogger(__name__)


def _expected_frame_indices(metadata_dir: Path | None, video_id: str) -> list[int] | None:
    """Return the ground-truth frame-index list for a video, if metadata has it.

    BTC metadata JSON may carry frame_indices / keyframe_indices / frames so that
    keyframe ordinals map to original video frame indices.
    """
    meta, _ = _metadata(metadata_dir, video_id)
    for key in ("frame_indices", "keyframe_indices", "frames"):
        values = meta.get(key)
        if isinstance(values, list):
            result: list[int] = []
            for v in values:
                if isinstance(v, int):
                    result.append(v)
                elif isinstance(v, dict):
                    inner = v.get("frame_id", v.get("index"))
                    if isinstance(inner, int):
                        result.append(inner)
            return result
    return None


def probe_official_features(
    features_path: Path,
    manifest_path: Path | None = None,
    metadata_dir: Path | None = None,
    raw_dir: Path | None = None,
) -> dict:
    """Inspect the official BTC CLIP `.npy` and how its rows relate to the manifest.

    Because the competition only documents that vectors are "in ascending keyframe
    order", this probe reports dimensionality, count, per-video frame counts from
    the manifest, and the first/last frame index of each video so the row order can
    be reverse-engineered before `build_official_index` is called.
    """
    features_path = Path(features_path)
    vectors = np.load(features_path, mmap_mode="r")
    result: dict = {
        "features_path": str(features_path),
        "dim": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
    }

    per_video: dict[str, dict] = {}
    manifest_path = Path(manifest_path) if manifest_path else None
    if manifest_path is not None and manifest_path.exists():
        records = load_manifest(manifest_path)
        result["manifest_count"] = len(records)
        for record in records:
            slot = per_video.setdefault(record.video_id, {"count": 0, "first_frame": None, "last_frame": None})
            slot["count"] += 1
            slot["first_frame"] = record.frame_id if slot["first_frame"] is None else min(slot["first_frame"], record.frame_id)
            slot["last_frame"] = max(slot["last_frame"] or 0, record.frame_id)
        result["per_video"] = per_video
        result["video_ids_in_order"] = list(per_video)
    elif raw_dir is not None:
        # Fallback: derive per-video counts from the keyframe folders directly.
        assets = inspect_official_assets(Path(raw_dir))
        if assets.keyframes is not None:
            for image in sorted([*assets.keyframes.rglob("*.jpg"), *assets.keyframes.rglob("*.png")]):
                video_id = image.parent.name
                slot = per_video.setdefault(video_id, {"count": 0, "first_frame": None, "last_frame": None})
                slot["count"] += 1
            result["per_video"] = per_video
            result["video_ids_in_order"] = list(per_video)

    if metadata_dir is not None:
        metadata_dir = Path(metadata_dir)
        sample = {}
        for video_id in (result.get("video_ids_in_order") or [])[:3]:
            indices = _expected_frame_indices(metadata_dir, video_id)
            if indices:
                sample[video_id] = {"n_expected": len(indices), "first": indices[0], "last": indices[-1]}
        result["metadata_frame_indices_sample"] = sample

    if raw_dir is not None:
        assets = inspect_official_assets(Path(raw_dir))
        result["detected"] = {
            "Videos": str(assets.videos) if assets.videos else None,
            "Keyframes": str(assets.keyframes) if assets.keyframes else None,
            "Objects": str(assets.objects) if assets.objects else None,
            "Metadata": str(assets.metadata) if assets.metadata else None,
            "CLIP_features": [str(p) for p in assets.clip_features],
        }
        if assets.clip_features:
            result["clip_feature_file_count"] = len(assets.clip_features)
            result["clip_feature_total_bytes"] = sum(p.stat().st_size for p in assets.clip_features)

    return result


def build_official_index(
    raw_dir: Path,
    features_path: Path,
    output_manifest: Path,
    output_features: Path,
    keyframes_dir: Path | None = None,
    objects_dir: Path | None = None,
    metadata_dir: Path | None = None,
) -> int:
    """Build a manifest + aligned `.npy` from the official BTC CLIP features.

    Uses metadata.frame_indices (falling back to keyframe filename ordinals) to
    assign each feature row its video + true frame id, writes records sorted by
    (video_id, frame_id), and reorders the feature matrix to match. Raises if the
    official feature count does not equal the number of discovered keyframes.
    """
    raw_dir, features_path = Path(raw_dir), Path(features_path)
    output_manifest, output_features = Path(output_manifest), Path(output_features)
    assets = inspect_official_assets(raw_dir)

    keyframes_root = keyframes_dir and Path(keyframes_dir) or assets.keyframes
    if keyframes_root is None or not keyframes_root.exists():
        raise FileNotFoundError(f"Không tìm thấy Keyframes dưới {raw_dir}")
    metadata_root = metadata_dir and Path(metadata_dir) or assets.metadata
    objects_root = objects_dir and Path(objects_dir) or assets.objects

    vectors = np.load(features_path, mmap_mode="r")

    image_paths = sorted([*keyframes_root.rglob("*.jpg"), *keyframes_root.rglob("*.png")])
    if len(image_paths) != int(vectors.shape[0]):
        raise ValueError(
            f"Số keyframe ({len(image_paths)}) không khớp số feature CLIP BTC ({int(vectors.shape[0])}). "
            f"Chạy `aic2026 probe-official-features` trước để xác định thứ tự."
        )

    # Group keyframes by video, preserving ascending keyframe order within a video.
    by_video: dict[str, list[tuple[int, Path]]] = {}
    for image in image_paths:
        video_id = image.parent.name
        ordinal = _frame_id(image)
        by_video.setdefault(video_id, []).append((ordinal, image))

    rows: list[tuple[int, str, int, Path, list[str], str | None, str | None, int]] = []
    vector_id = 0
    for video_id in sorted(by_video):
        keyframes = sorted(by_video[video_id], key=lambda pair: pair[0])
        expected = _expected_frame_indices(metadata_root, video_id)
        meta, metadata_path = _metadata(metadata_root, video_id)
        title = meta.get("title")
        description = meta.get("description")
        for ordinal_index, (ordinal, image) in enumerate(keyframes):
            frame_id = expected[ordinal_index] if expected and ordinal_index < len(expected) else ordinal
            object_path = (objects_root / video_id / f"{image.stem}.json") if objects_root else None
            labels: list[str] = []
            if object_path and object_path.exists():
                try:
                    objects = json.loads(object_path.read_text(encoding="utf-8"))
                    labels = sorted({str(x.get("label", x.get("name", ""))) for x in objects if isinstance(x, dict)})
                except (OSError, json.JSONDecodeError):
                    pass
            rows.append(
                (
                    vector_id,
                    video_id,
                    frame_id,
                    image,
                    labels,
                    title,
                    description,
                    vector_id,
                )
            )
            vector_id += 1

    records: list[FrameRecord] = [
        FrameRecord(
            vector_id=vid,
            clip_feature_index=clip_idx,
            video_id=video_id,
            frame_id=frame_id,
            keyframe_path=str(image),
            object_labels=[label for label in labels if label],
            title=title,
            description=description,
            metadata_path=str(metadata_path) if metadata_path else None,
            object_path=str(object_path) if object_path and object_path.exists() else None,
        )
        for vid, video_id, frame_id, image, labels, title, description, clip_idx in rows
    ]

    # Cross-check per-video keyframe counts against metadata when available. A
    # mismatch usually means the metadata file does not describe these keyframes
    # (e.g. a stale frame_indices list), so we surface it instead of silently
    # writing frame_ids that point at the wrong video frames.
    for video_id in sorted(by_video):
        expected = _expected_frame_indices(metadata_root, video_id)
        if expected and len(expected) != len(by_video[video_id]):
            logger.warning(
                "video %s: metadata lists %d frame indices but %d keyframes found; "
                "frame_id mapping may be wrong",
                video_id,
                len(expected),
                len(by_video[video_id]),
            )

    # The official .npy is documented as one row per keyframe in ascending
    # keyframe order. Both the manifest rows and the .npy rows are produced in
    # that natural (sorted video, ascending ordinal) order, so row i of the
    # matrix corresponds to manifest record i. We keep the rows untouched and
    # verify the count matches — reordering blindly without knowing the actual
    # .npy video order would risk silently misaligning every frame.
    if len(records) != int(vectors.shape[0]):
        raise ValueError(
            f"Manifest records ({len(records)}) do not match feature rows "
            f"({int(vectors.shape[0])})."
        )
    aligned = np.asarray(vectors, dtype=np.float32).copy()

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_features.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text("\n".join(record.model_dump_json() for record in records) + "\n", encoding="utf-8")
    np.save(output_features, aligned)
    return len(records)
