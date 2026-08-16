from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from aic2026.data_platform import inspect_official_assets
from aic2026.models import FrameRecord


# Ngưỡng score tối thiểu để giữ lại object entity (dùng chung với official_index).
_OBJECT_SCORE_THRESHOLD = 0.3


@dataclass(slots=True)
class VideoMetadata:
    """Video-level metadata cached once per video during manifest build.

    Avoids per-frame disk I/O and redundant keyword storage.
    """
    video_id: str
    title: str | None
    description: str | None
    metadata_keywords: list[str]
    metadata_path: str | None
    object_files: set[str]
    frame_indices: list[int] | None = None


def _frame_id(path: Path) -> int:
    digits = re.findall(r"\d+", path.stem)
    return int(digits[-1]) if digits else 0


def _metadata(metadata_dir: Path | None, video_id: str) -> tuple[dict, Path | None]:
    if metadata_dir is None:
        return {}, None
    path = metadata_dir / f"{video_id}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")), path
    except (OSError, json.JSONDecodeError):
        return {}, None


def _load_object_labels(object_path: Path | None) -> list[str]:
    """Đọc object JSON (dict chứa mảng song parallel) → entity có score >= threshold."""
    if not object_path or not object_path.exists():
        return []
    try:
        data = json.loads(object_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    entities = data.get("detection_class_entities") or []
    scored = data.get("detection_scores") or []
    if not entities:
        return []
    scores: list[float] = []
    for s in scored:
        try:
            scores.append(float(s))
        except (TypeError, ValueError):
            scores.append(1.0)
    keep: list[str] = []
    for i, ent in enumerate(entities):
        sc = scores[i] if i < len(scores) else 1.0
        if sc >= _OBJECT_SCORE_THRESHOLD:
            keep.append(str(ent))
    return sorted(set(keep))


def _load_metadata_keywords(meta: dict) -> list[str]:
    """Lấy keywords (summarize) từ metadata JSON — thay thế title/description dư thừa."""
    keywords = meta.get("keywords") or []
    if isinstance(keywords, list):
        return [str(k) for k in keywords if k]
    return []


def _manifest_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        try:
            return str(path.relative_to(base_dir.parent))
        except ValueError:
            return str(path)


def _actual_frame_id(metadata: dict, ordinal: int) -> int:
    for key in ("frame_indices", "keyframe_indices", "frames"):
        values = metadata.get(key)
        if isinstance(values, list) and ordinal < len(values):
            value = values[ordinal]
            if isinstance(value, int):
                return value
            if isinstance(value, dict):
                return int(value.get("frame_id", value.get("index", ordinal)))
    return ordinal


def build_manifest(raw_dir: Path, output: Path) -> int:
    """Discover Keyframes/Objects/Metadata folders and write one JSONL record per frame."""
    raw_dir, output = Path(raw_dir), Path(output)
    assets = inspect_official_assets(raw_dir)
    if assets.keyframes is None:
        raise FileNotFoundError(f"No Keyframes directory found below {raw_dir}")

    # Pre-index object files per video for O(1) lookup.
    object_files_by_video: dict[str, set[str]] = {}
    if assets.objects is not None and assets.objects.exists():
        for video_dir in assets.objects.iterdir():
            if video_dir.is_dir():
                object_files_by_video[video_dir.name] = {p.stem for p in video_dir.glob("*.json")}

    # Pre-load and cache video metadata once per video.
    video_meta_cache: dict[str, VideoMetadata] = {}
    for image in sorted([*assets.keyframes.rglob("*.jpg"), *assets.keyframes.rglob("*.png")]):
        video_id = image.parent.name
        if video_id not in video_meta_cache:
            meta, metadata_path = _metadata(assets.metadata, video_id)
            title = meta.get("title")
            description = meta.get("description")
            keywords = _load_metadata_keywords(meta)
            obj_files = object_files_by_video.get(video_id, set())
            # Extract frame_indices for frame_id mapping
            frame_indices = None
            for key in ("frame_indices", "keyframe_indices", "frames"):
                values = meta.get(key)
                if isinstance(values, list):
                    frame_indices = []
                    for v in values:
                        if isinstance(v, int):
                            frame_indices.append(v)
                        elif isinstance(v, dict):
                            inner = v.get("frame_id", v.get("index"))
                            if isinstance(inner, int):
                                frame_indices.append(inner)
                    break
            video_meta_cache[video_id] = VideoMetadata(
                video_id=video_id,
                title=title,
                description=description,
                metadata_keywords=keywords,
                metadata_path=_manifest_path(metadata_path, raw_dir) if metadata_path else None,
                object_files=obj_files,
                frame_indices=frame_indices,
            )

    records: list[FrameRecord] = []
    for vector_id, image in enumerate(sorted([*assets.keyframes.rglob("*.jpg"), *assets.keyframes.rglob("*.png")])):
        video_id = image.parent.name
        ordinal = _frame_id(image)
        vmeta = video_meta_cache[video_id]

        object_path = None
        labels: list[str] = []
        if vmeta.object_files:
            single = f"{ordinal:03d}"
            if single in vmeta.object_files:
                obj_path = (assets.objects / video_id / f"{single}.json") if assets.objects else None
                if obj_path:
                    object_path = _manifest_path(obj_path, raw_dir) if obj_path.exists() else None
                    labels = _load_object_labels(obj_path)

        video_path = assets.videos / f"{video_id}.mp4" if assets.videos else None
        # Use cached frame_indices for frame_id mapping
        frame_id = vmeta.frame_indices[ordinal - 1] if vmeta.frame_indices and ordinal - 1 < len(vmeta.frame_indices) else ordinal

        records.append(FrameRecord(
            vector_id=vector_id,
            video_id=video_id,
            frame_id=frame_id,
            keyframe_path=_manifest_path(image, raw_dir),
            object_labels=labels,
            metadata_keywords=vmeta.metadata_keywords,
            title=vmeta.title,
            description=vmeta.description,
            video_path=_manifest_path(video_path, raw_dir) if video_path and video_path.exists() else None,
            object_path=object_path,
            metadata_path=vmeta.metadata_path,
            clip_feature_index=vector_id,
        ))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(r.model_dump_json() for r in records) + ("\n" if records else ""), encoding="utf-8")
    return len(records)


def load_manifest(path: Path) -> list[FrameRecord]:
    return [FrameRecord.model_validate_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
