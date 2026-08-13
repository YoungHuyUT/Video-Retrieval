from __future__ import annotations

import json
import re
from pathlib import Path

from aic2026.data_platform import inspect_official_assets
from aic2026.models import FrameRecord


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
    records: list[FrameRecord] = []
    for vector_id, image in enumerate(sorted([*assets.keyframes.rglob("*.jpg"), *assets.keyframes.rglob("*.png")])):
            video_id = image.parent.name
            ordinal = _frame_id(image)
            meta, metadata_path = _metadata(assets.metadata, video_id)
            object_path = (assets.objects / video_id / f"{image.stem}.json") if assets.objects else None
            labels: list[str] = []
            if object_path and object_path.exists():
                try:
                    objects = json.loads(object_path.read_text(encoding="utf-8"))
                    labels = sorted({str(x.get("label", x.get("name", ""))) for x in objects if isinstance(x, dict)})
                except json.JSONDecodeError:
                    pass
            video_path = assets.videos / f"{video_id}.mp4" if assets.videos else None
            records.append(FrameRecord(
                vector_id=vector_id, video_id=video_id, frame_id=_actual_frame_id(meta, ordinal),
                keyframe_path=_manifest_path(image, raw_dir), object_labels=[x for x in labels if x],
                title=meta.get("title"), description=meta.get("description"),
                video_path=_manifest_path(video_path, raw_dir) if video_path and video_path.exists() else None,
                object_path=_manifest_path(object_path, raw_dir) if object_path and object_path.exists() else None,
                metadata_path=_manifest_path(metadata_path, raw_dir) if metadata_path else None,
                clip_feature_index=vector_id,
            ))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(r.model_dump_json() for r in records) + ("\n" if records else ""), encoding="utf-8")
    return len(records)


def load_manifest(path: Path) -> list[FrameRecord]:
    return [FrameRecord.model_validate_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
