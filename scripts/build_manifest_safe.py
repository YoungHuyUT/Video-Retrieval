"""Build official manifest (new schema) + aligned features .npy without OOM.

Chiến lược RAM-safe:
- Load từng file .npy, đọc object/keywords cho video (đọc 1 lúc theo video, không
  giữ toàn bộ vector trong RAM).
- Ghi manifest JSONL dòng-by-dòng (incremental).
- Cuối cùng, concat features bằng np.save với memmap để tránh materialize toàn bộ.

Dùng khi build lại official_manifest.jsonl sau khi đổi schema FrameRecord
(bỏ title/description, thêm metadata_keywords) + parser object dict-array mới.
"""
from __future__ import annotations

import csv
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aic2026.data_platform import inspect_official_assets
from aic2026.models import FrameRecord

_OBJECT_SCORE_THRESHOLD = 0.3


def _frame_id_from_csv(map_root: Path | None, video_id: str) -> list[int] | None:
    if map_root is None:
        return None
    p = map_root / "map-keyframes" / f"{video_id}.csv"
    if not p.exists():
        return None
    out: list[int] = []
    try:
        with p.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                fi = r.get("frame_idx")
                if fi:
                    out.append(int(fi))
    except OSError:
        return None
    return out or None


def _load_object_labels(obj_path: Path | None) -> list[str]:
    if not obj_path or not obj_path.exists():
        return []
    try:
        d = json.loads(obj_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(d, dict):
        return []
    ents = d.get("detection_class_entities") or []
    scored = d.get("detection_scores") or []
    if not ents:
        return []
    scores = []
    for s in scored:
        try:
            scores.append(float(s))
        except (TypeError, ValueError):
            scores.append(1.0)
    keep = []
    for i, e in enumerate(ents):
        sc = scores[i] if i < len(scores) else 1.0
        if sc >= _OBJECT_SCORE_THRESHOLD:
            keep.append(str(e))
    return sorted(set(keep))


def _metadata(meta_dir: Path | None, vid: str):
    if meta_dir is None:
        return {}, None
    p = meta_dir / f"{vid}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")), p
    except (OSError, json.JSONDecodeError):
        return {}, None


def _keywords(meta: dict) -> list[str]:
    kw = meta.get("keywords") or []
    if isinstance(kw, list):
        return [str(k) for k in kw if k]
    return []


def _meta_frame_indices(meta: dict) -> list[int] | None:
    for key in ("frame_indices", "keyframe_indices", "frames"):
        v = meta.get(key)
        if isinstance(v, list) and v and all(isinstance(x, int) for x in v):
            return v
    return None


def main():
    # Hardcode project root để chạy được kể cả khi cwd bị reset (background shell).
    project_root = Path(__file__).resolve().parent.parent
    import os
    os.chdir(project_root)
    raw_dir = project_root / "data" / "raw"
    out_manifest = project_root / "data" / "processed" / "official_manifest.jsonl"
    out_features = project_root / "data" / "processed" / "official_features.npy"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    assets = inspect_official_assets(raw_dir)
    if not assets.clip_features:
        print("No CLIP features found", file=sys.stderr); sys.exit(1)
    if assets.keyframes is None or not assets.keyframes.exists():
        print("No Keyframes dir", file=sys.stderr); sys.exit(1)
    objects_root = assets.objects
    metadata_root = assets.metadata
    map_root = assets.map_keyframes

    image_paths = sorted([*assets.keyframes.rglob("*.jpg"), *assets.keyframes.rglob("*.png")])
    image_by_slot: dict[tuple[str, int], Path] = {}
    for im in image_paths:
        image_by_slot.setdefault((im.parent.name, int(__import__("re").findall(r"\d+", im.stem)[-1]) if __import__("re").findall(r"\d+", im.stem) else 0), im)

    obj_files_by_video: dict[str, set[str]] = {}
    if objects_root and objects_root.exists():
        for vd in objects_root.iterdir():
            if vd.is_dir():
                obj_files_by_video[vd.name] = {p.stem for p in vd.glob("*.json")}

    # Total frame count for memmap allocation.
    total = 0
    per_video_shapes: list[tuple[Path, str, int]] = []
    for fpath in assets.clip_features:
        vid = fpath.stem
        arr = np.load(fpath, mmap_mode="r")
        n = int(arr.shape[0])
        dim = int(arr.shape[1])
        per_video_shapes.append((fpath, vid, n))
        total += n

    dim = per_video_shapes[0][2] if per_video_shapes else 0
    if dim == 0:
        print("No features", file=sys.stderr); sys.exit(1)

    # Allocate output memmap + manifest writer.
    out_features.parent.mkdir(parents=True, exist_ok=True)
    out_arr = np.lib.format.open_memmap(out_features, mode="w+", dtype=np.float32, shape=(total, dim))
    offset = 0
    with out_manifest.open("w", encoding="utf-8") as mfh:
        bar = tqdm(per_video_shapes, desc="Build manifest", unit="video")
        for fpath, vid, n in bar:
            meta, meta_path = _metadata(metadata_root, vid)
            kw = _keywords(meta)
            csv_frames = _frame_id_from_csv(map_root, vid)
            meta_frames = _meta_frame_indices(meta)
            expected = csv_frames or meta_frames
            obj_files = obj_files_by_video.get(vid)
            vec = np.load(fpath, mmap_mode="r")
            for oi in range(n):
                ordinal = oi + 1
                fid = expected[oi] if (expected and oi < len(expected)) else ordinal
                keyframe_path = ""
                im = image_by_slot.get((vid, ordinal))
                if im:
                    keyframe_path = str(im)
                labels = []
                obj_path = None
                if obj_files:
                    single = f"{ordinal:03d}"
                    if single in obj_files:
                        obj_path = objects_root / vid / f"{single}.json"
                        labels = _load_object_labels(obj_path)
                rec = FrameRecord(
                    vector_id=offset + oi,
                    video_id=vid,
                    frame_id=fid,
                    keyframe_path=keyframe_path,
                    object_labels=labels,
                    metadata_keywords=kw,
                    object_path=str(obj_path) if obj_path and obj_path.exists() else None,
                    metadata_path=str(meta_path) if meta_path else None,
                    clip_feature_index=offset + oi,
                )
                mfh.write(rec.model_dump_json() + "\n")
            out_arr[offset:offset + n] = np.asarray(vec, dtype=np.float32)
            del vec
            offset += n
            bar.set_postfix(frames=offset)
    out_arr.flush()
    del out_arr
    # np.save memmap + header; re-open as normal .npy.
    # open_memmap đã tạo .npy hợp lệ.
    print(f"\nDONE: {total} frames, {len(per_video_shapes)} videos")
    print(f"manifest: {out_manifest}")
    print(f"features: {out_features}")
