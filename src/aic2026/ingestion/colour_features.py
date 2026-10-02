"""Offline precompute of per-frame colour evidence.

CLIP's global embedding is nearly colour-blind, so "red car" and "blue car"
retrieve almost identically.  The original runtime reranker compensated by
decoding keyframe images *per query* with PIL to estimate colour fractions —
slow, limited to a top-N window, and blind for videos with no local images.

This module moves all the expensive pixel work to a **single offline pass**.
For every keyframe with an image it stores, in a tiny JSONL sidecar:

    {"vector_id": 98,
     "dom": "green",
     "fracs": {"red": 0.02, "green": 0.31, ...},
     "obj": {"person": ["red", 0.42], "car": ["blue", 0.50]}}

At query time the runtime reranker (``contrastive_clip_colour_rerank``) reads
colour evidence directly from decoded keyframe crops via the CLIP image tower —
no offline sidecar required.  This module still exists so the sidecar can be
precomputed for experiments/benchmarks that want a cheaper, dict-lookup colour
signal.  Frames without a local image are simply omitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from aic2026.models import FrameRecord
from aic2026.reranking.color import _COLOURS, colour_fraction

_COLOUR_LIST = sorted(_COLOURS)
_DEFAULT_OUTPUT = Path("data/processed/colour_features.jsonl")


def _resolve(image_path: str | None, video_id: str | None, root: Path) -> Path | None:
    """Find a keyframe/object file on disk given a manifest-relative path."""
    if not image_path:
        return None
    p = Path(image_path)
    if p.is_absolute() or p.exists():
        return p
    name = p.name
    if video_id:
        cand = root / video_id / name
        if cand.exists():
            return cand
    alt = Path.cwd() / image_path
    return alt if alt.exists() else None


def _frame_colour_profile(rgb: np.ndarray) -> tuple[dict[str, float], str]:
    """Return per-colour fractions and the dominant colour for an RGB image."""
    fracs = {c: float(colour_fraction(rgb, c)) for c in _COLOUR_LIST}
    dom = max(fracs, key=fracs.get)
    return fracs, dom


def _object_colours(
    rgb: np.ndarray, object_path: str | None, root: Path
) -> dict[str, list]:
    """Colour of each detected object box (OpenImages entities with score>=0.20)."""
    if not object_path:
        return {}
    p = _resolve(object_path, None, root)
    if p is None or not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    names = data.get("detection_class_entities") or []
    scores = data.get("detection_scores") or []
    boxes = data.get("detection_boxes") or []
    height, width = rgb.shape[:2]
    out: dict[str, list] = {}
    for name, raw_score, box in zip(names, scores, boxes):
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if score < 0.20 or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        top, left, bottom, right = (float(v) for v in box)
        if not (bottom > top and right > left):
            continue
        y0, y1 = int(max(0.0, top * height)), int(min(height, bottom * height))
        x0, x1 = int(max(0.0, left * width)), int(min(width, right * width))
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        _, dom = _frame_colour_profile(crop)
        fracs = {c: float(colour_fraction(crop, c)) for c in _COLOUR_LIST}
        frac = fracs[dom]
        if frac >= 0.10:
            # Store key lowercased so it matches query alias words downstream
            # (OpenImages names like "Car" lowercased land in the alias set).
            out[str(name).casefold()] = [dom, round(float(frac), 3)]
    return out


def _process_record(record: FrameRecord, keyframes_root: Path, objects_root: Path) -> dict | None:
    path = _resolve(record.keyframe_path, record.video_id, keyframes_root)
    if path is None:
        return None
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.thumbnail((192, 192))
            rgb = np.asarray(image.convert("RGB"))
    except (OSError, ValueError):
        return None
    if rgb.size == 0:
        return None
    fracs, dom = _frame_colour_profile(rgb)
    obj = _object_colours(rgb, record.object_path, objects_root)
    return {
        "vector_id": record.vector_id,
        "dom": dom,
        "fracs": {c: round(float(v), 4) for c, v in fracs.items()},
        "obj": obj,
    }


def build_colour_features(
    manifest_path: Path,
    keyframes_root: Path = Path("data/raw/Keyframes"),
    objects_root: Path = Path("data/raw/Objects"),
    output_path: Path = _DEFAULT_OUTPUT,
    limit: int = 0,
) -> int:
    """Precompute the colour sidecar over the manifest. Returns record count.

    Runs once. ``limit`` > 0 restricts to the first N manifest records (dev).
    Frames without a local keyframe image are skipped (no evidence).
    """
    from aic2026.ingestion import load_manifest

    records: list[FrameRecord] = load_manifest(manifest_path)
    if limit and limit > 0:
        records = records[:limit]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8", buffering=1) as fh:  # line-buffered
        for record in tqdm(records, desc="colour features", unit="frame"):
            entry = _process_record(record, Path(keyframes_root), Path(objects_root))
            if entry is None:
                continue
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()  # ghi ra đĩa ngay lập tức
            written += 1
    return written


def load_colour_features(path: Path = _DEFAULT_OUTPUT) -> dict[int, dict]:
    """Read the colour sidecar into ``vector_id`` → entry. Missing file → {}."""
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            vid = entry.get("vector_id")
            if vid is not None:
                out[int(vid)] = entry
    return out
