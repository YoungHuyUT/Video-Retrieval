"""Object detection for lightweight frames (spec §5, §13 — object evidence).

The lightweight pipeline's NEW (uniform/motion) frames have NO object labels —
BTC frames get theirs from the official OpenImages detection JSON, but NEW
frames are decoded fresh and never run through a detector.  Without labels, the
object-evidence reranker (``rerank_with_object_evidence``) and the object
modality of Adaptive Fusion are blind to NEW frames: a query like "chai nước"
can only fire on BTC keyframes, never on the denser uniform/motion coverage.

This module runs ``fasterrcnn_resnet50_fpn`` (torchvision, CPU) over every NEW
keyframe, keeps COCO entities above the same score thresholds the BTC loader
uses (``_OBJECT_SCORE_THRESHOLD = 0.3`` / ``_OBJECT_PRESENT_THRESHOLD = 0.4``),
and writes the labels back into the catalog so ``_regenerate_manifest`` carries
them into the manifest.  The COCO vocabulary overlaps the OpenImages/BTC terms
the query aliases cover (person, car, dog, bottle, …), so a single object
modality now spans both frame sources.

The step is RESUME-ABLE: frames already labelled (object_labels non-empty) are
skipped, so interrupting and re-running only processes the remainder.  All
writes go through :class:`CatalogLock` so a concurrent build can't corrupt the
SQLite catalog.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from aic2026.lightweight.catalog import CatalogLock, LightweightCatalog

logger = logging.getLogger(__name__)

# Mirrors ingestion/official_index.py so NEW-frame labels are scored identically
# to BTC labels (a frame with no entity >= 0.4 present-threshold is "blurry").
_OBJECT_SCORE_THRESHOLD = 0.3
_OBJECT_PRESENT_THRESHOLD = 0.4

# COCO instance category names (torchvision fasterrcnn_resnet50_fpn, 91 classes;
# index 0 = background). Padded to length 91 so a model output index never
# IndexErrors — any gap maps to 'N/A' and is dropped by the keep logic below.
_COCO_91 = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush", "N/A",
]
COCO_LABELS: list[str] = (_COCO_91 + ["N/A"])[:91]


def _keep_labels(boxes, labels, scores, score_thr, present_thr):
    """Return (sorted_unique_labels) for entities above thresholds.

    ``labels`` are 1-based COCO indices.  Mirrors ``_load_object_labels``: keep
    an entity if its score >= score_thr, and require at least one entity >=
    present_thr else return [] (blurry frame -> no object evidence).
    """
    if boxes is None or len(labels) == 0:
        return []
    keep: list[str] = []
    has_strong = False
    for lab, sc in zip(labels.tolist(), scores.tolist()):
        if sc < score_thr:
            continue
        name = COCO_LABELS[lab] if 0 <= lab < len(COCO_LABELS) else "N/A"
        if name in ("N/A", "__background__"):
            continue
        keep.append(name)
        if sc >= present_thr:
            has_strong = True
    if not has_strong:
        return []
    return sorted(set(keep))


def _load_model(device: str):
    """Load the detector once.

    We use ``ssdlite320_mobilenet_v3_large`` rather than FasterRCNN: on CPU it is
    ~38x faster (~0.26s vs ~10s per 720p frame) for the same COCO-80 vocabulary,
    at a small precision cost.  Object *presence* (what the reranker needs) is
    robust either way, and the 11h full-corpus runtime on CPU is the difference
    between feasible and not.  CUDA is used if available.
    """
    import torch
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )

    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights, score_thresh=0.05)
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    return model, dev


def _decode_image(path: Path, max_edge: int = 512):
    """Load a keyframe, resize so the long edge <= ``max_edge`` (keep aspect).

    FasterRCNN on a 1280x720 frame takes ~19s on CPU; shrinking to ~512px drops
    that to a fraction (fewer region proposals) with negligible label-quality
    loss for coarse object *presence* (the only thing the reranker needs).  The
    image is padded to a square so a whole batch shares one tensor shape.
    """
    from PIL import Image
    import torch

    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_edge / max(w, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.BILINEAR)
    arr = __import__("numpy").asarray(img)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    # Pad to square (max_edge x max_edge) so batches are uniform.
    pad_w = max_edge - nw
    pad_h = max_edge - nh
    if pad_w or pad_h:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))
    return tensor


def detect_objects(
    root: Path,
    *,
    sources: tuple[str, ...] = ("uniform", "motion"),
    device: str = "cpu",
    score_thr: float = _OBJECT_SCORE_THRESHOLD,
    present_thr: float = _OBJECT_PRESENT_THRESHOLD,
    limit: int | None = None,
) -> dict:
    """Detect objects for every NEW frame lacking labels, persist to catalog.

    Returns a summary dict.  Idempotent: frames whose ``object_labels`` column
    is already non-empty are skipped.
    """
    root = Path(root)
    catalog_path = root / "catalog.db"
    if not catalog_path.exists():
        raise FileNotFoundError(f"no catalog at {catalog_path} — run lw-merge-btc + lw-build first")

    with CatalogLock(root):
        catalog = LightweightCatalog(catalog_path)
        # Frames to process: specified sources, not yet labelled.
        placeholders = ",".join("?" for _ in sources) or "?"
        rows = catalog.conn.execute(
            f"""
            SELECT video_id, frame_id, path, object_labels
            FROM frames
            WHERE source IN ({placeholders})
            ORDER BY video_id ASC, frame_id ASC
            """,
            tuple(sources),
        ).fetchall()

        # Skip ones already labelled (resume support).
        pending = [
            (r["video_id"], int(r["frame_id"]), r["path"])
            for r in rows
            if not LightweightCatalog.parse_object_labels(r["object_labels"])
        ]
        catalog.close()

    if not pending:
        logger.info("detect-objects: nothing to do (all %s frames labelled)", sources)
        return {"processed": 0, "skipped": len(rows), "total": len(rows)}

    logger.info("detect-objects: %d %s frames to label (%d already done)",
                len(pending), sources, len(rows) - len(pending))

    model, dev = _load_model(device)
    import torch

    total = len(pending)
    processed = 0
    labelled_frames = 0
    t0 = time.time()

    # Inference in BATCHES of frames (one tensor stack per call) — dramatically
    # faster than per-frame inference on CPU (~19s/frame at 720p shrinks to a few
    # seconds per 8-frame batch at 512px).  Writes commit per BATCH_COMMIT frames
    # so progress survives an interrupt.
    INFER_BATCH = 8
    COMMIT_BATCH = 256

    pending_labels: list[tuple[str, int, list[str]]] = []
    last_video = None
    batch: list[tuple[str, int, list[str]]] = []

    def _flush() -> None:
        nonlocal labelled_frames
        if not batch:
            return
        with CatalogLock(root):
            cat = LightweightCatalog(catalog_path)
            for vid, fid, labels in batch:
                cat.set_frame_object_labels(vid, fid, labels)
                if labels:
                    labelled_frames += 1
            cat.commit_frames()
            cat.close()
        batch.clear()

    i = 0
    while i < total:
        chunk = pending[i : i + INFER_BATCH]
        i += len(chunk)
        tensors = []
        meta = []
        for vid, fid, path in chunk:
            try:
                t = _decode_image(Path(path)).to(dev)
                tensors.append(t)
                meta.append((vid, fid, path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("decode failed %s/%s: %s", vid, fid, exc)
                pending_labels.append((vid, fid, []))
        if tensors:
            stack = torch.stack(tensors, 0)
            try:
                with torch.inference_mode():
                    outs = model(stack)
            except Exception as exc:  # noqa: BLE001
                logger.warning("batch infer failed: %s", exc)
                outs = [None] * len(meta)
            for (vid, fid, _), out in zip(meta, outs):
                if out is None:
                    pending_labels.append((vid, fid, []))
                    continue
                labels = _keep_labels(
                    out["boxes"], out["labels"], out["scores"], score_thr, present_thr
                )
                pending_labels.append((vid, fid, labels))
        processed += len(chunk)
        # Roll pending labels into the commit batch; flush at video boundary or
        # every COMMIT_BATCH frames.
        while pending_labels:
            vid, fid, labels = pending_labels.pop(0)
            batch.append((vid, fid, labels))
            if vid != last_video or len(batch) >= COMMIT_BATCH or processed >= total:
                _flush()
            last_video = vid
        if limit is not None and processed >= limit:
            break
        if processed % 200 == 0:
            rate = processed / max(time.time() - t0, 1e-6)
            logger.info("detect-objects: %d/%d (%.1f fps)", processed, total, rate)

    # Final flush (handles the limit-break tail).
    _flush()

    # Carry the new labels into manifest_lw.jsonl (query-time reads object_labels
    # from the manifest, not the catalog directly). Mirrors lw-build's commit
    # protocol so the two stay row-aligned.
    from aic2026.lightweight.build import LightweightConfig, _regenerate_manifest

    cfg = LightweightConfig(root=root)
    with CatalogLock(root):
        cat = LightweightCatalog(catalog_path)
        _regenerate_manifest(cat, cfg)
        cat.close()

    summary = {
        "processed": processed,
        "labelled_frames": labelled_frames,
        "skipped": len(rows) - len(pending),
        "total": len(rows),
        "seconds": round(time.time() - t0, 1),
    }
    logger.info("detect-objects done: %s", summary)
    return summary
