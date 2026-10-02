#!/usr/bin/env python3
"""Integrate the archived AIC2026 batch 2 SigLIP2 vectors into production.

This is safe to rerun. It rebuilds from the preserved batch 1 backup and the
batch 2 source pair, writes a standards-compliant .npy file, and never expands
the multi-tens-of-GB keyframe ZIP archives.

Usage: python scripts/merge_batch2.py [--dry-run]
"""
from __future__ import annotations

import json
import csv
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "archive"
SIGLIP2_DIR = ROOT / "data" / "processed" / "siglip2"

B1_MANIFEST = SIGLIP2_DIR / "manifest_siglip2.jsonl"
B1_FEATURES = SIGLIP2_DIR / "features_siglip2.npy"
B1_MANIFEST_BACKUP = SIGLIP2_DIR / "manifest_siglip2.jsonl.bak_batch1"
B1_FEATURES_BACKUP = SIGLIP2_DIR / "features_siglip2.npy.bak_batch1"
B2_MANIFEST = ARCHIVE / "manifestb2.jsonl"
B2_FEATURES = ARCHIVE / "siglip2_featuresb2.npy"
MAP_ROOT = ROOT / "data" / "raw" / "aic26-b2-map-keyframes" / "map-keyframes"
META = SIGLIP2_DIR / "meta.json"
MODEL_NAME = "google/siglip2-base-patch16-224"
DIM = 768
CHUNK_ROWS = 8192


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_b2_map() -> dict[str, list[dict[str, str]]]:
    mapping = {}
    if MAP_ROOT.is_dir():
        for path in MAP_ROOT.glob("*.csv"):
            with path.open(encoding="utf-8-sig", newline="") as handle:
                mapping[path.stem] = list(csv.DictReader(handle))
    return mapping


def convert_b2(record: dict, vector_id: int, mapping: dict[str, list[dict[str, str]]]) -> dict:
    video_id = str(record["video_id"])
    ordinal = int(record["ordinal"])
    frame_id = int(record["frame_id"])
    frame_unit = "frame_idx"
    timestamp = None
    map_rows = mapping.get(video_id, [])
    if 1 <= ordinal <= len(map_rows):
        row = map_rows[ordinal - 1]
        if re.match(r"N\d{3}(?:-|_)", video_id.upper()):
            pts_time = row.get("pts_time")
            if pts_time in (None, ""):
                raise ValueError(f"Missing pts_time in BTC map for {video_id} ordinal {ordinal}")
            timestamp = float(pts_time)
            frame_id = round(timestamp * 1000)
            frame_unit = "milliseconds"
        else:
            mapped_idx = row.get("frame_idx")
            if mapped_idx not in (None, ""):
                frame_id = int(float(mapped_idx))
    elif re.match(r"N\d{3}(?:-|_)", video_id.upper()):
        raise ValueError(f"BTC map is missing {video_id} ordinal {ordinal}; refusing inaccurate frame_idx")

    keyframe_path = f"data/raw/Keyframes/{video_id}/{Path(record['keyframe_path']).name}"
    return {
        "vector_id": vector_id,
        "video_id": video_id,
        "frame_id": frame_id,
        "frame_unit": frame_unit,
        "keyframe_path": keyframe_path,
        "object_labels": [],
        "metadata_keywords": [],
        "title": None,
        "description": None,
        "video_path": None,
        "object_path": None,
        "metadata_path": None,
        # Batch 2 carries SigLIP2 vectors only, no aligned official CLIP row.
        "clip_feature_index": None,
        "ocr_done": False,
        "shot_id": None,
        "timestamp": timestamp,
        "source": "batch2",
        "model_version": MODEL_NAME,
        "embedding_version": MODEL_NAME,
    }


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    # Cleanup removes the archived batch 2 inputs after integration. In that
    # state, validate the canonical pair and exit cleanly instead of requiring
    # source files that were intentionally deleted.
    missing_b2 = [path for path in (B2_MANIFEST, B2_FEATURES) if not path.is_file()]
    if missing_b2:
        if not B1_MANIFEST.is_file() or not B1_FEATURES.is_file():
            raise FileNotFoundError(f"Missing canonical merged output: {B1_MANIFEST} or {B1_FEATURES}")
        records = load_jsonl(B1_MANIFEST)
        features = np.load(B1_FEATURES, mmap_mode="r")
        b2_count = sum(row.get("source") == "batch2" for row in records)
        if features.ndim != 2 or features.shape != (len(records), DIM) or features.dtype != np.float32:
            raise ValueError(f"Canonical merged pair is invalid: {features.shape}, {features.dtype}, {len(records)} manifest rows")
        if not b2_count:
            raise FileNotFoundError(f"Batch 2 source files are missing and production manifest has no batch 2 rows: {missing_b2}")
        print(f"Already integrated: {len(records)} vectors ({b2_count} batch 2); archived inputs have been cleaned.")
        return
    for path in (B2_MANIFEST, B2_FEATURES):
        if not path.is_file():
            raise FileNotFoundError(path)

    b2_records = load_jsonl(B2_MANIFEST)
    b2_features = np.load(B2_FEATURES, mmap_mode="r")
    mapping = load_b2_map()
    if b2_features.ndim != 2 or b2_features.shape[1] != DIM:
        raise ValueError(f"Batch 2 feature shape must be (N, {DIM}), got {b2_features.shape}")
    if len(b2_records) != b2_features.shape[0]:
        raise ValueError(f"Batch 2 manifest/features mismatch: {len(b2_records)} vs {b2_features.shape[0]}")

    # Preserve clean batch 1 inputs once. Rebuilding from these keeps reruns
    # idempotent and repairs an earlier raw-memmap merge without double-appending.
    backups = (B1_MANIFEST_BACKUP.is_file(), B1_FEATURES_BACKUP.is_file())
    if any(backups) and not all(backups):
        raise RuntimeError("Only one batch 1 backup exists; restore the matching pair before merging.")
    if all(backups):
        base_manifest_path, base_features_path = B1_MANIFEST_BACKUP, B1_FEATURES_BACKUP
    else:
        current_records = load_jsonl(B1_MANIFEST)
        if any(row.get("source") == "batch2" for row in current_records):
            raise RuntimeError("Batch 2 is already present but batch 1 backups are missing; refusing to append duplicates.")
        base_manifest_path, base_features_path = B1_MANIFEST, B1_FEATURES

    b1_records = load_jsonl(base_manifest_path)
    b1_features = np.load(base_features_path, mmap_mode="r")
    if b1_features.ndim != 2 or b1_features.shape[1] != DIM:
        raise ValueError(f"Batch 1 feature shape must be (N, {DIM}), got {b1_features.shape}")
    if len(b1_records) != b1_features.shape[0]:
        raise ValueError(f"Batch 1 manifest/features mismatch: {len(b1_records)} vs {b1_features.shape[0]}")

    b2_ids = [(str(row["video_id"]), int(row["ordinal"])) for row in b2_records]
    if len(set(b2_ids)) != len(b2_ids):
        raise ValueError("Batch 2 manifest contains duplicate (video_id, ordinal) keys")

    offset = len(b1_records)
    merged_manifest = b1_records + [
        convert_b2(row, offset + i, mapping) for i, row in enumerate(b2_records)
    ]
    total_rows = len(merged_manifest)
    feature_tmp = B1_FEATURES.with_name("features_siglip2.tmp.npy")
    manifest_tmp = B1_MANIFEST.with_name("manifest_siglip2.jsonl.tmp")

    print(f"Batch 1: {len(b1_records)} rows, shape={b1_features.shape}")
    print(f"Batch 2: {len(b2_records)} rows, shape={b2_features.shape}")
    print(f"Combined: {total_rows} rows x {DIM}; batch 2 starts at vector_id {offset}")
    print(f"Loaded BTC map for {len(mapping)} videos")
    print("Keyframes stay in ZIPs and will be resolved on demand.")
    if dry_run:
        return

    if not all(backups):
        shutil.copy2(base_manifest_path, B1_MANIFEST_BACKUP)
        shutil.copy2(base_features_path, B1_FEATURES_BACKUP)

    # Write a true NPY container (including its header); np.memmap would create
    # a raw binary file that np.load cannot open despite the .npy suffix.
    feature_tmp.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        feature_tmp, mode="w+", dtype=np.float32, shape=(total_rows, DIM)
    )
    for start in range(0, len(b1_records), CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, len(b1_records))
        output[start:end] = b1_features[start:end]
    for start in range(0, len(b2_records), CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, len(b2_records))
        output[offset + start:offset + end] = b2_features[start:end]
    output.flush()
    del output

    # Validate the staged container before replacing production assets.
    staged = np.load(feature_tmp, mmap_mode="r")
    if staged.shape != (total_rows, DIM) or staged.dtype != np.float32:
        raise ValueError(f"Staged .npy failed validation: {staged.shape}, {staged.dtype}")
    del staged

    with manifest_tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in merged_manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    feature_tmp.replace(B1_FEATURES)
    manifest_tmp.replace(B1_MANIFEST)

    metadata = {}
    if META.exists():
        metadata = json.loads(META.read_text(encoding="utf-8"))
    metadata.update({
        "model_name": MODEL_NAME,
        "embedding_dim": DIM,
        "dtype": "float32",
        "frame_count": total_rows,
        "video_count": len({str(row["video_id"]) for row in merged_manifest}),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batches": {"batch1_frames": len(b1_records), "batch2_frames": len(b2_records)},
    })
    META.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Integrated {total_rows} vectors. NPY size: {B1_FEATURES.stat().st_size:,} bytes")
    print("The app loads these default SigLIP2 files on its next restart.")


if __name__ == "__main__":
    main()
