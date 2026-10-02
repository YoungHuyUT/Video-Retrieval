#!/usr/bin/env python3
"""Download Batch 2 object JSON ZIPs and merge confident labels into SigLIP manifest.

Archives are handled one at a time and deleted after their labels are read, so
only the canonical JSONL manifest remains on disk. Frame joins use (video_id,
numeric filename stem), which is the keyframe ordinal used by Batch 2.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/processed/siglip2/manifest_siglip2.jsonl"
META = ROOT / "data/processed/siglip2/meta.json"
ARCHIVE_DIR = ROOT / "data/archive"
ARCHIVES = {
    "Objects_S01.zip": "https://aic-data.ledo.io.vn/Objects_S01.zip",
    "Objects_M01_M10.zip": "https://aic-data.ledo.io.vn/Objects_M01_M10.zip",
    "Objects_N001_N100.zip": "https://aic-data.ledo.io.vn/Objects_N001_N100.zip",
}


def _download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "AIC2026-object-label-enrichment"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        if offset and response.status != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        received = offset
        next_report = received + 64 * 1024 * 1024
        with partial.open(mode) as out:
            while block := response.read(8 * 1024 * 1024):
                out.write(block)
                received += len(block)
                if received >= next_report:
                    print(f"  downloaded {received / (1024 * 1024):.0f} MiB", flush=True)
                    next_report = received + 64 * 1024 * 1024
    if not zipfile.is_zipfile(partial):
        raise zipfile.BadZipFile(f"Downloaded archive is incomplete: {destination.name}")
    partial.replace(destination)


def _labels(data: object) -> list[str]:
    if not isinstance(data, dict):
        return []
    entities = data.get("detection_class_entities") or []
    scores = data.get("detection_scores") or []
    parsed: list[float] = []
    for score in scores:
        try:
            parsed.append(float(score))
        except (TypeError, ValueError):
            parsed.append(1.0)
    if not any(i < len(parsed) and parsed[i] >= 0.4 for i in range(len(entities))):
        return []
    return sorted({str(entity) for i, entity in enumerate(entities)
                   if i >= len(parsed) or parsed[i] >= 0.3})


def _read_labels(path: Path) -> tuple[dict[tuple[str, int], list[str]], int]:
    found: dict[tuple[str, int], list[str]] = {}
    bad = 0
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            rel = PurePosixPath(member.filename.replace("\\", "/"))
            if rel.suffix.lower() != ".json" or not rel.stem.isdigit() or len(rel.parts) < 2:
                continue
            try:
                data = json.loads(archive.read(member).decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError, OSError, zipfile.BadZipFile):
                bad += 1
                continue
            found[(rel.parent.name, int(rel.stem))] = _labels(data)
    return found, bad


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    keep_archives = "--keep-archives" in sys.argv
    keep_backup = "--backup" in sys.argv
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    batch2_rows: set[tuple[str, int]] = set()
    for line in MANIFEST.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("source") != "batch2":
            continue
        stem = Path(str(row.get("keyframe_path", ""))).stem
        if stem.isdigit():
            batch2_rows.add((str(row["video_id"]), int(stem)))
    if not batch2_rows:
        raise RuntimeError("No Batch 2 records were found in the canonical manifest")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    merged: dict[tuple[str, int], list[str]] = {}
    unmatched_members = 0
    for name, url in ARCHIVES.items():
        archive_path = ARCHIVE_DIR / name
        if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
            archive_path.unlink(missing_ok=True)
            print(f"Downloading {name} ...", flush=True)
            _download(url, archive_path)
        print(f"Reading {name} ...", flush=True)
        labels, bad = _read_labels(archive_path)
        selected = {key: value for key, value in labels.items() if key in batch2_rows}
        merged.update(selected)
        unmatched_members += bad
        print(f"  {len(labels):,} frames read; {len(selected):,} matched manifest rows; {bad} invalid JSON", flush=True)
        if not keep_archives:
            archive_path.unlink(missing_ok=True)

    if unmatched_members:
        raise RuntimeError(f"Found {unmatched_members} invalid object JSON files; manifest left unchanged")
    if len(merged) < len(batch2_rows) * 0.95:
        missing = len(batch2_rows) - len(merged)
        raise RuntimeError(f"Only matched {len(merged):,}/{len(batch2_rows):,} Batch 2 rows ({missing:,} missing); manifest left unchanged")
    if dry_run:
        print(f"Dry run: {len(merged):,} rows have object data; manifest unchanged")
        return 0

    temporary = MANIFEST.with_name(MANIFEST.name + ".tmp")
    backup = MANIFEST.with_name(MANIFEST.name + ".bak_before_objects")
    if keep_backup and not backup.exists():
        import shutil
        shutil.copy2(MANIFEST, backup)
    enriched = 0
    with MANIFEST.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8", newline="\n") as target:
        for line in source:
            row = json.loads(line)
            if row.get("source") == "batch2":
                stem = Path(str(row.get("keyframe_path", ""))).stem
                labels = merged.get((str(row["video_id"]), int(stem))) if stem.isdigit() else None
                if labels is not None:
                    row["object_labels"] = labels
                    enriched += bool(labels)
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(MANIFEST)
    if META.exists():
        metadata = json.loads(META.read_text(encoding="utf-8"))
        metadata["batch2_object_labels"] = {
            "matched_frames": len(merged),
            "frames_with_labels": enriched,
            "frames_without_confident_labels": len(batch2_rows) - enriched,
            "minimum_label_score": 0.3,
            "minimum_object_presence_score": 0.4,
            "archives": list(ARCHIVES),
        }
        META.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    backup_note = f"; backup: {backup.name}" if keep_backup else ""
    print(f"Updated {MANIFEST}: {enriched:,} Batch 2 frames now have object labels{backup_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
