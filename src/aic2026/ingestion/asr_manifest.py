from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from aic2026.ingestion.manifest import load_manifest
from aic2026.models import FrameRecord

logger = logging.getLogger(__name__)


def _load_pts_map(map_keyframes_dir: Path | None, video_id: str) -> dict[int, float]:
    """Read CSV map-keyframes to build frame_idx -> pts_time mapping."""
    if map_keyframes_dir is None:
        return {}
    candidates = [
        map_keyframes_dir / "map-keyframes" / f"{video_id}.csv",
        map_keyframes_dir / f"{video_id}.csv",
    ]
    csv_path = next((c for c in candidates if c.exists()), None)
    if csv_path is None:
        return {}
    pts_map: dict[int, float] = {}
    try:
        with csv_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                f_idx = row.get("frame_idx")
                pts = row.get("pts_time")
                if f_idx and pts:
                    try:
                        pts_map[int(f_idx)] = float(pts)
                    except ValueError:
                        pass
    except (OSError, ValueError, KeyError):
        pass
    return pts_map


def _load_asr_segments(asr_dir: Path, video_id: str) -> list[dict[str, Any]]:
    """Load timestamped ASR segments for a given video."""
    json_path = asr_dir / f"{video_id}.json"
    if not json_path.exists():
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [
                seg for seg in data
                if isinstance(seg, dict) and "start" in seg and "end" in seg and "text" in seg
            ]
    except (OSError, json.JSONDecodeError):
        pass
    return []


def enrich_manifest_with_asr(
    manifest_path: Path,
    asr_dir: Path,
    output_path: Path | None = None,
    map_keyframes_dir: Path | None = None,
    time_window: float = 1.5,
    default_fps: float = 25.0,
    video_prefix: str = "",
    resume: bool = False,
    progress_callback: callable | None = None,
) -> tuple[int, int]:
    """Enrich FrameRecords in manifest with ASR speech transcripts matching keyframe timestamps.

    Returns: ``(total_records_written, enriched_frames_count)``.
    """
    manifest_path = Path(manifest_path)
    asr_dir = Path(asr_dir)
    target = Path(output_path) if output_path is not None else manifest_path

    records = load_manifest(manifest_path)
    if not records:
        logger.warning("Manifest %s is empty", manifest_path)
        return 0, 0

    prefixes = [p.strip() for p in video_prefix.split(",") if p.strip()]

    # Cache per-video ASR segments & PTS time maps
    asr_cache: dict[str, list[dict[str, Any]]] = {}
    pts_cache: dict[str, dict[int, float]] = {}

    unique_videos = {r.video_id for r in records}
    for vid in unique_videos:
        if prefixes and not any(vid.startswith(p) for p in prefixes):
            continue
        asr_cache[vid] = _load_asr_segments(asr_dir, vid)
        if map_keyframes_dir:
            pts_cache[vid] = _load_pts_map(map_keyframes_dir, vid)

    enriched_count = 0
    written = 0
    total = len(records)

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as out_f:
        for idx, record in enumerate(records):
            vid = record.video_id
            if prefixes and not any(vid.startswith(p) for p in prefixes):
                out_f.write(record.model_dump_json() + "\n")
                written += 1
                continue

            if resume and getattr(record, "asr_done", False):
                out_f.write(record.model_dump_json() + "\n")
                written += 1
                if record.asr_text:
                    enriched_count += 1
                continue

            segments = asr_cache.get(vid, [])
            if not segments:
                # No ASR for this video; mark done and pass through
                rec = record.model_copy(update={"asr_done": True})
                out_f.write(rec.model_dump_json() + "\n")
                written += 1
                continue

            # Determine timestamp T
            pts_map = pts_cache.get(vid, {})
            if record.frame_id in pts_map:
                t = pts_map[record.frame_id]
            else:
                t = float(record.frame_id) / default_fps

            # Find matching segments in window [start - time_window, end + time_window]
            matched_texts = []
            for seg in segments:
                start_t = float(seg["start"]) - time_window
                end_t = float(seg["end"]) + time_window
                if start_t <= t <= end_t:
                    txt = str(seg["text"]).strip()
                    if txt and txt not in matched_texts:
                        matched_texts.append(txt)

            existing = list(record.asr_text or [])
            merged = list(dict.fromkeys([*existing, *matched_texts]))

            if merged:
                enriched_count += 1

            rec = record.model_copy(update={"asr_text": merged, "asr_done": True})
            out_f.write(rec.model_dump_json() + "\n")
            written += 1

            if progress_callback is not None and (idx + 1) % 1000 == 0:
                progress_callback(idx + 1, total, enriched_count)

    logger.info(
        "Enriched %d / %d manifest records with ASR text -> %s",
        enriched_count,
        written,
        target,
    )
    return written, enriched_count
