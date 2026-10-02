"""Merge the existing BTC keyframes into the lightweight manifest (spec §2, §13).

The official BTC frames are the curated baseline. We DO NOT re-decode or copy
them — we simply record each one as a ``source="btc"`` FrameRecord pointing at
its existing ``keyframe_path`` on disk, with a *real* timestamp recovered from
the ``map-keyframes`` CSV (ordinal ``n`` → ``frame_idx`` → ``pts_time`` seconds).

The BTC CLIP vectors already live in ``official_features.npy`` and are reused
directly by :mod:`aic2026.lightweight.embed` (no re-embedding). This merge only
produces the manifest + a ``clip_index`` back-reference into the official matrix.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from aic2026.models import FrameRecord

logger = logging.getLogger(__name__)

EMBEDDING_VERSION = "lightweight-v1"
MODEL_VERSION = "openai/ViT-B-32 + google/siglip2-so400m-patch14-384"


@dataclass
class MergeBtcConfig:
    root: Path = Path("data/processed/new_lightweight")
    official_manifest: Path = Path("data/processed/official_manifest.jsonl")
    map_dir: Path = Path("data/raw/map-keyframes-aic25-b1/map-keyframes")
    # Cache the parsed timestamp map so repeated runs are cheap.
    _ts_cache: dict[str, dict[int, float]] = field(default_factory=dict, repr=False)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest_lw.jsonl"

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    # -- helpers ----------------------------------------------------------
    def _load_ts_map(self, video_id: str) -> dict[int, float]:
        if video_id in self._ts_cache:
            return self._ts_cache[video_id]
        csv_path = self.map_dir / f"{video_id}.csv"
        ts: dict[int, float] = {}
        if csv_path.exists():
            with csv_path.open(encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    try:
                        ts[int(row["frame_idx"])] = float(row["pts_time"])
                    except (KeyError, ValueError):
                        continue
        self._ts_cache[video_id] = ts
        return ts

    def _next_frame_id(self, cfg_root: Path, video_id: str) -> int:
        """Continue frame_id per-video so BTC + uniform + motion never collide."""
        # We keep a per-(video) running counter persisted in a sidecar so a
        # resumed run continues numbering. Stored as JSON {video_id: max_frame_id}.
        sidecar = cfg_root / "frame_id_counter.json"
        if not sidecar.exists():
            return 0
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            return int(data.get(video_id, 0)) + 1
        except (json.JSONDecodeError, ValueError):
            return 0

    def merge(self, limit: int | None = None) -> int:
        """Write BTC FrameRecords to ``manifest_lw.jsonl``. Returns #records."""
        self.ensure_dirs()
        # Build per-video BTC records, assigning lightweight frame_ids that start
        # after the highest BTC frame_id for that video (uniform/motion append).
        per_video_max: dict[str, int] = {}
        records: list[FrameRecord] = []
        seen_videos: set[str] = set()

        with self.official_manifest.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                video_id = o["video_id"]
                seen_videos.add(video_id)
                ts_map = self._load_ts_map(video_id)
                src_frame_id = int(o.get("frame_id", 0))
                timestamp = ts_map.get(src_frame_id, None)
                if timestamp is None:
                    # Fallback: BTC ordinal unavailable — skip timestamp rather
                    # than guess (zoom-in needs real seconds).
                    logger.warning(
                        "no timestamp for %s frame %s", video_id, src_frame_id
                    )
                    timestamp = 0.0
                # Use the REAL BTC frame_id (already a true frame index from the
                # map-keyframes CSV via official_index). Do NOT re-number — the
                # competition ground truth is keyed on these real frame indices.
                lw_frame_id = src_frame_id
                per_video_max[video_id] = max(per_video_max.get(video_id, -1), lw_frame_id)
                rec = FrameRecord(
                    vector_id=int(o.get("vector_id", 0)),
                    video_id=video_id,
                    frame_id=lw_frame_id,
                    keyframe_path=str(o.get("keyframe_path", "")),
                    object_labels=list(o.get("object_labels", []) or []),
                    clip_feature_index=int(o["clip_feature_index"]),
                    shot_id=None,
                    timestamp=timestamp,
                    source="btc",
                    model_version=MODEL_VERSION,
                    embedding_version=EMBEDDING_VERSION,
                )
                records.append(rec)
                if limit is not None and len(records) >= limit:
                    break

        # Persist manifest (overwrite — BTC is the deterministic base layer),
        # unless a prior merge already wrote uniform/motion rows (don't wipe a
        # completed build on a re-merge).
        from aic2026.lightweight.catalog import CatalogLock, LightweightCatalog

        have_new = False
        if self.manifest_path.exists():
            try:
                import json as _json

                with self.manifest_path.open("r", encoding="utf-8") as _fh:
                    for _line in _fh:
                        _line = _line.strip()
                        if not _line:
                            continue
                        if _json.loads(_line).get("source") != "btc":
                            have_new = True
                            break
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                have_new = False
        if not have_new:
            self.manifest_path.write_text(
                "\n".join(r.model_dump_json() for r in records) + ("\n" if records else ""),
                encoding="utf-8",
            )
        # Mirror BTC rows into the catalog as well. The catalog becomes the
        # COMPLETE source of truth (BTC + uniform + motion), so manifest_lw.jsonl
        # is always regenerable from it — a crash mid-build heals on resume.
        # Held under CatalogLock so a concurrent lw-build/lw-merge can't corrupt it.
        with CatalogLock(self.root):
            cat = LightweightCatalog(self.root / "catalog.db")
            try:
                cat.conn.execute("BEGIN")
                cat.conn.executemany(
                    """
                    INSERT INTO videos (video_id, fps, frame_count, source_hash, is_built)
                    VALUES (?, 0, 0, NULL, 0)
                    ON CONFLICT(video_id) DO UPDATE SET video_id=excluded.video_id
                    """,
                    [(rec.video_id,) for rec in records],
                )
                cat.conn.executemany(
                    """
                    INSERT INTO frames
                        (video_id, frame_id, timestamp, source, path, clip_index, model_version, embedding_version, object_labels)
                    VALUES (?, ?, ?, 'btc', ?, ?, ?, ?, ?)
                    ON CONFLICT(video_id, frame_id) DO UPDATE SET
                        -- BTC is the authoritative baseline.  A dynamic frame
                        -- can share the same original frame_id; it must never
                        -- overwrite the BTC row or make that baseline vanish.
                        timestamp=excluded.timestamp, source='btc', path=excluded.path,
                        clip_index=excluded.clip_index,
                        model_version=excluded.model_version,
                        embedding_version=excluded.embedding_version,
                        object_labels=COALESCE(excluded.object_labels, frames.object_labels)
                    """,
                    [
                        (
                            rec.video_id,
                            int(rec.frame_id),
                            float(rec.timestamp if rec.timestamp is not None else 0.0),
                            rec.keyframe_path,
                            int(rec.clip_feature_index) if rec.clip_feature_index is not None else None,
                            rec.model_version,
                            rec.embedding_version,
                            json.dumps(sorted(set(rec.object_labels or [])), ensure_ascii=False)
                            if rec.object_labels
                            else None,
                        )
                        for rec in records
                    ],
                )
                cat.conn.commit()
            finally:
                cat.close()
        # The catalog is the durable source of truth.  In particular, the UPSERT
        # above may have repaired an old dynamic/BTC primary-key collision, so
        # regenerate even when a previous run already added dynamic rows.
        from aic2026.lightweight.build import _regenerate_manifest

        with CatalogLock(self.root):
            cat = LightweightCatalog(self.root / "catalog.db")
            try:
                _regenerate_manifest(cat, MergeBtcConfig(root=self.root))
            finally:
                cat.close()
        # Repaired BTC rows may replace dynamic rows at the same original frame
        # index.  Their old vector rows no longer describe the regenerated
        # manifest, so never let a later query use them by accident.  They are
        # derived artifacts and `lw-embed` recreates them from this manifest.
        for feature_name in ("features_clip.npy", "features_siglip2.npy"):
            feature_path = self.root / feature_name
            if feature_path.exists():
                feature_path.unlink()
                logger.warning("invalidated stale %s after BTC merge", feature_path)
        # Remember per-video max frame_id so uniform/motion append cleanly.
        counter_path = self.root / "frame_id_counter.json"
        counter = {
            v: per_video_max.get(v, 0) for v in seen_videos
        }
        counter_path.write_text(json.dumps(counter), encoding="utf-8")
        logger.info("merge-btc: %d records from %d videos", len(records), len(seen_videos))
        return len(records)


def load_manifest(path: Path) -> list[FrameRecord]:
    out: list[FrameRecord] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(FrameRecord.model_validate_json(line))
    return out
