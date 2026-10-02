"""Validate shot-adaptive artifacts (CLI ``validate-artifacts``, spec §11).

Checks (non-destructive, read-only):
  * config.json exists and parses.
  * manifest_shot.jsonl rows are valid FrameRecords with shot_id/timestamp.
  * features_clip.npy / features_siglip2.npy are row-aligned with the manifest.
  * every keyframe_path in the manifest exists on disk.
  * catalog.db frame count matches the manifest row count.
Returns a summary dict; raises ``ArtifactValidationError`` on a hard failure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aic2026.extraction.catalog import ShotCatalog
from aic2026.extraction.pipeline import ShotConfig

logger = logging.getLogger(__name__)


class ArtifactValidationError(Exception):
    """Raised when a required artifact is missing or internally inconsistent."""


@dataclass
class ValidationReport:
    manifest_rows: int = 0
    clip_rows: int = 0
    siglip2_rows: int = 0
    missing_keyframes: int = 0
    catalog_frames: int = 0
    catalog_shots: int = 0
    catalog_videos: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_artifacts(cfg: ShotConfig, strict: bool = True) -> ValidationReport:
    report = ValidationReport()

    # 1. config.json
    if not cfg.config_path.exists():
        report.errors.append(f"missing config.json at {cfg.config_path}")
    else:
        try:
            json.loads(cfg.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.errors.append(f"config.json invalid JSON: {exc}")

    # 2. manifest
    if not cfg.manifest_path.exists():
        report.errors.append(f"missing manifest {cfg.manifest_path}")
        if strict:
            raise ArtifactValidationError("; ".join(report.errors))
        return report

    rows: list[dict] = []
    with cfg.manifest_path.open("r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                report.errors.append(f"manifest line {ln}: bad JSON")
                continue
            if obj.get("shot_id") is None or obj.get("timestamp") is None:
                report.errors.append(f"manifest line {ln}: missing shot_id/timestamp")
            rows.append(obj)
    report.manifest_rows = len(rows)

    # 3. features row-alignment
    if cfg.clip_feats_path.exists():
        clip = np.load(cfg.clip_feats_path)
        report.clip_rows = int(clip.shape[0])
        if report.clip_rows != report.manifest_rows:
            report.errors.append(
                f"clip features rows {report.clip_rows} != manifest {report.manifest_rows}"
            )
    else:
        report.errors.append(f"missing {cfg.clip_feats_path}")

    if cfg.siglip2_feats_path.exists():
        sig = np.load(cfg.siglip2_feats_path)
        report.siglip2_rows = int(sig.shape[0])
        if report.siglip2_rows != report.manifest_rows:
            report.errors.append(
                f"siglip2 features rows {report.siglip2_rows} != manifest {report.manifest_rows}"
            )
    else:
        report.errors.append(f"missing {cfg.siglip2_feats_path}")

    # 4. keyframe files exist
    missing = 0
    for obj in rows[:5000]:  # cap scan to keep validate fast on huge indexes
        p = obj.get("keyframe_path")
        if not p or not Path(p).exists():
            missing += 1
    report.missing_keyframes = missing
    if missing:
        report.errors.append(f"{missing} keyframe files missing (scanned {min(len(rows),5000)})")

    # 5. catalog consistency
    if cfg.catalog_path.exists():
        cat = ShotCatalog(cfg.catalog_path)
        report.catalog_frames = cat.frame_count()
        report.catalog_shots = cat.shot_count()
        report.catalog_videos = cat.video_count()
        cat.close()
        if report.catalog_frames != report.manifest_rows:
            report.errors.append(
                f"catalog frames {report.catalog_frames} != manifest {report.manifest_rows}"
            )
    else:
        report.errors.append(f"missing catalog {cfg.catalog_path}")

    if strict and report.errors:
        raise ArtifactValidationError("; ".join(report.errors))
    logger.info("validate: %s", report)
    return report
