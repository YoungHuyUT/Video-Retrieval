"""Shot-adaptive offline extraction pipeline (spec §1-§13).

Public surface used by the CLI and retrieval layer:

* :class:`ShotCatalog` — SQLite provenance/catalog (spec §10).
* :class:`ShotDetector` / :func:`get_detector` — swappable shot detector (§1).
* :class:`AdaptiveFrameSampler` / :class:`MotionChangeDetector` — sampling (§2,§3).
* :func:`extract_video_shots` / :func:`build_index` — offline orchestration (§4,§5,§11).
* :class:`ShotConfig` — all tunables, nothing hard-coded downstream (§10).
"""

from __future__ import annotations

from aic2026.extraction.catalog import ShotCatalog
from aic2026.extraction.detector import (
    DEFAULT_DETECTOR,
    PySceneDetectDetector,
    ShotDetector,
    TransNetV2Detector,
    get_detector,
)
from aic2026.extraction.pipeline import (
    EMBEDDING_VERSION,
    ShotConfig,
    build_index,
    decode_frame_at,
    extract_video_shots,
)
from aic2026.extraction.sampler import (
    AdaptiveFrameSampler,
    MotionChangeDetector,
    merge_timestamps,
)
from aic2026.extraction.validate import ArtifactValidationError, validate_artifacts

__all__ = [
    "ShotCatalog",
    "ShotDetector",
    "PySceneDetectDetector",
    "TransNetV2Detector",
    "get_detector",
    "DEFAULT_DETECTOR",
    "AdaptiveFrameSampler",
    "MotionChangeDetector",
    "merge_timestamps",
    "ShotConfig",
    "extract_video_shots",
    "build_index",
    "decode_frame_at",
    "validate_artifacts",
    "ArtifactValidationError",
    "EMBEDDING_VERSION",
]
