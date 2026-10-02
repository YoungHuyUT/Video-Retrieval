"""Lightweight visual extraction pipeline (spec §2, §5, §6, §10, §11).

BTC frames (official baseline) + FFmpeg/OpenCV uniform ~1 FPS + lightweight
motion/change frames -> perceptual dedup -> SigLIP2 + CLIP -> FAISS -> temporal
grouping -> temporal zoom-in -> fine reranking. Keeps ``official_features.npy``
untouched; builds into ``data/processed/new_lightweight/`` in parallel.
"""

from aic2026.lightweight.build import (
    LightweightConfig,
    lw_build,
    lw_embed,
    lw_index,
    lw_validate,
    video_source_hash,
)
from aic2026.lightweight.catalog import LightweightCatalog
from aic2026.lightweight.dedup import DedupConfig, dedup_frames
from aic2026.lightweight.merge_btc import MergeBtcConfig, load_manifest

__all__ = [
    "LightweightConfig",
    "LightweightCatalog",
    "MergeBtcConfig",
    "load_manifest",
    "dedup_frames",
    "DedupConfig",
    "lw_build",
    "lw_embed",
    "lw_index",
    "lw_validate",
    "video_source_hash",
]
