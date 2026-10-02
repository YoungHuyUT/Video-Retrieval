"""Perceptual dedup for the merged BTC + uniform + motion frame set (spec §2).

Philosophy (spec): TEMPORAL COVERAGE > STORAGE SAVING. We therefore dedup
*lightly* — only drop a frame when it is (a) very close in time to another frame
AND (b) nearly identical in embedding space. Different timestamps are kept even
if visually similar, because they preserve the temporal signal the reranker needs.

Inputs are parallel arrays: ``records`` (FrameRecord) and ``vectors`` (np.ndarray,
L2-normalized rows). We never touch BTC frames' already-computed CLIP vectors, so
the dedup matrix reuses them where available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from aic2026.models import FrameRecord

logger = logging.getLogger(__name__)


@dataclass
class DedupConfig:
    # Cosine similarity above this AND within ``min_gap_s`` -> drop the weaker
    # (later / non-BTC) frame. Keep it high so we don't over-prune coverage.
    sim_threshold: float = 0.98
    # Two frames closer than this in seconds are eligible for sim-dedup.
    min_gap_s: float = 0.5
    # Always keep BTC frames (they are the curated baseline); never dedup them
    # away even if a uniform frame lands on top.
    protect_btc: bool = True


def dedup_frames(
    records: list[FrameRecord],
    vectors: np.ndarray,
    cfg: DedupConfig | None = None,
) -> tuple[list[FrameRecord], np.ndarray, list[int]]:
    """Return ``(kept_records, kept_vectors, kept_indices)``.

    ``kept_indices`` are the surviving positions into the *original* ``records``
    array (useful for mapping back to feature rows). Idempotent and deterministic
    (stable sort by timestamp, BTC always wins a tie).
    """
    cfg = cfg or DedupConfig()
    if not records:
        return [], vectors, []

    kept_positions: list[int] = []
    # Timestamps restart at zero for each video.  The old global scan compared
    # frames from unrelated videos and was O(N²).  Partitioning plus a rolling
    # time window makes the work local and preserves cross-video recall.
    by_video: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        by_video.setdefault(rec.video_id, []).append(i)
    for positions in by_video.values():
        positions.sort(key=lambda i: (records[i].timestamp or 0.0, 0 if records[i].source == "btc" else 1))
        recent: list[int] = []
        for i in positions:
            rec_i = records[i]
            ts_i = rec_i.timestamp or 0.0
            recent = [j for j in recent if ts_i - (records[j].timestamp or 0.0) <= cfg.min_gap_s]
            drop = False
            for j_pos in recent:
                rec_j = records[j_pos]
                sim = float(np.dot(vectors[i], vectors[j_pos]))
                if sim >= cfg.sim_threshold and (
                    (cfg.protect_btc and rec_j.source == "btc") or rec_i.source != "btc"
                ):
                    drop = True
                    break
            if not drop:
                recent.append(i)
                kept_positions.append(i)

    kept_positions.sort()
    kept_records = [records[i] for i in kept_positions]
    kept_vectors = vectors[kept_positions] if vectors.size else vectors
    return kept_records, kept_vectors, kept_positions
