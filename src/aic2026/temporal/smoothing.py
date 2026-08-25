from __future__ import annotations

import math
from collections import defaultdict

from aic2026.models import Candidate


def apply_temporal_smoothing(
    candidates: list[Candidate],
    sigma: float = 1.5,
    window: int = 3,
    weight: float = 0.20,
) -> list[Candidate]:
    """Apply 1D Gaussian temporal smoothing to candidate frame scores within each video.

    Exploits temporal continuity in video shots: if keyframe t has a high similarity
    score, neighboring keyframes t-1, t+1, t-2, t+2 in the same video shot receive a
    smooth resonance bonus. This boosts continuous relevant shots and suppresses
    isolated single-frame false positive outliers.

    Parameters
    ----------
    candidates : list[Candidate]
        List of retrieved candidates.
    sigma : float
        Gaussian standard deviation in index units (default 1.5).
    window : int
        Maximum neighbor distance in keyframe sequence (default 3).
    weight : float
        Blending weight for the temporal neighborhood bonus (default 0.20).

    Returns
    -------
    list[Candidate]
        Smoothed and re-ranked candidates sorted by updated score descending.
    """
    if not candidates or weight <= 0 or window <= 0:
        return candidates

    by_video: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_video[c.video_id].append(c)

    # Precompute Gaussian kernel weights for relative offsets
    kernel: dict[int, float] = {}
    for offset in range(1, window + 1):
        kernel[offset] = math.exp(- (offset ** 2) / (2.0 * sigma * sigma))
    max_kernel_sum = 2.0 * sum(kernel.values()) if kernel else 1.0

    smoothed: list[Candidate] = []
    for video_id, frames in by_video.items():
        # Sort chronologically by frame_id
        ordered = sorted(frames, key=lambda c: c.frame_id)
        n = len(ordered)

        for i, cand in enumerate(ordered):
            neighbor_sum = 0.0
            weight_sum = 0.0

            for offset in range(1, window + 1):
                k_w = kernel[offset]
                # Left neighbor
                if i - offset >= 0:
                    neighbor_sum += ordered[i - offset].score * k_w
                    weight_sum += k_w
                # Right neighbor
                if i + offset < n:
                    neighbor_sum += ordered[i + offset].score * k_w
                    weight_sum += k_w

            if weight_sum > 0:
                bonus = neighbor_sum / max_kernel_sum
                new_score = float(cand.score + weight * bonus)
            else:
                new_score = float(cand.score)

            smoothed.append(cand.model_copy(update={"score": new_score}))

    return sorted(smoothed, key=lambda c: c.score, reverse=True)
