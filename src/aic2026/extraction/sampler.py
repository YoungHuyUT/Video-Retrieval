"""Adaptive representative-frame sampling (spec §2, §3, §9).

A shot (start_s, end_s) is sampled into a *small* set of representative frames
whose COUNT scales with shot duration (NOT a fixed 1 FPS cadence):

    duration < 5s   -> 3 frames
    5s..15s        -> 5 frames
    15s..30s       -> 7 frames
    > 30s          -> 7 frames + motion/visual-change peaks

Temporal coverage is guaranteed: the endpoints (start/middle/end) are always
included and the rest are spread evenly, so a query landing anywhere in the
shot still has a nearby representative frame (RECALL priority from the spec).

MotionChangeDetector (spec §3) is an AUXILIARY signal only: it finds
visual-change peaks inside long shots and we add a few of them on top of the
evenly-spaced base, but it never *replaces* semantic/uniform coverage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveFrameSampler:
    """Pick representative timestamps for a shot given its [start_s, end_s]."""

    # Count thresholds (spec §2). Benchmark config C uses a flat 5/shot via the
    # ``fixed_per_shot`` override; the adaptive curve is the default for D/E.
    n_short: int = 3   # duration < 5s
    n_mid: int = 5     # 5s..15s
    n_long: int = 7    # >= 15s
    long_threshold_s: float = 30.0
    mid_threshold_s: float = 15.0
    short_threshold_s: float = 5.0
    # Max extra frames added from motion peaks for very long shots (spec §2).
    max_motion_extra: int = 4
    # Override: if set (benchmark config B/C), ignore duration and always take N.
    fixed_per_shot: int | None = None

    def count_for(self, duration_s: float) -> int:
        if self.fixed_per_shot is not None:
            return int(self.fixed_per_shot)
        if duration_s < self.short_threshold_s:
            return self.n_short
        if duration_s < self.mid_threshold_s:
            return self.n_mid
        return self.n_long

    def sample(self, shot: tuple[float, float]) -> list[float]:
        start_s, end_s = float(shot[0]), float(shot[1])
        duration = end_s - start_s
        n = self.count_for(duration)
        # Guard: degenerate shot (< 2 distinct timestamps) -> just the midpoint.
        if n <= 0 or duration <= 0:
            return [start_s + duration / 2.0]
        if n == 1:
            return [(start_s + end_s) / 2.0]

        # Evenly spaced base including both endpoints (temporal coverage).
        base = np.linspace(start_s, end_s, num=n)
        return [float(t) for t in base]


@dataclass
class MotionChangeDetector:
    """Auxiliary motion/visual-change peak finder (spec §3).

    Computes a cheap per-frame change signal (grayscale diff + histogram diff)
    over the shot, then returns up to ``max_peaks`` timestamps of the strongest
    *local* changes. Used only to enrich long shots; not a retrieval signal.
    """

    max_peaks: int = 4
    # Minimum gap (s) between reported peaks so we don't stack duplicates.
    peak_min_gap_s: float = 1.0
    # Decoded stride (frames) while scanning — keeps CPU cost bounded; we only
    # need a coarse change curve, not every frame.
    scan_stride: int = 4

    def _load(self):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install video extra: uv sync --extra video") from exc
        self._cv2 = cv2

    def peaks(self, video_path: Path, shot: tuple[float, float]) -> list[float]:
        """Return motion-peak timestamps (seconds) within the shot span."""
        self._load()
        cv2 = self._cv2
        video_path, start_s, end_s = Path(video_path), float(shot[0]), float(shot[1])
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("MotionChangeDetector: cannot open %s", video_path)
            return []
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            fps = float(fps or 0.0)
            if fps <= 0:
                return []
            start_frame = max(0, int(start_s * fps))
            end_frame = max(start_frame + 1, int(end_s * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            prev_gray = None
            prev_hist = None
            frames: list[tuple[int, float]] = []  # (frame_idx, change_score)
            for frame_idx in range(start_frame, end_frame + 1, self.scan_stride):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, bgr = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
                cv2.normalize(hist, hist)
                score = 0.0
                if prev_gray is not None:
                    score += float(cv2.absdiff(gray, prev_gray).mean()) / 255.0
                if prev_hist is not None:
                    score += float(cv2.compareHist(hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA))
                frames.append((frame_idx, score))
                prev_gray, prev_hist = gray, hist
        finally:
            cap.release()

        if len(frames) < 3:
            return []

        scores = np.asarray([s for _, s in frames], dtype=np.float32)
        # Local maxima (interior points strictly/not-lower than both neighbours).
        peaks: list[tuple[int, float]] = []  # (frame_idx, score)
        for i in range(1, len(scores) - 1):
            if scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1] and scores[i] > 0:
                peaks.append((frames[i][0], float(scores[i])))
        if not peaks:
            return []

        # Keep the strongest peaks, enforcing a temporal gap.
        order = np.argsort(-np.asarray([s for _, s in peaks]))  # ranks over the peaks themselves
        chosen: list[float] = []
        for rank in order:
            fidx, _ = peaks[rank]
            ts = fidx / fps
            if ts < start_s or ts > end_s:
                continue
            if any(abs(ts - c) < self.peak_min_gap_s for c in chosen):
                continue
            chosen.append(ts)
            if len(chosen) >= self.max_peaks:
                break
        return chosen


def merge_timestamps(
    base: list[float],
    extra: list[float],
    start_s: float,
    end_s: float,
    dedupe_gap_s: float = 0.1,
) -> list[float]:
    """Merge evenly-spaced base with motion peaks, clip to span, dedupe, sort."""
    merged = list(base) + list(extra)
    merged = [t for t in merged if start_s - 1e-6 <= t <= end_s + 1e-6]
    merged.sort()
    out: list[float] = []
    for t in merged:
        if not out or abs(t - out[-1]) >= dedupe_gap_s:
            out.append(t)
    return out
