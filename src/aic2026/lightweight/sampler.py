"""Lightweight frame sampling (spec §2): uniform ~1 FPS + motion/change peaks.

Two cheap, CPU-only signals — no optical flow, no heavy model:

* ``UniformSampler`` decodes one frame every ``1/UNIFORM_FPS`` seconds via
  ``cv2.CAP_PROP_POS_MSEC``. This is the temporal backbone and is what catches
  *static* but semantically distinct states (a new object appearing, a sign
  changing) that a motion filter would skip.
* ``MotionChangeDetector`` decodes at a coarse stride, scores each sampled frame
  with grayscale frame-diff + histogram Bhattacharyya, and keeps the strongest
  *local* peaks. This catches short actions a 1 FPS grid misses (a quick gesture,
  a bottle being poured) WITHOUT re-decoding the whole video densely.

Both return ``(timestamp_s, frame_idx)`` pairs; the caller dedups + merges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UniformSampler:
    """Decode one representative frame every ``1/fps`` seconds (spec §2)."""

    fps: float = 1.0  # target sampling rate in Hz (configurable 0.5/1.0/2.0)

    def sample(self, video_path: Path, duration_s: float) -> list[tuple[float, int]]:
        """Return ``[(timestamp_s, frame_idx), ...]`` on the uniform grid.

        ``frame_idx`` is the nearest source frame (for provenance / cache keys).
        """
        if self.fps <= 0 or duration_s <= 0:
            return []
        step_s = 1.0 / self.fps
        # Start at half a step so we don't always land on frame 0; cover 0..duration.
        ts = step_s / 2.0
        out: list[tuple[float, int]] = []
        cap, cv2 = _open(video_path)
        try:
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            src_fps = float(src_fps or 0.0)
            while ts <= duration_s:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, ts) * 1000.0)
                ok, _ = cap.read()
                if not ok:
                    break
                fidx = int(round(ts * src_fps)) if src_fps > 0 else 0
                out.append((float(ts), fidx))
                ts += step_s
        finally:
            cap.release()
        return out


@dataclass
class MotionChangeDetector:
    """Lightweight motion/visual-change peak finder over the whole video (spec §3).

    Coarse-stride decode → per-step change score (grayscale diff + histogram
    Bhattacharyya) → keep strongest interior local maxima, enforcing a temporal
    gap so peaks don't stack. Returns ``[(timestamp_s, frame_idx), ...]``.
    """

    # Report peaks above this 0..1 change score (configurable).
    threshold: float = 0.35
    # Max peaks kept per video (coverage, not density — keep this small).
    max_peaks: int = 30
    # Minimum gap (s) between reported peaks.
    peak_min_gap_s: float = 1.0
    # Decoded stride (frames) while scanning — coarse is enough for change.
    scan_stride: int = 8

    def peaks(self, video_path: Path, duration_s: float) -> list[tuple[float, int]]:
        cap, cv2 = _open(video_path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            fps = float(fps or 0.0)
            if fps <= 0:
                return []
            end_frame = max(1, int(duration_s * fps))
            # Single forward pass (no per-frame seeks) — far cheaper than
            # cap.set(POS_FRAMES) on every stride step.
            prev_gray = None
            prev_hist = None
            fidx = 0
            scored: list[tuple[int, float]] = []  # (frame_idx, change_score)
            while fidx <= end_frame:
                ok, bgr = cap.read()
                if not ok:
                    break
                if fidx % self.scan_stride == 0:
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
                    cv2.normalize(hist, hist)
                    score = 0.0
                    if prev_gray is not None:
                        score += float(cv2.absdiff(gray, prev_gray).mean()) / 255.0
                    if prev_hist is not None:
                        score += float(cv2.compareHist(hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA))
                    score /= 2.0  # mean of the two 0..1 signals
                    scored.append((fidx, score))
                    prev_gray, prev_hist = gray, hist
                fidx += 1
        finally:
            cap.release()

        if len(scored) < 3:
            return []

        scores = np.asarray([s for _, s in scored], dtype=np.float32)
        local: list[tuple[int, float]] = []
        for i in range(1, len(scores) - 1):
            if scores[i] > self.threshold and scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1]:
                local.append((scored[i][0], float(scores[i])))
        if not local:
            return []

        order = np.argsort(-np.asarray([s for _, s in local]))
        chosen: list[tuple[float, int]] = []
        for rank in order:
            fidx, _ = local[rank]
            ts = fidx / fps
            if ts < 0.0 or ts > duration_s:
                continue
            if any(abs(ts - c) < self.peak_min_gap_s for c, _ in chosen):
                continue
            chosen.append((float(ts), int(fidx)))
            if len(chosen) >= self.max_peaks:
                break
        return chosen


def _open(video_path: Path):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install video extra: uv sync --extra video") from exc
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode video: {video_path}")
    return cap, cv2
