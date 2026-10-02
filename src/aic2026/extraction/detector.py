"""Shot boundary detection (spec §1, §13).

RAW VIDEO ──▶ ShotDetector ──▶ list[(start_s, end_s)]

The system standardises on **PySceneDetect** (content-aware detector) as the
default shot detector.  A thin :class:`ShotDetector` protocol lets us swap in a
learned detector (TransNetV2) later without touching the downstream sampler,
embedder, or CLI — satisfying the spec's "interface allows swapping later"
requirement.

Detectors return shot boundaries in **seconds** (float), already converted from
frame indices using the video fps, so the sampler never has to know about frames.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_DETECTOR = "pyscenedetect"


@runtime_checkable
class ShotDetector(Protocol):
    """Minimal contract every shot detector must satisfy."""

    name: str

    def detect(self, video_path: Path) -> list[tuple[float, float]]:
        """Return ``[(start_s, end_s), ...]`` covering the whole video.

        The union of the returned intervals must span ``[0, duration]`` with no
        gaps so the sampler never produces unsampled gaps.
        """
        ...


class PySceneDetectDetector:
    """Content-aware shot detector backed by PySceneDetect.

    Uses the ``ContentDetector`` (histogram + edge change) which is the
    CPU-friendly default.  If a video yields *zero* cuts (one long static shot)
    we fall back to a single shot spanning the whole clip so the upstream
    pipeline always has at least one interval to sample.
    """

    name = "pyscenedetect"

    def __init__(self, threshold: float = 27.0, min_scene_len: int = 8) -> None:
        self.threshold = threshold
        self.min_scene_len = min_scene_len

    def _load(self):
        try:
            import scenedetect
            from scenedetect import ContentDetector, detect, open_video
        except ImportError as exc:
            raise RuntimeError(
                "Install video extra: uv sync --extra video "
                "(PySceneDetect is required for shot detection)"
            ) from exc
        self._scenedetect = scenedetect
        self._ContentDetector = ContentDetector
        self._detect = detect
        self._open_video = open_video

    def detect(self, video_path: Path) -> list[tuple[float, float]]:
        self._load()
        video_path = Path(video_path)
        # Pass the path directly: scenedetect's ``detect`` opens the video
        # itself (it accepts a path or an already-open VideoStream). Duration
        # for the no-cut fallback is computed by the caller via cv2, so we just
        # return a single span covering the whole clip when no cuts are found.
        scene_list = self._detect(
            str(video_path),
            self._ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len),
        )

        if not scene_list:
            # No cuts found: caller fills duration; we return a placeholder
            # whole-video span (0..0) which the pipeline clips to real duration.
            return [(0.0, 0.0)]

        # ``scene_list`` is a list of (start_timecode, end_timecode); convert to seconds.
        shots: list[tuple[float, float]] = []
        for start_tc, end_tc in scene_list:
            start_s = float(start_tc.get_seconds())
            end_s = float(end_tc.get_seconds())
            shots.append((start_s, end_s))
        return shots


class TransNetV2Detector:
    """Learned shot detector stub (spec §13 "swap later").

    Not installed by default.  Raises a clear error if selected so the system
    fails loudly instead of silently falling back and giving misleading results.
    """

    name = "transnetv2"

    def detect(self, video_path: Path) -> list[tuple[float, float]]:
        raise NotImplementedError(
            "TransNetV2Detector is a placeholder. Install TransNetV2 and implement "
            "frame-wise shot probability decoding here, then return [(start_s, end_s)] "
            "like PySceneDetectDetector. Until then use --shot-detector pyscenedetect."
        )


_DETECTORS = {
    "pyscenedetect": PySceneDetectDetector,
    "transnetv2": TransNetV2Detector,
}


def get_detector(name: str = DEFAULT_DETECTOR, **kwargs) -> ShotDetector:
    """Factory so callers pass a string from config/CLI, not a class."""
    key = (name or DEFAULT_DETECTOR).lower()
    if key not in _DETECTORS:
        raise ValueError(
            f"Unknown shot detector {name!r}; choose from {sorted(_DETECTORS)}"
        )
    return _DETECTORS[key](**kwargs)
