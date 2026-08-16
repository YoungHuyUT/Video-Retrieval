"""Coarse-to-fine temporal refinement for TRAKE."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from aic2026.models import Candidate
from aic2026.temporal.alignment import align_events_dp

logger = logging.getLogger(__name__)


def _window_frame_ids(
    event_frames: list[int], fps: float, *, window_seconds: float,
    sample_fps: float, frame_count: int,
) -> list[int]:
    stride = max(1, round(fps / sample_fps))
    radius = max(1, round(window_seconds * fps))
    selected: set[int] = set()
    for center in event_frames:
        start = max(0, int(center) - radius)
        end = min(frame_count - 1, int(center) + radius)
        selected.update(range(start, end + 1, stride))
        # Always include the window boundaries so a window that reaches the
        # video edge (or lands between strided samples) still covers its full
        # [start, end] extent instead of stopping short at the last on-grid frame.
        selected.add(start)
        selected.add(end)
        selected.add(min(max(0, int(center)), frame_count - 1))
    return sorted(selected)


def _read_frames(video_path: Path, frame_ids: list[int]) -> tuple[list[int], list[object]]:
    try:
        import cv2
        from PIL import Image
    except ImportError:
        return [], []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], []
    ids: list[int] = []
    images: list[object] = []
    try:
        for frame_id in frame_ids:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, bgr = capture.read()
            if not ok:
                continue
            ids.append(int(frame_id))
            images.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    return ids, images


def refine_trake_candidates(
    candidates: list[Candidate], event_embeddings: np.ndarray,
    encode_images: Callable[[list[object]], np.ndarray] | None, *,
    video_root: Path = Path("data/raw/Videos"), top_videos: int = 20,
    sample_fps: float = 3.0, window_seconds: float = 2.0,
    penalty_weight: float = 0.005,
) -> list[Candidate]:
    """Refine provisional TRAKE times from short, densely sampled video windows.

    Missing videos or optional image dependencies leave the original keyframe
    candidate untouched, so this is safe to enable by default.
    """
    if not candidates or encode_images is None or top_videos <= 0 or sample_fps <= 0:
        return candidates
    try:
        import cv2
    except ImportError:
        logger.info("TRAKE dense refinement skipped: OpenCV is unavailable")
        return candidates

    queries = np.asarray(event_embeddings, dtype=np.float32)
    queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    refined: list[Candidate] = []
    for rank, candidate in enumerate(candidates):
        if rank >= top_videos or not candidate.event_frames:
            refined.append(candidate)
            continue
        video_path = video_root / f"{candidate.video_id}.mp4"
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            refined.append(candidate)
            continue
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if frame_count <= 0:
            refined.append(candidate)
            continue
        frame_ids = _window_frame_ids(
            candidate.event_frames, fps, window_seconds=window_seconds,
            sample_fps=sample_fps, frame_count=frame_count,
        )
        dense_ids, images = _read_frames(video_path, frame_ids)
        if len(dense_ids) < len(queries):
            refined.append(candidate)
            continue
        try:
            vectors = np.asarray(encode_images(images), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TRAKE dense encode failed for %s: %s", candidate.video_id, exc)
            refined.append(candidate)
            continue
        if vectors.ndim != 2 or vectors.shape != (len(dense_ids), queries.shape[1]):
            refined.append(candidate)
            continue
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        similarity = queries @ vectors.T
        try:
            score, positions = align_events_dp(similarity, penalty_weight=penalty_weight)
        except ValueError:
            refined.append(candidate)
            continue
        event_frames = [dense_ids[position] for position in positions]
        representative = int(np.argmax([similarity[event, pos] for event, pos in enumerate(positions)]))
        refined.append(candidate.model_copy(update={
            "frame_id": event_frames[representative],
            "score": float(score / len(queries)),
            "vector_id": None,
            "keyframe_path": None,
            "event_frames": event_frames,
        }))
    return sorted(refined, key=lambda item: (-item.score, item.video_id))
