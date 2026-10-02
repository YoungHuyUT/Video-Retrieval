from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aic2026.models import FrameRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionReport:
    video_id: str
    decoded_frames: int
    kept_frames: int
    output_dir: Path
    feature_path: Path


class OpenCLIPFrameEncoder:
    """Batch image encoder. Use the same CLIP checkpoint for image and query text embeddings."""
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str | None = None):
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError("Install video/model extras: uv sync --extra video --extra models") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        self.model.eval()

    def encode_images(self, images: Iterable[object]) -> np.ndarray:
        batch = self.torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with self.torch.no_grad():
            values = self.model.encode_image(batch)
            values = values / values.norm(dim=-1, keepdim=True)
        return values.detach().cpu().float().numpy()

    def encode_images_in_chunks(self, images: Iterable[object], batch_size: int = 4) -> list[np.ndarray]:
        items = list(images)
        if not items:
            return []
        outputs: list[np.ndarray] = []
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            outputs.append(self.encode_images(chunk))
        return outputs


def _encode_images_in_chunks(encoder: object, images: Iterable[object], batch_size: int = 4) -> np.ndarray:
    items = list(images)
    if not items:
        return np.empty((0, 0), dtype=np.float32)
    if hasattr(encoder, "encode_images_in_chunks"):
        outputs = encoder.encode_images_in_chunks(items, batch_size=batch_size)
        if not outputs:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(outputs)
    vectors: list[np.ndarray] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        chunk_vectors = encoder.encode_images(chunk)
        if chunk_vectors.size:
            vectors.append(chunk_vectors)
    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(vectors)


def embed_existing_keyframes(keyframes_root: Path, features_root: Path, encoder: object, batch_size: int = 4, video_id: str | None = None) -> list[Path]:
    """Encode existing JPG keyframes on disk into per-video .npz archives."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install pillow to read existing keyframes") from exc

    keyframes_root, features_root = Path(keyframes_root), Path(features_root)
    features_root.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    if video_id is not None:
        selected_dirs = [keyframes_root / video_id]
        if not selected_dirs[0].exists() or not selected_dirs[0].is_dir():
            raise FileNotFoundError(f"Keyframe folder not found: {selected_dirs[0]}")
    else:
        selected_dirs = [path for path in sorted(keyframes_root.glob("*")) if path.is_dir()]

    for video_dir in selected_dirs:
        frame_paths = sorted(
            [*video_dir.glob("*.jpg"), *video_dir.glob("*.png")]
        )
        if not frame_paths:
            continue

        images: list[object] = []
        frame_ids: list[int] = []
        for frame_path in frame_paths:
            try:
                image = Image.open(frame_path)
                frame_id = int(frame_path.stem)
            except (ValueError, OSError, TypeError):
                # Skip files whose name is not a plain frame index (e.g.
                # "L21_V001_frame001.jpg"); the pipeline only handles numeric ids.
                continue
            images.append(image.convert("RGB"))
            frame_ids.append(frame_id)

        if not images:
            continue

        vectors = _encode_images_in_chunks(encoder, images, batch_size=batch_size)
        archive_path = features_root / f"{video_dir.name}.npz"
        np.savez_compressed(archive_path, frame_ids=np.asarray(frame_ids, dtype=np.int64), features=np.asarray(vectors, dtype=np.float32))
        written_paths.append(archive_path)

    return written_paths


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12))


def extract_deduplicated_keyframes(
    video_path: Path,
    keyframes_root: Path,
    features_root: Path,
    encoder: OpenCLIPFrameEncoder,
    sample_interval_sec: float | None = 1.0,
    cosine_threshold: float = 0.985,
    batch_size: int = 32,
    image_quality: int = 92,
    progress_callback: callable | None = None,
) -> ExtractionReport:
    """Sample keyframes uniformly in time and CLIP-embed only the sampled frames.

    Two modes:

    * ``sample_interval_sec`` set (default 1.0s) — **coarse time-uniform sampling**.
      Frames are kept at a fixed cadence (``round(fps * interval)`` frames apart,
      always including frame 0). Only those sampled frames are decoded-and-embedded,
      so cost scales with wall-clock time, not frame count: a 10-min 30fps video
      becomes ~600 frames instead of ~18k. This is the fast, coverage-safe path for
      retrieval (no scene is skipped) and is the recommended mode for building the
      index. The cosine dedup is NOT applied here — sampling already controls density.
    * ``sample_interval_sec is None`` — **legacy mode**: decode EVERY frame, embed
      each, and keep only frames whose cosine to the last kept frame is ``<
      cosine_threshold``. Retained for backward compatibility / re-ingest of short
      clips where per-frame semantics matter.

    Original 0-based video frame IDs are used as filenames in both modes.
    """
    if sample_interval_sec is not None and sample_interval_sec <= 0:
        raise ValueError("sample_interval_sec must be > 0 (or None for legacy mode)")
    if sample_interval_sec is None and not 0 < cosine_threshold < 1:
        raise ValueError("cosine_threshold must be in (0, 1) in legacy mode")
    try:
        import cv2
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install video extra: uv sync --extra video") from exc

    video_path, keyframes_root, features_root = Path(video_path), Path(keyframes_root), Path(features_root)
    video_id = video_path.stem
    output_dir = keyframes_root / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    features_root.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode video: {video_path}")

    # ``CAP_PROP_FPS`` may be absent in lightweight test fakes; fall back to 0
    # (yields frame_step=1, i.e. keep every frame) so sampling stays safe.
    _fps_prop = getattr(cv2, "CAP_PROP_FPS", None)
    fps = capture.get(_fps_prop) if _fps_prop is not None else 0.0
    fps = float(fps or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if hasattr(cv2, "CAP_PROP_FRAME_COUNT") else None
    frame_step = max(1, round(fps * sample_interval_sec)) if sample_interval_sec else 1

    retained_vectors: list[np.ndarray] = []
    retained_ids: list[int] = []
    last_vector: np.ndarray | None = None
    pending_images: list[object] = []
    pending_ids: list[int] = []
    decoded = 0

    def flush() -> None:
        if not pending_images:
            return
        vectors = _encode_images_in_chunks(encoder, pending_images, batch_size=batch_size)
        if sample_interval_sec is None:
            # Legacy: keep only frames differing from the last retained one.
            for image, frame_id, vector in zip(pending_images, pending_ids, vectors):
                if last_vector is None or _cosine(last_vector, vector) < cosine_threshold:
                    image.save(output_dir / f"{frame_id:09d}.jpg", quality=image_quality, optimize=True)
                    retained_vectors.append(vector)
                    retained_ids.append(frame_id)
                    last_vector = vector
        else:
            # Coarse mode: keep every sampled frame (already time-diverse).
            for image, frame_id, vector in zip(pending_images, pending_ids, vectors):
                image.save(output_dir / f"{frame_id:09d}.jpg", quality=image_quality, optimize=True)
                retained_vectors.append(vector)
                retained_ids.append(frame_id)
                last_vector = vector
        if progress_callback is not None:
            progress_callback(decoded, total_frames or decoded, len(retained_ids))
        pending_images.clear()
        pending_ids.clear()

    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frame_id = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            decoded += 1
            # Coarse mode: only keep frames at the sampling cadence (frame 0 always).
            if sample_interval_sec is not None and frame_id % frame_step != 0:
                if progress_callback is not None:
                    progress_callback(decoded, total_frames or decoded, len(retained_ids))
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pending_images.append(Image.fromarray(rgb))
            pending_ids.append(frame_id)
            if progress_callback is not None:
                progress_callback(decoded, total_frames or decoded, len(retained_ids))
            if len(pending_images) >= batch_size:
                flush()
        flush()
    finally:
        capture.release()

    feature_path = features_root / f"{video_id}.npz"
    np.savez_compressed(feature_path, frame_ids=np.asarray(retained_ids, dtype=np.int64), features=np.asarray(retained_vectors, dtype=np.float32))
    return ExtractionReport(video_id, decoded, len(retained_ids), output_dir, feature_path)


def _resolve_frame_path(keyframes_root: Path, video_id: str, frame_id: int) -> Path | None:
    """Resolve a keyframe file allowing any zero-padding width (3-9 digits).

    BTC keyframes may use ``0000.jpg``, ``001.jpg``, or ``000000001.jpg``.
    Returns the first existing candidate, or ``None`` when nothing matches.
    """
    digits = f"{int(frame_id):d}"
    candidates: list[Path] = []
    for ext in ("jpg", "png"):
        for video in (video_id, video_id.lower()):
            root = keyframes_root / video
            for width in range(9, 2, -1):
                if len(digits) <= width:
                    candidates.append(root / f"{int(frame_id):0{width}d}.{ext}")
            candidates.append(root / f"{digits}.{ext}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_derived_artifacts(keyframes_root: Path, features_root: Path, manifest_path: Path, features_path: Path, video_ids: Iterable[str] | None = None) -> int:
    """Merge per-video retained CLIP archives into one manifest + `.npy` index input."""
    keyframes_root, features_root = Path(keyframes_root), Path(features_root)
    selected_video_ids = set(video_ids or [])
    records: list[FrameRecord] = []
    all_features: list[np.ndarray] = []
    vector_id = 0
    for archive_path in sorted(features_root.glob("*.npz")):
        video_id = archive_path.stem
        if selected_video_ids and video_id not in selected_video_ids:
            continue
        # Skip empty/corrupt archives (e.g. a 0-byte leftover from a crashed run).
        if archive_path.stat().st_size == 0:
            logger.warning("Skipping empty feature archive %s", archive_path)
            continue
        try:
            archive = np.load(archive_path)
            frame_ids, features = archive["frame_ids"], archive["features"]
        except (OSError, EOFError, ValueError):
            logger.warning("Skipping corrupt feature archive %s", archive_path)
            continue
        if len(frame_ids) != len(features):
            raise ValueError(f"{archive_path}: frame_ids and features have different sizes")
        for frame_id, feature in zip(frame_ids, features):
            image = _resolve_frame_path(keyframes_root, video_id, int(frame_id))
            if image is None or not image.exists():
                logger.warning(
                    "Missing retained frame %s (frame %s); skipping record",
                    video_id,
                    int(frame_id),
                )
                continue
            records.append(FrameRecord(vector_id=vector_id, clip_feature_index=vector_id, video_id=video_id, frame_id=int(frame_id), keyframe_path=str(image)))
            all_features.append(feature)
            vector_id += 1
    if not records:
        raise ValueError("No .npz derived feature archives found")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(record.model_dump_json() for record in records) + "\n", encoding="utf-8")
    np.save(features_path, np.asarray(all_features, dtype=np.float32))
    return len(records)
