"""Shot-adaptive extraction orchestration (spec §1-§5, §11).

``extract_video_shots`` is the per-video worker:

    RAW VIDEO
      └─ ShotDetector.detect        -> [(start_s, end_s), ...]
      └─ AdaptiveFrameSampler.sample (± MotionChangeDetector.peaks for long shots)
      └─ decode representative frame at each timestamp (cv2)
      └─ embed with CLIP + SigLIP2  -> [features_clip.npy, features_siglip2.npy]
      └─ write keyframe JPGs, manifest_shot.jsonl, catalog.db rows
      └─ write per-video shots/<VID>.jsonl (resume/cache)

The whole pipeline root lives under ``data/processed/shot_adaptive/`` so it runs
in parallel with the official index (baseline) and never overwrites it.  Resume:
a video is skipped if its ``shots/<VID>.jsonl`` exists AND the manifest already
holds its frames (catalog.is_video_done) — a crashed run continues from where it
stopped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aic2026.models import FrameRecord
from aic2026.extraction.catalog import ShotCatalog
from aic2026.extraction.detector import ShotDetector, get_detector
from aic2026.extraction.sampler import (
    AdaptiveFrameSampler,
    MotionChangeDetector,
    merge_timestamps,
)

logger = logging.getLogger(__name__)

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
SIGLIP2_MODEL = "google/siglip2-so400m-patch14-384"
EMBEDDING_VERSION = "shot-adaptive-v1"


@dataclass
class ShotConfig:
    """All tunables for the pipeline (spec §10). None are hard-coded downstream."""

    root: Path = Path("data/processed/shot_adaptive")
    shot_detector: str = "pyscenedetect"
    sampler: AdaptiveFrameSampler = field(default_factory=AdaptiveFrameSampler)
    with_motion_peaks: bool = False  # config E flips this on for long shots
    clip_model: str = CLIP_MODEL
    clip_pretrained: str = CLIP_PRETRAINED
    siglip2_model: str = SIGLIP2_MODEL
    image_quality: int = 92
    model_version: str = "openai/ViT-B-32 + google/siglip2-so400m-patch14-384"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest_shot.jsonl"

    @property
    def clip_feats_path(self) -> Path:
        return self.root / "features_clip.npy"

    @property
    def siglip2_feats_path(self) -> Path:
        return self.root / "features_siglip2.npy"

    @property
    def keyframes_root(self) -> Path:
        return self.root / "keyframes"

    @property
    def shots_root(self) -> Path:
        return self.root / "shots"

    @property
    def catalog_path(self) -> Path:
        return self.root / "catalog.db"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.keyframes_root.mkdir(parents=True, exist_ok=True)
        self.shots_root.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return {
            "shot_detector": self.shot_detector,
            "KEYFRAMES_PER_SHOT_SHORT": self.sampler.n_short,
            "KEYFRAMES_PER_SHOT_MID": self.sampler.n_mid,
            "KEYFRAMES_PER_SHOT_LONG": self.sampler.n_long,
            "WITH_MOTION_PEAKS": self.with_motion_peaks,
            "clip_model": f"{self.clip_model}/{self.clip_pretrained}",
            "siglip2_model": self.siglip2_model,
            "embedding_version": EMBEDDING_VERSION,
            "model_version": self.model_version,
        }

    def save_config(self) -> None:
        self.config_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )


def video_source_hash(video_path: Path) -> str:
    """Cheap, stable hash from path + size + mtime (not content — keeps it fast)."""
    st = video_path.stat()
    raw = f"{video_path.resolve()}|{st.st_size}|{st.st_mtime}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _open_decode(video_path: Path):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install video extra: uv sync --extra video") from exc
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fps = float(fps or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if hasattr(cv2, "CAP_PROP_FRAME_COUNT") else 0
    duration = (frame_count / fps) if (fps > 0 and frame_count > 0) else 0.0
    return cap, cv2, fps, frame_count, duration


def decode_frame_at(video_path: Path, timestamp_s: float):
    """Return an RGB PIL image for the frame nearest ``timestamp_s`` (seconds)."""
    from PIL import Image

    cap, cv2, fps, _frame_count, _dur = _open_decode(video_path)
    try:
        if fps > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_s) * 1000.0)
        ok, bgr = cap.read()
        if not ok:
            return None
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert("RGB")
    finally:
        cap.release()


def _load_encoders(cfg: ShotConfig):
    from aic2026.data_platform.video_frames import OpenCLIPFrameEncoder
    from aic2026.embeddings.siglip2 import Siglip2TextEmbedder

    clip_enc = OpenCLIPFrameEncoder(model_name=cfg.clip_model, pretrained=cfg.clip_pretrained)
    sig_enc = Siglip2TextEmbedder(model_name=cfg.siglip2_model)
    return clip_enc, sig_enc


def extract_video_shots(
    video_path: Path,
    cfg: ShotConfig,
    detector: ShotDetector | None = None,
    clip_enc: object | None = None,
    sig_enc: object | None = None,
    force: bool = False,
    max_shots: int | None = None,
) -> int:
    """Extract one video's representative frames + embeddings. Returns #frames added."""
    video_path = Path(video_path)
    video_id = video_path.stem
    cfg.ensure_dirs()
    detector = detector or get_detector(cfg.shot_detector)
    catalogue = ShotCatalog(cfg.catalog_path)

    # Resume guard: skip if per-video shot list + frames already present.
    shot_list_path = cfg.shots_root / f"{video_id}.jsonl"
    if not force and shot_list_path.exists() and catalogue.is_video_done(video_id):
        logger.info("skip %s (already extracted)", video_id)
        catalogue.close()
        return 0

    cap, cv2, fps, frame_count, duration = _open_decode(video_path)
    cap.release()

    shots = detector.detect(video_path)
    # Normalise: ensure coverage from 0..duration (clip any out-of-range).
    norm_shots: list[tuple[float, float]] = []
    for s, e in shots:
        s, e = max(0.0, float(s)), min(max(0.0, float(e)), duration or float(e))
        if e > s:
            norm_shots.append((s, e))
    if not norm_shots:
        norm_shots = [(0.0, duration)]

    # Debug/verify only: cap the number of shots processed (keeps a long-video
    # dry-run under the test timeout). Production passes None -> all shots.
    if max_shots is not None and max_shots > 0:
        norm_shots = norm_shots[:max_shots]

    if clip_enc is None or sig_enc is None:
        clip_enc, sig_enc = _load_encoders(cfg)

    motion = MotionChangeDetector() if cfg.with_motion_peaks else None

    # Persist per-video shot list first (cheap resume marker).
    shot_list_path.parent.mkdir(parents=True, exist_ok=True)
    shot_records: list[dict] = []

    # Accumulators stay small (records = metadata; vectors = KB-scale). We do NOT
    # buffer decoded PIL images across the whole video — that blew past this
    # 7.7 GB machine's RAM on a 21-min clip (1249 RGB frames ≈ 3.2 GB) and killed
    # the process AFTER decoding but BEFORE commit (frames on disk, 0 in catalog).
    # Instead we encode each shot's frames immediately and drop the images, so
    # peak RAM stays bounded to one shot's worth of frames.
    records: list[FrameRecord] = []
    clip_buf: list[np.ndarray] = []
    sig_buf: list[np.ndarray] = []

    global_vector_id = _next_vector_id(cfg)

    catalogue.upsert_video(video_id, fps, frame_count, video_source_hash(video_path))

    frame_id_counter = 0
    for shot_idx, (s, e) in enumerate(norm_shots):
        base_ts = cfg.sampler.sample((s, e))
        extra_ts: list[float] = []
        if motion is not None and (e - s) > cfg.sampler.long_threshold_s:
            extra_ts = motion.peaks(video_path, (s, e))
        ts_list = merge_timestamps(base_ts, extra_ts, s, e)
        shot_db_id = catalogue.add_shot(video_id, s, e, n_frames=len(ts_list))

        shot_images: list[object] = []
        shot_frame_records: list[FrameRecord] = []
        for ts in ts_list:
            img = decode_frame_at(video_path, ts)
            if img is None:
                continue
            out_path = (
                cfg.keyframes_root
                / video_id
                / f"{frame_id_counter:09d}.jpg"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, quality=cfg.image_quality, optimize=True)

            rec = FrameRecord(
                vector_id=global_vector_id,
                video_id=video_id,
                frame_id=frame_id_counter,
                keyframe_path=str(out_path),
                clip_feature_index=global_vector_id,
                shot_id=shot_db_id,
                timestamp=ts,
                model_version=cfg.model_version,
                embedding_version=EMBEDDING_VERSION,
            )
            shot_frame_records.append(rec)
            shot_images.append(img)
            frame_id_counter += 1
            global_vector_id += 1
        # Encode this shot's frames NOW and free the (large) images immediately,
        # so RAM never holds more than one shot's decoded frames at once.
        if shot_images:
            clip_buf.append(clip_enc.encode_images(shot_images))
            sig_buf.append(sig_enc.encode_images(shot_images))
            records.extend(shot_frame_records)
            del shot_images, shot_frame_records
        shot_records.append(
            {"shot_id": shot_db_id, "start_s": s, "end_s": e, "n_frames": len(ts_list)}
        )

    # Stack the per-shot batches into the full frame matrix (KB-scale, cheap).
    clip_vectors = np.vstack(clip_buf) if clip_buf else np.empty((0, 0), np.float32)
    sig_vectors = np.vstack(sig_buf) if sig_buf else np.empty((0, 0), np.float32)

    # Write manifest + per-feature files + catalog rows (now bounded in RAM).
    _append_manifest(cfg.manifest_path, records)
    _append_features(cfg.clip_feats_path, clip_vectors)
    _append_features(cfg.siglip2_feats_path, sig_vectors)
    for rec, sv in zip(records, sig_vectors):
        catalogue.add_frame(
            video_id=rec.video_id,
            frame_id=rec.frame_id,
            shot_id=rec.shot_id,
            timestamp=rec.timestamp,
            path=rec.keyframe_path,
            model_version=rec.model_version,
            embedding_version=rec.embedding_version,
        )

    catalogue.commit_frames()
    catalogue.register_model_version(
        "clip", f"{cfg.clip_model}/{cfg.clip_pretrained}",
        dim=int(clip_vectors.shape[1]) if clip_vectors.size else None,
        dtype="float32",
    )
    catalogue.register_model_version(
        "siglip2", cfg.siglip2_model,
        dim=int(sig_vectors.shape[1]) if sig_vectors.size else None,
        dtype="float32",
    )
    catalogue.close()

    shot_list_path.write_text(
        json.dumps(
            {"video_id": video_id, "duration": duration, "fps": fps, "shots": shot_records},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("extracted %s: %d shots, %d frames", video_id, len(norm_shots), len(records))
    return len(records)


def _next_vector_id(cfg: ShotConfig) -> int:
    """Continue vector_id after the last manifest row (supports resume)."""
    if not cfg.manifest_path.exists():
        return 0
    last = 0
    with cfg.manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                last = max(last, int(obj.get("vector_id", 0)) + 1)
            except json.JSONDecodeError:
                continue
    return last


def _append_manifest(path: Path, records: list[FrameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")


def _append_features(path: Path, vectors: np.ndarray) -> None:
    """Append rows to a row-aligned ``.npy`` (memmap-friendly, no full reload)."""
    if vectors.size == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = np.load(path)
        combined = np.vstack([existing, vectors.astype(np.float32)])
    else:
        combined = vectors.astype(np.float32)
    np.save(path, combined)


def build_index(cfg: ShotConfig) -> dict:
    """Register final artifacts + emit a small summary dict (spec §5, §11)."""
    catalogue = ShotCatalog(cfg.catalog_path)
    n_frames = catalogue.frame_count()
    n_shots = catalogue.shot_count()
    n_videos = catalogue.video_count()
    catalogue.register_artifact(
        "features_clip", str(cfg.clip_feats_path),
        model_name="clip", model_version=f"{cfg.clip_model}/{cfg.clip_pretrained}",
        dim=None, dtype="float32",
    )
    catalogue.register_artifact(
        "features_siglip2", str(cfg.siglip2_feats_path),
        model_name="siglip2", model_version=cfg.siglip2_model,
        dim=None, dtype="float32",
    )
    catalogue.register_artifact(
        "manifest", str(cfg.manifest_path), model_name="manifest", model_version=EMBEDDING_VERSION
    )
    catalogue.close()
    cfg.save_config()
    summary = {
        "videos": n_videos,
        "shots": n_shots,
        "frames": n_frames,
        "clip_dim": int(np.load(cfg.clip_feats_path).shape[1]) if cfg.clip_feats_path.exists() else None,
        "siglip2_dim": int(np.load(cfg.siglip2_feats_path).shape[1]) if cfg.siglip2_feats_path.exists() else None,
    }
    logger.info("build-index: %s", summary)
    return summary
