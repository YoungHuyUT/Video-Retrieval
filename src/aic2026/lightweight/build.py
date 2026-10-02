"""Lightweight extraction orchestration (spec §2, §5, §6, §10, §11).

Public entry points (each maps to a CLI command):

* ``lw_build``  — decode uniform + motion frames, append to manifest + catalog.
* ``lw_embed``  — assemble CLIP (reuse official for BTC) + SigLIP2 matrices.
* ``lw_index``  — build FAISS ``VectorIndex`` per modality + register artifacts.
* ``lw_validate`` — sanity-check row alignment + artifact existence.

Design rules honored from the spec:
* BTC frames are NEVER re-decoded; uniform/motion are decoded fresh into
  ``keyframes/<VID>/``. Manifest is the single source of truth for frame order.
* Resume: a video is skipped if its frames are already in the catalog.
* Config + model/source-hash versioning so a stale run can be detected.
* Deterministic, no hard-coded FPS (UNIFORM_FPS from config).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from aic2026.lightweight.catalog import LightweightCatalog
from aic2026.lightweight.embed import EmbedConfig, assemble_features
from aic2026.lightweight.merge_btc import EMBEDDING_VERSION, MODEL_VERSION
from aic2026.lightweight.sampler import MotionChangeDetector, UniformSampler
from aic2026.models import FrameRecord

logger = logging.getLogger(__name__)


@dataclass
class LightweightConfig:
    root: Path = Path("data/processed/siglip2")
    raw_videos_dir: Path = Path("data/raw/Videos")
    uniform_fps: float = 1.0
    motion_threshold: float = 0.35
    # Disabled by default: temporal coverage matters more than a small disk
    # saving, and an unbenchmarked dedup pass can remove useful evidence.
    dedup_sim: float = 1.0
    # query-time zoom-in (spec §6)
    temporal_window_s: float = 4.0
    min_local_frames: int = 5
    enable_decode_fallback: bool = True
    cache_dynamic_frames: bool = True
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    siglip2_model: str = "google/siglip2-so400m-patch14-384"
    image_quality: int = 92
    embedding_version: str = EMBEDDING_VERSION

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest_siglip2.jsonl"

    @property
    def catalog_path(self) -> Path:
        return self.root / "catalog.db"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def keyframes_root(self) -> Path:
        return self.root / "keyframes"

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.keyframes_root.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["root"] = str(d["root"])
        d["raw_videos_dir"] = str(d["raw_videos_dir"])
        return d

    def save_config(self) -> None:
        self.config_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "LightweightConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # Path fields come back as strings.
        for k in ("root", "raw_videos_dir"):
            if k in data:
                data[k] = Path(data[k])
        return cls(**data)


def video_source_hash(video_path: Path) -> str:
    st = video_path.stat()
    raw = f"{video_path.resolve()}|{st.st_size}|{st.st_mtime}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _list_videos(raw_dir: Path) -> list[Path]:
    out: list[Path] = []
    if not raw_dir.exists():
        return out
    # Videos are nested (e.g. Videos_L21_a/video/L21_V001.mp4); walk to any depth.
    for mp4 in sorted(raw_dir.glob("**/*.mp4")):
        out.append(mp4)
    return out


def lw_build(cfg: LightweightConfig, force: bool = False, limit: int | None = None) -> int:
    """Decode uniform + motion frames for every video, append to manifest+catalog.

    BTC frames must already be merged via ``MergeBtcConfig.merge`` (this reads the
    existing ``manifest_lw.jsonl`` as the base and appends uniform/motion rows).
    Returns #frames added.
    """
    cfg.ensure_dirs()
    # Only ONE build mutates the catalog at a time. The lock makes a second
    # concurrent lw-build wait briefly then error clearly instead of corrupting
    # the SQLite catalog through write contention.
    from aic2026.lightweight.catalog import CatalogLock

    with CatalogLock(cfg.root):
        catalog = LightweightCatalog(cfg.catalog_path)
        # The catalog is the committed source of truth (BTC merged in lw-merge-btc +
        # uniform/motion appended here). manifest_lw.jsonl is always *regenerated*
        # from the catalog so a crash mid-build heals on the next resume — no
        # divergence between the two, and peak RAM stays flat (we never buffer the
        # whole corpus in memory; the previous all-in-RAM version exhausted address
        # space on video 3).
        _regenerate_manifest(catalog, cfg)

        videos = _list_videos(cfg.raw_videos_dir)
        if limit is not None:
            videos = videos[:limit]

        uniform = UniformSampler(fps=cfg.uniform_fps)
        motion = MotionChangeDetector(threshold=cfg.motion_threshold)
        added = 0

        for vid in videos:
            video_id = vid.stem
            if not force and catalog.is_video_done(video_id):
                logger.info("skip %s (already built)", video_id)
                continue
            cap_info = _video_meta(vid)
            if cap_info is None:
                logger.warning("cannot open %s", video_id)
                continue
            fps, frame_count, duration = cap_info
            catalog.upsert_video(video_id, fps, frame_count, video_source_hash(vid))

            # Already-used frame_ids for this video (BTC rows merged earlier, plus
            # any prior uniform/motion rows). NEW frames reuse the REAL frame index
            # captured during decode (== frame you'd land on with Ctrl+G), so we must
            # skip any target that already maps to an existing frame (it's literally
            # the same keyframe BTC already has). This keeps frame_id == true frame
            # index for submission, with no per-video offset.
            used_ids = set(catalog.frame_ids(video_id))

            video_added = 0

            # Plan all target timestamps (uniform grid + motion peaks) once.
            uniform_ts = [(ts, "uniform") for ts, _ in uniform.sample(vid, duration)]
            motion_pts = motion.peaks(vid, duration)
            # Drop motion peaks that land within 0.3s of a uniform frame (the dedup
            # pass catches closer dups, but this avoids decoding obvious repeats).
            uniform_set = {ts for ts, _ in uniform_ts}
            motion_ts = [
                (ts, "motion")
                for ts, _ in motion_pts
                if not any(abs(ts - u) < 0.3 for u in uniform_set)
            ]
            targets = sorted(uniform_ts + motion_ts, key=lambda x: x[0])

            # SINGLE forward decode pass — capture the frame nearest each target ts.
            # Returns the TRUE frame index (fidx) alongside the image, so the saved
            # frame_id matches what Ctrl+G would land on (no fps-rounding drift).
            # `_single_pass_decode` is a generator.  Each image is saved and its
            # catalog row queued before the next frame is decoded, so a long video
            # never accumulates thousands of full-resolution PIL images in RAM.
            for fidx, ts, source, img in _single_pass_decode(vid, targets, cfg):
                # Skip if this exact frame index is already present (BTC duplicate).
                if fidx in used_ids:
                    continue
                rec = _make_record(video_id, fidx, ts, source, vid, cfg)
                rec.keyframe_path = str(_save_frame(img, rec, cfg))
                catalog.add_frame(
                    video_id=rec.video_id,
                    frame_id=rec.frame_id,
                    timestamp=rec.timestamp,
                    source=rec.source,
                    path=rec.keyframe_path,
                    model_version=MODEL_VERSION,
                    embedding_version=cfg.embedding_version,
                )
                video_added += 1
                used_ids.add(fidx)

            added += video_added
            # Commit catalog + regenerate manifest per video so progress survives
            # crashes (the manifest is a pure reflection of the committed catalog).
            catalog.commit_frames()
            _regenerate_manifest(catalog, cfg)
            # Mark done ONLY after the commit succeeded, so a video interrupted
            # mid-build (is_built stays 0) is resumed by the next run.
            catalog.mark_video_built(video_id)
            logger.info("built %s: +%d frames (uniform+motion)", video_id, video_added)

        cfg.save_config()
        catalog.close()
    logger.info("lw_build done: +%d frames", added)
    return added


def _regenerate_manifest(catalog: "LightweightCatalog", cfg: LightweightConfig) -> None:
    """Rebuild manifest_lw.jsonl from the committed catalog (crash-safe source).

    The catalog holds BTC (merged by lw-merge-btc) + uniform + motion rows with
    the same fields as FrameRecord, ordered by ``frame_id`` (== manifest row
    position). The lightweight CLIP/SigLIP2 matrices are row-aligned to this
    manifest, so it must always reflect the catalog exactly.
    """
    rows = catalog.all_frames()
    lines: list[str] = []
    for line_idx, r in enumerate(rows):
        clip_idx = r["clip_index"]
        # vector_id == manifest LINE index == features matrix ROW index.
        # features_clip.npy / features_siglip2.npy are assembled row-aligned to
        # this manifest in assemble_features(), so a candidate's vector_id must
        # point back to the SAME row that was searched. Using -1 here (the old
        # value) broke every downstream stage that keys on vector_id:
        #   * RetrievalAgent._deduplicate_best skips vector_id is None AND dedups
        #     by vector_id, so all -1 candidates collapsed to a single result;
        #   * rerank_with_object_evidence is guarded by `if item.vector_id is not
        #     None`, so object evidence was silently skipped for every frame.
        rec = FrameRecord(
            vector_id=line_idx,
            video_id=r["video_id"],
            frame_id=int(r["frame_id"]),
            keyframe_path=r["path"],
            clip_feature_index=int(clip_idx) if clip_idx is not None else None,
            shot_id=None,
            timestamp=float(r["timestamp"]),
            source=r["source"],
            model_version=r["model_version"],
            embedding_version=r["embedding_version"],
            object_labels=LightweightCatalog.parse_object_labels(r["object_labels"]),
        )
        lines.append(rec.model_dump_json())
    # Publish manifest atomically; feature files are row-aligned to it and a
    # half-written JSONL would otherwise silently remap every retrieval result.
    tmp = cfg.manifest_path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp.replace(cfg.manifest_path)


def _video_meta(vid: Path):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("uv sync --extra video") from exc
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (frame_count / fps) if (fps > 0 and frame_count > 0) else 0.0
        return fps, frame_count, duration
    finally:
        cap.release()


def _make_record(video_id, frame_id, ts, source, vid, cfg) -> FrameRecord:
    return FrameRecord(
        vector_id=-1,  # assigned at index time
        video_id=video_id,
        frame_id=frame_id,
        keyframe_path="",  # filled after decode
        clip_feature_index=None,
        shot_id=None,
        timestamp=ts,
        source=source,
        model_version=MODEL_VERSION,
        embedding_version=cfg.embedding_version,
    )


def _single_pass_decode(
    vid: Path, targets: list[tuple[float, str]], cfg
) -> Iterator[tuple[int, float, str, object]]:
    """Forward-decode ``vid`` once, capturing the frame nearest each target ts.

    ``targets`` = sorted list of ``(timestamp_s, source)``. Yields captured
    ``(fidx, ts, source, PIL_image)`` tuples in target order, where ``fidx`` is the
    TRUE frame index read from the video (the frame you'd land on with Ctrl+G).
    This is used directly as the submission ``frame_id`` — no fps-driven rounding,
    so it stays exact regardless of 25/29.97/30 fps.

    A single ``cap.read()`` loop walks the video; whenever the playhead passes the
    next target we grab the closest frame. Much faster than per-target seeks.
    """
    from PIL import Image

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("uv sync --extra video") from exc
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        return []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not targets or fps <= 0:
        cap.release()
        return
    target_frame_idx = [(max(0.0, ts) * fps, src) for ts, src in targets]
    tgt_pos = 0
    fidx = 0
    try:
        while tgt_pos < len(target_frame_idx):
            tgt_f, tgt_src = target_frame_idx[tgt_pos]
            try:
                ok, bgr = cap.read()
            except Exception:  # cv2 can raise SystemError on a corrupt frame
                ok, bgr = False, None
            if not ok:
                # End of stream OR a decode error — stop (remaining targets
                # simply won't be captured for this video).
                break
            # Capture when we're at or just past the target frame.
            if fidx >= tgt_f:
                try:
                    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert("RGB")
                except Exception:
                    fidx += 1
                    continue
                yield int(fidx), float(targets[tgt_pos][0]), tgt_src, img
                tgt_pos += 1
                # Fast-forward: skip frames strictly between captures by seeking
                # only if there's a big gap (keeps the loop cheap for sparse grids).
                while tgt_pos < len(target_frame_idx) and target_frame_idx[tgt_pos][0] <= fidx + 1:
                    # target essentially at same frame — grab a clone.
                    yield (
                        int(fidx), float(targets[tgt_pos][0]),
                        target_frame_idx[tgt_pos][1], img.copy(),
                    )
                    tgt_pos += 1
            fidx += 1
    finally:
        cap.release()


def _save_frame(img, rec, cfg) -> Path:
    out = cfg.keyframes_root / rec.video_id / f"{rec.frame_id:09d}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=cfg.image_quality, optimize=True)
    return out


def lw_embed(cfg: LightweightConfig, force: bool = False) -> dict:
    from aic2026.lightweight.merge_btc import load_manifest

    records = load_manifest(cfg.manifest_path)
    embed_cfg = EmbedConfig(
        root=cfg.root,
        official_features=cfg.root / "features_siglip2.npy",
        clip_model=cfg.clip_model,
        clip_pretrained=cfg.clip_pretrained,
        siglip2_model=cfg.siglip2_model,
        image_quality=cfg.image_quality,
        dedup_sim=cfg.dedup_sim,
    )
    return assemble_features(records, embed_cfg, force=force)


def lw_index(cfg: LightweightConfig) -> dict:
    """Build FAISS VectorIndex for SigLIP2 modality + register artifacts."""
    from aic2026.retrieval.index import VectorIndex

    catalog = LightweightCatalog(cfg.catalog_path)
    sig_path = cfg.root / "features_siglip2.npy"
    if not sig_path.exists():
        raise FileNotFoundError(f"missing {sig_path} — run lw-embed first")
    sig_idx = VectorIndex.from_npy(str(sig_path), mmap=False)
    sig_dim = int(sig_idx.vectors.shape[1])
    catalog.register_artifact(
        "features_siglip2", str(sig_path), model_name="siglip2",
        model_version=cfg.siglip2_model, dim=sig_dim, dtype="float32",
    )
    catalog.register_model_version(
        "siglip2", cfg.siglip2_model, dim=sig_dim, dtype="float32"
    )
    summary = {"siglip2_dim": sig_dim, "siglip2": True}
    catalog.close()
    logger.info("lw_index: %s", summary)
    return summary


def lw_validate(cfg: LightweightConfig) -> dict:
    """Spec §11 validation: row alignment + artifact existence (SigLIP2 only)."""
    from aic2026.lightweight.merge_btc import load_manifest

    errors: list[str] = []
    records = load_manifest(cfg.manifest_path)
    n_manifest = len(records)
    sig_path = cfg.root / "features_siglip2.npy"
    if not sig_path.exists():
        errors.append("features_siglip2.npy missing")
    if sig_path.exists():
        sshape = np.load(sig_path, mmap_mode="r").shape
        if sshape[0] != n_manifest:
            errors.append(f"siglip2 rows {sshape[0]} != manifest {n_manifest}")
    if n_manifest:
        ids = [r.vector_id for r in records]
        if ids != list(range(n_manifest)):
            errors.append("manifest vector_id must equal its row index")
        keys = [(r.video_id, r.frame_id) for r in records]
        if len(set(keys)) != n_manifest:
            errors.append("duplicate (video_id, frame_id) rows in manifest")
    # Check numerical validity in chunks.
    if sig_path.exists():
        matrix = np.load(sig_path, mmap_mode="r")
        for start in range(0, matrix.shape[0], 8192):
            if not np.isfinite(matrix[start:start + 8192]).all():
                errors.append("siglip2 contains NaN/Inf")
                break
    # Source coverage.
    from collections import Counter

    src = Counter(r.source for r in records)
    summary = {
        "frames": n_manifest,
        "sources": dict(src),
        "errors": errors,
        "ok": len(errors) == 0,
    }
    if errors:
        logger.warning("lw_validate FAILED: %s", errors)
    else:
        logger.info("lw_validate OK: %d frames %s", n_manifest, dict(src))
    return summary
