"""Embedding assembly for the lightweight pipeline (spec §4, §5, §11).

Three frame sources share ONE row-aligned manifest. Their vectors are assembled
into two modality matrices:

* CLIP (512-d): BTC frames reuse ``official_features.npy`` rows via
  ``FrameRecord.clip_feature_index`` — we do NOT re-embed 177k curated frames.
  Uniform/motion (NEW) frames are freshly embedded with ``OpenCLIPFrameEncoder``.
* SigLIP2 (768-d): ONLY computed for NEW frames (uniform + motion). BTC frames
  keep their CLIP vectors and contribute NOTHING to the SigLIP2 matrix (their
  rows are left as zero vectors). This avoids re-embedding 177k BTC frames with
  SigLIP2 on CPU (~46h) — the whole point of choosing option (b).

Both output matrices are row-aligned with ``manifest_lw.jsonl`` so the FAISS
index and the manifest stay in lockstep. BTC SigLIP2 rows being zero means a
SigLIP2 search returns score 0 for BTC frames, so they never enter the SigLIP2
candidate pool — exactly the intended architecture (BTC -> CLIP only; NEW ->
CLIP + SigLIP2).
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aic2026.models import FrameRecord

logger = logging.getLogger(__name__)

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
SIGLIP2_MODEL = "google/siglip2-so400m-patch14-384"
# SigLIP2-SO400M: 1152-d. Hard-coded because BTC frames carry no
# SigLIP2 vector, so we cannot infer the dim from the (all-zero) BTC rows.
SIGLIP2_DIM = 1152


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    """Write a complete NPY then atomically publish it.

    A cancelled embedding job must leave the previous, aligned feature matrix
    intact rather than a valid-looking file containing partially filled rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npy", dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        np.save(tmp, values)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _temporary_matrix(path: Path, shape: tuple[int, int]) -> tuple[Path, np.memmap]:
    """Allocate an NPY-compatible matrix on disk, not in process RAM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npy", dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    return tmp, np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=shape)


def _all_finite(matrix: np.ndarray, rows_per_chunk: int = 8192) -> bool:
    """Validate a large memmap without allocating a full boolean matrix."""
    for start in range(0, matrix.shape[0], rows_per_chunk):
        if not np.isfinite(matrix[start : start + rows_per_chunk]).all():
            return False
    return True


@dataclass
class EmbedConfig:
    root: Path = Path("data/processed/siglip2")
    official_features: Path = Path("data/processed/siglip2/features_siglip2.npy")
    clip_model: str = CLIP_MODEL
    clip_pretrained: str = CLIP_PRETRAINED
    siglip2_model: str = SIGLIP2_MODEL
    image_quality: int = 92
    # Perceptual dedup (spec §2): drop near-identical frames. Set to 1.0 to
    # disable (keep every frame). Lower = more aggressive pruning. Default OFF
    # (1.0) so the build is fast + deterministic; enable when storage matters.
    dedup_sim: float = 1.0

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest_siglip2.jsonl"

    @property
    def clip_feats_path(self) -> Path:
        return self.root / "features_siglip2.npy"

    @property
    def siglip2_feats_path(self) -> Path:
        return self.root / "features_siglip2.npy"

    @property
    def keyframes_root(self) -> Path:
        return self.root / "keyframes"

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.keyframes_root.mkdir(parents=True, exist_ok=True)


def _load_encoders(cfg: EmbedConfig):
    from aic2026.data_platform.video_frames import OpenCLIPFrameEncoder
    from aic2026.embeddings.siglip2 import Siglip2TextEmbedder

    clip_enc = OpenCLIPFrameEncoder(model_name=cfg.clip_model, pretrained=cfg.clip_pretrained)
    sig_enc = Siglip2TextEmbedder(model_name=cfg.siglip2_model)
    return clip_enc, sig_enc


def _decode_image(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB").copy()


def assemble_features(
    records: list[FrameRecord],
    cfg: EmbedConfig,
    clip_enc: object | None = None,
    sig_enc: object | None = None,
    force: bool = False,
) -> dict:
    """Embed/assemble CLIP + SigLIP2 matrices row-aligned to ``records``.

    Returns a summary dict. Writes ``features_clip.npy`` / ``features_siglip2.npy``.
    Skips if both matrices exist and ``force`` is False.
    """
    cfg.ensure_dirs()
    if (
        not force
        and cfg.clip_feats_path.exists()
        and cfg.siglip2_feats_path.exists()
    ):
        logger.info("assemble_features: outputs exist, skip (force to rebuild)")
        return {"skipped": True, "frames": len(records)}

    if clip_enc is None or sig_enc is None:
        clip_enc, sig_enc = _load_encoders(cfg)

    # --- CLIP: reuse official rows for BTC, embed the rest -----------------
    official_clip = np.load(cfg.official_features, mmap_mode="r")
    n = len(records)
    clip_dim = int(official_clip.shape[1])
    # These matrices are ~1 GB together on the full corpus.  Build them as
    # temporary NPY memmaps so peak resident RAM is bounded by an image batch;
    # only replace the published artifacts once all rows are valid.
    clip_tmp, clip_mat = _temporary_matrix(cfg.clip_feats_path, (n, clip_dim))
    sig_tmp, sig_mat = _temporary_matrix(cfg.siglip2_feats_path, (n, SIGLIP2_DIM))
    for i, rec in enumerate(records):
        if rec.source == "btc" and rec.clip_feature_index is not None:
            clip_mat[i] = np.asarray(official_clip[int(rec.clip_feature_index)], dtype=np.float32)
        else:
            clip_mat[i] = np.nan  # placeholder; filled below
    # Collect NEW (non-BTC) images to embed once (CLIP + SigLIP2) in batches.
    # Process in small batches so we never hold all 7k+ decoded tensors in RAM
    # at once (loading every NEW frame simultaneously OOMs on CPU builds).
    new_idx: list[int] = [i for i, r in enumerate(records) if r.source != "btc"]
    sig_mat[:] = 0.0
    if new_idx:
        BATCH = 64
        for start in range(0, len(new_idx), BATCH):
            chunk = new_idx[start : start + BATCH]
            imgs = [_decode_image(Path(records[i].keyframe_path)) for i in chunk]
            try:
                new_clip = clip_enc.encode_images(imgs)
                # SigLIP2 ONLY for NEW frames — never for BTC (avoids ~46h re-embed).
                new_sig = sig_enc.encode_images(imgs)
            finally:
                for image in imgs:
                    image.close()
            for k, i in enumerate(chunk):
                clip_mat[i] = new_clip[k]
                sig_mat[i] = new_sig[k]
    else:
        new_sig = np.empty((0, SIGLIP2_DIM), dtype=np.float32)

    if not _all_finite(clip_mat) or not _all_finite(sig_mat):
        raise ValueError("Embedding produced NaN/Inf; feature files were not replaced")

    # --- Optional perceptual dedup (spec §2) --------------------------------
    # Prune near-identical frames (cosine >= dedup_sim within a small temporal
    # window) to save storage while preserving temporal coverage. BTC frames
    # are protected. Disabled by default (dedup_sim == 1.0) for speed.
    deduped = 0
    if 0.0 < cfg.dedup_sim < 1.0:
        from aic2026.lightweight.dedup import DedupConfig, dedup_frames

        kept_recs, kept_clip, kept_idx = dedup_frames(
            records, clip_mat, DedupConfig(sim_threshold=cfg.dedup_sim)
        )
        deduped = n - len(kept_recs)
        if deduped:
            kept_sig = sig_mat[kept_idx]
            _atomic_save_npy(cfg.clip_feats_path, kept_clip.astype(np.float32, copy=False))
            _atomic_save_npy(cfg.siglip2_feats_path, kept_sig.astype(np.float32, copy=False))
            cfg.manifest_path.write_text(
                "\n".join(r.model_dump_json() for r in kept_recs) + "\n", encoding="utf-8"
            )
            records = kept_recs
            n = len(records)
            # The deduplicated matrices were atomically published above.
            del clip_mat, sig_mat
            clip_tmp.unlink(missing_ok=True)
            sig_tmp.unlink(missing_ok=True)
        else:
            clip_mat.flush()
            sig_mat.flush()
            # Windows cannot rename an open mmap; release both handles first.
            del clip_mat, sig_mat
            clip_tmp.replace(cfg.clip_feats_path)
            sig_tmp.replace(cfg.siglip2_feats_path)
    else:
        clip_mat.flush()
        sig_mat.flush()
        del clip_mat, sig_mat
        clip_tmp.replace(cfg.clip_feats_path)
        sig_tmp.replace(cfg.siglip2_feats_path)

    summary = {
        "frames": n,
        "clip_dim": clip_dim,
        "siglip2_dim": SIGLIP2_DIM,
        "clip_reused_from_official": sum(
            1 for r in records if r.source == "btc" and r.clip_feature_index is not None
        ),
        "clip_newly_embedded": len(new_idx),
        "siglip2_embedded_new_only": len(new_idx),
        "siglip2_zero_rows_btc": sum(1 for r in records if r.source == "btc"),
        "deduped": deduped,
    }
    logger.info("assemble_features: %s", summary)
    return summary
