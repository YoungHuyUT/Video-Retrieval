"""BGE-M3 dense text encoder for NOTEBOOK ASR retrieval (P0).

Provides multilingual dense embeddings for ASR segment texts, enabling
semantic search alongside BM25 lexical search.

Model: BAAI/bge-m3 (1024-dim, supports Vietnamese natively).
- ~2.3GB download on first use, cached in HuggingFace home (~/.cache/huggingface)
- CPU inference: ~5-15ms per segment (batched)
- GPU: ~1-3ms per segment (if CUDA available)

Design:
- Singleton per model (avoid reloading 2GB weights per query).
- Lazy load (only import sentence-transformers when first needed).
- Warmup on first load (eliminates first-query latency).
- Batch encode for efficiency (single forward pass for N texts).
- L2-normalized output (cosine similarity via dot product).

Precomputation:
    python -m aic2026.cli build-dense-index \\
        --sidecar data/processed/asr_sidecar.jsonl \\
        --out data/processed/asr_dense.npy

Stores a (N_segments, 1024) float32 array aligned with sidecar segment order.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BGE_M3_MODEL = "BAAI/bge-m3"
DENSE_EMBEDDING_DIM = 1024

# ---------------------------------------------------------------------------
# Singleton cache (one model per process, keyed by model name)
# ---------------------------------------------------------------------------
_CACHE: dict[str, "DenseTextEncoder"] = {}


class DenseTextEncoder:
    """BGE-M3 text encoder with singleton access and lazy loading.

    Usage:
        encoder = DenseTextEncoder.get_or_create()
        vec = encoder.encode("một phụ nữ đang nói chuyện")  # (1024,) float32
        vecs = encoder.batch_encode(["text1", "text2"])      # (2, 1024) float32
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BGE_M3_MODEL,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device  # None = auto (cuda if available, else cpu)
        self._model = None
        self._embedding_dim = DENSE_EMBEDDING_DIM
        self._query_cache: dict[str, np.ndarray] = {}

    @classmethod
    def get_or_create(
        cls,
        model_name: str = DEFAULT_BGE_M3_MODEL,
        device: str | None = None,
    ) -> "DenseTextEncoder":
        """Return a singleton encoder for the given model (avoids reload)."""
        key = model_name
        if key not in _CACHE:
            _CACHE[key] = cls(model_name=model_name, device=device)
        return _CACHE[key]

    def load(self) -> None:
        """Load the BGE-M3 model (lazy, only on first use)."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for dense retrieval: "
                "uv sync --extra retrieval  (or pip install sentence-transformers)"
            ) from exc

        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        logger.info("DenseTextEncoder: loading %s ...", self.model_name)
        t0 = time.time()

        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )
        # Warmup: dummy encode to eliminate first-query latency
        self._model.encode(
            ["warmup"],
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False,
        )
        logger.info(
            "DenseTextEncoder: loaded %s in %.1fs",
            self.model_name,
            time.time() - t0,
        )

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text → (1024,) float32 L2-normalized vector.

        Uses an internal query cache to avoid re-encoding identical queries
        within the same process.
        """
        # Check cache first
        if text in self._query_cache:
            return self._query_cache[text]

        if self._model is None:
            self.load()
        vec = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vec = np.asarray(vec, dtype=np.float32)
        # Cache the result
        self._query_cache[text] = vec
        return vec

    def batch_encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode multiple texts → (N, 1024) float32 L2-normalized vectors.

        Much faster than N × encode() — single forward pass (batched).
        """
        if not texts:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        if self._model is None:
            self.load()

        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
        return np.asarray(vecs, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


# ---------------------------------------------------------------------------
# Precomputation helper
# ---------------------------------------------------------------------------

def precompute_asr_dense_embeddings(
    sidecar_path: str | Path,
    out_path: str | Path | None = None,
    model_name: str = DEFAULT_BGE_M3_MODEL,
    batch_size: int = 64,
) -> Path:
    """Precompute BGE-M3 embeddings for all ASR segments.

    Reads the sidecar, encodes every segment's text, and saves a
    (N_segments, 1024) float32 .npy file.  The segment order matches the
    sidecar iteration order (video_id → segment_index), which the
    DenseRetriever uses at query time.

    Args:
        sidecar_path: Path to asr_sidecar.jsonl
        out_path: Output .npy path (default: same dir as sidecar, named asr_dense.npy)
        model_name: BGE-M3 model name
        batch_size: Batch size for encoding

    Returns:
        Path to the saved .npy file
    """
    from aic2026.ingestion.asr import load_transcripts_sidecar

    sidecar_path = Path(sidecar_path)
    if out_path is None:
        out_path = sidecar_path.parent / "asr_dense.npy"
    out_path = Path(out_path)

    logger.info("Loading ASR sidecar from %s ...", sidecar_path)
    transcripts = load_transcripts_sidecar(sidecar_path)

    # Collect segment texts in order (must match DenseRetriever's iteration)
    segment_texts: list[str] = []
    segment_meta: list[tuple[str, int]] = []  # (video_id, seg_idx)
    for video_id, transcript in transcripts.items():
        for seg_idx, seg in enumerate(transcript.segments):
            text = seg.text.strip()
            if text:
                segment_texts.append(text)
                segment_meta.append((video_id, seg_idx))

    if not segment_texts:
        logger.warning("No ASR segments to encode")
        return out_path

    logger.info(
        "Encoding %d ASR segments with %s (batch=%d) ...",
        len(segment_texts), model_name, batch_size,
    )
    t0 = time.time()

    encoder = DenseTextEncoder.get_or_create(model_name=model_name)
    embeddings = encoder.batch_encode(segment_texts, batch_size=batch_size)

    elapsed = time.time() - t0
    logger.info(
        "Encoded %d segments in %.1fs (%.1f seg/s)",
        len(segment_texts), elapsed,
        len(segment_texts) / max(elapsed, 0.001),
    )

    # Save embeddings + metadata
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), embeddings)

    # Save metadata (segment order) for alignment verification
    meta_path = out_path.with_suffix(".meta.json")
    import json
    meta = {
        "model": model_name,
        "dim": DENSE_EMBEDDING_DIM,
        "count": len(segment_texts),
        "segment_keys": segment_meta,  # [(video_id, seg_idx), ...]
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    logger.info("Saved dense embeddings to %s (%s)", out_path, f"{embeddings.shape}")
    return out_path


__all__ = [
    "DenseTextEncoder",
    "DEFAULT_BGE_M3_MODEL",
    "DENSE_EMBEDDING_DIM",
    "precompute_asr_dense_embeddings",
]
