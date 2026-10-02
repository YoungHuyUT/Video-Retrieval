"""Dense ASR segment retrieval using BGE-M3 embeddings (P0)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from aic2026.ingestion.asr import load_transcripts_sidecar
from aic2026.notebook.types import ASRSegmentMatch

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense semantic retriever for ASR segments using BGE-M3 embeddings."""

    def __init__(
        self,
        sidecar_path: str | None = None,
        dense_embeddings_path: str | None = None,
        transcripts: dict[str, "VideoTranscript"] | None = None,
    ) -> None:
        self.sidecar_path = Path(sidecar_path) if sidecar_path else None
        self._dense_path: Path | None = None
        self._embeddings: np.ndarray | None = None
        self._seg_keys: list[tuple[str, int]] = []
        self._transcripts: dict = {}
        self._encoder = None
        self._available = False

        if dense_embeddings_path:
            self._dense_path = Path(dense_embeddings_path)
        else:
            # Check multiple candidate paths (user may have placed it in siglip2 dir).
            # Does NOT depend on sidecar_path — works with pre-loaded transcripts too.
            candidates = [
                Path("data/processed/asr_dense.npy"),
                Path("data/processed/asr_dense_features.npy"),
                Path("data/processed/siglip2/asr_dense.npy"),
                Path("data/processed/siglip2/asr_dense_features.npy"),
            ]
            self._dense_path = None
            for cand in candidates:
                if cand.exists():
                    self._dense_path = cand
                    logger.info("DenseRetriever: found embeddings at %s", cand)
                    break

        if transcripts is not None:
            self._transcripts = transcripts
            self._seg_keys = []
            for video_id, transcript in transcripts.items():
                for seg_idx, seg in enumerate(transcript.segments):
                    if seg.text.strip():
                        self._seg_keys.append((video_id, seg_idx))
            self._load_embeddings()
        else:
            self._load()

    def _load(self) -> None:
        if self.sidecar_path is None or not self.sidecar_path.exists():
            logger.warning("DenseRetriever: sidecar not found at %s", self.sidecar_path)
            return

        self._transcripts = load_transcripts_sidecar(self.sidecar_path)
        if not self._transcripts:
            logger.warning("DenseRetriever: sidecar is empty")
            return

        self._seg_keys = []
        for video_id, transcript in self._transcripts.items():
            for seg_idx, seg in enumerate(transcript.segments):
                if seg.text.strip():
                    self._seg_keys.append((video_id, seg_idx))

        self._load_embeddings()

    def _load_embeddings(self) -> None:
        """Load the dense embeddings .npy file and align with segment keys."""
        if self._dense_path is None or not self._dense_path.exists():
            logger.info(
                "DenseRetriever: dense embeddings not found at any candidate path - "
                "dense retrieval disabled. Run `aic2026 build-dense-index`.",
            )
            return

        try:
            self._embeddings = np.load(str(self._dense_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("DenseRetriever: failed to load %s: %s", self._dense_path, exc)
            return

        if self._embeddings.shape[0] != len(self._seg_keys):
            logger.warning(
                "DenseRetriever: embedding count (%d) != segment count (%d). "
                "Re-run `aic2026 build-dense-index`.",
                self._embeddings.shape[0], len(self._seg_keys),
            )
            self._embeddings = None
            return

        self._available = True
        logger.info(
            "DenseRetriever: loaded %d embeddings (%d-dim) from %s",
            self._embeddings.shape[0], self._embeddings.shape[1], self._dense_path,
        )

    @property
    def is_available(self) -> bool:
        return self._available and self._embeddings is not None

    def search_segments(self, query: str, top_k: int = 30, video_ids: set[str] | None = None) -> list[ASRSegmentMatch]:
        if not self.is_available or not query.strip():
            return []

        if self._encoder is None:
            from aic2026.notebook.dense_encoder import DenseTextEncoder
            self._encoder = DenseTextEncoder.get_or_create()

        t0 = time.time()
        q_vec = self._encoder.encode(query)
        scores = self._embeddings @ q_vec

        results: list[ASRSegmentMatch] = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            video_id, seg_idx = self._seg_keys[idx]
            if video_ids is not None and video_id not in video_ids:
                continue
            segment = self._transcripts[video_id].segments[seg_idx]
            results.append(ASRSegmentMatch(
                video_id=video_id, segment_text=segment.text,
                start=segment.start, end=segment.end, frame=segment.frame,
                matched_query=query, bm25_score=0.0, dense_score=float(score),
            ))

        results.sort(key=lambda m: m.dense_score, reverse=True)
        elapsed = time.time() - t0
        if elapsed > 0.1:
            logger.debug("DenseRetriever: search took %.3fs", elapsed)
        return results[:top_k]

    def search_queries(self, queries: Sequence[str], top_k_per_query: int = 20, video_ids: set[str] | None = None) -> list[ASRSegmentMatch]:
        """Search for multiple queries in batch (faster than one-by-one encoding)."""
        if not self.is_available or not queries:
            return []

        if self._encoder is None:
            from aic2026.notebook.dense_encoder import DenseTextEncoder
            self._encoder = DenseTextEncoder.get_or_create()

        t0 = time.time()
        # Batch-encode all queries at once (Nx faster than one-by-one).
        q_matrix = np.stack([self._encoder.encode(q) for q in queries])  # [Q, D]
        # Matrix multiply: [Q, D] @ [D, N] → [Q, N]
        all_scores = q_matrix @ self._embeddings.T  # [Q, N]

        all_matches: list[ASRSegmentMatch] = []
        for q_idx, query in enumerate(queries):
            scores = all_scores[q_idx]
            for idx, score in enumerate(scores):
                if score <= 0:
                    continue
                video_id, seg_idx = self._seg_keys[idx]
                if video_ids is not None and video_id not in video_ids:
                    continue
                segment = self._transcripts[video_id].segments[seg_idx]
                all_matches.append(ASRSegmentMatch(
                    video_id=video_id, segment_text=segment.text,
                    start=segment.start, end=segment.end, frame=segment.frame,
                    matched_query=query, bm25_score=0.0, dense_score=float(score),
                ))

        all_matches.sort(key=lambda m: m.dense_score, reverse=True)
        elapsed = time.time() - t0
        if elapsed > 0.1:
            logger.debug("DenseRetriever: batch search took %.3fs for %d queries", elapsed, len(queries))
        return all_matches[:top_k_per_query * len(queries)]

    def retrieve(self, plan: "NotebookPlan", top_k_per_query: int = 10, max_total_matches: int = 100) -> list[ASRSegmentMatch]:
        if not self.is_available or not plan.asr_queries:
            return []
        matches = self.search_queries(plan.asr_queries, top_k_per_query=top_k_per_query)
        if len(matches) > max_total_matches:
            matches = matches[:max_total_matches]
        logger.info("DenseRetriever: %d matches for %d queries", len(matches), len(plan.asr_queries))
        return matches


__all__ = ["DenseRetriever"]
