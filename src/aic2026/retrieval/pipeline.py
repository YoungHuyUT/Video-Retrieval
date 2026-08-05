from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING
import numpy as np

from aic2026.models import Candidate, FrameRecord
from .index import VectorIndex

if TYPE_CHECKING:
    from .bm25_index import BM25Index


# RRF rank-fusion constant (standard value; higher → dampens rank advantage)
_RRF_K = 60


class RetrievalPipeline:
    """Vector-only and hybrid (BM25 + vector) retrieval over a frame manifest."""

    def __init__(
        self,
        index: VectorIndex,
        manifest: list[FrameRecord],
        frames_per_video: int = 3,
    ) -> None:
        if len(index.vectors) != len(manifest):
            raise ValueError("Feature count must equal manifest record count")
        self.index = index
        self.manifest = manifest
        self.frames_per_video = frames_per_video

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        ranked_lists: list[np.ndarray],
        k: int = _RRF_K,
    ) -> dict[int, float]:
        """Reciprocal Rank Fusion over multiple ranked id arrays.

        Each list contributes ``1 / (k + 1-based_rank)`` to every document it
        contains. Documents absent from a list receive 0 contribution from it.
        """
        rrf: dict[int, float] = {}
        for ids in ranked_lists:
            for rank, idx in enumerate(ids):
                doc = int(idx)
                rrf[doc] = rrf.get(doc, 0.0) + 1.0 / (k + rank + 1)
        return rrf

    def _build_candidates(
        self,
        rrf_scores: dict[int, float],
        max_answers: int,
    ) -> list[Candidate]:
        """Convert {manifest_idx: rrf_score} → sorted, per-video-capped Candidates."""
        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

        # Rank videos by their best candidate score for stable ordering
        video_best: dict[str, float] = {}
        for idx in sorted_ids:
            vid = self.manifest[idx].video_id
            if vid not in video_best:
                video_best[vid] = rrf_scores[idx]

        per_video: dict[str, int] = defaultdict(int)
        candidates: list[Candidate] = []
        for idx in sorted_ids:
            record = self.manifest[idx]
            if per_video[record.video_id] >= self.frames_per_video:
                continue
            per_video[record.video_id] += 1
            candidates.append(
                Candidate(
                    video_id=record.video_id,
                    frame_id=record.frame_id,
                    score=rrf_scores[idx],
                    vector_id=record.vector_id,
                    keyframe_path=record.keyframe_path,
                )
            )
            if len(candidates) == max_answers:
                break
        return candidates

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        text_embedding: np.ndarray,
        top_frames: int = 500,
        max_answers: int = 100,
    ) -> list[Candidate]:
        """Vector-only cosine retrieval (backward-compatible)."""
        ids, scores = self.index.search(text_embedding, top_frames)

        # Treat vector ranks as a single-list RRF so the candidate builder
        # is shared. Raw cosine scores become rrf_scores here.
        rrf_scores: dict[int, float] = {
            int(idx): float(score) for idx, score in zip(ids, scores)
        }
        return self._build_candidates(rrf_scores, max_answers)

    def hybrid_retrieve(
        self,
        text_query: str,
        text_embedding: np.ndarray,
        bm25_index: "BM25Index",
        top_frames: int = 500,
        max_answers: int = 100,
    ) -> list[Candidate]:
        """Hybrid BM25 + vector retrieval fused with Reciprocal Rank Fusion.

        ``text_query``    — raw query string, passed to BM25 for lexical matching
        ``text_embedding``— encoded query vector, passed to VectorIndex for cosine search
        ``bm25_index``    — pre-built BM25Index over the same manifest

        RRF ensures neither modality dominates: a document ranked high by *both*
        rises to the top even if its individual scores are mediocre.
        """
        # 1. Vector search
        vec_ids, _ = self.index.search(text_embedding, top_frames)

        # 2. BM25 search
        bm25_ids, _ = bm25_index.search(text_query, top_frames)

        # 3. RRF fusion — only ranks matter, not raw scores
        rrf_scores = self._rrf_fuse([vec_ids, bm25_ids])

        return self._build_candidates(rrf_scores, max_answers)
