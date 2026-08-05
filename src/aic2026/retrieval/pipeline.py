from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from aic2026.models import Candidate, FrameRecord

from .index import VectorIndex

if TYPE_CHECKING:
    from .bm25_index import BM25Index


# Standard Reciprocal Rank Fusion constant.
_RRF_K = 60


class RetrievalPipeline:
    """Frame retrieval with separate raw and diversified candidate stages.

    Raw retrieval preserves high-recall evidence without applying a per-video
    limit. Diversification is a separate stage intended mainly for KIS-style
    result selection.
    """

    def __init__(
        self,
        index: VectorIndex,
        manifest: list[FrameRecord],
        frames_per_video: int = 3,
    ) -> None:
        if len(index.vectors) != len(manifest):
            raise ValueError("Feature count must equal manifest record count")

        if frames_per_video <= 0:
            raise ValueError("frames_per_video must be greater than zero")

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
        """Fuse ranked manifest indices using Reciprocal Rank Fusion."""

        fused_scores: dict[int, float] = {}

        for ids in ranked_lists:
            for rank, idx in enumerate(ids):
                manifest_idx = int(idx)
                contribution = 1.0 / (k + rank + 1)
                fused_scores[manifest_idx] = (
                    fused_scores.get(manifest_idx, 0.0) + contribution
                )

        return fused_scores

    def _candidates_from_scores(
        self,
        scores_by_manifest_idx: dict[int, float],
        limit: int | None = None,
    ) -> list[Candidate]:
        """Convert manifest-index scores into globally ranked candidates.

        This method deliberately does not apply a per-video limit.
        """

        if limit is not None and limit <= 0:
            return []

        sorted_ids = sorted(
            scores_by_manifest_idx,
            key=scores_by_manifest_idx.__getitem__,
            reverse=True,
        )

        if limit is not None:
            sorted_ids = sorted_ids[:limit]

        candidates: list[Candidate] = []

        for manifest_idx in sorted_ids:
            record = self.manifest[manifest_idx]

            candidates.append(
                Candidate(
                    video_id=record.video_id,
                    frame_id=record.frame_id,
                    score=float(scores_by_manifest_idx[manifest_idx]),
                    vector_id=record.vector_id,
                    keyframe_path=record.keyframe_path,
                )
            )

        return candidates

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def diversify_candidates(
        self,
        candidates: list[Candidate],
        max_answers: int = 100,
        frames_per_video: int | None = None,
    ) -> list[Candidate]:
        """Apply a per-video cap after raw retrieval.

        This is suitable for KIS, where returning many nearly identical frames
        from one video reduces result diversity. QA and TRAKE should normally
        consume the raw candidate pool instead.
        """

        if max_answers <= 0:
            return []

        per_video_limit = (
            self.frames_per_video
            if frames_per_video is None
            else frames_per_video
        )

        if per_video_limit <= 0:
            raise ValueError("frames_per_video must be greater than zero")

        per_video_count: dict[str, int] = defaultdict(int)
        selected: list[Candidate] = []

        ranked = sorted(
            candidates,
            key=lambda candidate: candidate.score,
            reverse=True,
        )

        for candidate in ranked:
            if per_video_count[candidate.video_id] >= per_video_limit:
                continue

            per_video_count[candidate.video_id] += 1
            selected.append(candidate)

            if len(selected) >= max_answers:
                break

        return selected

    # ------------------------------------------------------------------
    # Vector retrieval
    # ------------------------------------------------------------------

    def retrieve_raw(
        self,
        text_embedding: np.ndarray,
        top_frames: int = 500,
    ) -> list[Candidate]:
        """Return globally ranked vector candidates without diversification."""

        ids, scores = self.index.search(text_embedding, top_frames)

        scores_by_manifest_idx = {
            int(idx): float(score)
            for idx, score in zip(ids, scores)
        }

        return self._candidates_from_scores(
            scores_by_manifest_idx,
            limit=top_frames,
        )

    def retrieve(
        self,
        text_embedding: np.ndarray,
        top_frames: int = 500,
        max_answers: int = 100,
    ) -> list[Candidate]:
        """Vector retrieval followed by per-video diversification.

        This preserves the previous public behavior for KIS and existing code.
        """

        raw_candidates = self.retrieve_raw(
            text_embedding=text_embedding,
            top_frames=top_frames,
        )

        return self.diversify_candidates(
            candidates=raw_candidates,
            max_answers=max_answers,
        )

    # ------------------------------------------------------------------
    # Hybrid retrieval
    # ------------------------------------------------------------------

    def hybrid_retrieve_raw(
        self,
        text_query: str,
        text_embedding: np.ndarray,
        bm25_index: BM25Index,
        top_frames: int = 500,
    ) -> list[Candidate]:
        """Return raw candidates from vector and BM25 retrieval using RRF."""

        vector_ids, _ = self.index.search(
            text_embedding,
            top_frames,
        )

        bm25_ids, _ = bm25_index.search(
            text_query,
            top_frames,
        )

        fused_scores = self._rrf_fuse(
            [vector_ids, bm25_ids],
        )

        return self._candidates_from_scores(
            fused_scores,
            limit=top_frames,
        )

    def hybrid_retrieve(
        self,
        text_query: str,
        text_embedding: np.ndarray,
        bm25_index: BM25Index,
        top_frames: int = 500,
        max_answers: int = 100,
    ) -> list[Candidate]:
        """Hybrid retrieval followed by per-video diversification."""

        raw_candidates = self.hybrid_retrieve_raw(
            text_query=text_query,
            text_embedding=text_embedding,
            bm25_index=bm25_index,
            top_frames=top_frames,
        )

        return self.diversify_candidates(
            candidates=raw_candidates,
            max_answers=max_answers,
        )