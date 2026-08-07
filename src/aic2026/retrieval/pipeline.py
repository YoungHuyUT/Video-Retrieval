from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from aic2026.models import Candidate, FrameRecord
from aic2026.temporal import align_events_dp

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
        self._video_to_manifest_indices = (
            self._build_video_manifest_index()
        )

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

    def _build_video_manifest_index(
        self,
    ) -> dict[str, np.ndarray]:
        """Group manifest rows by video and order them by frame ID."""

        grouped: defaultdict[
            str,
            dict[int, int],
        ] = defaultdict(dict)

        for manifest_index, record in enumerate(
            self.manifest
        ):
            # setdefault prevents duplicate frame IDs from creating
            # two temporal positions for the same video frame.
            grouped[record.video_id].setdefault(
                record.frame_id,
                manifest_index,
            )

        return {
            video_id: np.asarray(
                [
                    frame_to_manifest_index[frame_id]
                    for frame_id in sorted(
                        frame_to_manifest_index
                    )
                ],
                dtype=np.int64,
            )
            for video_id, frame_to_manifest_index
            in grouped.items()
        }

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

    def retrieve_trake(
        self,
        event_embeddings: np.ndarray,
        top_videos: int = 100,
        prefilter_frames_per_event: int = 500,
        penalty_weight: float = 0.005,
    ) -> list[Candidate]:
        """Rank videos and align one ordered frame to each event."""

        if top_videos <= 0:
            return []

        if prefilter_frames_per_event <= 0:
            raise ValueError(
                "prefilter_frames_per_event must be greater than zero"
            )

        if penalty_weight < 0:
            raise ValueError(
                "penalty_weight must not be negative"
            )

        query_matrix = np.asarray(
            event_embeddings,
            dtype=np.float32,
        )

        if query_matrix.ndim != 2:
            raise ValueError(
                "event_embeddings must be a 2D matrix"
            )

        event_count, embedding_dimension = (
            query_matrix.shape
        )

        if event_count == 0:
            raise ValueError(
                "TRAKE requires at least one event embedding"
            )

        expected_dimension = (
            self.index.vectors.shape[1]
        )

        if embedding_dimension != expected_dimension:
            raise ValueError(
                "Event embedding dimension does not match "
                "the indexed frame vectors: "
                f"{embedding_dimension} != "
                f"{expected_dimension}"
            )

        norms = np.linalg.norm(
            query_matrix,
            axis=1,
            keepdims=True,
        )

        query_matrix = query_matrix / np.maximum(
            norms,
            1e-12,
        )

        # Stage 1: pre-filter video candidates using each event separately.
        candidate_videos: set[str] = set()

        for event_embedding in query_matrix:
            manifest_indices, _ = self.index.search(
                event_embedding,
                prefilter_frames_per_event,
            )

            for manifest_index in manifest_indices:
                record = self.manifest[
                    int(manifest_index)
                ]

                candidate_videos.add(
                    record.video_id
                )

        ranked_candidates: list[Candidate] = []

        # Stage 2: calculate the complete events × frames matrix
        # for every candidate video and apply monotonic DP.
        for video_id in sorted(
            candidate_videos
        ):
            manifest_indices = (
                self._video_to_manifest_indices.get(
                    video_id
                )
            )

            if manifest_indices is None:
                continue

            if len(manifest_indices) < event_count:
                continue

            frame_vectors = self.index.vectors[
                manifest_indices
            ]

            similarity_matrix = (
                query_matrix
                @ frame_vectors.T
            )

            try:
                alignment_score, aligned_positions = (
                    align_events_dp(
                        similarity_matrix=similarity_matrix,
                        penalty_weight=penalty_weight,
                    )
                )
            except ValueError:
                continue

            aligned_manifest_indices = [
                int(
                    manifest_indices[position]
                )
                for position in aligned_positions
            ]

            event_frames = [
                self.manifest[
                    manifest_index
                ].frame_id
                for manifest_index
                in aligned_manifest_indices
            ]

            event_scores = [
                float(
                    similarity_matrix[
                        event_index,
                        frame_position,
                    ]
                )
                for event_index, frame_position
                in enumerate(aligned_positions)
            ]

            representative_event_index = int(
                np.argmax(event_scores)
            )

            representative_record = self.manifest[
                aligned_manifest_indices[
                    representative_event_index
                ]
            ]

            ranked_candidates.append(
                Candidate(
                    video_id=video_id,
                    frame_id=(
                        representative_record.frame_id
                    ),
                    score=float(
                        alignment_score
                        / event_count
                    ),
                    vector_id=(
                        representative_record.vector_id
                    ),
                    keyframe_path=(
                        representative_record.keyframe_path
                    ),
                    event_frames=event_frames,
                )
            )

        ranked_candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.video_id,
            )
        )

        return ranked_candidates[:top_videos]

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