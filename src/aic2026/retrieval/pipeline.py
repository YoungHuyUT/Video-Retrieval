from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from aic2026.models import Candidate, FrameRecord
from aic2026.temporal import align_events_dp

from .index import VectorIndex

if TYPE_CHECKING:
    from .bm25_index import BM25Index
    from .vectordb import ChromaVectorStore


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
        index: VectorIndex | ChromaVectorStore,
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
        # Cache video_id per row so the FAISS/NumPy backend can mask by video.
        if hasattr(self.index, "manifest_video_ids"):
            self.index.manifest_video_ids = [record.video_id for record in manifest]
        self._video_to_manifest_indices = (
            self._build_video_manifest_index()
        )
        # Mean-pool các frame vector của mỗi video thành 1 vector đại diện (video-level
        # embedding) để làm coarse filter trước khi chạy DP trên dataset lớn. Vector đã
        # L2-normalize, mean-pool xong normalize lại cho chắc (cosine không đổi hướng).
        self._video_embeddings: dict[str, np.ndarray] = {}
        for video_id, idx in self._video_to_manifest_indices.items():
            vec = self.index.vectors[idx].mean(axis=0)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            self._video_embeddings[video_id] = vec

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def filter_videos_by_metadata(
        self,
        terms: list[str],
        video_ids: set[str] | None = None,
    ) -> list[str]:
        """Return videos whose metadata/object labels contain any of *terms*.

        This is a hard pre-filter used before CLIP search when a query contains a
        concept that lives only in metadata (e.g. "trong cửa hàng"): narrowing the
        candidate video set before embedding retrieval cuts many false frames and
        raises precision. Terms are matched case-insensitively and accent-insensitively.
        """
        if not terms:
            return sorted(video_ids or {record.video_id for record in self.manifest})

        import unicodedata

        def _fold(value: str) -> str:
            value = unicodedata.normalize("NFC", value).casefold()
            return "".join(c for c in unicodedata.normalize("NFD", value) if unicodedata.category(c) != "Mn")

        folded_terms = [_fold(term) for term in terms]
        allowed = video_ids or {record.video_id for record in self.manifest}
        matched: set[str] = set()

        for record in self.manifest:
            if record.video_id not in allowed:
                continue
            haystack = " ".join(
                [
                    *(record.object_labels or []),
                    record.title or "",
                    record.description or "",
                ]
            )
            folded_haystack = _fold(haystack)
            if any(term and term in folded_haystack for term in folded_terms):
                matched.add(record.video_id)

        # Fallback: no video matched the metadata filter — return everything so we
        # never lose recall because metadata was empty or the term was a paraphrase.
        if not matched:
            return sorted(allowed)
        return sorted(matched)

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

    @staticmethod
    def _mask_video_ids(
        candidates: list[Candidate],
        video_ids: set[str] | None,
    ) -> list[Candidate]:
        """Drop candidates whose video is not in *video_ids* (metadata pre-filter)."""

        if not video_ids:
            return candidates
        allowed = set(video_ids)
        return [candidate for candidate in candidates if candidate.video_id in allowed]

    def filter_terms_to_video_ids(
        self,
        terms: list[str],
        video_ids: set[str] | None = None,
    ) -> set[str] | None:
        """Resolve metadata filter terms to an allowed video set (or None = no filter).

        Returns ``None`` when *terms* is empty so callers can skip masking entirely.
        """
        if not terms:
            return None
        matched = self.filter_videos_by_metadata(terms, video_ids)
        return set(matched)

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

    def search_with_filter(
        self,
        text_embedding: np.ndarray,
        top_frames: int,
        video_ids: set[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vector search, optionally pushed down to the DB via ``video_ids``.

        Chroma applies the filter inside the ANN scan; FAISS/NumPy masks the
        returned rows. Either way the metadata pre-filter runs at the index
        tier instead of post-hoc Python masking.
        """
        if video_ids:
            return self.index.search_filtered(text_embedding, top_frames, video_ids)
        return self.index.search(text_embedding, top_frames)

    def retrieve_raw(
        self,
        text_embedding: np.ndarray,
        top_frames: int = 500,
        video_ids: set[str] | None = None,
    ) -> list[Candidate]:
        """Return globally ranked vector candidates without diversification."""

        ids, scores = self.search_with_filter(text_embedding, top_frames, video_ids)

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
        video_ids: set[str] | None = None,
    ) -> list[Candidate]:
        """Vector retrieval followed by per-video diversification.

        This preserves the previous public behavior for KIS and existing code.
        """

        raw_candidates = self.retrieve_raw(
            text_embedding=text_embedding,
            top_frames=top_frames,
            video_ids=video_ids,
        )

        return self.diversify_candidates(
            candidates=raw_candidates,
            max_answers=max_answers,
        )

    def video_level_rerank(
        self,
        candidates: list[Candidate],
        top_videos: int | None = None,
        frames_per_video: int | None = None,
        aggregation_top_k: int = 3,
    ) -> list[Candidate]:
        """Re-rank frame candidates at the video level (KIS rerank + coarse filter).

        KIS ground truth is video-level (``video_id`` + ``ranges``), so a purely
        frame-level ranking is brittle: a single high-scoring outlier frame can
        lift the wrong video, while the correct video may have no single top
        frame even though many of its frames match the concept well.

        This stage fixes that with three steps:

        1. **Aggregate** each video's strongest frame scores into one
           ``video_score`` (log-sum-exp of the top-``aggregation_top_k`` frames).
           Log-sum-exp rewards videos with *several* strongly matching frames
           rather than a lone outlier, which matches how a correct KIS video
           typically looks (the whole scene matches the query).
        2. **Coarse filter** — keep only the top-``top_videos`` videos by
           ``video_score``; the rest are dropped before they can pollute the
           final list. ``top_videos <= 0`` or ``>=`` the number of videos keeps
           everything (backward-compatible, used on small datasets).
        3. **Re-emit** per-frame candidates ordered by their video-aware score
           (whole video jumps together by ``video_score``, frames inside a video
           keep a stable ``1e-4`` tiebreak) so the correct video dominates the
           top of the result while the best representative frame leads each video.

        The returned candidates retain every original field; only ``score`` is
        replaced by the video-aware value so downstream MMR/diversity still works.
        """

        if not candidates:
            return []

        if frames_per_video is None:
            frames_per_video = self.frames_per_video
        if frames_per_video <= 0:
            raise ValueError("frames_per_video must be greater than zero")

        by_video: dict[str, list[Candidate]] = defaultdict(list)
        for cand in candidates:
            by_video[cand.video_id].append(cand)

        # Sort each video's frames by frame score (desc) before aggregation.
        for frames in by_video.values():
            frames.sort(key=lambda c: c.score, reverse=True)

        # Step 1 — aggregate per-frame scores into a single video score.
        video_scores: dict[str, float] = {}
        top_k = max(1, aggregation_top_k)
        for video_id, frames in by_video.items():
            top = [frame.score for frame in frames[:top_k]]
            video_scores[video_id] = (
                float(np.logaddexp.reduce(top)) if top else 0.0
            )

        # Step 2 — coarse filter: keep only the top-K videos by video score.
        ranked_videos = sorted(
            video_scores,
            key=lambda vid: video_scores[vid],
            reverse=True,
        )
        if top_videos is not None and 0 < top_videos < len(ranked_videos):
            kept = set(ranked_videos[:top_videos])
        else:
            kept = set(ranked_videos)

        # Step 3 — re-emit frames ordered by video-aware score.
        reordered: list[Candidate] = []
        for video_id in ranked_videos:
            if video_id not in kept:
                continue
            for rank, frame in enumerate(by_video[video_id][:frames_per_video]):
                boosted = video_scores[video_id] + rank * 1e-4
                reordered.append(
                    frame.model_copy(update={"score": boosted})
                )

        return reordered

    def retrieve_trake(
        self,
        event_embeddings: np.ndarray,
        top_videos: int = 100,
        prefilter_frames_per_event: int = 500,
        penalty_weight: float = 0.005,
        video_ids: set[str] | None = None,
        coarse_top_k: int = 200,
    ) -> list[Candidate]:
        """Rank videos and align one ordered frame to each event.

        Stage 1 là **video-level coarse filter**: thay vì gom mọi video có ít nhất 1
        frame khớp 1 event (có thể vài nghìn video trên dataset lớn → DP chậm/OOM),
        ta tính centroid của các event query rồi xếp hạng video theo cosine với
        video-embedding (mean-pool). Chỉ giữ top-``coarse_top_k`` video liên quan nhất
        đưa vào Stage 2 (DP alignment). ``coarse_top_k <= 0`` hoặc >= số video thì xét
        hết (backward-compatible).
        """

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

        # Stage 1: video-level coarse filter (thay vì gom mọi video khớp 1 event).
        # Centroid của các event query, normalize -> đo cosine với video-embedding.
        query_centroid = query_matrix.mean(axis=0)
        centroid_norm = float(np.linalg.norm(query_centroid))
        if centroid_norm > 0:
            query_centroid = query_centroid / centroid_norm

        if coarse_top_k > 0 and len(self._video_embeddings) > coarse_top_k:
            video_ids_list = list(self._video_embeddings)
            mat = np.stack(
                [self._video_embeddings[v] for v in video_ids_list]
            )  # [V × 512]
            scores = mat @ query_centroid  # [V]
            top_idx = np.argsort(scores)[-coarse_top_k:][::-1]
            candidate_videos = {
                video_ids_list[int(i)] for i in top_idx
            }
        else:
            # Backward-compatible: xét hết mọi video (dataset nhỏ / coarse tắt).
            candidate_videos = set(self._video_embeddings.keys())

        # Giao cắt với metadata pre-filter nếu có.
        if video_ids is not None:
            candidate_videos &= video_ids

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
        video_ids: set[str] | None = None,
    ) -> list[Candidate]:
        """Return raw candidates from vector and BM25 retrieval using RRF.

        The vector tier is pushed down with *video_ids* (Chroma ``where`` /
        FAISS mask) so off-topic videos never enter the fused pool.
        """

        vector_ids, _ = self.search_with_filter(
            text_embedding,
            top_frames,
            video_ids,
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
        video_ids: set[str] | None = None,
    ) -> list[Candidate]:
        """Hybrid retrieval followed by per-video diversification."""

        raw_candidates = self.hybrid_retrieve_raw(
            text_query=text_query,
            text_embedding=text_embedding,
            bm25_index=bm25_index,
            top_frames=top_frames,
            video_ids=video_ids,
        )

        return self.diversify_candidates(
            candidates=raw_candidates,
            max_answers=max_answers,
        )