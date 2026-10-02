from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Callable

import numpy as np

from aic2026.models import Candidate, FrameRecord
from aic2026.temporal import align_events_dp

from .index import VectorIndex
from .video_metadata import VideoMetadataStore

if TYPE_CHECKING:
    from .bm25_index import BM25Index
    from .vectordb import ChromaVectorStore


# Standard Reciprocal Rank Fusion constant.
_RRF_K = 60

# Pool tối đa để RRF quét xuyên toàn bộ corpus khi có full data (BM25 sống). BM25 vốn
# đã quét toàn bộ video qua text; mở rộng thêm vector pool giúp những video có frame khớp
# nằm ở rank sâu (cosine thấp) vẫn lọt vào top sau RRF. Hằng số này cũng dùng trong
# tools.py (import từ đây) để tính adaptive RRF k.
FULL_DATA_CORPUS_TOP_FRAMES = 2000

# Minimum and maximum k for adaptive RRF — small candidate pools (e.g. pool=500
# in the KIS fallback path) compress the rank contribution 1/(k+rank+1) so much
# that the dynamic range collapses and ties become common.  We scale k down
# toward 1 for small pools so rank-position still carries discriminative weight.
_RRF_K_MIN = 1
_RRF_K_MAX = 60



def dedupe_temporal(
    candidates: list[Candidate],
    records: dict[int, FrameRecord] | None = None,
    window_seconds: float = 1.5,
) -> list[Candidate]:
    """Group same-video frames within a temporal window, keep highest score.

    Improvement.md Task 6: Temporal dedupe before video-level rerank.
    Near-duplicate frames (same video, within ``window_seconds``) are grouped
    and only the highest-scoring frame per group is kept.  This prevents
    temporal clusters of similar frames from inflating a video's log-sum-exp
    score via repeated near-duplicates.
    """
    if not candidates:
        return []

    by_video: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_video.setdefault(c.video_id, []).append(c)

    deduped: list[Candidate] = []
    for video_id, frames in by_video.items():
        if len(frames) <= 1:
            deduped.extend(frames)
            continue

        def _sort_key(c: Candidate) -> float:
            if records and c.vector_id is not None:
                rec = records.get(c.vector_id)
                if rec is not None:
                    ts = getattr(rec, "timestamp_seconds", None)
                    if ts is None:
                        ts = getattr(rec, "timestamp", None)
                    if ts is not None:
                        return float(ts)
            if c.timestamp is not None:
                return float(c.timestamp)
            return float(c.frame_id)

        frames.sort(key=_sort_key)

        groups: list[list[Candidate]] = [[frames[0]]]
        for frame in frames[1:]:
            last_group = groups[-1]
            last_ts = _sort_key(last_group[-1])
            cur_ts = _sort_key(frame)
            if (cur_ts - last_ts) <= window_seconds:
                last_group.append(frame)
            else:
                groups.append([frame])

        for group in groups:
            best = max(group, key=lambda c: c.score)
            deduped.append(best)

    return deduped

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
        video_metadata: VideoMetadataStore | None = None,
    ) -> None:
        if len(index.vectors) != len(manifest):
            raise ValueError("Feature count must equal manifest record count")

        if frames_per_video <= 0:
            raise ValueError("frames_per_video must be greater than zero")

        self.index = index
        self.manifest = manifest
        self.frames_per_video = frames_per_video
        # Per-video metadata (title/description/keywords) lives outside the
        # manifest now to avoid 200x duplication. When provided, metadata
        # filters and BM25 tokenization consult it instead of scanning every
        # FrameRecord.
        self.video_metadata = video_metadata or VideoMetadataStore.empty()
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

        Cost: O(V) over :attr:`video_metadata` (873 docs) plus a single O(N) pass
        to merge per-frame ``object_labels`` for the matched candidate set only.
        Previously this scanned all 200k FrameRecords — the duplication of video
        metadata per frame made that both slow and BM25-IDF-incorrect.
        """
        if not terms:
            return sorted(video_ids or {record.video_id for record in self.manifest})

        import unicodedata

        def _fold(value: str) -> str:
            value = unicodedata.normalize("NFC", value).casefold()
            # đ/Đ (U+0111/U+0110) không có decomposition Unicode nên NFD+filter Mn
            # không bỏ được dấu gạch ngang — phải transliterate riêng, nếu không
            # metadata chứa "cửa hàng" không khớp term "cua hang" khi lọc.
            value = value.translate(str.maketrans({"đ": "d"}))
            return "".join(c for c in unicodedata.normalize("NFD", value) if unicodedata.category(c) != "Mn")

        folded_terms = [_fold(term) for term in terms]
        allowed = video_ids or {record.video_id for record in self.manifest}
        matched: set[str] = set()

        # Stage 1: video-level text (title/description/keywords). O(V) over the
        # small VideoMetadataStore. Even if the store is empty (legacy manifest
        # without video_metadata.jsonl), we skip directly to stage 2.
        store = self.video_metadata
        if len(store) > 0:
            for vm in store.all():
                if vm.video_id not in allowed:
                    continue
                haystack = " ".join(
                    [
                        vm.title or "",
                        vm.description or "",
                        *vm.metadata_keywords,
                    ]
                )
                folded_haystack = _fold(haystack)
                if any(term and term in folded_haystack for term in folded_terms):
                    matched.add(vm.video_id)

        # Stage 2: per-frame object labels (objects live on frames, not videos).
        # We scan the manifest once, but only emit videos NOT already matched by
        # stage 1 — when stage 1 already covers them we save the per-frame scan
        # by short-circuiting. Most queries either match no videos (then we
        # fall back to "return all") or match several; the inner work is small.
        if not matched:
            for record in self.manifest:
                if record.video_id not in allowed:
                    continue
                haystack = " ".join(record.object_labels or [])
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
        k: int | None = None,
        weights: list[float] | None = None,
    ) -> dict[int, float]:
        """Fuse ranked manifest indices using Reciprocal Rank Fusion.

        Optional *weights* parameter scales the RRF contribution of each input list,
        allowing primary queries (weight 1.0) to take priority over partial/fine-detail
        sub-queries (weights 0.5 - 0.8).
        """

        if k is None:
            max_len = max((len(ids) for ids in ranked_lists), default=0)
            # Linear interpolation: 0 items → 1, full corpus → 60.
            # This keeps the reciprocal-rank curve steep for short lists while
            # matching standard RRF behavior on large ones.
            k = int(round(_RRF_K_MIN + (_RRF_K_MAX - _RRF_K_MIN) * (max_len / FULL_DATA_CORPUS_TOP_FRAMES)))
            k = max(_RRF_K_MIN, min(_RRF_K_MAX, k))

        fused_scores: dict[int, float] = {}

        for list_idx, ids in enumerate(ranked_lists):
            w = weights[list_idx] if weights and list_idx < len(weights) else 1.0
            for rank, idx in enumerate(ids):
                manifest_idx = int(idx)
                contribution = w / (k + rank + 1)
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
                    timestamp=record.timestamp,
                    frame_unit=record.frame_unit,
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

    def prefixes_to_video_ids(
        self,
        prefixes: list[str] | None,
        video_ids: set[str] | None = None,
    ) -> set[str] | None:
        """Resolve ``video_id`` prefixes to an allowed video set (or None = no filter).

        Used to restrict a retrieval task to a specific dataset split (e.g. QA
        focuses on ``L25`` online-course videos, TRAKE on ``L26``). Returns
        ``None`` when *prefixes* is empty so callers can skip masking; when a
        base ``video_ids`` set is supplied the two are intersected so a prefix
        can further narrow a metadata-derived pool.
        """

        if not prefixes:
            return None
        manifest = getattr(self, "manifest", None)
        if not manifest:
            return None
        wanted = set(prefixes)
        matched = {
            record.video_id
            for record in manifest
            if any(
                record.video_id.startswith(prefix) for prefix in wanted
            )
        }
        if video_ids is not None:
            matched &= set(video_ids)
        return matched

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
        frame_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vector search, optionally pushed down to the DB via ``video_ids``.

        Chroma applies the filter inside the ANN scan; FAISS/NumPy masks the
        returned rows. Either way the metadata pre-filter runs at the index
        tier instead of post-hoc Python masking.
        """
        if frame_indices is not None and hasattr(self.index, "search_filtered_indices"):
            return self.index.search_filtered_indices(text_embedding, top_frames, frame_indices)
        if video_ids:
            return self.index.search_filtered(text_embedding, top_frames, video_ids)
        return self.index.search(text_embedding, top_frames)

    def search_many_with_filter(
        self,
        embeddings: list[np.ndarray],
        top_frames: int,
        video_ids: set[str] | None = None,
        frame_indices: np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Batch vector search for multiple embeddings.

        Uses ``index.search_many_filtered`` for a single matrix multiply
        (2-4x faster than per-embedding loops). Returns a list of
        (ids, scores) tuples, one per embedding.
        """
        if not embeddings:
            return []
        matrix = np.asarray(embeddings, dtype=np.float32)
        if frame_indices is not None and hasattr(self.index, "search_many_filtered_indices"):
            return self.index.search_many_filtered_indices(matrix, top_frames, frame_indices)
        if hasattr(self.index, "search_many_filtered"):
            return self.index.search_many_filtered(matrix, top_frames, video_ids)
        # Fallback: per-embedding search (Chroma or legacy).
        return [
            self.search_with_filter(emb, top_frames, video_ids)
            for emb in embeddings
        ]

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
        keep_frame_scores: bool = False,
        frame_scores: dict[int, float] | None = None,
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

        ``keep_frame_scores`` (used in the *soft* adaptive mode for 2-3 videos)
        changes step 3: videos are still ordered by ``video_score``, but each
        frame **keeps its own retrieval score** instead of being overwritten by
        the video blob. On a small/competing set this preserves the fine-grained
        RRF order inside the correct video instead of flattening it — the bug that
        made the single-video sample rank everything as a tie.
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

        # Aggregate a small number of strongest frames.  This keeps an isolated
        # high-scoring false frame from outranking a video with consistent visual
        # evidence, while still limiting the final result per video.
        video_scores: dict[str, float] = {}
        top_k = max(1, aggregation_top_k)
        for video_id, frames in by_video.items():
            frames.sort(key=lambda c: c.score, reverse=True)
            top = [frame.score for frame in frames[:top_k]]
            video_scores[video_id] = float(np.logaddexp.reduce(top)) if top else 0.0

        ranked_videos = sorted(video_scores, key=video_scores.__getitem__, reverse=True)
        if top_videos is not None and 0 < top_videos < len(ranked_videos):
            ranked_videos = ranked_videos[:top_videos]

        reranked: list[Candidate] = []
        for video_id in ranked_videos:
            for rank, frame in enumerate(by_video[video_id][:frames_per_video]):
                if keep_frame_scores:
                    score = frame_scores.get(frame.vector_id, frame.score) if frame_scores else frame.score
                else:
                    score = video_scores[video_id] - rank * 1e-4
                reranked.append(frame.model_copy(update={"score": score}))

        return sorted(reranked, key=lambda c: c.score, reverse=True)

    def retrieve_trake(
        self,
        event_embeddings: np.ndarray,
        top_videos: int = 100,
        prefilter_frames_per_event: int = 500,
        penalty_weight: float = 0.005,
        video_ids: set[str] | None = None,
        coarse_top_k: int = 200,
        object_adjustment: Callable[[int, int], float] | None = None,
        # Precomputed object-evidence adjustment matrices from
        # ``build_object_adjustment_matrices`` (shape (event_count, F) per video,
        # aligned with this video's manifest order). Passing these avoids the
        # 40k+ per-(event,frame) Python closure calls that previously made TRAKE
        # take ~145s per query — the matrix is added once with NumPy. The legacy
        # ``object_adjustment`` callable is still supported as a fallback.
        object_adjustment_matrices: dict[str, np.ndarray] | None = None,
        # Soft *preference* (NOT a restriction) for certain video_id prefixes
        # (e.g. ["L26"]). Preferred videos get a small bounded score nudge so they
        # rank above equally-similar non-preferred videos, but a non-preferred
        # video is never dropped. BTC event queries are generic and match many
        # splits, so a hard restrict would zero recall — this only biases ranking.
        preferred_prefixes: list[str] | None = None,
        use_temporal_decay: bool = False,
        temporal_decay_alpha: float = 0.01,
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
            # On the memory-mapped load path the index matrix is NOT normalized
            # in place (to save RAM). Normalize the sliced frame block here so
            # the matrix product below is a true cosine similarity regardless of
            # how the index was loaded. Cheap: only candidate-video frames.
            frame_norms = np.maximum(
                np.linalg.norm(frame_vectors, axis=1, keepdims=True),
                1e-12,
            )
            frame_vectors = frame_vectors / frame_norms

            similarity_matrix = (
                query_matrix
                @ frame_vectors.T
            )
            if object_adjustment_matrices is not None and video_id in object_adjustment_matrices:
                # Vectorized path: add the precomputed (event_count, F) matrix in
                # one NumPy op instead of calling a Python closure 40k+ times.
                adjustment = object_adjustment_matrices[video_id]
                if adjustment.shape == similarity_matrix.shape:
                    similarity_matrix = similarity_matrix + adjustment
            elif object_adjustment is not None:
                for event_index in range(event_count):
                    similarity_matrix[event_index] += np.asarray(
                        [
                            object_adjustment(event_index, int(manifest_index))
                            for manifest_index in manifest_indices
                        ],
                        dtype=np.float32,
                    )

            try:
                alignment_score, aligned_positions = (
                    align_events_dp(
                        similarity_matrix=similarity_matrix,
                        penalty_weight=penalty_weight,
                        use_decay=use_temporal_decay,
                        decay_alpha=temporal_decay_alpha,
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
                    timestamp=representative_record.timestamp,
                    frame_unit=representative_record.frame_unit,
                    event_frames=event_frames,
                )
            )

        # Soft preference nudge (NOT a restriction): preferred-prefix videos get
        # a small bounded bonus so they rise above equally-similar non-preferred
        # videos without ever suppressing a non-preferred match. Alignment scores
        # sit ~0.2-0.4, so a 0.02 bonus is enough to break ties without reversing
        # a genuinely better non-preferred video.
        if preferred_prefixes:
            wanted = set(preferred_prefixes)
            for candidate in ranked_candidates:
                if any(
                    candidate.video_id.startswith(p) for p in wanted
                ):
                    candidate.score += 0.02

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

    def retrieve_raw_per_video(
        self,
        text_embedding: np.ndarray,
        frames_per_video: int = 10,
        video_ids: set[str] | None = None,
    ) -> list[Candidate]:
        """Vector retrieval returning the top frames PER VIDEO (not one global
        top-k).

        This guarantees every video contributes its strongest frames, so retrieval
        scans the WHOLE corpus at the video level — essential when the dataset has
        many videos and a correct video's best frame may sit beyond a global top-k
        cutoff (which would otherwise be silently dropped).

        Accelerator: instead of a Python loop + matmul over every video
        (O(V·F) Python, chậm trên dataset lớn), we issue ONE ANN query over the
        ENTIRE corpus (FAISS IndexFlatIP / Chroma) with k = num_videos ×
        frames_per_video, then group by video and keep the top-N per video in
        NumPy. On a 177k-frame index this drops to ~1 ANN scan (< 50 ms on FAISS)
        + vectorized per-video top-k.
        """
        if frames_per_video <= 0:
            raise ValueError("frames_per_video must be greater than zero")
        if not self.manifest:
            return []

        q = np.asarray(text_embedding, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)

        video_ids_set = set(video_ids) if video_ids else None
        # Số video cần thăm dự.
        if video_ids_set is not None:
            target_videos = [v for v in self._video_to_manifest_indices if v in video_ids_set]
        else:
            target_videos = list(self._video_to_manifest_indices)
        n_videos = len(target_videos)
        # Lấy đủ frame để mỗi video đều có frames_per_video kết quả. Dùng margin
        # an toàn vì ANN trả về global top-k (có thể tận dụng nhiều ở 1 vài video).
        k_pool = min(n_videos * frames_per_video, len(self.manifest))
        if k_pool <= 0:
            return []

        ids, scores = self.search_with_filter(q, k_pool, video_ids_set)
        if len(ids) == 0:
            return []

        # Group returned manifest indices by video_id, keep top-N per video.
        # ids/scores align 1-1; sort each video's frame by score, take top-N.
        by_video: dict[str, list[tuple[float, int]]] = {}
        manifest = self.manifest
        for manifest_idx, score in zip(ids, scores):
            rec = manifest[int(manifest_idx)]
            by_video.setdefault(rec.video_id, []).append((float(score), int(manifest_idx)))

        candidates: list[Candidate] = []
        for video_id in target_videos:
            frames = by_video.get(video_id)
            if not frames:
                continue
            # Top frames_per_video cho video này (đã sắp xếp score giảm dần).
            frames.sort(key=lambda t: t[0], reverse=True)
            for score, manifest_idx in frames[:frames_per_video]:
                rec = manifest[manifest_idx]
                candidates.append(
                    Candidate(
                        video_id=video_id,
                        frame_id=rec.frame_id,
                        score=score,
                        vector_id=rec.vector_id,
                        keyframe_path=rec.keyframe_path,
                        timestamp=rec.timestamp,
                        frame_unit=rec.frame_unit,
                    )
                )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def hybrid_retrieve_raw_full(
        self,
        text_query: str,
        text_embedding: np.ndarray,
        bm25_index: BM25Index,
        frames_per_video: int = 10,
        video_ids: set[str] | None = None,
        limit: int = 5000,
    ) -> list[Candidate]:
        """Full-corpus hybrid retrieval: per-video vector top-k (covers every
        video) fused with BM25 lexical via RRF. Use when the dataset has many
        videos and we must not miss any video."""
        vec_candidates = self.retrieve_raw_per_video(
            text_embedding, frames_per_video=frames_per_video, video_ids=video_ids
        )
        vec_ids = np.asarray(
            [c.vector_id for c in vec_candidates], dtype=np.int64
        )
        bm25_ids, _ = bm25_index.search(
            text_query, min(len(self.manifest), FULL_DATA_CORPUS_TOP_FRAMES)
        )
        fused = self._rrf_fuse([vec_ids, bm25_ids])
        return self._candidates_from_scores(fused, limit=limit)
