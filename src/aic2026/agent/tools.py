from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from aic2026.models import Candidate, FrameRecord

logger = logging.getLogger(__name__)
from aic2026.reranking import (
    late_interaction_rerank,
    contrastive_clip_colour_rerank,
    rerank_with_colour_evidence,
    object_evidence_adjustment,
    rerank_with_metadata,
    rerank_with_object_evidence,
)
from aic2026.retrieval import RetrievalPipeline
from aic2026.temporal import align_events, refine_trake_candidates

if TYPE_CHECKING:
    from aic2026.retrieval.bm25_index import BM25Index


_TASKS_REQUIRING_DENSE_VIDEO_EVIDENCE = frozenset(
    {
        "qa",
        "trake",
    }
)

@dataclass
class RetrievalTools:
    """The agent's controlled interface to competition evidence."""

    pipeline: RetrievalPipeline
    encode_text: Callable[[str], np.ndarray]
    visual_answerer: (
        Callable[[str, list[Candidate]], dict[int, str]] | None
    ) = None
    encode_images: Callable[[list[object]], np.ndarray] | None = None
    bm25_index: BM25Index | None = None
    # Trọng số metadata bonus (Direction B): nhỏ, cộng trực tiếp lên RRF gốc
    # (KHÔNG normalize). RRF base ~0.01-0.03; weight 0.01 → bonus tối đa +0.01,
    # đủ nudging mà không đảo ngược thứ tự RRF.
    rerank_weight: float = 0.01
    # Trọng số late-interaction (ColBERT-style MaxSim): encode_text ~200 lần trong
    # retrieve(), RẤT nặng cho thi tốc độ. Mặc định 0.0 (TẮT) — chỉ bật nếu cần
    # tăng recall query dài nhiều từ (rare cho BTC). Có thể bật qua runtime config.
    late_interaction_weight: float = 0.0
    # HSV colour evidence reads only the top image candidates and never calls a
    # VLM.  It is deliberately bounded so CLIP remains the primary signal.
    colour_rerank_weight: float = 0.04
    contrastive_colour_rerank_weight: float = 0.06
    video_filter_terms: list[str] | None = None
    # Số video tối đa giữ lại sau bước video-level coarse filter (KIS). Dataset
    # lớn (> số video này) thì chỉ top-K video theo video_score mới được giữ, còn
    # lại loại bỏ trước khi vào MMR — tránh video nhiễu (1 frame outlier) lọt top.
    # 0 / <= 0 hoặc >= số video = xét hết (backward-compatible, dataset nhỏ).
    coarse_top_k: int = 200
    # Coarse keyframes select the video; only its short event windows are then
    # decoded from the source video for finer TRAKE timestamps.
    trake_dense_refine: bool = True
    trake_refine_top_videos: int = 20
    trake_refine_sample_fps: float = 3.0
    trake_refine_window_seconds: float = 2.0
    trake_video_root: Path = Path("data/raw/Videos")

    @property
    def has_lexical_objects(self) -> bool:
        """Whether a frame-level objects/OCR BM25 index is available."""
        return self.bm25_index is not None and not self.bm25_index.is_empty

    def retrieve(
        self,
        query: str | list[str],
        limit: int,
        task_type: str | None = None,
    ) -> list[Candidate]:
        """Retrieve task-appropriate candidates.

        KIS receives diversified results. QA and TRAKE receive raw candidates
        so that multiple relevant frames from the same video are preserved.
        Every non-TRAKE path is then reranked by lexical keyword overlap with
        object/metadata text (a small deterministic score bonus), so frames
        whose labels/titles mention the query terms float above pure vector
        matches.

        ``query`` may be a single string or a list of query variants
        (multi-query expansion). When multiple variants are given, each is
        encoded and the resulting ranked lists are fused with Reciprocal Rank
        Fusion before reranking, improving recall.
        """

        if limit <= 0:
            return []

        queries = [query] if isinstance(query, str) else list(query)
        embeddings = [self.encode_text(q) for q in queries]

        requires_dense_evidence = (
            task_type in _TASKS_REQUIRING_DENSE_VIDEO_EVIDENCE
        )

        # Metadata pre-filter: narrow the candidate video set before the
        # (relatively expensive) embedding/BM25 retrieval so frames from
        # off-topic videos never enter the pool.
        allowed_video_ids = self.pipeline.filter_terms_to_video_ids(
            self.video_filter_terms or []
        )
        if allowed_video_ids is not None:
            logger.info(
                "metadata filter: %d/%d videos matched (%s)",
                len(allowed_video_ids),
                len({record.video_id for record in self.pipeline.manifest}),
                ", ".join(sorted(allowed_video_ids))[:200],
            )

        if self.bm25_index is not None and self.bm25_index.is_empty:
            logger.warning(
                "BM25 skipped: manifest has no Objects/Metadata text. "
                "Run `aic2026 prepare` to load them for lexical retrieval."
            )

        # Bound the retrieval work to the requested candidate pool.  A global
        # per-video scan is disproportionately expensive for a static CLIP index.
        pool = limit

        if self.bm25_index is not None and not self.bm25_index.is_empty:
            logger.debug("BM25 lexical index active; fusing with vector retrieval.")
            if requires_dense_evidence:
                candidates = self.pipeline.hybrid_retrieve_raw(
                    text_query=queries[0],
                    text_embedding=embeddings[0],
                    bm25_index=self.bm25_index,
                    top_frames=pool,
                    video_ids=allowed_video_ids,
                )
            else:
                candidates = self.pipeline.hybrid_retrieve_raw(
                    text_query=queries[0],
                    text_embedding=embeddings[0],
                    bm25_index=self.bm25_index,
                    top_frames=pool,
                    video_ids=allowed_video_ids,
                )
        elif requires_dense_evidence:
            candidates = self.pipeline.retrieve_raw(
                text_embedding=embeddings[0],
                top_frames=pool,
                video_ids=allowed_video_ids,
            )
        elif len(embeddings) > 1:
            # Multi-query expansion: RRF-fuse per-variant vector rankings.
            ranked_lists = []
            for emb in embeddings:
                ids, _ = self.pipeline.search_with_filter(
                    emb, pool, allowed_video_ids
                )
                ranked_lists.append(np.asarray(ids, dtype=np.int64))
            fused = self.pipeline._rrf_fuse(ranked_lists)
            candidates = self.pipeline._candidates_from_scores(fused, limit=limit)
        else:
            candidates = self.pipeline.retrieve_raw(
                text_embedding=embeddings[0],
                top_frames=pool,
                video_ids=allowed_video_ids,
            )

        # Safety net: drop any candidate whose video slipped past the DB filter
        # (e.g. matched only via BM25 lexical text, not the vector tier).
        if allowed_video_ids is not None:
            candidates = self.pipeline._mask_video_ids(candidates, allowed_video_ids)

        # Lexical metadata bonus: nudge frames whose object/title/description
        # text literally mentions the query terms (deterministic, not learned).
        candidates = rerank_with_metadata(
            query=queries[0],
            candidates=candidates,
            records=self._record_lookup(),
            weight=self.rerank_weight,
        )
        # Object detector evidence is a stronger, signed signal than generic
        # lexical overlap: an exact/synonym match is promoted; a known object
        # mismatch is softly penalized.  No advanced UI field is required.
        candidates = rerank_with_object_evidence(
            query=queries[0],
            candidates=candidates,
            records=self._record_lookup(),
        )

        candidates = rerank_with_colour_evidence(
            query=queries[0],
            candidates=candidates,
            records=self._record_lookup(),
            weight=self.colour_rerank_weight,
        )
        candidates = contrastive_clip_colour_rerank(
            query=queries[0],
            candidates=candidates,
            records=self._record_lookup(),
            encode_text=self.encode_text,
            encode_images=self.encode_images,
            weight=self.contrastive_colour_rerank_weight,
        )

        # KIS is ranked per frame.  Do not aggregate/cap by video here: a
        # candidate rises or falls only on its own retrieval evidence.

        # Late-interaction (ColBERT-style MaxSim): bắt khớp cục bộ theo từng facet
        # của query — một frame khớp BẤT KỲ facet nào (vd "water bottle") đều được
        # nâng, điều vector CLIP pooled đơn lẻ không làm được. Chỉ chạy khi index
        # thực sự chứa ĐỦ toàn bộ corpus: Chroma nếu chỉ build 1 phần (thiếu mấy
        # chục nghìn vector) thì vector_id sẽ lệch → skip an toàn, fallback về
        # RRF + metadata bonus, không crash.
        if self.encode_text is not None and candidates:
            index = self.pipeline.index
            manifest_size = len(self.pipeline.manifest)
            index_has_all = True
            if hasattr(index, "collection_count"):  # ChromaVectorStore
                try:
                    if index.collection_count != manifest_size:
                        index_has_all = False
                        logger.warning(
                            "late-interaction skipped: Chroma index has %d vectors "
                            "but manifest has %d — index is partial, vector_id would "
                            "mismatch. Rebuild with `aic2026 build-chroma-index`.",
                            index.collection_count,
                            manifest_size,
                        )
                except Exception as exc:  # noqa: BLE001
                    index_has_all = False
                    logger.warning("late-interaction skipped (index count error): %s", exc)
            elif getattr(index, "vectors", None) is None:
                index_has_all = False

            if index_has_all:
                try:
                    frame_vectors = np.stack(
                        [index.vectors[c.vector_id] for c in candidates]
                    )
                    candidates = late_interaction_rerank(
                        query=queries[0],
                        candidates=candidates,
                        encode_text=self.encode_text,
                        frame_vectors=frame_vectors,
                        top_n=200,
                        weight=self.late_interaction_weight,
                    )
                    # late_interaction chỉ sửa score tại chỗ, cần sort lại.
                    candidates = sorted(
                        candidates, key=lambda c: c.score, reverse=True
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("late-interaction rerank skipped: %s", exc)

        return candidates

    def _record_lookup(self) -> dict[int, FrameRecord]:
        """Map ``vector_id`` → manifest record, for reranking."""

        manifest = getattr(self.pipeline, "manifest", None)
        if not manifest:
            return {}
        return {
            record.vector_id: record
            for record in manifest
        }

    def retrieve_trake(
        self,
        events: list[str],
        limit: int,
        prefilter_frames_per_event: int = 500,
        penalty_weight: float = 0.005,
        coarse_top_k: int = 200,
    ) -> list[Candidate]:
        """Run deterministic event-wise TRAKE retrieval and alignment.

        ``coarse_top_k`` giới hạn số video đưa vào DP alignment (xem
        ``RetrievalPipeline.retrieve_trake``): chỉ top-K video theo coarse
        video-level similarity mới được xét, tránh DP chạy trên vài nghìn video.
        Object labels do not change this cap; it remains the guardrail that
        keeps dynamic-programming alignment bounded on the full corpus.
        """

        cleaned_events = [
            event.strip()
            for event in events
            if event.strip()
        ]

        if not cleaned_events:
            raise ValueError(
                "TRAKE requires at least one non-empty event"
            )

        if limit <= 0:
            return []

        event_embeddings = np.stack(
            [
                np.asarray(
                    self.encode_text(event),
                    dtype=np.float32,
                ).reshape(-1)
                for event in cleaned_events
            ],
            axis=0,
        )

        # Metadata pre-filter (TRAKE): restrict candidate videos before the
        # per-event vector scan inside retrieve_trake (pushed down to the DB).
        allowed_video_ids = self.pipeline.filter_terms_to_video_ids(
            self.video_filter_terms or []
        )

        # Unit-test/lightweight pipelines may not expose manifest records.  The
        # production pipeline always does; in that case apply evidence inside
        # the event×frame matrix, before temporal DP selects an alignment.
        object_adjustment = None
        manifest = getattr(self.pipeline, "manifest", None)
        if manifest:
            object_adjustment = lambda event_index, manifest_index: object_evidence_adjustment(
                cleaned_events[event_index], manifest[manifest_index], weight=0.08
            ) or 0.0

        candidates = self.pipeline.retrieve_trake(
            event_embeddings=event_embeddings,
            top_videos=limit,
            prefilter_frames_per_event=(
                prefilter_frames_per_event
            ),
            penalty_weight=penalty_weight,
            video_ids=allowed_video_ids,
            coarse_top_k=coarse_top_k,
            object_adjustment=object_adjustment,
        )

        # Safety net for any candidate the DB filter could not exclude.
        if allowed_video_ids is not None:
            candidates = self.pipeline._mask_video_ids(candidates, allowed_video_ids)

        if self.trake_dense_refine:
            candidates = refine_trake_candidates(
                candidates,
                event_embeddings,
                self.encode_images,
                video_root=self.trake_video_root,
                top_videos=self.trake_refine_top_videos,
                sample_fps=self.trake_refine_sample_fps,
                window_seconds=self.trake_refine_window_seconds,
                penalty_weight=penalty_weight,
            )

        return candidates

    def candidates_for_video(
        self,
        candidates: list[Candidate],
        video_id: str,
    ) -> list[Candidate]:
        return [
            candidate
            for candidate in candidates
            if candidate.video_id == video_id
        ]

    def temporal_alignment(
        self,
        candidates: list[Candidate],
        event_count: int,
    ) -> list[int]:
        return align_events(candidates, event_count)

    def answer_question(
        self,
        question: str,
        candidates: list[Candidate],
    ) -> dict[int, str]:
        if self.visual_answerer is None:
            return {}

        return self.visual_answerer(
            question,
            candidates,
        )

    @staticmethod
    def evidence(
        candidates: list[Candidate],
        maximum: int = 40,
    ) -> list[dict]:
        fields = {
            "vector_id",
            "video_id",
            "frame_id",
            "score",
            "keyframe_path",
        }

        return [
            candidate.model_dump(include=fields)
            for candidate in candidates[:maximum]
        ]

    @staticmethod
    def by_vector_id(
        candidates: list[Candidate],
    ) -> dict[int, Candidate]:
        return {
            candidate.vector_id: candidate
            for candidate in candidates
            if candidate.vector_id is not None
        }

    @staticmethod
    def group_by_video(
        candidates: list[Candidate],
    ) -> dict[str, list[Candidate]]:
        grouped: dict[str, list[Candidate]] = defaultdict(list)

        for candidate in candidates:
            grouped[candidate.video_id].append(candidate)

        return grouped
