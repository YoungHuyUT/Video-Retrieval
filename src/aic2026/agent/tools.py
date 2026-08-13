from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import logging
import numpy as np

from aic2026.models import Candidate, FrameRecord

logger = logging.getLogger(__name__)
from aic2026.reranking import rerank_with_metadata
from aic2026.retrieval import RetrievalPipeline
from aic2026.temporal import align_events

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
    bm25_index: BM25Index | None = None
    rerank_weight: float = 0.05
    video_filter_terms: list[str] | None = None
    # Số video tối đa giữ lại sau bước video-level coarse filter (KIS). Dataset
    # lớn (> số video này) thì chỉ top-K video theo video_score mới được giữ, còn
    # lại loại bỏ trước khi vào MMR — tránh video nhiễu (1 frame outlier) lọt top.
    # 0 / <= 0 hoặc >= số video = xét hết (backward-compatible, dataset nhỏ).
    coarse_top_k: int = 200

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
            logger.debug(
                "BM25 skipped: manifest has no Objects/Metadata text. "
                "Run `aic2026 prepare` to load them for lexical retrieval."
            )

        if self.bm25_index is not None and not self.bm25_index.is_empty:
            logger.debug("BM25 lexical index active; fusing with vector retrieval.")
            # Hybrid path: RRF fuses vector + BM25. Push the video filter down
            # to the vector tier where possible; the Python mask is the safety
            # net for any candidate the DB filter could not exclude.
            if requires_dense_evidence:
                candidates = self.pipeline.hybrid_retrieve_raw(
                    text_query=queries[0],
                    text_embedding=embeddings[0],
                    bm25_index=self.bm25_index,
                    top_frames=limit,
                    video_ids=allowed_video_ids,
                )
            else:
                candidates = self.pipeline.hybrid_retrieve(
                    text_query=queries[0],
                    text_embedding=embeddings[0],
                    bm25_index=self.bm25_index,
                    top_frames=limit,
                    max_answers=limit,
                    video_ids=allowed_video_ids,
                )
        elif requires_dense_evidence:
            candidates = self.pipeline.retrieve_raw(
                text_embedding=embeddings[0],
                top_frames=limit,
                video_ids=allowed_video_ids,
            )
        elif len(embeddings) > 1:
            # Multi-query expansion: RRF-fuse per-variant vector rankings.
            ranked_lists = []
            for emb in embeddings:
                ids, _ = self.pipeline.search_with_filter(
                    emb, limit, allowed_video_ids
                )
                ranked_lists.append(np.asarray(ids, dtype=np.int64))
            fused = self.pipeline._rrf_fuse(ranked_lists)
            candidates = self.pipeline._candidates_from_scores(fused, limit=limit)
        else:
            candidates = self.pipeline.retrieve(
                text_embedding=embeddings[0],
                top_frames=limit,
                max_answers=limit,
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

        # KIS-only: re-rank at the video level. KIS ground truth is video-level,
        # so a purely frame-level ranking (above) is brittle — a single outlier
        # frame can lift the wrong video while the correct video has merely
        # "many good frames". This aggregates frame scores → video_score →
        # coarse filter top-K videos → re-emit frames video-aware. QA needs
        # multiple strong frames per video kept intact (no coarse cap), and
        # TRAKE has its own DP-based coarse filter, so this step is KIS-only.
        if task_type == "kis":
            candidates = self.pipeline.video_level_rerank(
                candidates=candidates,
                top_videos=self.coarse_top_k,
                frames_per_video=self.pipeline.frames_per_video,
            )

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

        candidates = self.pipeline.retrieve_trake(
            event_embeddings=event_embeddings,
            top_videos=limit,
            prefilter_frames_per_event=(
                prefilter_frames_per_event
            ),
            penalty_weight=penalty_weight,
            video_ids=allowed_video_ids,
            coarse_top_k=coarse_top_k,
        )

        # Safety net for any candidate the DB filter could not exclude.
        if allowed_video_ids is not None:
            candidates = self.pipeline._mask_video_ids(candidates, allowed_video_ids)

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