from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from aic2026.models import Candidate
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

    def retrieve(
        self,
        query: str,
        limit: int,
        task_type: str | None = None,
    ) -> list[Candidate]:
        """Retrieve task-appropriate candidates.

        KIS receives diversified results. QA and TRAKE receive raw candidates
        so that multiple relevant frames from the same video are preserved.
        """

        if limit <= 0:
            return []

        embedding = self.encode_text(query)

        requires_dense_evidence = (
            task_type in _TASKS_REQUIRING_DENSE_VIDEO_EVIDENCE
        )

        if self.bm25_index is not None:
            if requires_dense_evidence:
                return self.pipeline.hybrid_retrieve_raw(
                    text_query=query,
                    text_embedding=embedding,
                    bm25_index=self.bm25_index,
                    top_frames=limit,
                )

            return self.pipeline.hybrid_retrieve(
                text_query=query,
                text_embedding=embedding,
                bm25_index=self.bm25_index,
                top_frames=limit,
                max_answers=limit,
            )

        if requires_dense_evidence:
            return self.pipeline.retrieve_raw(
                text_embedding=embedding,
                top_frames=limit,
            )

        return self.pipeline.retrieve(
            text_embedding=embedding,
            top_frames=limit,
            max_answers=limit,
        )

    def retrieve_trake(
        self,
        events: list[str],
        limit: int,
        prefilter_frames_per_event: int = 500,
        penalty_weight: float = 0.005,
    ) -> list[Candidate]:
        """Run deterministic event-wise TRAKE retrieval and alignment."""

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

        return self.pipeline.retrieve_trake(
            event_embeddings=event_embeddings,
            top_videos=limit,
            prefilter_frames_per_event=(
                prefilter_frames_per_event
            ),
            penalty_weight=penalty_weight,
        )

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