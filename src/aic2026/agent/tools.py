from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable
import numpy as np

from aic2026.models import Candidate
from aic2026.retrieval import RetrievalPipeline
from aic2026.temporal import align_events


@dataclass
class RetrievalTools:
    """The agent's only route to competition evidence."""
    pipeline: RetrievalPipeline
    encode_text: Callable[[str], np.ndarray]
    visual_answerer: Callable[[str, list[Candidate]], dict[int, str]] | None = None

    def retrieve(self, query: str, limit: int) -> list[Candidate]:
        return self.pipeline.retrieve(self.encode_text(query), top_frames=limit, max_answers=limit)

    def candidates_for_video(self, candidates: list[Candidate], video_id: str) -> list[Candidate]:
        return [candidate for candidate in candidates if candidate.video_id == video_id]

    def temporal_alignment(self, candidates: list[Candidate], event_count: int) -> list[int]:
        return align_events(candidates, event_count)

    def answer_question(self, question: str, candidates: list[Candidate]) -> dict[int, str]:
        if self.visual_answerer is None:
            return {}
        return self.visual_answerer(question, candidates)

    @staticmethod
    def evidence(candidates: list[Candidate], maximum: int = 40) -> list[dict]:
        return [candidate.model_dump(include={"vector_id", "video_id", "frame_id", "score", "keyframe_path"}) for candidate in candidates[:maximum]]

    @staticmethod
    def by_vector_id(candidates: list[Candidate]) -> dict[int, Candidate]:
        return {candidate.vector_id: candidate for candidate in candidates if candidate.vector_id is not None}
