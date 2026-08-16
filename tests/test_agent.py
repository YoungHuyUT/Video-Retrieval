from __future__ import annotations

import numpy as np

from aic2026.agent import RetrievalAgent
from aic2026.agent.tools import RetrievalTools
from aic2026.agent.types import AgentPlan
from aic2026.models import Candidate, Query


class FakePipeline:
    """Minimal deterministic retrieval backend used by agent unit tests.

    Mirrors only the methods ``RetrievalTools`` actually calls so the agent's
    deterministic pipeline can run end-to-end without a real index.
    """

    manifest: list | None = None
    frames_per_video: int | None = None

    def filter_terms_to_video_ids(
        self,
        terms: list[str],
        video_ids: set[str] | None = None,
    ) -> set[str] | None:
        # No metadata filter → return None so the agent considers all videos.
        return None

    def prefixes_to_video_ids(
        self,
        prefixes: list[str] | None,
        video_ids: set[str] | None = None,
    ) -> set[str] | None:
        # No prefix restriction → return None so the agent considers all videos.
        return None

    def filter_videos_by_metadata(
        self,
        terms: list[str],
        video_ids: set[str] | None = None,
    ) -> list[str]:
        return sorted(video_ids or [])

    def retrieve(
        self,
        text_embedding: np.ndarray,
        top_frames: int,
        max_answers: int,
        video_ids: set[str] | None = None,
    ) -> list[Candidate]:
        return [
            Candidate(
                video_id="L01_V001",
                frame_id=505,
                score=0.90,
                vector_id=7,
            )
        ]

    def retrieve_raw(
        self,
        text_embedding: np.ndarray,
        top_frames: int,
        video_ids: set[str] | None = None,
    ) -> list[Candidate]:
        return [
            Candidate(
                video_id="L21_V001",
                frame_id=100,
                score=0.95,
                vector_id=1,
            ),
            Candidate(
                video_id="L21_V001",
                frame_id=200,
                score=0.90,
                vector_id=2,
            ),
            Candidate(
                video_id="L21_V001",
                frame_id=300,
                score=0.85,
                vector_id=3,
            ),
        ]

    def retrieve_trake(
        self,
        event_embeddings: np.ndarray,
        top_videos: int,
        prefilter_frames_per_event: int,
        penalty_weight: float,
        video_ids: set[str] | None = None,
        coarse_top_k: int = 200,
        object_adjustment=None,
        preferred_prefixes: list[str] | None = None,
    ) -> list[Candidate]:
        assert event_embeddings.shape[0] == 3
        assert top_videos > 0
        assert prefilter_frames_per_event > 0
        assert penalty_weight >= 0

        return [
            Candidate(
                video_id="L21_V001",
                frame_id=200,
                score=0.92,
                vector_id=2,
                event_frames=[
                    100,
                    200,
                    300,
                ],
            )
        ]

    @staticmethod
    def _mask_video_ids(
        candidates: list[Candidate],
        video_ids: set[str] | None,
    ) -> list[Candidate]:
        if not video_ids:
            return candidates
        allowed = set(video_ids)
        return [c for c in candidates if c.video_id in allowed]

    # KIS / multi-query expansion paths (keep candidates intact for the test).
    def _rrf_fuse(
        self,
        ranked_lists: list[np.ndarray],
        k: int = 60,
    ) -> dict[int, float]:
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, vid in enumerate(ranked.tolist()):
                scores[vid] = scores.get(vid, 0.0) + 1.0 / (rank + 1)
        return scores

    def _candidates_from_scores(
        self,
        scores_by_manifest_idx: dict[int, float],
        limit: int | None = None,
    ) -> list[Candidate]:
        items = sorted(
            scores_by_manifest_idx.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        out = [
            Candidate(video_id="L01_V001", frame_id=i, score=s, vector_id=i)
            for i, s in items
        ]
        return out[:limit] if limit else out

    def search_with_filter(
        self,
        text_embedding: np.ndarray,
        top_frames: int,
        video_ids: set[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        ids = np.asarray([7, 1, 2], dtype=np.int64)
        scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
        return ids, scores

    def video_level_rerank(
        self,
        candidates: list[Candidate],
        top_videos: int | None = None,
        frames_per_video: int | None = None,
        aggregation_top_k: int = 3,
    ) -> list[Candidate]:
        return candidates


class BuggyLLM:
    """Simulates an LLM that throws an unexpected programming error.

    The deterministic pipeline calls the LLM only for translation and is
    expected to *swallow* that failure (falling back to the original text) so
    retrieval still runs — an LLM hiccup must never abort the whole pipeline.
    """

    def structured(
        self,
        system: str,
        user: str,
        schema: type,
    ):
        raise RuntimeError("programming bug")


def make_tools() -> RetrievalTools:
    return RetrievalTools(
        pipeline=FakePipeline(),
        encode_text=lambda _: np.asarray(
            [1.0],
            dtype=np.float32,
        ),
    )


def test_agent_can_only_return_retrieved_evidence() -> None:
    """The deterministic agent returns only candidates produced by retrieval."""
    agent = RetrievalAgent(
        make_tools(),
        llm=BuggyLLM(),
    )

    result = agent.run(
        Query(
            query_id="q-kis",
            type="kis",
            text="red speaker",
        )
    )

    assert [
        candidate.vector_id
        for candidate in result.candidates
    ] == [7]


def test_explicit_trake_events_override_planner_events() -> None:
    """TRAKE uses the query's explicit events, not any planner-supplied ones."""
    query = Query(
        query_id="q-trake",
        type="trake",
        text="A person enters, sits, and starts speaking.",
        events=[
            "The person enters the room.",
            "The person sits down.",
            "The person starts speaking.",
        ],
    )

    # The BuggyLLM would raise if the planner were still consulted; the
    # deterministic pipeline must not call it for TRAKE/KIS at all.
    agent = RetrievalAgent(
        make_tools(),
        llm=BuggyLLM(),
    )

    result = agent.run(query)

    assert result.plan.events == query.events
    assert len(result.candidates) == 1

    candidate = result.candidates[0]

    assert candidate.video_id == "L21_V001"
    assert candidate.event_frames == [
        100,
        200,
        300,
    ]
    assert len(candidate.event_frames) == len(
        query.events
    )


def test_plan_fallback_preserves_explicit_trake_events() -> None:
    """When nothing else is available, the plan keeps the query's events."""
    query = Query(
        query_id="q-fallback",
        type="trake",
        text="A three-event sequence.",
        events=[
            "Event one.",
            "Event two.",
            "Event three.",
        ],
    )

    agent = RetrievalAgent(
        make_tools(),
        llm=BuggyLLM(),
    )

    # The deterministic plan is built without an LLM; its events are the
    # query's events verbatim.
    result = agent.run(query)

    assert isinstance(result.plan, AgentPlan)
    assert result.plan.events == query.events
    assert result.plan.query_variants == [query.text]


def test_llm_failure_is_swallowed_not_aborting_pipeline() -> None:
    """An unexpected LLM/translation error must not abort retrieval.

    The deterministic pipeline is expected to catch translation failures and
    fall back to the original text, so a buggy LLM yields results rather than
    a propagated RuntimeError.
    """
    agent = RetrievalAgent(
        make_tools(),
        llm=BuggyLLM(),
    )

    query = Query(
        query_id="q-bug",
        type="kis",
        text="test query",
    )

    # Must NOT raise RuntimeError("programming bug").
    result = agent.run(query)

    assert result is not None
    assert [c.vector_id for c in result.candidates] == [7]
