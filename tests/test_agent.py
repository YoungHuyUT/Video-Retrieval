from __future__ import annotations

import numpy as np
import pytest

from aic2026.agent import RetrievalAgent
from aic2026.agent.local_llm import LLMInvocationError
from aic2026.agent.tools import RetrievalTools
from aic2026.agent.types import AgentDecision, AgentPlan
from aic2026.models import Candidate, Query


class FakePipeline:
    """Small deterministic retrieval backend used by agent unit tests."""

    def retrieve(
        self,
        text_embedding: np.ndarray,
        top_frames: int,
        max_answers: int,
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


class EvidenceSelectingLLM:
    def structured(
        self,
        system: str,
        user: str,
        schema: type,
    ):
        if schema is AgentPlan:
            return AgentPlan(
                query_variants=["red speaker"],
                rationale="Visual rewrite.",
            )

        return AgentDecision(
            action="finish",
            selected_vector_ids=[7, 999999],
            rationale="Use retrieved evidence only.",
        )


class CollapsingTrakeLLM:
    """Simulates a planner that incorrectly compresses three events into one."""

    def structured(
        self,
        system: str,
        user: str,
        schema: type,
    ):
        if schema is AgentPlan:
            return AgentPlan(
                query_variants=["person performing a sequence"],
                events=["compressed sequence"],
                rationale="Incorrectly compressed the event sequence.",
            )

        return AgentDecision(
            action="finish",
            selected_vector_ids=[1, 2, 3],
            rationale="Selected one video's evidence.",
        )


class FailingLLM:
    def structured(
        self,
        system: str,
        user: str,
        schema: type,
    ):
        raise LLMInvocationError("Ollama is unavailable.")


class BuggyLLM:
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
    agent = RetrievalAgent(
        EvidenceSelectingLLM(),
        make_tools(),
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

    agent = RetrievalAgent(
        CollapsingTrakeLLM(),
        make_tools(),
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
        FailingLLM(),
        make_tools(),
    )

    plan = agent._plan(query)

    assert plan.query_variants == [query.text]
    assert plan.events == query.events


def test_unexpected_programming_error_is_not_swallowed() -> None:
    agent = RetrievalAgent(
        BuggyLLM(),
        make_tools(),
    )

    query = Query(
        query_id="q-bug",
        type="kis",
        text="test query",
    )

    with pytest.raises(
        RuntimeError,
        match="programming bug",
    ):
        agent._plan(query)