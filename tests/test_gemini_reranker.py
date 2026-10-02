"""Unit tests for Gemini Multimodal Reranker, Resilience, Gate, and Circuit Breaker."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aic2026.models import Candidate, FrameRecord
from aic2026.query.plan import (
    Constraint,
    ConstraintKind,
    CountOp,
    Event,
    QueryPlan,
)
from aic2026.reranking.gemini_reranker import (
    CircuitBreaker,
    CircuitBreakerState,
    GeminiEvaluationResult,
    GeminiFlashLiteProvider,
    GeminiGate,
    GeminiReranker,
    MultimodalRerankerProvider,
)


@pytest.fixture
def sample_candidates() -> list[Candidate]:
    return [
        Candidate(video_id="v1", frame_id=10, score=0.80, vector_id=1, keyframe_path="/tmp/v1_10.jpg"),
        Candidate(video_id="v2", frame_id=20, score=0.78, vector_id=2, keyframe_path="/tmp/v2_20.jpg"),
        Candidate(video_id="v3", frame_id=30, score=0.60, vector_id=3, keyframe_path="/tmp/v3_30.jpg"),
    ]


@pytest.fixture
def sample_records() -> dict[int, FrameRecord]:
    return {
        1: FrameRecord(video_id="v1", frame_id=10, vector_id=1, keyframe_path="/tmp/v1_10.jpg"),
        2: FrameRecord(video_id="v2", frame_id=20, vector_id=2, keyframe_path="/tmp/v2_20.jpg"),
        3: FrameRecord(video_id="v3", frame_id=30, vector_id=3, keyframe_path="/tmp/v3_30.jpg"),
    }


def test_gemini_gate_trigger_conditions(sample_candidates: list[Candidate]) -> None:
    gate = GeminiGate(ambiguity_margin=0.05, enabled=True)

    # Simple plan without count/spatial/attributes/events
    simple_plan = QueryPlan(raw_text="a man walks")
    assert not gate.should_call(simple_plan, [
        Candidate(video_id="v1", frame_id=1, score=0.90),
        Candidate(video_id="v2", frame_id=2, score=0.70),
    ])

    # 1. Ambiguity margin trigger (top1 - top2 < 0.05)
    assert gate.should_call(simple_plan, sample_candidates)  # 0.80 vs 0.78 diff 0.02

    # 2. Count constraint trigger
    count_plan = QueryPlan(
        raw_text="exactly 3 people",
        constraints=[Constraint(kind=ConstraintKind.COUNT, subject="people", operator=CountOp.EQ, value=3)],
    )
    assert gate.should_call(count_plan, [Candidate(video_id="v1", frame_id=1, score=0.90)])

    # 3. Spatial relations trigger
    spatial_plan = QueryPlan(raw_text="cup on left of table", spatial_relations=["left of"])
    assert gate.should_call(spatial_plan, [Candidate(video_id="v1", frame_id=1, score=0.90)])

    # 4. Multiple visual attributes trigger
    attr_plan = QueryPlan(raw_text="red blue shirt", attributes=["red", "blue"])
    assert gate.should_call(attr_plan, [Candidate(video_id="v1", frame_id=1, score=0.90)])

    # 5. Multi-event chain trigger
    event_plan = QueryPlan(
        raw_text="first enters then sits",
        events=[Event(index=0, description="enters"), Event(index=1, description="sits")],
    )
    assert gate.should_call(event_plan, [Candidate(video_id="v1", frame_id=1, score=0.90)])


def test_circuit_breaker_transitions() -> None:
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=0.1)

    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.allow_request()

    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitBreakerState.CLOSED

    cb.record_failure()  # Trips threshold 3
    assert cb.state == CircuitBreakerState.OPEN
    assert not cb.allow_request()

    # Wait for cooldown
    import time
    time.sleep(0.15)
    assert cb.allow_request()  # Transitions to HALF_OPEN
    assert cb.state == CircuitBreakerState.HALF_OPEN

    cb.record_success()
    assert cb.state == CircuitBreakerState.CLOSED


def test_gemini_reranker_missing_api_key(
    sample_candidates: list[Candidate],
    sample_records: dict[int, FrameRecord],
) -> None:
    provider = GeminiFlashLiteProvider(api_key=None)
    reranker = GeminiReranker(enabled=True, provider=provider, ambiguity_margin=0.10)
    plan = QueryPlan(raw_text="test query")

    # API key missing -> provider returns empty dict -> candidates unchanged
    res = reranker.rerank(plan, sample_candidates, sample_records)
    assert len(res) == len(sample_candidates)
    assert res[0].score == sample_candidates[0].score


def test_gemini_reranker_successful_fusion(
    sample_candidates: list[Candidate],
    sample_records: dict[int, FrameRecord],
) -> None:
    mock_provider = MagicMock(spec=MultimodalRerankerProvider)
    mock_provider.rerank_candidates.return_value = {
        "v1": GeminiEvaluationResult(candidate_id="v1", overall_match=0.50),
        "v2": GeminiEvaluationResult(candidate_id="v2", overall_match=1.00),
    }

    reranker = GeminiReranker(
        enabled=True,
        local_weight=0.60,
        gemini_weight=0.40,
        ambiguity_margin=0.10,
        provider=mock_provider,
    )
    plan = QueryPlan(raw_text="test query", attributes=["red", "blue"])

    res = reranker.rerank(plan, sample_candidates, sample_records)

    # v1 score: 0.60 * 0.80 + 0.40 * 0.50 = 0.48 + 0.20 = 0.68
    # v2 score: 0.60 * 0.78 + 0.40 * 1.00 = 0.468 + 0.40 = 0.868
    # v2 should now rank FIRST
    assert res[0].video_id == "v2"
    assert pytest.approx(res[0].score, 0.001) == 0.868
    assert res[1].video_id == "v1"
    assert pytest.approx(res[1].score, 0.001) == 0.680


def test_gemini_reranker_http_error_resilience(
    sample_candidates: list[Candidate],
    sample_records: dict[int, FrameRecord],
) -> None:
    mock_provider = MagicMock(spec=MultimodalRerankerProvider)
    mock_provider.rerank_candidates.side_effect = Exception("HTTP 429 Too Many Requests")

    reranker = GeminiReranker(
        enabled=True,
        ambiguity_margin=0.10,
        provider=mock_provider,
    )
    plan = QueryPlan(raw_text="test query", attributes=["red", "blue"])

    # Exception caught, original candidates returned safely
    res = reranker.rerank(plan, sample_candidates, sample_records)
    assert len(res) == len(sample_candidates)
    assert res[0].video_id == "v1"
    assert res[0].score == 0.80


def test_gemini_reranker_malformed_json(
    sample_candidates: list[Candidate],
) -> None:
    provider = GeminiFlashLiteProvider(api_key="fake_key")
    evals = provider._parse_and_validate_response("THIS IS NOT JSON", sample_candidates)
    assert evals == {}


def test_gemini_reranker_caching(
    sample_candidates: list[Candidate],
    sample_records: dict[int, FrameRecord],
) -> None:
    mock_provider = MagicMock(spec=MultimodalRerankerProvider)
    mock_provider.rerank_candidates.return_value = {
        "v1": GeminiEvaluationResult(candidate_id="v1", overall_match=0.90),
    }

    reranker = GeminiReranker(
        enabled=True,
        ambiguity_margin=0.10,
        provider=mock_provider,
        cache_enabled=True,
    )
    plan = QueryPlan(raw_text="test query", attributes=["red", "blue"])

    res1 = reranker.rerank(plan, sample_candidates, sample_records)
    res2 = reranker.rerank(plan, sample_candidates, sample_records)

    # Provider called only ONCE due to cache hit
    assert mock_provider.rerank_candidates.call_count == 1
    assert res1[0].score == res2[0].score
