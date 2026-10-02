"""Integration tests for Cascaded KIS Multimodal Pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from aic2026.agent.tools import RetrievalTools
from aic2026.models import Candidate, FrameRecord
from aic2026.query.expansion import expand_text
from aic2026.query.parser import parse_query
from aic2026.query.routing import compute_modality_weights
from aic2026.retrieval.pipeline import RetrievalPipeline


@pytest.fixture
def mock_pipeline() -> RetrievalPipeline:
    manifest = [
        FrameRecord(video_id="v1", frame_id=10, vector_id=0, keyframe_path="/tmp/v1_10.jpg"),
        FrameRecord(video_id="v1", frame_id=20, vector_id=1, keyframe_path="/tmp/v1_20.jpg"),
        FrameRecord(video_id="v2", frame_id=15, vector_id=2, keyframe_path="/tmp/v2_15.jpg"),
    ]
    vectors = np.random.randn(3, 512).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    class MockIndex:
        def __init__(self, vecs: np.ndarray):
            self.vectors = vecs

        def search(self, emb: np.ndarray, top_k: int):
            return np.array([0, 1, 2]), np.array([0.9, 0.8, 0.7])

        def search_filtered(self, emb: np.ndarray, top_k: int, video_ids: set[str]):
            return np.array([0, 1, 2]), np.array([0.9, 0.8, 0.7])

    pipeline = RetrievalPipeline(index=MockIndex(vectors), manifest=manifest)
    return pipeline


def test_query_decomposition_and_spatial() -> None:
    query = "A man wearing glasses enters room, then grabs a red cup on left of table"
    plan = parse_query(query)

    assert len(plan.events) == 2
    assert "glasses" in plan.entities or "room" in plan.entities
    assert "left of" in plan.spatial_relations


def test_modality_routing_weights() -> None:
    query = "A man wearing a red shirt on left of room"
    plan = parse_query(query)
    weights = compute_modality_weights(plan)

    assert weights.visual > 0.0
    assert weights.semantic > 0.0
    # Weights re-normalized
    total = weights.semantic + weights.visual + weights.object + weights.asr + weights.ocr + weights.metadata
    assert pytest.approx(total, 0.01) == 1.0


def test_controlled_expansion_trust_bounds() -> None:
    text = "picks up a red cup"
    variants = expand_text(text, max_variants=4)

    assert len(variants) >= 1
    assert variants[0] == text  # Original receives highest trust/priority


def test_kis_pipeline_returns_single_best_frame(mock_pipeline: RetrievalPipeline) -> None:
    tools = RetrievalTools(
        pipeline=mock_pipeline,
        encode_text=lambda q: np.random.randn(512).astype(np.float32),
        use_gemini_rerank=True,
        gemini_model="gemini-2.5-flash-lite",
        gemini_top_k=10,
    )

    # Call retrieve for KIS
    results = tools.retrieve(
        query="A man wearing glasses enters a room and picks up a red cup",
        limit=5,
        task_type="kis",
    )

    assert isinstance(results, list)
    assert len(results) > 0
    # Confirm result objects are candidates
    assert isinstance(results[0], Candidate)
