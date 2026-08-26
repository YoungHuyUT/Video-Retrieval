from __future__ import annotations

import json
import numpy as np
import pytest

from aic2026.agent.local_llm import _strip_to_json
from aic2026.agent.runtime import RetrievalAgent
from aic2026.agent.tools import RetrievalTools
from aic2026.agent.translator import (
    formulate_visual_facets,
    offline_decompose_facets,
    offline_translate,
)
from aic2026.agent.types import VisualQueryFacets
from aic2026.models import Candidate, FrameRecord, Query
from aic2026.retrieval import RetrievalPipeline, VectorIndex


class _FakeJSONLLM:
    """Mock LLM returning noisy structured JSON output."""

    def __init__(self, reply: str) -> None:
        self._reply = reply

    def structured(self, system: str, user: str, schema: type):
        parsed = _strip_to_json(self._reply)
        return schema.model_validate_json(parsed)


# =========================================================================
# Tầng 1: LLM Visual Query Formulator (Typos, Visual Phrasing, Context)
# =========================================================================

def test_visual_formulator_llm_structured_output() -> None:
    fake_reply = (
        "<think>Formulating visual search query and decomposing facets...</think>\n"
        "```json\n"
        "{\n"
        '  "q_core": "a person holding a large rough gemstone close to face to observe in an open-pit mine",\n'
        '  "q_entity": "a man in dark blue suit, holding a large rough gemstone",\n'
        '  "q_action": "holding rough gemstone close to face to observe carefully",\n'
        '  "q_scene": "panoramic top-down view of an open-pit mining quarry in daylight"\n'
        "}\n"
        "```"
    )
    llm = _FakeJSONLLM(fake_reply)
    facets = formulate_visual_facets(
        "nột người cầm đá quý thô soi gần mặt ở mỏ lộ thiên",
        llm=llm,
        use_llm=True,
    )

    assert isinstance(facets, VisualQueryFacets)
    assert "rough gemstone" in facets.q_core.lower()
    assert "dark blue suit" in facets.q_entity.lower()
    assert "holding" in facets.q_action.lower()
    assert "open-pit" in facets.q_scene.lower()
    assert len(facets.to_list()) == 4


def test_visual_formulator_offline_typo_and_gemstone_translation() -> None:
    # Test typo correction in offline dictionary:
    # "nột người" -> "a person", "đĩa da ngăn" -> "multi-compartment plate", "đá quý thô" -> "rough gemstone"
    t1 = offline_translate("nột người")
    assert "person" in t1.lower()

    t2 = offline_translate("đĩa da ngăn")
    assert "multi-compartment plate" in t2.lower()

    t3 = offline_translate("đá quý thô ở mỏ lộ thiên")
    assert "rough gemstone" in t3.lower()
    assert "open-pit mine quarry" in t3.lower()


# =========================================================================
# Tầng 2: Generalized Query Expansion (GQE) & Multi-Facet Decomposition
# =========================================================================

def test_offline_decompose_facets_extracts_four_facets() -> None:
    facets = offline_decompose_facets("Người đàn ông mặc áo vest đen đang cầm đá quý thô quan sát trong trường quay")

    assert isinstance(facets, VisualQueryFacets)
    assert facets.q_core != ""
    assert "vest" in facets.q_entity.lower() or "man" in facets.q_entity.lower() or "suit" in facets.q_entity.lower()
    assert "holding" in facets.q_action.lower() or "observing" in facets.q_action.lower() or "action" in facets.q_action.lower()
    assert "studio" in facets.q_scene.lower() or "scene" in facets.q_scene.lower()

    # Canonical order list
    facet_list = facets.to_list()
    assert len(facet_list) >= 1
    assert facets.q_core in facet_list


def test_visual_query_facets_to_list_deduplication() -> None:
    facets = VisualQueryFacets(
        q_core="a person speaking",
        q_entity="a person speaking",
        q_action="speaking",
        q_scene="in a studio",
    )
    facet_list = facets.to_list()
    # "a person speaking" should appear only once
    assert len(facet_list) == 3
    assert facet_list[0] == "a person speaking"
    assert facet_list[1] == "speaking"
    assert facet_list[2] == "in a studio"


# =========================================================================
# Tầng 3: Score-Reflected Reciprocal Rank Fusion (SRRF)
# =========================================================================

def build_test_pipeline() -> RetrievalPipeline:
    # 5 frames with 4-dimensional feature vectors (L2-normalized)
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],  # Frame 0: matches facet 1 perfectly
            [0.0, 1.0, 0.0, 0.0],  # Frame 1: matches facet 2 perfectly
            [0.5, 0.5, 0.5, 0.5],  # Frame 2: matches all 4 facets moderately well (cosine = 0.5 each)
            [0.0, 0.0, 1.0, 0.0],  # Frame 3: matches facet 3 perfectly
            [0.0, 0.0, 0.0, 1.0],  # Frame 4: matches facet 4 perfectly
        ],
        dtype=np.float32,
    )
    manifest = [
        FrameRecord(vector_id=0, video_id="V1", frame_id=10, keyframe_path="V1/10.jpg"),
        FrameRecord(vector_id=1, video_id="V1", frame_id=20, keyframe_path="V1/20.jpg"),
        FrameRecord(vector_id=2, video_id="V2", frame_id=30, keyframe_path="V2/30.jpg"),
        FrameRecord(vector_id=3, video_id="V3", frame_id=40, keyframe_path="V3/40.jpg"),
        FrameRecord(vector_id=4, video_id="V4", frame_id=50, keyframe_path="V4/50.jpg"),
    ]
    return RetrievalPipeline(index=VectorIndex(vectors), manifest=manifest)


def test_srrf_formula_mathematical_correctness() -> None:
    pipeline = build_test_pipeline()

    # Ranked results from 2 facets:
    # Facet 1: id 0 (sim 0.9, rank 0), id 2 (sim 0.6, rank 1)
    # Facet 2: id 2 (sim 0.8, rank 0), id 0 (sim 0.2, rank 1)
    res1 = (np.array([0, 2]), np.array([0.9, 0.6]))
    res2 = (np.array([2, 0]), np.array([0.8, 0.2]))

    fused = pipeline._srrf_fuse([res1, res2], k=60)

    # Calculation:
    # id 0: 0.9 / (60 + 0 + 1) + 0.2 / (60 + 1 + 1) = 0.9/61 + 0.2/62 ≈ 0.014754 + 0.003226 = 0.017980
    # id 2: 0.6 / (60 + 1 + 1) + 0.8 / (60 + 0 + 1) = 0.6/62 + 0.8/61 ≈ 0.009677 + 0.013115 = 0.022792
    assert fused[2] > fused[0]
    expected_0 = 0.9 / 61.0 + 0.2 / 62.0
    expected_2 = 0.6 / 62.0 + 0.8 / 61.0
    assert pytest.approx(fused[0], rel=1e-4) == expected_0
    assert pytest.approx(fused[2], rel=1e-4) == expected_2


def test_srrf_noise_resistance_vs_standard_rrf() -> None:
    pipeline = build_test_pipeline()

    # Scenario: Frame A has strong match in core facet (sim 0.95, rank 0) but rank 1 in noisy facet (sim 0.10)
    # Frame B has a fluke weak match (sim 0.12, rank 0) in noisy facet, and rank 1 (sim 0.11) in core facet.
    # In standard RRF:
    # Frame A: 1/61 + 1/62 = 0.03253
    # Frame B: 1/61 + 1/62 = 0.03253 -> TIE!
    # In SRRF:
    # Frame A: 0.95/61 + 0.10/62 = 0.01557 + 0.00161 = 0.01718
    # Frame B: 0.11/62 + 0.12/61 = 0.00177 + 0.00197 = 0.00374
    # Frame A clearly and correctly dominates in SRRF!
    core_res = (np.array([0, 1]), np.array([0.95, 0.11]))
    noisy_res = (np.array([1, 0]), np.array([0.12, 0.10]))

    srrf_fused = pipeline._srrf_fuse([core_res, noisy_res], k=60)
    assert srrf_fused[0] > 4.0 * srrf_fused[1]  # Frame 0 is ~4.5x higher score


def test_multi_facet_retrieve_raw_end_to_end() -> None:
    pipeline = build_test_pipeline()

    # 4 facet query vectors
    q_facets = [
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    ]

    candidates = pipeline.multi_facet_retrieve_raw(q_facets, top_frames=5)
    assert len(candidates) == 5

    # Frame 2 has cosine 0.5 with ALL 4 facets, so:
    # Frame 2 receives 4 * (0.5 / (60 + rank + 1))
    # Whereas frames 0, 1, 3, 4 only match 1 facet each (1.0) and 0 for others.
    # Total score for Frame 2: ~ 4 * (0.5 / 62) ≈ 0.03225
    # Total score for Frame 0: 1.0 / 61 ≈ 0.01639
    # Frame 2 which satisfies all multi-facet constraints ranks #1!
    assert candidates[0].vector_id == 2


# =========================================================================
# Integration Test with RetrievalAgent
# =========================================================================

def test_retrieval_agent_runs_with_gqe_srrf() -> None:
    pipeline = build_test_pipeline()
    tools = RetrievalTools(
        pipeline=pipeline,
        encode_text=lambda q: np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    )
    agent = RetrievalAgent(tools=tools, answer_limit=5)
    agent.translate = True

    # Run agent on a query
    query = Query(
        query_id="q-test",
        type="kis",
        text="người đàn ông mặc áo vest cầm đá quý ở mỏ lộ thiên",
    )
    result = agent.run(query)

    assert result is not None
    assert len(result.candidates) > 0
    assert result.plan.facets is not None
    assert isinstance(result.plan.facets, VisualQueryFacets)

    # Check traces contain gqe_facets step
    trace_steps = [t.step for t in result.trace]
    assert "gqe_facets" in trace_steps
    assert "retrieve" in trace_steps

    facet_trace = next(t for t in result.trace if t.step == "gqe_facets")
    parsed_facets = json.loads(facet_trace.detail)
    assert "q_core" in parsed_facets
    assert "q_entity" in parsed_facets
    assert "q_action" in parsed_facets
    assert "q_scene" in parsed_facets
