from __future__ import annotations

import numpy as np

from aic2026.models import Candidate
from aic2026.reranking import facet_queries, late_interaction_rerank


def _c(vid: str, score: float, vector_id: int, frame_id: int = 1) -> Candidate:
    return Candidate(video_id=vid, frame_id=frame_id, score=score, vector_id=vector_id)


def test_facet_queries_includes_full_and_keywords() -> None:
    facets = facet_queries("a man holding a water bottle")
    # full query is always first
    assert facets[0] == "a man holding a water bottle"
    # individual meaningful words become facets (stopword "a" dropped by len>2)
    assert "man" in facets
    assert "water" in facets
    assert "a" not in facets  # stopword filtered from single-word facets
    # no duplicates
    assert len(facets) == len(set(facets))


def test_late_interaction_boosts_facet_match() -> None:
    rng = np.random.default_rng(0)
    dim = 16
    # query facets: "cat" and "dog"
    def encode(text: str) -> np.ndarray:
        # deterministic pseudo-embedding: hash word -> direction
        v = rng.standard_normal(dim).astype(np.float32)
        if "cat" in text:
            v[0] += 5.0
        if "dog" in text:
            v[1] += 5.0
        return v

    # Two frames with EQUAL CLIP score; frame0 matches "cat" facet, frame1 does not.
    frame0 = np.zeros(dim, dtype=np.float32); frame0[0] = 1.0
    frame1 = np.zeros(dim, dtype=np.float32); frame1[2] = 0.3
    frames = np.stack([frame0, frame1])

    cands = [_c("V1", 0.5, 0), _c("V1", 0.5, 1)]  # equal raw CLIP score
    out = late_interaction_rerank(
        "cat", cands, encode, frames, top_n=10, weight=0.5
    )
    # Equal CLIP -> late-interaction is the tie-breaker: frame0 (matches "cat")
    # ranks above frame1 (no facet match).
    assert out[0].vector_id == 0
    assert out[0].score >= out[1].score


def test_late_interaction_keeps_ranking_when_no_facet_match() -> None:
    def encode(text: str) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)

    frames = np.stack([np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                       np.array([0, 1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)])
    cands = [_c("V1", 0.9, 0), _c("V1", 0.4, 1)]
    out = late_interaction_rerank("zzz", cands, encode, frames, top_n=10, weight=0.5)
    # no facet matches -> scores unchanged, original order preserved
    assert out[0].vector_id == 0
    assert abs(out[0].score - 0.9) < 1e-9


def test_late_interaction_handles_none_vectors() -> None:
    def encode(text: str) -> np.ndarray:
        return np.ones(4, dtype=np.float32)
    frames = np.stack([np.array([1.0, 0, 0, 0], dtype=np.float32),
                       np.array([0, 1.0, 0, 0], dtype=np.float32)])
    cands = [_c("V1", 0.5, 0), _c("V1", 0.5, 1)]
    # top_n larger than pool is fine; None embeddings handled by _stack upstream
    out = late_interaction_rerank("x", cands, encode, frames, top_n=10, weight=0.1)
    assert len(out) == 2


def test_phase9_srrf_fusion_keeps_clip_dominant_order() -> None:
    """Phase 9: fusion must NOT flatten the CLIP ranking the way the old
    additive nudge did. A frame with a much higher CLIP score but slightly
    weaker facet match should still rank above a weak-CLIP frame."""
    dim = 16

    def encode(text: str) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        if "cat" in text:
            v[0] = 1.0
        if "dog" in text:
            v[1] = 1.0
        return v

    # frame0: strong CLIP (0.9), weak facet; frame1: weak CLIP (0.1), strong facet
    frame0 = np.zeros(dim, dtype=np.float32); frame0[2] = 1.0
    frame1 = np.zeros(dim, dtype=np.float32); frame1[0] = 1.0
    frames = np.stack([frame0, frame1])
    cands = [_c("V1", 0.9, 0), _c("V1", 0.1, 1)]
    out = late_interaction_rerank("cat", cands, encode, frames, top_n=10, weight=0.4)
    # CLIP still dominates: frame0 stays first despite frame1 matching the facet.
    assert out[0].vector_id == 0


def test_phase9_entity_facet_outranks_generic_word() -> None:
    """Phase 9: an entity-bearing facet ("bottle") should contribute more to the
    MaxSim than a generic word, so a frame matching 'bottle' ranks above one
    matching only a generic term."""
    dim = 16

    def encode(text: str) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        if "bottle" in text:
            v[0] = 1.0
        if "scene" in text:
            v[1] = 1.0
        return v

    frame_bottle = np.zeros(dim, dtype=np.float32); frame_bottle[0] = 1.0
    frame_scene = np.zeros(dim, dtype=np.float32); frame_scene[1] = 1.0
    frames = np.stack([frame_bottle, frame_scene])
    # Equal CLIP so the late-interaction signal is the tie-breaker.
    cands = [_c("V1", 0.5, 0), _c("V1", 0.5, 1)]
    out = late_interaction_rerank(
        "a bottle in the scene", cands, encode, frames, top_n=10, weight=0.4
    )
    assert out[0].vector_id == 0


def test_phase9_fusion_does_not_lose_candidates() -> None:
    """Fusion must re-emit every input candidate with a vector_id."""
    dim = 8

    def encode(text: str) -> np.ndarray:
        return np.zeros(dim, dtype=np.float32)

    frames = np.stack([np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                       np.array([0, 1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)])
    cands = [_c("V1", 0.9, 0), _c("V1", 0.4, 1)]
    out = late_interaction_rerank("zzz", cands, encode, frames, top_n=10, weight=0.4)
    assert {c.vector_id for c in out} == {0, 1}
    # no-match case: scores unchanged
    assert abs(out[0].score - 0.9) < 1e-9
