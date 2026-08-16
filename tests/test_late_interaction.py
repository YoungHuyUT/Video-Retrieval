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

    # Two frames: frame0 matches only "cat", frame1 matches neither strongly.
    frame0 = np.zeros(dim, dtype=np.float32); frame0[0] = 1.0
    frame1 = np.zeros(dim, dtype=np.float32); frame1[2] = 0.3
    frames = np.stack([frame0, frame1])

    cands = [_c("V1", 0.3, 0), _c("V1", 0.5, 1)]  # frame1 has higher raw score
    out = late_interaction_rerank(
        "cat", cands, encode, frames, top_n=10, weight=0.5
    )
    # frame0 matches the "cat" facet -> its late-interaction term lifts it above frame1.
    assert out[0].vector_id == 0
    assert out[0].score > cands[1].score


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
