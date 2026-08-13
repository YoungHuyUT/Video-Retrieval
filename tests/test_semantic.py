from __future__ import annotations

import numpy as np

from aic2026.evaluation.metrics import relevance
from aic2026.evaluation.semantic import SemanticMatcher
from aic2026.models import Candidate, GroundTruth, Query


class _FakeEmbedder:
    """Tiny deterministic embedder: char-n-gram hashing into a dense vector."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().replace("màu ", "").split():
            idx = sum(ord(c) for c in token) % self.dim
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec


def test_exact_match() -> None:
    matcher = SemanticMatcher()
    assert matcher.match("màu xanh", "màu xanh")


def test_normalized_match_different_case() -> None:
    matcher = SemanticMatcher()
    assert matcher.match("XANH", "xanh")


def test_color_prefix_equivalent() -> None:
    matcher = SemanticMatcher()
    assert matcher.match("xanh", "màu xanh")
    assert matcher.match("màu đỏ", "đỏ")


def test_synonym_group() -> None:
    matcher = SemanticMatcher()
    assert matcher.match("màu xanh lá", "xanh lá cây")
    assert matcher.match("đang đi bộ", "đi")


def test_wrong_color_is_false() -> None:
    matcher = SemanticMatcher()
    assert not matcher.match("màu trắng", "màu xanh")
    assert not matcher.match("trắng", "xanh")


def test_empty_is_false() -> None:
    matcher = SemanticMatcher()
    assert not matcher.match("", "xanh")
    assert not matcher.match("xanh", "")


def test_embedder_threshold() -> None:
    matcher = SemanticMatcher(embedder=_FakeEmbedder(), threshold=0.5)
    assert matcher.match("xanh", "màu xanh")
    assert not matcher.match("trắng", "xanh")


def test_qa_relevance_semantic() -> None:
    query = Query(query_id="q1", type="qa", text="mô tả", question="màu gì")
    truth = GroundTruth(video_id="v1", ranges=[(800, 900)], answer="màu xanh")

    correct = Candidate(
        video_id="v1", frame_id=850, score=1.0, answer="xanh"
    )
    wrong = Candidate(
        video_id="v1", frame_id=850, score=1.0, answer="màu trắng"
    )

    assert relevance(query, correct, truth) == 1.0
    assert relevance(query, wrong, truth) == 0.0
