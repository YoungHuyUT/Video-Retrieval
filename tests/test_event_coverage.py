"""Unit tests for Event Coverage scoring (spec §3, §4)."""

from __future__ import annotations

import numpy as np
import pytest

from aic2026.models import Candidate
from aic2026.reranking.event_coverage import (
    EventCoverageResult,
    event_aware_rerank,
    event_coverage_scores,
)


def _emb(vec: list[float]) -> np.ndarray:
    a = np.asarray(vec, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_event_coverage_prefers_multi_event_frame():
    """Frame B (explains many events) should beat Frame A (one strong match)."""
    # Two events in 3-dim space.
    e1 = _emb([1.0, 0.0, 0.0])
    e2 = _emb([0.0, 1.0, 0.0])
    e3 = _emb([0.0, 0.0, 1.0])
    events = np.stack([e1, e2, e3])

    # Frame A: strong on E1, weak on E2/E3.  Frame B: balanced on all three.
    fa = _emb([1.0, 0.05, 0.05])
    fb = _emb([0.6, 0.6, 0.55])
    frames = np.stack([fa, fb])
    vids = [10, 20]

    res = event_coverage_scores(events, frames, vids)
    assert isinstance(res, EventCoverageResult)
    # min-max normalized: best frame → 1.0, worst → 0.0
    cov_b = res.coverage[20]
    cov_a = res.coverage[10]
    assert cov_b >= cov_a
    assert cov_b == pytest.approx(1.0, abs=1e-5)
    assert cov_a == pytest.approx(0.0, abs=1e-5)


def test_event_coverage_empty_on_bad_inputs():
    assert event_coverage_scores(None, None, []) == EventCoverageResult({}, {}, {}, 0)
    assert event_coverage_scores(np.zeros((2, 4)), np.zeros((3, 3)), [1, 2, 3]).n_events == 0


def test_event_weights_zero_remove_event():
    e1 = _emb([1.0, 0.0])
    e2 = _emb([0.0, 1.0])
    events = np.stack([e1, e2])
    fa = _emb([1.0, 0.0])  # perfect E1
    fb = _emb([0.0, 1.0])  # perfect E2
    frames = np.stack([fa, fb])
    vids = [1, 2]
    # Weight E2 = 0 → only E1 matters → fa (perfect E1) should win.
    res = event_coverage_scores(events, frames, vids, event_weights=[1.0, 0.0])
    assert res.coverage[1] >= res.coverage[2]


def test_event_aware_rerank_blends_and_keeps_order():
    e1 = _emb([1.0, 0.0, 0.0])
    e2 = _emb([0.0, 1.0, 0.0])
    e3 = _emb([0.0, 0.0, 1.0])
    events = np.stack([e1, e2, e3])

    # index_vectors matrix: row 0 = fa (E1 only), row 1 = fb (all three).
    fa = _emb([1.0, 0.05, 0.05])
    fb = _emb([0.6, 0.6, 0.55])
    index_vectors = np.stack([fa, fb])
    cands = [
        Candidate(video_id="V", frame_id=0, score=0.9, vector_id=0),
        Candidate(video_id="V", frame_id=1, score=0.1, vector_id=1),
    ]
    out = event_aware_rerank(cands, events, index_vectors, blend=0.7)
    # Event coverage should pull fb (vector_id 1) to the top despite low incoming.
    assert out[0].vector_id == 1


def test_event_aware_rerank_graceful_on_failure():
    cands = [Candidate(video_id="V", frame_id=0, score=0.9, vector_id=0)]
    # event_embeddings=None → returns original list unchanged.
    out = event_aware_rerank(cands, None, np.zeros((1, 4)))
    assert out == cands
