"""Unit tests for normalize_score abstraction (spec §15) and ASR temporal
localization (spec §ASR TEMPORAL LOCALIZATION)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aic2026.reranking.lexical import normalize_score
from aic2026.reranking.modality_scores import asr_temporal_scores


def test_normalize_score_minmax():
    vals = [0.0, 0.5, 1.0]
    out = normalize_score(vals, "minmax")
    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.5)


def test_normalize_score_maxdiv():
    vals = [0.2, 0.4]
    out = normalize_score(vals, "maxdiv")
    assert out[1] == pytest.approx(1.0)
    assert out[0] == pytest.approx(0.5)


def test_normalize_score_sigmoid_centered():
    # sigmoid of 0 → 0.5; symmetric inputs → centered around 0.5.
    out = normalize_score([-1.0, 0.0, 1.0], "sigmoid", temperature=1.0)
    assert out[1] == pytest.approx(0.5, abs=1e-6)
    assert out[0] < 0.5 < out[2]


def test_normalize_score_percentile_clips():
    vals = [0.0, 0.1, 0.2, 0.9, 1.0]
    out = normalize_score(vals, "percentile", percentile=90.0)
    assert all(0.0 <= v <= 1.0 for v in out)
    assert out[-1] == pytest.approx(1.0)


def test_normalize_score_empty():
    assert normalize_score([], "minmax") == []


def _seg(text, start, end, frame=None):
    return SimpleNamespace(text=text, start=start, end=end, frame=frame)


def _transcript(segments):
    return SimpleNamespace(segments=segments, text="")


def test_asr_temporal_scores_localizes():
    # Candidate frame at t=5 should match the segment centered at 5 (temporal win).
    segs = [_seg("remember the red hat", 4.0, 6.0, frame=50)]
    transcripts = {"V": _transcript(segs)}
    cands = [
        # inside window (t=5) vs outside (t=120)
        SimpleNamespace(video_id="V", frame_id=50, vector_id=50, timestamp=5.0),
        SimpleNamespace(video_id="V", frame_id=999, vector_id=999, timestamp=120.0),
    ]
    scores = asr_temporal_scores("remember the red hat", cands, transcripts, alpha=0.01)
    assert 50 in scores
    # The frame inside the temporal window (t=5) must score higher than the far
    # frame (t=120): the far frame only earns weak *decayed* credit (ratio·decay·0.3,
    # the recall-preserving weak signal), never more than the localized match.
    assert scores[50] > scores.get(999, 0.0)
    # Inside-window score is the full ratio (all 3 query terms match).
    assert scores[50] == pytest.approx(1.0, abs=1e-6)


def test_asr_temporal_scores_returns_empty_without_sidecar():
    assert asr_temporal_scores("hello", [], None) == {}


def test_asr_temporal_scores_no_folded_match():
    segs = [_seg("completely different words", 4.0, 6.0, frame=50)]
    transcripts = {"V": _transcript(segs)}
    cands = [SimpleNamespace(video_id="V", frame_id=50, vector_id=50, timestamp=5.0)]
    out = asr_temporal_scores("unrelated query terms here", cands, transcripts)
    assert out == {}
