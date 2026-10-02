from __future__ import annotations

import math

from aic2026.reranking.lexical import adaptive_modality_fusion, minmax_normalize


def test_minmax_normalizes_to_unit_range() -> None:
    # Eq.1: (s - min) / (max - min + eps) maps scores onto [0, 1], preserving
    # the value ordering of the original scores (higher input -> higher output).
    scores = [0.2, 0.35, 0.25, 0.3]
    norm = minmax_normalize(scores)
    assert math.isclose(float(min(norm)), 0.0)   # lowest score -> 0
    assert math.isclose(float(max(norm)), 1.0)   # highest score -> 1
    # value ordering preserved: each output tracks its input rank
    pairs = sorted(zip(scores, norm))
    for (_, a), (_, b) in zip(pairs, pairs[1:]):
        assert float(a) <= float(b)


def test_adaptive_fusion_weighted_sum_eq2() -> None:
    # Two modalities with disjoint indices; verify S(f) = w_m * s_norm_m(f).
    modality_ids = {
        "semantic": [1, 2],   # scores [0.2, 0.4] -> norm [0, 1]
        "ocr": [2, 3],        # scores [0.0, 1.0] -> norm [0, 1]
    }
    modality_scores = {
        "semantic": [0.2, 0.4],
        "ocr": [0.0, 1.0],
    }
    weights = {"semantic": 0.7, "ocr": 0.3}

    fused = adaptive_modality_fusion(modality_ids, modality_scores, weights)

    # idx 1 only in semantic -> 0.7 * 0 = 0
    assert math.isclose(fused[1], 0.0)
    # idx 2 in both -> 0.7 * 1 + 0.3 * 0 = 0.7 (semantic max, ocr min)
    assert math.isclose(fused[2], 0.7)
    # idx 3 only in ocr -> 0.3 * 1 = 0.3
    assert math.isclose(fused[3], 0.3)


def test_adaptive_fusion_skips_zero_weight_modality() -> None:
    # Modality with w_m <= 0 must contribute nothing (Eq.2 over active only).
    modality_ids = {
        "semantic": [1, 2],
        "asr": [2, 3],
    }
    modality_scores = {
        "semantic": [0.2, 0.4],
        "asr": [5.0, 9.0],
    }
    weights = {"semantic": 1.0, "asr": 0.0}

    fused = adaptive_modality_fusion(modality_ids, modality_scores, weights)
    # idx 3 only in asr (w=0) -> absent
    assert 3 not in fused
    # idx 1 only in semantic -> 1.0 * 0 = 0
    assert math.isclose(fused[1], 0.0)


def test_adaptive_fusion_empty_when_no_active_modality() -> None:
    modality_ids = {"semantic": [1, 2], "ocr": [3]}
    modality_scores = {"semantic": [0.2, 0.4], "ocr": [0.0]}
    # planner predicted all-zero weights -> fusion stays off
    fused = adaptive_modality_fusion(
        modality_ids, modality_scores, {"semantic": 0.0, "ocr": 0.0}
    )
    assert fused == {}


def test_adaptive_fusion_mismatched_length_raises() -> None:
    modality_ids = {"semantic": [1, 2]}
    modality_scores = {"semantic": [0.2]}  # length mismatch
    weights = {"semantic": 1.0}
    try:
        adaptive_modality_fusion(modality_ids, modality_scores, weights)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_adaptive_fusion_returns_union_indices() -> None:
    # Indices absent from every active modality never appear (S=0 dropped).
    modality_ids = {"semantic": [10, 20]}
    modality_scores = {"semantic": [0.2, 0.9]}
    fused = adaptive_modality_fusion(
        modality_ids, modality_scores, {"semantic": 1.0}
    )
    assert set(fused.keys()) == {10, 20}
