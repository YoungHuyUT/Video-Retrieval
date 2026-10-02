"""Tests for the refactor's Adaptive Fusion final tier + missing-modality
renormalization (paper Eq.1 min-max + Eq.2 weighted sum with renormalization).
"""

from __future__ import annotations

import math

from aic2026.reranking.lexical import adaptive_modality_fusion, minmax_normalize


def test_renormalize_drops_missing_modality() -> None:
    # semantic (0.45) + object (0.30) present; asr (0.25) configured but the
    # transcript scores are absent → asr must be dropped and its weight folded
    # into the remaining two so the score still spans [0, 1].
    modality_ids = {
        "semantic": [1, 2],   # scores [0.2, 0.4] -> norm [0, 1]
        "object": [2, 3],     # scores [0.0, 1.0] -> norm [0, 1]
        # "asr" intentionally missing
    }
    modality_scores = {
        "semantic": [0.2, 0.4],
        "object": [0.0, 1.0],
    }
    weights = {"semantic": 0.45, "object": 0.30, "asr": 0.25}
    fused = adaptive_modality_fusion(modality_ids, modality_scores, weights)

    # active sum = 0.45 + 0.30 = 0.75 → divide each term by 0.75.
    # idx 1: only semantic (norm 0) -> 0
    assert math.isclose(fused[1], 0.0)
    # idx 2: semantic norm 1, object norm 0 -> (0.45*1 + 0.30*0)/0.75 = 0.6
    assert math.isclose(fused[2], 0.6)
    # idx 3: object norm 1 -> (0.30*1)/0.75 = 0.4
    assert math.isclose(fused[3], 0.4)

    # The three scores must still span the full [0, 1]-equivalent range after
    # renormalization (max == 0.6 here, which is the renormalized top).
    assert math.isclose(max(fused.values()), 0.6)


def test_no_renormalize_when_all_active() -> None:
    # All configured modalities present and weights sum to 1.0 → behaviour is
    # identical to the legacy weighted sum (no division).
    modality_ids = {"semantic": [1, 2], "object": [2, 3]}
    modality_scores = {"semantic": [0.2, 0.4], "object": [0.0, 1.0]}
    weights = {"semantic": 0.7, "object": 0.3}
    fused = adaptive_modality_fusion(modality_ids, modality_scores, weights, renormalize=True)
    assert math.isclose(fused[1], 0.0)
    assert math.isclose(fused[2], 0.7)
    assert math.isclose(fused[3], 0.3)


def test_minmax_formula_matches_spec() -> None:
    # Verify s_norm = (s - min) / (max - min + eps) exactly.
    scores = [0.1, 0.3, 0.5, 0.5]
    norm = minmax_normalize(scores)
    assert math.isclose(norm[0], 0.0)              # lowest -> 0
    assert math.isclose(norm[-1], 1.0)             # highest -> 1
    assert math.isclose(norm[1], 0.5)              # (0.3-0.1)/(0.5-0.1)
