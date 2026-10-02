from __future__ import annotations

import math

from aic2026.temporal.alignment import (
    TAU,
    final_trake_gating,
    temporal_lambdas,
)


# --- Eq.3: lambda_i = exp(-alpha * delta_t) -------------------------------

def test_temporal_lambdas_first_event_anchored() -> None:
    # Event 0 has no predecessor -> lambda = 1.0.
    lambdas = temporal_lambdas([10, 20, 30], use_decay=True, decay_alpha=0.01)
    assert len(lambdas) == 3
    assert math.isclose(lambdas[0], 1.0)


def test_temporal_lambdas_decay_on_with_gap() -> None:
    # 2 events at [0, 100] normalise to [0, 1] (span 100), so delta_t = 1.
    # lambda_1 = exp(-alpha * 1) = exp(-0.01).
    lambdas = temporal_lambdas([0, 100], use_decay=True, decay_alpha=0.01)
    assert math.isclose(lambdas[1], math.exp(-0.01))


def test_temporal_lambdas_no_decay_returns_ones() -> None:
    # use_decay=False -> every lambda = 1.0 (legacy additive path preserved).
    lambdas = temporal_lambdas([5, 9, 12, 40], use_decay=False, decay_alpha=0.05)
    assert lambdas == [1.0, 1.0, 1.0, 1.0]


def test_temporal_lambdas_larger_alpha_suppresses_more() -> None:
    base = temporal_lambdas([0, 100, 200], use_decay=True, decay_alpha=0.05)
    loose = temporal_lambdas([0, 100, 200], use_decay=True, decay_alpha=0.002)
    # stronger alpha -> smaller lambda
    assert base[1] < loose[1]
    assert 0 < base[1] < 1


# --- Eq.6-7: SS(final) = sum_i s_i * lambda_i * b_i -----------------------

def test_final_gating_multiplicative_sum() -> None:
    event_scores = [0.9, 0.7, 0.5]
    lambdas = [1.0, 0.8, 0.6]
    fine_rerank = [1.0, 1.0, 1.0]  # b_i default when no fine reranker
    expected = 0.9 * 1.0 + 0.7 * 0.8 + 0.5 * 0.6
    assert math.isclose(
        final_trake_gating(event_scores, lambdas, fine_rerank), expected
    )


def test_final_gating_b_i_defaults_to_one() -> None:
    # No fine reranker supplied -> b_i = 1.0 -> degrades to additive decay score.
    event_scores = [0.9, 0.7, 0.5]
    lambdas = [1.0, 0.8, 0.6]
    expected = 0.9 * 1.0 + 0.7 * 0.8 + 0.5 * 0.6
    assert math.isclose(final_trake_gating(event_scores, lambdas), expected)


def test_final_gating_low_b_i_suppresses_sequence() -> None:
    # A near-zero b_i on one event collapses that event's contribution
    # (hard multi-facet gate), lowering the whole sequence score.
    event_scores = [1.0, 1.0, 1.0]
    lambdas = [1.0, 1.0, 1.0]
    full = final_trake_gating(event_scores, lambdas, [1.0, 1.0, 1.0])
    gated = final_trake_gating(event_scores, lambdas, [1.0, 0.0, 1.0])
    assert gated < full
    assert math.isclose(gated, 2.0)


def test_final_gating_missing_b_i_floored_at_tau() -> None:
    # b_i = None or <= 0 should be floored at TAU (never NaN / negative collapse).
    event_scores = [1.0, 1.0]
    lambdas = [1.0, 1.0]
    val = final_trake_gating(event_scores, lambdas, [1.0, None])
    assert val > 0
    assert math.isclose(val, 1.0 + TAU)


def test_final_gating_empty_returns_zero() -> None:
    assert final_trake_gating([], []) == 0.0


def test_final_gating_pads_short_arrays() -> None:
    # Mismatched lengths are tolerated by padding the shorter with 1.0.
    event_scores = [0.5, 0.5, 0.5]
    lambdas = [1.0, 1.0]        # shorter than events -> padded to 1.0
    fine_rerank = [2.0]         # shorter -> padded to 1.0
    expected = 0.5 * 1.0 * 2.0 + 0.5 * 1.0 * 1.0 + 0.5 * 1.0 * 1.0
    assert math.isclose(
        final_trake_gating(event_scores, lambdas, fine_rerank), expected
    )
