from __future__ import annotations

import numpy as np
import pytest

from aic2026.temporal import align_events_dp


def test_dp_returns_one_ordered_frame_per_event() -> None:
    similarity = np.asarray(
        [
            [0.90, 0.20, 0.10, 0.00],
            [0.10, 0.85, 0.30, 0.10],
            [0.00, 0.10, 0.40, 0.95],
        ],
        dtype=np.float32,
    )

    score, positions = align_events_dp(
        similarity_matrix=similarity,
    )

    assert positions == [
        0,
        1,
        3,
    ]

    assert score == pytest.approx(
        2.685,
        abs=1e-5,
    )


def test_dp_rejects_semantically_good_but_reversed_order() -> None:
    similarity = np.asarray(
        [
            [0.10, 0.90, 0.20],
            [0.95, 0.20, 0.80],
        ],
        dtype=np.float32,
    )

    _, positions = align_events_dp(
        similarity_matrix=similarity,
    )

    assert positions == [
        1,
        2,
    ]


def test_gap_penalty_prefers_compact_sequence() -> None:
    similarity = np.asarray(
        [
            [0.95, 0.00, 0.00, 0.00],
            [0.00, 0.75, 0.00, 1.00],
        ],
        dtype=np.float32,
    )

    _, positions = align_events_dp(
        similarity_matrix=similarity,
        penalty_weight=0.20,
    )

    assert positions == [
        0,
        1,
    ]


def test_dp_rejects_fewer_frames_than_events() -> None:
    similarity = np.asarray(
        [
            [0.90, 0.10],
            [0.20, 0.80],
            [0.30, 0.70],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="fewer frames than events",
    ):
        align_events_dp(
            similarity_matrix=similarity,
        )


def test_dp_rejects_invalid_values() -> None:
    similarity = np.asarray(
        [
            [0.90, np.nan],
            [0.20, 0.80],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        align_events_dp(
            similarity_matrix=similarity,
        )