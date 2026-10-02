"""Tests for Phase 6 temporal decay + beam + first/last semantics.

These exercise the opt-in extensions to ``align_events_dp``. The OFF path
(``use_decay=False``) is covered by ``test_temporal_alignment.py`` and must
stay byte-identical to the legacy DP.
"""

from __future__ import annotations

import numpy as np
import pytest

from aic2026.temporal import (
    BEAM_THRESHOLD,
    align_events_dp,
    decay_alpha_from_wording,
)


def test_decay_alpha_from_wording() -> None:
    # Strongest marker wins; default when no marker.
    assert decay_alpha_from_wording("do A, then B") < 0.1
    assert (
        decay_alpha_from_wording("do A immediately then B")
        > decay_alpha_from_wording("do A, then B")
    )
    assert (
        decay_alpha_from_wording("do A later then B")
        < decay_alpha_from_wording("do A, then B")
    )
    assert decay_alpha_from_wording("something neutral") == 0.01


def test_decay_uniform_frames_matches_legacy_when_no_gap() -> None:
    # With no time gaps (monotonic consecutive frames) decay adds a constant
    # offset per event, so the *path* is unchanged vs legacy DP.
    similarity = np.asarray(
        [
            [0.90, 0.20, 0.10, 0.00],
            [0.10, 0.85, 0.30, 0.10],
            [0.00, 0.10, 0.40, 0.95],
        ],
        dtype=np.float32,
    )
    score_legacy, positions_legacy = align_events_dp(similarity_matrix=similarity)
    score_decay, positions_decay = align_events_dp(
        similarity_matrix=similarity,
        use_decay=True,
        decay_alpha=0.01,
        frame_positions=np.arange(4, dtype=np.float32),
    )
    assert positions_decay == positions_legacy == [0, 1, 3]


def test_decay_prefers_compact_temporal_sequence() -> None:
    # Two equally-good-from-similarity alignments; decay should favour the
    # tighter (smaller normalised time gap) one.
    similarity = np.asarray(
        [
            [0.95, 0.00, 0.00, 0.00],
            [0.00, 0.75, 0.00, 1.00],
        ],
        dtype=np.float32,
    )
    # frame positions: event1 at 0 or 3; event2 at 1 or 2/3. With strong decay,
    # choosing frame 0 then 1 (tight) should beat 0 then 3 (wide gap).
    score, positions = align_events_dp(
        similarity_matrix=similarity,
        use_decay=True,
        decay_alpha=1.0,
        frame_positions=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    )
    assert positions == [0, 1]


def test_beam_matches_dp_on_small_matrix() -> None:
    similarity = np.asarray(
        [
            [0.90, 0.20, 0.10, 0.00],
            [0.10, 0.85, 0.30, 0.10],
            [0.00, 0.10, 0.40, 0.95],
        ],
        dtype=np.float32,
    )
    score_dp, dp_path = align_events_dp(similarity_matrix=similarity)
    score_beam, beam_path = align_events_dp(
        similarity_matrix=similarity,
        use_beam=True,
        beam_width=4,  # covers all frames -> exact
        frame_positions=np.arange(4, dtype=np.float32),
    )
    assert beam_path == dp_path
    assert score_beam == score_dp


def test_beam_switches_for_many_events() -> None:
    # A matrix with >= BEAM_THRESHOLD events exercises the beam branch.
    rng = np.random.default_rng(0)
    similarity = rng.random((BEAM_THRESHOLD + 2, BEAM_THRESHOLD + 5)).astype(
        np.float32
    )
    score_dp, dp_path = align_events_dp(similarity_matrix=similarity)
    frame_count = similarity.shape[1]
    # Exact when beam_width >= frame_count (beam explores every frame).
    score_beam, beam_path = align_events_dp(
        similarity_matrix=similarity,
        use_beam=True,
        beam_width=frame_count,
    )
    assert beam_path == dp_path
    # Score matches to float32 precision (carry/recurrence parity).
    assert score_beam == pytest.approx(score_dp, rel=1e-5)
    assert all(
        beam_path[i] < beam_path[i + 1] for i in range(len(beam_path) - 1)
    )


def test_beam_narrow_still_monotonic() -> None:
    # A narrow beam is an approximation but must still return a valid,
    # strictly increasing alignment path.
    rng = np.random.default_rng(1)
    similarity = rng.random((BEAM_THRESHOLD + 4, BEAM_THRESHOLD + 6)).astype(
        np.float32
    )
    score_beam, beam_path = align_events_dp(
        similarity_matrix=similarity,
        use_beam=True,
        beam_width=8,
    )
    assert list(beam_path) == sorted(beam_path)
    assert len(beam_path) == similarity.shape[0]


def test_first_role_selects_earliest_matching_frame() -> None:
    # Column 0 has identical top similarity at two positions; "first" event
    # should pick the EARLIEST one, not the last argmax column.
    similarity = np.asarray(
        [
            [0.90, 0.90, 0.10],
            [0.10, 0.20, 0.95],
        ],
        dtype=np.float32,
    )
    score, positions = align_events_dp(
        similarity_matrix=similarity,
        event_roles=["first", "last"],
    )
    assert positions[0] == 0  # earliest frame with the max (0.90)
    assert positions[1] == 2  # "last" -> argmax (0.95)


def test_last_role_selects_highest_score_frame() -> None:
    similarity = np.asarray(
        [
            [0.10, 0.90, 0.20],
            [0.00, 0.10, 0.95],
        ],
        dtype=np.float32,
    )
    score, positions = align_events_dp(
        similarity_matrix=similarity,
        event_roles=["first", "last"],
    )
    assert positions[0] == 1  # first -> earliest max (0.90 at col 1)
    assert positions[1] == 2  # last -> argmax (0.95 at col 2)
