import numpy as np

from aic2026.temporal.dense_refinement import _window_frame_ids


def test_dense_windows_are_bounded_sampled_and_keep_event_centres() -> None:
    ids = _window_frame_ids(
        [30, 90], fps=30.0, window_seconds=1.0, sample_fps=3.0, frame_count=100
    )
    assert ids == sorted(set(ids))
    assert 30 in ids and 90 in ids
    assert ids[0] == 0
    assert ids[-1] == 99


def test_dense_window_stride_tracks_requested_fps() -> None:
    ids = _window_frame_ids(
        [100], fps=30.0, window_seconds=1.0, sample_fps=3.0, frame_count=300
    )
    assert set(range(70, 131, 10)).issubset(ids)
