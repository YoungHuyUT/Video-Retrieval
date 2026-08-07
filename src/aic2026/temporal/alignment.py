from __future__ import annotations

import numpy as np

from aic2026.models import Candidate


def align_events(
    candidates: list[Candidate],
    event_count: int,
) -> list[int]:
    """Legacy score-based baseline kept for compatibility."""

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            candidate.frame_id,
        ),
    )

    frames = sorted(
        candidate.frame_id
        for candidate in ordered[:event_count]
    )

    if len(frames) != event_count:
        raise ValueError(
            "Not enough candidate frames to align all events"
        )

    return frames


def align_events_dp(
    similarity_matrix: np.ndarray,
    penalty_weight: float = 0.005,
) -> tuple[float, list[int]]:
    """Find the highest-scoring monotonic event-to-frame alignment.

    The returned positions are column indices inside similarity_matrix.
    Each event receives exactly one frame and consecutive events must
    select strictly increasing frame positions.

    The optimized objective is:

        sum(event-to-frame similarities)
        - penalty_weight * sum(position gaps)
    """

    scores = np.asarray(
        similarity_matrix,
        dtype=np.float32,
    )

    if scores.ndim != 2:
        raise ValueError(
            "similarity_matrix must be a 2D matrix"
        )

    event_count, frame_count = scores.shape

    if event_count == 0:
        raise ValueError(
            "At least one event is required"
        )

    if frame_count == 0:
        raise ValueError(
            "At least one frame is required"
        )

    if frame_count < event_count:
        raise ValueError(
            "There are fewer frames than events"
        )

    if penalty_weight < 0:
        raise ValueError(
            "penalty_weight must not be negative"
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "similarity_matrix must contain finite values"
        )

    dp = np.full(
        (event_count, frame_count),
        -np.inf,
        dtype=np.float32,
    )

    predecessor = np.full(
        (event_count, frame_count),
        -1,
        dtype=np.int32,
    )

    # Event đầu tiên có thể chọn bất kỳ frame nào.
    dp[0, :] = scores[0, :]

    for event_index in range(
        1,
        event_count,
    ):
        running_best_score = -np.inf
        running_best_index = -1

        # Ít nhất phải chừa event_index frame ở phía trước.
        for frame_index in range(
            event_index,
            frame_count,
        ):
            previous_index = frame_index - 1

            previous_score = (
                dp[
                    event_index - 1,
                    previous_index,
                ]
                + penalty_weight * previous_index
            )

            if previous_score > running_best_score:
                running_best_score = previous_score
                running_best_index = previous_index

            if running_best_index < 0:
                continue

            if not np.isfinite(
                running_best_score
            ):
                continue

            dp[
                event_index,
                frame_index,
            ] = (
                scores[
                    event_index,
                    frame_index,
                ]
                + running_best_score
                - penalty_weight * frame_index
            )

            predecessor[
                event_index,
                frame_index,
            ] = running_best_index

    final_position = int(
        np.argmax(
            dp[-1, :]
        )
    )

    final_score = float(
        dp[-1, final_position]
    )

    if not np.isfinite(final_score):
        raise ValueError(
            "No valid monotonic alignment exists"
        )

    path = [0] * event_count
    current_position = final_position

    for event_index in range(
        event_count - 1,
        -1,
        -1,
    ):
        path[event_index] = current_position

        if event_index == 0:
            continue

        current_position = int(
            predecessor[
                event_index,
                current_position,
            ]
        )

        if current_position < 0:
            raise RuntimeError(
                "Broken DP predecessor chain"
            )

    return final_score, path