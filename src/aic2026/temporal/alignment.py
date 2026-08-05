from __future__ import annotations

from aic2026.models import Candidate


def align_events(candidates: list[Candidate], event_count: int) -> list[int]:
    """Baseline: pick ordered, high-score semantic frames from the selected video."""
    ordered = sorted(candidates, key=lambda c: (-c.score, c.frame_id))
    frames = sorted(c.frame_id for c in ordered[:event_count])
    if len(frames) != event_count:
        raise ValueError("Not enough candidate frames to align all events")
    return frames
