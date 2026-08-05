from __future__ import annotations

from aic2026.models import Candidate, GroundTruth, Query


def _normalized(value: str | None) -> str:
    return " ".join((value or "").lower().strip().split())


def relevance(query: Query, candidate: Candidate, truth: GroundTruth) -> float:
    if candidate.video_id != truth.video_id:
        return 0.0
    if query.type == "trake":
        frames = candidate.event_frames or []
        if len(frames) != len(truth.ranges):
            return 0.0
        return sum(start <= frame <= end for frame, (start, end) in zip(frames, truth.ranges)) / len(truth.ranges)
    in_range = any(start <= candidate.frame_id <= end for start, end in truth.ranges)
    if not in_range:
        return 0.0
    if query.type == "qa":
        return float(_normalized(candidate.answer) == _normalized(truth.answer))
    return 1.0


def final_score(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return sum(max(scores[:k], default=0.0) for k in (1, 5, 20, 50, 100)) / 5


def evaluate_query(query: Query, candidates: list[Candidate], truth: GroundTruth) -> dict[str, float]:
    scores = [relevance(query, candidate, truth) for candidate in candidates[:100]]
    return {**{f"R@{k}": max(scores[:k], default=0.0) for k in (1, 5, 20, 50, 100)}, "final_score": final_score(scores)}
