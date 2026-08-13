from __future__ import annotations

from aic2026.models import Candidate, GroundTruth, Query

from .semantic import SemanticMatcher

# Reusable deterministic matcher for Q&A relevance. Exact + synonym groups by
# default; can be swapped for embedding/LLM-backed matching by callers.
_DEFAULT_QA_MATCHER = SemanticMatcher()


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
        # The AIC 2026 PDF requires the answer to match semantically, not as an
        # exact string. Exact normalized equality is the first rule inside the
        # matcher, so this stays consistent for the strict case too.
        return float(
            _DEFAULT_QA_MATCHER.match(
                candidate.answer or "",
                truth.answer or "",
            )
        )
    return 1.0


def final_score(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return sum(max(scores[:k], default=0.0) for k in (1, 5, 20, 50, 100)) / 5


def evaluate_query(query: Query, candidates: list[Candidate], truth: GroundTruth) -> dict[str, float]:
    # R@k is only meaningful in score-descending order (best first). Some
    # callers (e.g. the CLI `evaluate` command loading raw JSON) may not have
    # pre-sorted candidates, so sort here to make the metric independent of input order.
    ranked = sorted(
        candidates,
        key=lambda candidate: candidate.score,
        reverse=True,
    )
    scores = [relevance(query, candidate, truth) for candidate in ranked[:100]]
    return {**{f"R@{k}": max(scores[:k], default=0.0) for k in (1, 5, 20, 50, 100)}, "final_score": final_score(scores)}
