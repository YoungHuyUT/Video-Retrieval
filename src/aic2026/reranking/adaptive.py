from __future__ import annotations

from collections import defaultdict
from typing import Literal

from aic2026.models import Candidate

# Three adaptive rerank modes, resolved from how many *competing* videos actually
# appear in the top-K retrieval results (not the total video count — a 1000-video
# dataset can still collapse to one relevant video for a specific query).
#
#   flat : only 1 competing video  -> skip video-level rerank, keep RRF order
#                                (this is what made the single-video sample "ngon")
#   soft : 2-3 competing videos    -> rerank by video but KEEP per-frame scores,
#                                so the correct video floats up without flattening
#                                the fine-grained frame order inside it
#   full : >=4 competing videos    -> full video-level rerank + MMR diversity
#                                (coarse-filter weak videos, avoid one video dominating)
RerankMode = Literal["flat", "soft", "full"]

_SOFT_MAX = 3  # <= this many competing videos => soft
_FULL_MIN = 4  # >= this many competing videos => full


def count_competing_videos(candidates: list[Candidate], k: int = 50) -> int:
    """Number of distinct videos appearing in the top-``k`` candidates by score.

    This, not the total video count, decides whether video-level reranking is
    helpful or harmful: with a single relevant video, per-video aggregation just
    collapses every frame into one undifferentiated blob.
    """
    if not candidates:
        return 0
    top = sorted(candidates, key=lambda c: c.score, reverse=True)[: max(0, k)]
    return len({c.video_id for c in top})


def resolve_mode(
    candidates: list[Candidate],
    k: int = 50,
    soft_max: int = _SOFT_MAX,
    full_min: int = _FULL_MIN,
) -> RerankMode:
    """Pick the rerank mode from the competing-video count in the top-``k``."""
    n = count_competing_videos(candidates, k=k)
    if n <= 1:
        return "flat"
    if n < full_min:  # 2..(full_min-1)
        return "soft"
    return "full"


def video_rank_order(candidates: list[Candidate], aggregation_top_k: int = 3) -> dict[str, float]:
    """Map each video_id to an aggregated ``video_score`` (log-sum-exp of its
    top-``aggregation_top_k`` frame scores). Used by soft/full reranking to order
    videos without losing per-frame scores.
    """
    by_video: dict[str, list[float]] = defaultdict(list)
    for cand in candidates:
        by_video[cand.video_id].append(cand.score)
    scores: dict[str, float] = {}
    top_k = max(1, aggregation_top_k)
    for video_id, frame_scores in by_video.items():
        top = sorted(frame_scores, reverse=True)[:top_k]
        scores[video_id] = float(__import__("numpy").logaddexp.reduce(top)) if top else 0.0
    return scores
