"""Frame → Moment clustering (spec §9, §10, §11).

The architecture shift of §9: stop treating each frame as an independent
candidate all the way to the end of the pipeline.  After retrieval we group
*nearby-in-time* frames (same video, within a shot / temporal neighbourhood)
into **moments**, rerank the *moments* first, and only then pick a
representative frame.  This matters most for KIS / QA / TRAKE, where the user
wants THE evidence frame, not three near-duplicate frames from the same beat.

    frames
      ↓  temporal clustering
    moment candidates {video_id, start, end, representatives[], frame_scores[]}
      ↓  rerank moments (aggregate frame scores)
    best moment
      ↓  pick representative frame
    best evidence frame

A Moment is built **cheaply** from the per-frame ``timestamp`` / ``shot_id``
already in the manifest — no extra frame extraction, no re-encoding (spec §9:
"Không cần extract thêm hàng trăm nghìn frame").  Two consecutive frames belong
to the same moment when (a) they share a ``shot_id`` (when available) OR (b)
their timestamps are within ``gap_seconds`` (default 3 s).  This mirrors the
spec's "Các frame gần nhau trong cùng shot/moment nên được group".

The aggregation is configurable (spec §15): default ``max`` picks the strongest
member frame as the moment score (good for "best evidence frame"); ``mean`` /
``lse`` (log-sum-exp) variants are provided for benchmarking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np

from aic2026.models import Candidate

if TYPE_CHECKING:
    from aic2026.models import FrameRecord

logger = logging.getLogger(__name__)

MomentAggregation = Literal["max", "mean", "lse"]


@dataclass
class Moment:
    """A contiguous temporal group of candidate frames within one video."""

    video_id: str
    start_time: float
    end_time: float
    representative_frames: list[int] = field(default_factory=list)  # frame_ids
    frame_scores: list[float] = field(default_factory=list)
    vector_ids: list[int] = field(default_factory=list)  # manifest ids
    keyframe_paths: list[str] = field(default_factory=list)
    shot_id: str | None = None
    # Aggregated moment score (filled by ``cluster_frames_to_moments`` /
    # ``rerank_moments``).
    score: float = 0.0
    # Debug info from original candidates (preserved through reranking)
    debugs: list[dict | None] = field(default_factory=list)


def _default_agg_fn(mode: MomentAggregation) -> Callable[[np.ndarray], float]:
    if mode == "mean":
        return lambda a: float(a.mean())
    if mode == "lse":
        # log-sum-exp ~ soft-max of scores; bounded, monotonic in the max.
        return lambda a: float(np.logaddexp.reduce(a)) if len(a) else 0.0
    # default "max"
    return lambda a: float(a.max()) if len(a) else 0.0


def cluster_frames_to_moments(
    candidates: list[Candidate],
    records: dict[int, "FrameRecord"] | None = None,
    *,  # keyword-only, config-driven
    gap_seconds: float = 3.0,
    use_shot_id: bool = True,
    min_frames: int = 1,
    aggregation: MomentAggregation = "max",
) -> list[Moment]:
    """Group temporally-adjacent candidate frames into moments, per video.

    Sorting is by (video_id, timestamp) so consecutive frames in the same beat
    fall together.  A new moment starts when the gap to the previous frame
    exceeds ``gap_seconds`` (or the ``shot_id`` changes, when available and
    ``use_shot_id`` is on).  Each moment's score is the aggregated frame score.

    Parameters
    ----------
    candidates: the (already reranked) candidate pool.
    records: optional vector_id→FrameRecord lookup for shot_id / timestamps.
    gap_seconds: max timestamp gap to stay in the same moment.
    use_shot_id: split moments when shot_id changes (when shot_id is present).
    min_frames: moments with fewer than this many frames are dropped (default 1 =
        keep all).  Raise to filter noise in very long pools.
    aggregation: how to combine member frame scores into the moment score.

    Returns
    -------
    A list of :class:`Moment` (one per cluster), sorted by score descending
    (best moment first).  Empty input → empty list.  Frames without a timestamp
    are clustered by frame_id gap as a fallback so the function never crashes.
    """
    if not candidates:
        return []

    agg = _default_agg_fn(aggregation)

    def _ts(c: Candidate) -> float:
        rec = records.get(c.vector_id) if (records and c.vector_id is not None) else None
        return float(getattr(rec, "timestamp", None) or c.timestamp or 0.0)

    def _shot(c: Candidate) -> str | None:
        if not use_shot_id:
            return None
        rec = records.get(c.vector_id) if (records and c.vector_id is not None) else None
        return getattr(rec, "shot_id", None)

    # Sort by video, then time so adjacency is meaningful.
    ordered = sorted(candidates, key=lambda c: (c.video_id, _ts(c), c.frame_id))

    moments: list[Moment] = []
    cur: Moment | None = None
    for c in ordered:
        ts = _ts(c)
        shot = _shot(c)
        if cur is None:
            cur = Moment(
                video_id=c.video_id,
                start_time=ts,
                end_time=ts,
                shot_id=shot,
            )
            _append_to_moment(cur, c)
            continue
        same_video = c.video_id == cur.video_id
        same_shot = (shot is not None and cur.shot_id is not None and shot == cur.shot_id)
        within_gap = (ts - cur.end_time) <= gap_seconds
        # Continue the current moment if: same video AND (explicit shot match OR
        # within the time gap).  If shot ids are absent, only the time gap rules.
        if same_video and (same_shot or within_gap):
            cur.end_time = max(cur.end_time, ts)
            _append_to_moment(cur, c)
        else:
            moments.append(cur)
            cur = Moment(
                video_id=c.video_id,
                start_time=ts,
                end_time=ts,
                shot_id=shot,
            )
            _append_to_moment(cur, c)

    if cur is not None:
        moments.append(cur)

    # Aggregate each moment's frame scores, drop too-small moments.
    out: list[Moment] = []
    for m in moments:
        if len(m.frame_scores) < min_frames:
            continue
        arr = np.asarray(m.frame_scores, dtype=np.float32)
        m.score = agg(arr)
        out.append(m)

    out.sort(key=lambda m: m.score, reverse=True)
    return out


def _append_to_moment(m: Moment, c: Candidate) -> None:
    m.representative_frames.append(c.frame_id)
    m.frame_scores.append(float(c.score))
    if c.vector_id is not None:
        m.vector_ids.append(c.vector_id)
    if c.keyframe_path:
        m.keyframe_paths.append(c.keyframe_path)
    m.debugs.append(c.debug)


def best_frame_from_moment(moment: Moment) -> Candidate:
    """Build a single :class:`Candidate` for the highest-scoring member frame.

    The returned candidate carries the moment's aggregated score plus the
    representative frame's own metadata, so downstream code (which expects a
    Candidate) keeps working while the score reflects the whole moment.
    """
    best_idx = int(np.argmax(np.asarray(moment.frame_scores, dtype=np.float32)))
    vector_id = moment.vector_ids[best_idx] if best_idx < len(moment.vector_ids) else None
    frame_id = (
        moment.representative_frames[best_idx]
        if best_idx < len(moment.representative_frames)
        else 0
    )
    path = (
        moment.keyframe_paths[best_idx]
        if best_idx < len(moment.keyframe_paths)
        else None
    )
    debug = moment.debugs[best_idx] if best_idx < len(moment.debugs) else None
    return Candidate(
        video_id=moment.video_id,
        frame_id=frame_id,
        score=float(moment.score),
        vector_id=vector_id,
        keyframe_path=path,
        timestamp=float(moment.start_time),
        debug=debug,
    )


def rerank_moments_to_frames(
    candidates: list[Candidate],
    records: dict[int, "FrameRecord"] | None = None,
    *,
    gap_seconds: float = 3.0,
    use_shot_id: bool = True,
    min_frames: int = 1,
    aggregation: MomentAggregation = "max",
    keep_structure: bool = False,
) -> list[Candidate]:
    """Convenience: cluster frames→moments→pick the best frame of each moment.

    Returns the best representative frame of each moment, ordered by moment score
    (best moment's frame first).  This collapses near-duplicate frames per beat
    while still surfacing the top-N distinct moments (the spec's "best evidence
    frame" for KIS/QA; for TRAKE each event gets its own frame downstream).

    If ``keep_structure`` is True, ALL member frames of each (ranked) moment are
    re-emitted in moment order — useful when the caller wants the full set but
    sorted by temporal coherence rather than pure per-frame score.
    """
    if not candidates:
        return []
    moments = cluster_frames_to_moments(
        candidates,
        records=records,
        gap_seconds=gap_seconds,
        use_shot_id=use_shot_id,
        min_frames=min_frames,
        aggregation=aggregation,
    )
    if not moments:
        return candidates
    if keep_structure:
        out: list[Candidate] = []
        for m in moments:
            out.extend(_moment_members_as_candidates(m))
        return out
    return [best_frame_from_moment(m) for m in moments]


def _moment_members_as_candidates(m: Moment) -> list[Candidate]:
    """Re-emit every member frame of a moment as a Candidate (moment score kept)."""
    out: list[Candidate] = []
    for i, frame_id in enumerate(m.representative_frames):
        out.append(
            Candidate(
                video_id=m.video_id,
                frame_id=frame_id,
                score=float(m.frame_scores[i]),
                vector_id=(
                    m.vector_ids[i] if i < len(m.vector_ids) else None
                ),
                keyframe_path=(
                    m.keyframe_paths[i] if i < len(m.keyframe_paths) else None
                ),
                timestamp=float(m.start_time),
            )
        )
    return out
