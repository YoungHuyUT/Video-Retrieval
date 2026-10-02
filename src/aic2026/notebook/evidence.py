"""Evidence aggregation for NOTEBOOK (spec §13, §14).

Groups ASR segment matches by video_id and builds NotebookEvidence objects
with event coverage, temporal consistency, and modality signals.

Per memory file: ASR transcripts are 100% Vietnamese; the asr_queries
must be Vietnamese to match.
"""

from __future__ import annotations

import logging
from typing import Sequence

from aic2026.notebook.types import (
    ASRSegmentMatch,
    EventCoverage,
    ModalitySignal,
    NotebookEvidence,
    NotebookPlan,
    NotebookEvent,
    TemporalEvidence,
)
from aic2026.temporal.alignment import (
    decay_alpha_from_wording,
    temporal_lambdas,
)

logger = logging.getLogger(__name__)


def group_matches_by_video(
    matches: Sequence[ASRSegmentMatch],
) -> dict[str, list[ASRSegmentMatch]]:
    """Group ASR segment matches by video_id."""
    grouped: dict[str, list[ASRSegmentMatch]] = {}
    for m in matches:
        grouped.setdefault(m.video_id, []).append(m)
    return grouped


def compute_event_coverage(
    matches: Sequence[ASRSegmentMatch],
    events: Sequence[NotebookEvent],
) -> EventCoverage:
    """Compute which planned events are satisfied by the segment matches.

    An event is considered matched if any of its key_concepts appears in
    a segment's matched_query or segment_text.

    Coverage = matched_events / total_events (spec §13: event coverage).
    """
    matched_indices: set[int] = set()
    for ev in events:
        for concept in ev.key_concepts:
            for m in matches:
                if (
                    concept.lower() in m.matched_query.lower()
                    or concept.lower() in m.segment_text.lower()
                ):
                    matched_indices.add(ev.index)
                    break
            if ev.index in matched_indices:
                break

    # Weighted coverage: sum(matched_importance) / sum(all_importance)
    total_importance = sum(ev.importance for ev in events) or 1.0
    matched_importance = sum(
        ev.importance for ev in events if ev.index in matched_indices
    )
    weighted_cov = matched_importance / total_importance

    return EventCoverage(
        total_events=len(events),
        matched_events=len(matched_indices),
        matched_event_indices=sorted(matched_indices),
        weighted_coverage=weighted_cov,
    )


def compute_temporal_evidence(
    matches: Sequence[ASRSegmentMatch],
    has_strict_temporal: bool,
    alpha: float = 0.01,
) -> TemporalEvidence:
    """Compute temporal consistency of ASR evidence segments.

    Measures whether matched segments appear in chronological order (which
    is expected for a temporal event query). Uses temporal_lambdas from
    aic2026.temporal.alignment for the decay computation (spec §9).

    Improvements:
    - Kendall's tau for sequence ordering quality (P1 upgrade)
    - Gap penalty for large temporal gaps between evidence segments
    - Better lambda decay with gap-aware weighting

    Args:
        matches: Sorted list of ASRSegmentMatch (by score or by timestamp)
        has_strict_temporal: Whether the query requires strict ordering
        alpha: Temporal decay coefficient (exp(-alpha * delta_t))
    """
    # Sort by start timestamp for temporal analysis
    sorted_by_time = sorted(matches, key=lambda m: m.start)
    timestamps = [m.start for m in sorted_by_time if m.start is not None]

    if not timestamps or len(sorted_by_time) < 2:
        return TemporalEvidence(
            timestamps=timestamps,
            order_matches=True,
            violations=0,
            consistency=1.0,
        )

    # Check monotonicity of timestamps
    violations = 0
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            violations += 1

    # Compute Kendall's tau for sequence ordering quality
    tau = _kendall_tau(timestamps)

    # Compute temporal decay using the existing utility
    # Use frame positions as proxy for temporal positions
    frame_positions = [m.frame if m.frame is not None else int(m.start * 30) for m in sorted_by_time]
    lambdas = temporal_lambdas(
        frame_positions,
        use_decay=has_strict_temporal,
        decay_alpha=alpha,
    )

    # Gap penalty: penalize large gaps between consecutive evidence segments
    gap_penalty = _compute_gap_penalty(timestamps)

    # Consistency score: combine tau, lambda product, and gap penalty
    if has_strict_temporal:
        # Product of lambdas as consistency (penalizes large gaps)
        product = 1.0
        for l in lambdas[1:]:  # skip first (always 1.0)
            product *= l
        # Combine: tau * lambda_product * gap_penalty
        consistency = tau * product * gap_penalty * (1.0 - violations / len(timestamps))
    else:
        # For non-strict temporal: use tau with gap penalty
        consistency = tau * gap_penalty

    order_matches = violations == 0
    consistency = max(0.0, min(1.0, consistency))

    return TemporalEvidence(
        timestamps=timestamps,
        order_matches=order_matches,
        violations=violations,
        consistency=consistency,
    )


def _kendall_tau(timestamps: list[float]) -> float:
    """Compute Kendall's tau for sequence ordering quality.

    Returns a value in [0, 1] where:
    - 1.0 = perfect chronological order
    - 0.5 = random order
    - 0.0 = fully reversed order

    This is a simplified version that measures how many pairs are in order.
    """
    n = len(timestamps)
    if n < 2:
        return 1.0

    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if timestamps[i] <= timestamps[j]:
                concordant += 1

    # tau = (concordant - discordant) / total
    # Map to [0, 1]: (tau + 1) / 2
    tau_raw = (2 * concordant - total) / total if total > 0 else 1.0
    return (tau_raw + 1.0) / 2.0


def _compute_gap_penalty(timestamps: list[float], sigma: float = 30.0) -> float:
    """Compute Gaussian temporal proximity penalty across consecutive segments.

    Kernel: K(delta_t) = exp(- (delta_t)^2 / (2 * sigma^2))
    Returns a smooth value in [0, 1] where 1.0 = smooth continuous flow.
    """
    if len(timestamps) < 2:
        return 1.0

    import math
    gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    avg_gap = sum(gaps) / len(gaps)

    return math.exp(- (avg_gap ** 2) / (2 * (sigma ** 2)))


def compute_modality_signals(
    matches: Sequence[ASRSegmentMatch],
    plan: NotebookPlan,
) -> list[ModalitySignal]:
    """Compute modality-level score contributions (spec §14).

    For ASR-only version: just the ASR signal.
    Visual/object signals are included but zero-weight when use_visual=False.
    """
    signals: list[ModalitySignal] = []

    # ASR modality signal
    asr_score = sum(m.bm25_score for m in matches) / max(len(matches), 1)
    asr_weight = plan.modality_weights.get("asr", 1.0)
    signals.append(
        ModalitySignal(
            name="asr",
            raw_score=asr_score,
            normalised=0.0,  # will be normalised later across all videos
            weight=asr_weight,
        )
    )

    # Dense modality signal (BGE-M3 semantic scores)
    dense_matches = [m for m in matches if m.dense_score > 0]
    if dense_matches:
        dense_raw = sum(m.dense_score for m in dense_matches) / len(dense_matches)
        signals.append(
            ModalitySignal(
                name="dense",
                raw_score=dense_raw,
                normalised=0.0,  # will be normalised later
                weight=plan.modality_weights.get("dense", 1.0),
            )
        )

    # Object modality (from plan objects) — zero weight when use_visual=False
    object_weight = plan.modality_weights.get("object", 0.0)
    if plan.use_visual and plan.objects:
        object_score = len(plan.objects) * 0.1  # placeholder
        signals.append(
            ModalitySignal(
                name="object",
                raw_score=object_score,
                normalised=0.0,
                weight=object_weight,
            )
        )

    return signals


def build_evidence(
    video_id: str,
    matches: Sequence[ASRSegmentMatch],
    plan: NotebookPlan,
) -> NotebookEvidence:
    """Build a NotebookEvidence object for one video from its segment matches.

    Args:
        video_id: The video this evidence belongs to
        matches: All ASR segment matches for this video
        plan: The NotebookPlan driving this search

    Returns:
        NotebookEvidence with coverage, temporal, and modality signals
    """
    # Sort by best available score (dense_score > bm25_score)
    def _best_score(m: ASRSegmentMatch) -> float:
        return max(m.dense_score, m.bm25_score)

    sorted_matches = sorted(matches, key=_best_score, reverse=True)

    # Compute event coverage
    coverage = compute_event_coverage(sorted_matches, plan.events)

    # Compute temporal evidence
    alpha = decay_alpha_from_wording(plan.raw_text)
    temporal = compute_temporal_evidence(
        sorted_matches,
        plan.has_strict_temporal,
        alpha=alpha,
    )

    # Compute modality signals
    modality_signals = compute_modality_signals(sorted_matches, plan)

    # Evidence timestamps — top segments' start times for display (spec §15)
    # Use diversity-aware selection: pick top segments spread across the video
    evidence_timestamps = _select_diverse_timestamps(sorted_matches, max_count=10)

    return NotebookEvidence(
        video_id=video_id,
        segment_matches=sorted_matches,
        matched_event_indices=coverage.matched_event_indices,
        temporal=temporal,
        coverage=coverage,
        modality_signals=modality_signals,
        evidence_timestamps=evidence_timestamps,
    )


def _select_diverse_timestamps(
    matches: Sequence[ASRSegmentMatch],
    max_count: int = 10,
    min_gap: float = 5.0,
) -> list[float]:
    """Select timestamps that are diverse (spread across the video).

    Picks top segments but ensures minimum time gap between selected timestamps
    for better evidence coverage of the video.
    """
    timestamps: list[float] = []
    for m in matches:
        if m.start is None or m.start < 0:
            continue
        ts = round(m.start, 2)
        # Check minimum gap from already selected timestamps
        if timestamps and min(abs(ts - t) for t in timestamps) < min_gap:
            continue
        timestamps.append(ts)
        if len(timestamps) >= max_count:
            break
    timestamps.sort()
    return timestamps


def aggregate_evidence(
    matches: Sequence[ASRSegmentMatch],
    plan: NotebookPlan,
) -> dict[str, NotebookEvidence]:
    """Aggregate ASR segment matches into per-video Evidence objects.

    This is the main entry point for evidence aggregation. Groups matches
    by video_id and builds NotebookEvidence for each.

    Args:
        matches: All ASR segment matches from the retriever
        plan: The NotebookPlan driving the search

    Returns:
        Dict mapping video_id -> NotebookEvidence
    """
    grouped = group_matches_by_video(matches)
    logger.info(
        "Evidence: aggregating %d segments into %d videos",
        len(matches), len(grouped),
    )

    evidence_by_video: dict[str, NotebookEvidence] = {}
    for video_id, video_matches in grouped.items():
        evidence_by_video[video_id] = build_evidence(
            video_id, video_matches, plan,
        )

    return evidence_by_video


__all__ = [
    "group_matches_by_video",
    "compute_event_coverage",
    "compute_temporal_evidence",
    "compute_modality_signals",
    "build_evidence",
    "aggregate_evidence",
]
