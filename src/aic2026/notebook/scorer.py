"""Video-level scorer for NOTEBOOK (spec §14, §15).

Combines ASR evidence, event coverage, temporal consistency, and modality
signals into a single video-level score. This is the scoring function that
ranks candidate videos for the final Top-N output.

Score formula (spec §14):
  S_video = w_asr * normalised_asr_score
          + w_coverage * event_coverage
          + w_temporal * temporal_consistency
          + w_modality * sum(modality_signal * weight)

All components are normalised to [0, 1] within the candidate pool.
"""

from __future__ import annotations

import logging
from typing import Sequence

from aic2026.notebook.types import (
    ModalitySignal,
    NotebookCandidate,
    NotebookEvidence,
    NotebookPlan,
)

logger = logging.getLogger(__name__)

# Default fusion weights (spec §14: video-level fusion).
# These can be overridden by the plan's modality_weights.
_DEFAULT_WEIGHTS = {
    "asr": 0.35,
    "dense": 0.15,  # BGE-M3 dense semantic signal (P0 upgrade)
    "coverage": 0.25,
    "temporal": 0.15,
    "object": 0.10,
}


def _normalise(values: list[float]) -> list[float]:
    """Min-max normalise a list of scores to [0, 1].

    Returns all zeros if all values are equal (degenerate case).
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]


def score_video(
    evidence: NotebookEvidence,
    plan: NotebookPlan,
    weights: dict[str, float] | None = None,
) -> float:
    """Score a single video's evidence into a scalar in [0, 1].

    Uses the plan's modality_weights and the default fusion weights.
    The final score is a weighted combination of normalised ASR score,
    event coverage, temporal consistency, and modality signals.

    Args:
        evidence: NotebookEvidence for this video
        plan: The NotebookPlan driving the query
        weights: Optional weight overrides (defaults to _DEFAULT_WEIGHTS)

    Returns:
        Normalised score in [0, 1]
    """
    if weights is None:
        weights = _DEFAULT_WEIGHTS

    # Merge plan modality weights with default fusion weights
    w = {**_DEFAULT_WEIGHTS, **(weights or {})}

    # 1. ASR signal (normalised BM25 score)
    asr_score = 0.0
    if evidence.modality_signals:
        for ms in evidence.modality_signals:
            if ms.name == "asr":
                asr_score = ms.normalised
                break
    # If not yet normalised, use raw score capped
    if asr_score == 0.0 and evidence.segment_matches:
        raw_asr = sum(m.bm25_score for m in evidence.segment_matches)
        asr_score = min(1.0, raw_asr / 10.0)  # rough normalisation

    # 2. Dense signal (BGE-M3 cosine similarity)
    dense_score = 0.0
    if evidence.modality_signals:
        for ms in evidence.modality_signals:
            if ms.name == "dense":
                dense_score = ms.normalised
                break

    # 3. Event coverage (already in [0, 1])
    coverage_score = evidence.coverage.weighted_coverage or evidence.coverage.coverage

    # 4. Temporal consistency (already in [0, 1])
    temporal_score = evidence.temporal.consistency

    # 5. Modality signal contribution
    modality_score = 0.0
    for ms in evidence.modality_signals:
        modality_score += ms.normalised * ms.weight

    # Weighted sum
    score = (
        w.get("asr", 0.35) * asr_score
        + w.get("dense", 0.15) * dense_score
        + w.get("coverage", 0.25) * coverage_score
        + w.get("temporal", 0.15) * temporal_score
        + w.get("object", 0.10) * modality_score
    )

    return max(0.0, min(1.0, score))


def normalise_modality_signals(
    all_evidence: dict[str, NotebookEvidence],
) -> dict[str, NotebookEvidence]:
    """Normalise modality signal raw_scores to [0, 1] across all videos.

    Mutates the evidence objects in-place, setting each ModalitySignal.normalised
    field based on min-max scaling across the candidate pool.
    """
    # Collect all ASR raw scores
    for mod_name in ("asr", "dense", "object", "visual"):
        raw_scores = []
        video_signals: dict[str, ModalitySignal | None] = {}
        for vid, ev in all_evidence.items():
            ms = next((s for s in ev.modality_signals if s.name == mod_name), None)
            video_signals[vid] = ms
            if ms is not None:
                raw_scores.append(ms.raw_score)

        if not raw_scores:
            continue

        normalised = _normalise(raw_scores)
        idx = 0
        for vid, ms in video_signals.items():
            if ms is not None:
                ms.normalised = normalised[idx]
                idx += 1

    return all_evidence


def score_candidates(
    evidence_by_video: dict[str, NotebookEvidence],
    plan: NotebookPlan,
    top_n: int = 5,
) -> list[NotebookCandidate]:
    """Score all candidate videos and return the top-N.

    This is the main entry point for scoring. Takes per-video evidence,
    normalises modality signals, computes scores, and returns sorted candidates.

    Args:
        evidence_by_video: Dict of video_id -> NotebookEvidence
        plan: The NotebookPlan driving this query
        top_n: Maximum number of candidates to return

    Returns:
        List of NotebookCandidate sorted by score (descending), capped at top_n
    """
    if not evidence_by_video:
        return []

    # Normalise modality signals across the pool
    normalise_modality_signals(evidence_by_video)

    # Score each video
    candidates: list[NotebookCandidate] = []
    for video_id, evidence in evidence_by_video.items():
        score = score_video(evidence, plan)

        # Build evidence timestamps from top segments
        evidence_timestamps = list(evidence.evidence_timestamps)

        # Matched event indices
        matched_events = list(evidence.matched_event_indices)

        # ASR snippets for display
        asr_snippets = evidence.asr_snippets

        # Evidence count
        evidence_count = len(evidence.segment_matches)

        # Best evidence frame: highest-scoring ASR segment for quick true/false check.
        best_frame_id: int | None = None
        best_keyframe_path: str | None = None
        best_timestamp: float | None = None
        if evidence.segment_matches:
            best = max(evidence.segment_matches, key=lambda m: m.bm25_score + m.dense_score)
            best_frame_id = best.frame
            best_timestamp = best.start
            # Resolve keyframe path from manifest if frame_id is known.
            if best_frame_id is not None:
                from aic2026.models import FrameRecord
                # Try to find the keyframe path from the pipeline manifest.
                best_keyframe_path = f"{video_id}/{best_frame_id}.jpg"

        candidates.append(
            NotebookCandidate(
                video_id=video_id,
                score=round(score, 6),
                evidence_timestamps=evidence_timestamps,
                matched_events=matched_events,
                asr_snippets=asr_snippets,
                event_coverage=evidence.coverage.coverage,
                temporal_consistency=evidence.temporal.consistency,
                modality_signals=evidence.modality_signals,
                evidence_count=evidence_count,
                frame_id=best_frame_id,
                keyframe_path=best_keyframe_path,
                evidence_timestamp=best_timestamp,
            )
        )

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)

    # Apply top-N cutoff
    top_candidates = candidates[:top_n]

    logger.info(
        "Scorer: %d candidates ranked, top %d scores: %s",
        len(candidates),
        len(top_candidates),
        [round(c.score, 4) for c in top_candidates[:5]],
    )

    return top_candidates


def confidence_gap(candidates: list[NotebookCandidate]) -> float:
    """Compute confidence gap = score_top1 - score_top2.

    If gap < threshold, a second LLM pass may be triggered (spec §12).
    """
    if len(candidates) < 2:
        return 1.0  # max confidence when only one candidate
    return candidates[0].score - candidates[1].score


__all__ = [
    "score_video",
    "score_candidates",
    "normalise_modality_signals",
    "confidence_gap",
]
