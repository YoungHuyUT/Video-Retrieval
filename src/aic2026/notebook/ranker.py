"""Ranker for NOTEBOOK (spec §15, §12).

Ranks candidate videos and applies optional second-LLM-pass for low-confidence
gaps. Default: NO second pass (only 1 LLM call for the planner).

Spec §12: confidence_gap = score_top1 - score_top2 → optional 2nd LLM pass
only when gap < threshold.
"""

from __future__ import annotations

import logging
import time

from aic2026.notebook.types import NotebookCandidate, NotebookResult
from aic2026.notebook.scorer import score_candidates, confidence_gap

logger = logging.getLogger(__name__)

# Threshold below which a second LLM pass is considered (spec §12).
# A gap this small means the top-2 are close — need disambiguation.
_CONFIDENCE_GAP_THRESHOLD = 0.15


def rank_videos(
    evidence_by_video: dict[str, "NotebookEvidence"],  # NotebookEvidence imported lazily
    plan: "NotebookPlan",  # NotebookPlan imported lazily
    top_n: int = 5,
) -> list[NotebookCandidate]:
    """Rank videos by evidence quality and return Top-N candidates.

    Delegates to scorer.score_candidates which:
    1. Normalises modality signals across the pool
    2. Computes per-video scores
    3. Returns sorted Top-N

    Args:
        evidence_by_video: Dict of video_id -> NotebookEvidence
        plan: The NotebookPlan driving the query
        top_n: Number of top videos to return

    Returns:
        Sorted list of NotebookCandidate
    """
    from aic2026.notebook.scorer import score_candidates as _score

    return _score(evidence_by_video, plan, top_n=top_n)


def maybe_second_pass(
    candidates: list[NotebookCandidate],
    plan: "NotebookPlan",
    use_llm: bool = True,
    llm_model: str = "qwen2.5:1.5b",
    llm_base_url: str = "http://127.0.0.1:11434",
) -> tuple[list[NotebookCandidate], bool]:
    """Optionally trigger a second LLM pass when confidence is low.

    Per spec §12: confidence_gap = score_top1 - score_top2.
    Only a second pass if gap < threshold AND use_llm is True.
    DEFAULT: only 1 LLM call (planner only), so second_pass defaults to False.

    The second pass sends the top-2 candidates' evidence summaries to the LLM
    and asks which video better explains the query. If the LLM picks #2, the
    top-2 are swapped.

    Args:
        candidates: Already-ranked candidates
        plan: The NotebookPlan
        use_llm: Whether LLM is available for second pass
        llm_model: Model to use for second pass
        llm_base_url: Ollama base URL

    Returns:
        (candidates, second_pass_triggered)
    """
    if not candidates:
        return candidates, False

    if len(candidates) < 2:
        return candidates, False

    gap = confidence_gap(candidates)
    logger.info("Ranker: confidence_gap=%.4f (threshold=%.2f)", gap, _CONFIDENCE_GAP_THRESHOLD)

    if gap < _CONFIDENCE_GAP_THRESHOLD and use_llm:
        logger.info("Ranker: triggering second LLM pass for disambiguation")
        try:
            swapped = _llm_disambiguate(candidates[:2], plan, llm_model, llm_base_url)
            if swapped:
                # LLM preferred #2 — swap top-2
                candidates[0], candidates[1] = candidates[1], candidates[0]
                logger.info("Ranker: LLM preferred candidate '%s' over '%s'",
                            candidates[0].video_id, candidates[1].video_id)
            return candidates, True
        except Exception:  # noqa: BLE001
            logger.exception("Ranker: second LLM pass failed, keeping original order")
            return candidates, True

    return candidates, False


def _llm_disambiguate(
    top2: list[NotebookCandidate],
    plan: "NotebookPlan",
    llm_model: str,
    llm_base_url: str,
) -> bool:
    """Ask the LLM which of the top-2 candidates better explains the query.

    Returns True if the LLM prefers the SECOND candidate (index 1).
    """
    import json
    import urllib.request

    events_desc = "; ".join(
        f"E{i+1}: {e.description}" for i, e in enumerate(plan.events[:8])
    )
    asr_queries = ", ".join(plan.asr_queries[:4])

    def _summarise(c: NotebookCandidate) -> str:
        snippets = (c.asr_snippets or [])[:3]
        ts = (c.evidence_timestamps or [])[:5]
        parts = [f"video_id={c.video_id}", f"score={c.score:.4f}"]
        if c.event_coverage is not None:
            parts.append(f"coverage={c.event_coverage:.2f}")
        if c.temporal_consistency is not None:
            parts.append(f"temporal={c.temporal_consistency:.2f}")
        if ts:
            parts.append(f"timestamps={', '.join(ts)}")
        if snippets:
            parts.append(f"asr_snippets=[{'; '.join(s[:80] for s in snippets)}]")
        return " | ".join(parts)

    prompt = (
        "You are a video retrieval judge. Given a query and two candidate videos, "
        "decide which video better matches the query.\n\n"
        f"Query: {plan.raw_text}\n"
        f"Expected events: {events_desc}\n"
        f"ASR keywords: {asr_queries}\n\n"
        f"Candidate A: {_summarise(top2[0])}\n"
        f"Candidate B: {_summarise(top2[1])}\n\n"
        "Reply with ONLY a single character: 'A' if candidate A is better, "
        "'B' if candidate B is better. No explanation."
    )

    payload = json.dumps({
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 4},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{llm_base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())

    reply = (body.get("message", {}).get("content", "") or "").strip().upper()
    logger.info("Ranker: second pass LLM reply=%r", reply)
    return reply.startswith("B")


__all__ = [
    "rank_videos",
    "maybe_second_pass",
    "confidence_gap",
]
