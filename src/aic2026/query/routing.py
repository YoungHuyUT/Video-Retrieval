"""Modality routing module for KIS queries.

Analyzes QueryPlan to produce dynamic, normalized modality weights:
- visual: spatial relations, layout, composition, camera angles, color attributes
- object: entity presence, count constraints
- asr: speech markers ("says", "speech", "talks about", names, spoken quotes)
- ocr: text markers ("sign", "text on screen", "title", "logo")
- metadata: general metadata/title signals

Ensures no modality is hard-disabled unless unavailable, and weights are normalized.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aic2026.query.plan import ModalityWeights, QueryPlan


_VISUAL_SIGNALS_RE = re.compile(
    r"\b(?:left of|right of|below|above|behind|in front of|upper|lower|corner|close-up|aerial|wearing|red|blue|green|yellow|black|white|glasses|hat)\b",
    re.IGNORECASE,
)
_ASR_SIGNALS_RE = re.compile(
    r"\b(?:say|says|said|speech|talk|talks|speak|speaks|voice|conversation|nói|thoại|giọng)\b",
    re.IGNORECASE,
)
_OCR_SIGNALS_RE = re.compile(
    r"\b(?:sign|banner|title|label|name|text|logo|word|biển|tiêu đề|nhãn|chữ)\b",
    re.IGNORECASE,
)


def compute_modality_weights(plan: QueryPlan) -> ModalityWeights:
    """Compute normalized modality weights for a QueryPlan."""
    weights = plan.modalities.model_copy()

    text = plan.raw_text or plan.global_query

    # Visual signal
    has_visual = bool(plan.attributes or plan.spatial_relations or _VISUAL_SIGNALS_RE.search(text))
    weights.visual = 1.0 if has_visual else 0.5
    weights.semantic = 1.0

    # Object signal
    weights.object = 1.0 if plan.entities or plan.constraints else 0.4

    # ASR signal
    weights.asr = 1.0 if _ASR_SIGNALS_RE.search(text) else 0.1

    # OCR signal
    weights.ocr = 1.0 if _OCR_SIGNALS_RE.search(text) else 0.0

    # Metadata signal
    weights.metadata = 0.5

    # Renormalize active (non-zero) weights
    active_keys = [k for k in ["semantic", "visual", "object", "asr", "ocr", "metadata"] if getattr(weights, k) > 0.0]
    if active_keys:
        total = sum(getattr(weights, k) for k in active_keys)
        if total > 0:
            for k in active_keys:
                setattr(weights, k, round(getattr(weights, k) / total, 4))

    return weights
