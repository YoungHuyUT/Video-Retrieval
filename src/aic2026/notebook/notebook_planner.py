"""NOTEBOOK query planner (Improvement.md §540-657, §633-657).

Turns a raw natural-language query into a :class:`NotebookPlan` using a local
Ollama LLM (qwen2.5:1.5b, no-think). The plan is the SINGLE source of truth
for retrieval and scoring — never the raw query string.

Design (spec §11, §12, §20, §21):
- Local only (no paid API).
- Fast: small model, temperature=0, bounded output (~250-300 tokens).
- No thinking (think=False).
- JSON output only.
- Falls back to rule-based parsing on any LLM failure.
- Default model: qwen2.5:1.5b (user explicitly chose this over qwen3:0.6b).

The planner MUST distinguish:
  ASR-searchable information (entities, ingredients, concepts that might
  appear in spoken transcript)
  vs
  visual-only clues (spatial layout, colors, counts, camera view — these
  belong to visual_clues, NOT asr_queries).

Spec §3: ASR transcripts are Vietnamese. The asr_queries must be in Vietnamese
so they can match the ASR sidecar text.
"""

from __future__ import annotations

import logging
import re

from aic2026.agent.local_llm import LLMInvocationError, OllamaLLM
from aic2026.notebook.types import (
    NotebookPlan,
    NotebookEvent,
    NotebookTemporalRelation,
)

logger = logging.getLogger(__name__)

# User explicitly chose qwen2.5:1.5b over the spec's default qwen3:1.7b.
# qwen3:0.6b was deleted ("xoa cai 0.6b nha").
DEFAULT_NOTEBOOK_MODEL = "qwen2.5:1.5b"
_FALLBACK_NOTEBOOK_MODEL = "qwen2.5:3b"

# Per-process cache so the same query is never planned twice (spec §18).
_PLAN_CACHE: dict[tuple[str, str | None, str, str], NotebookPlan] = {}


_SYSTEM_PROMPT = """\
You are a video-search query-understanding engine for a "NotebookLM Lite" retrieval
system. Your job is to decompose a long, complex natural-language query into a
structured plan that drives ASR (speech) and visual retrieval.

OUTPUT RULES (must follow or the plan is rejected):
- Respond with JSON ONLY. NO markdown fences, NO commentary, NO explanation.
- Use EXACTLY these top-level keys, no others:
  raw_text, language, events, asr_queries, visual_clues, objects, constraints,
  temporal_relations, modality_weights, has_strict_temporal, use_visual
- For enum fields use EXACTLY the listed values (case-sensitive, lowercase):
  language:           "en" | "vi" | "unknown"
  temporal_role:      "start" | "middle" | "end" | "first" | "last"
  relation.kind:      "before" | "after" | "first" | "last" | "ordered"
- If a list would be empty, OMIT the key entirely (do not emit empty arrays).
- modality_weights: an OBJECT with keys "visual", "asr", "object" and float values in [0, 1]. Example: {"visual":1.0,"asr":0.8,"object":0.5}
- has_strict_temporal: true when the query expresses a strict ordering.
- use_visual: false (first version is ASR-only per spec §10).

GUIDANCE:
- ASR-searchable = entities/concepts in spoken transcript.
- Visual-only = spatial layout, colors, counts, camera view, scene description.
- ASR is Vietnamese: asr_queries in Vietnamese.
"""


def _ollama_client(model: str, base_url: str, timeout: float) -> OllamaLLM:
    """Build Ollama client for the notebook planner (think=False, ~300 tokens)."""
    return OllamaLLM(
        model=model,
        base_url=base_url,
        temperature=0.0,
        num_predict=300,
        num_ctx=2048,
        think=False,
        timeout_seconds=timeout,
        keep_alive="-1",
    )


def plan_notebook_llm(
    text: str,
    asr_query: str | None = None,
    model: str = DEFAULT_NOTEBOOK_MODEL,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 15.0,
) -> NotebookPlan:
    """Build a NotebookPlan with the local LLM.

    Tries ``model`` first; falls back to ``_FALLBACK_NOTEBOOK_MODEL`` if unavailable.
    """
    models_to_try = [model]
    if model != _FALLBACK_NOTEBOOK_MODEL:
        models_to_try.append(_FALLBACK_NOTEBOOK_MODEL)

    last_err: Exception | None = None
    for m in models_to_try:
        try:
            client = _ollama_client(m, base_url, timeout_seconds)
            plan = client.structured(
                system=_SYSTEM_PROMPT,
                user=f"Query: {text}\n\n"
                     f"ASR query (Vietnamese): {asr_query or text}\n\n"
                     f"Return the structured plan as JSON.",
                schema=NotebookPlan,
            )
            if not plan.raw_text:
                plan.raw_text = text
            return plan
        except LLMInvocationError as exc:
            last_err = exc
            logger.warning("Notebook LLM planner model %s failed: %s", m, exc)

    raise last_err or LLMInvocationError("All notebook planner models failed.")


def _rule_based_notebook_plan(
    text: str,
    asr_query: str | None = None,
) -> NotebookPlan:
    """Deterministic fallback: adapt :func:`parse_query` output into NotebookPlan.

    Guarantees the pipeline never breaks when the LLM is unavailable.
    """
    from aic2026.query.parser import parse_query

    base_plan = parse_query(text)
    vi_text = asr_query or text

    # --- Events ---
    # NOTE: key_concepts use actions only (safe). Entities from the parser
    # have false positives for Vietnamese ("ong"→bee, "cua"→crab) so we
    # exclude them — event coverage matching uses description text instead.
    events: list[NotebookEvent] = []
    for i, ev in enumerate(base_plan.events):
        events.append(
            NotebookEvent(
                index=i,
                description=ev.description,
                vi_text=vi_text,
                en_text="",
                key_concepts=list(ev.actions),
                temporal_role=ev.temporal_role.value,
            )
        )
    if not events:
        events.append(
            NotebookEvent(
                index=0,
                description=text,
                vi_text=vi_text,
                key_concepts=base_plan.entities,
                temporal_role="middle",
            )
        )

    # --- ASR queries: use original text + event descriptions (Vietnamese) ---
    # NOTE: We do NOT use base_plan.entities directly because the parser's
    # _requested_concepts() produces false positives for Vietnamese — e.g.
    # "ong" in "người đàn ông" matches "bee", "cua" in "mở cửa" matches "crab".
    # Instead, use the raw query text and event descriptions which are reliable.
    asr_queries: list[str] = []
    # Primary: the original query (or asr_query override) — most reliable for ASR.
    asr_queries.append(vi_text)
    # Secondary: each event description (also Vietnamese, from the parser's segmentation).
    for ev in events:
        desc = ev.description.strip()
        if desc and desc != vi_text:
            asr_queries.append(desc)
    # Add actions (verbs are generally safe — "mở", "đi", "ngồi" etc.)
    for action in base_plan.actions:
        if len(action) >= 2:
            asr_queries.append(action)
    # NOTE: entities intentionally skipped — parser false positives for VI
    # ("ong"→bee, "cua"→crab) corrupt ASR retrieval.

    # --- Visual clues: spatial/layout/color patterns ---
    visual_clues: list[str] = []
    spatial_patterns = [
        r"(?:goc|corner|left|right|trai|phai|tren|duoi|tren cung|duoi cung)",
        r"(?:bo co|layout|arrangement)",
        r"(?:mau|color|do|den|trang|xanh|vang|nam|tim|hong)",
        r"(?:so luong|co \d+|\\d+ (?:cai|chiec|nguoi|vat))",
    ]
    for pat in spatial_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            visual_clues.extend(matches)

    # NOTE: base_plan.entities has false positives for Vietnamese (parser bug).
    # Use visual_clues + actions as safer object proxies.
    objects = list(set(base_plan.actions) - {""})
    constraints = [c.raw for c in base_plan.constraints if c.raw]

    # --- Temporal relations ---
    temporal_relations: list[NotebookTemporalRelation] = []
    for rel in base_plan.temporal_relations:
        temporal_relations.append(
            NotebookTemporalRelation(
                kind=rel.kind.value,
                source=rel.source,
                target=rel.target,
            )
        )

    # --- Modality weights ---
    has_speech_signal = bool(
        base_plan.modalities.asr > 0
        or any(
            keyword in text.lower()
            for keyword in ("noi", "thoai", "giuong", "tro chuyen",
                            "speech", "said", "talk", "speak", "voice", "conversation")
        )
    )
    has_visual_clues = len(visual_clues) > 0
    modality_weights = {
        "visual": 1.0 if has_visual_clues or objects else 0.8,
        "asr": 1.0 if has_speech_signal else 0.7,
        "object": 0.5 if objects else 0.0,
    }

    has_strict_temporal = (
        len(temporal_relations) > 0 or base_plan.is_multi_event
    )

    return NotebookPlan(
        raw_text=text,
        language=base_plan.language,
        events=events,
        asr_queries=asr_queries,
        visual_clues=visual_clues,
        objects=objects,
        constraints=constraints,
        temporal_relations=temporal_relations,
        modality_weights=modality_weights,
        has_strict_temporal=has_strict_temporal,
        use_visual=False,
    )


def plan_notebook(
    text: str,
    asr_query: str | None = None,
    use_llm: bool = True,
    model: str = DEFAULT_NOTEBOOK_MODEL,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 15.0,
) -> NotebookPlan:
    """Plan a NOTEBOOK query (single entry point).

    Uses the LLM when ``use_llm`` is True (default), falls back to the
    deterministic rule-based parser on any LLM failure. Results are cached
    per (text, asr_query, model, base_url) so the LLM is never called twice
    for the same query (spec §18).
    """
    cache_key = (text, asr_query, model, base_url, use_llm)
    if cache_key in _PLAN_CACHE:
        logger.debug("NotebookPlan cache hit for query: %s...", text[:60])
        return _PLAN_CACHE[cache_key]

    plan: NotebookPlan
    if use_llm:
        try:
            plan = plan_notebook_llm(
                text, asr_query, model=model, base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
            logger.info(
                "NotebookPlan (LLM): %d events, %d asr_queries, %d visual_clues",
                len(plan.events), len(plan.asr_queries), len(plan.visual_clues),
            )
            _PLAN_CACHE[cache_key] = plan
            return plan
        except LLMInvocationError as exc:
            logger.warning(
                "Notebook LLM planner failed (%s); using rule-based fallback.",
                exc,
            )

    plan = _rule_based_notebook_plan(text, asr_query)
    logger.info(
        "NotebookPlan (rule-based): %d events, %d asr_queries, %d visual_clues",
        len(plan.events), len(plan.asr_queries), len(plan.visual_clues),
    )
    _PLAN_CACHE[cache_key] = plan
    return plan


__all__ = [
    "DEFAULT_NOTEBOOK_MODEL",
    "plan_notebook",
    "plan_notebook_llm",
    "NotebookPlan",
]
