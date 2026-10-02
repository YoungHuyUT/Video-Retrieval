"""LLM-backed QueryPlan planner (III, IV) — local, free, fast.

This module turns a raw query into a deterministic :class:`QueryPlan` using a
**local Ollama LLM** (default ``qwen3:0.6b`` — 522MB Q4, text-only, runs on
CPU/GPU locally, no paid API, per XXIV).  The LLM is prompted to emulate the
AI-City-2025 winning approach (arxiv 2512.12935v1): decompose the query into
modality-specific cues and assign *dynamic* per-modality weights based on how
distinctive each cue is for identifying the target frame.

Design constraints (from the user brief + XXIV):
* **Local only** — never calls a cloud/paid LLM.  Uses ``OllamaLLM``.
* **Fast** — small model, ``temperature=0``, bounded ``num_predict``.
* **Explainable** — the LLM returns strict JSON; the plan is logged verbatim
  (XXIII).  On any failure we fall back to the deterministic rule-based
  ``parse_query`` so retrieval never breaks (graceful degradation).
* **A/B-able** — ``plan_query`` accepts a ``use_llm`` flag so the rule-based and
  LLM paths can be benchmarked side by side (XX).

The :class:`QueryPlan` schema is the single contract both paths must satisfy,
so downstream retrieval code is unchanged regardless of which planner ran.
"""

from __future__ import annotations

import logging
from typing import Optional

from aic2026.agent.local_llm import LLMInvocationError, LocalLLM, OllamaLLM
from aic2026.query.plan import (
    Constraint,
    ConstraintKind,
    CountOp,
    Event,
    ModalityWeights,
    QueryPlan,
    RelationKind,
    TemporalRelation,
    TemporalRole,
)
from aic2026.query.parser import parse_query

logger = logging.getLogger(__name__)

# Lightweight, local, free TEXT model.  Per the user's "lựa cái nhẹ nhất" (pick
# the lightest), the DEFAULT planner is **qwen3:0.6b** (522MB Q4, text-only) —
# the smallest model available locally.  It produces valid QueryPlan JSON with
# ``think=False`` in ~23-29s warm and correctly splits action chains into
# start/middle/end events.  qwen2.5:1.5b (986MB) is kept as the fallback for
# when a higher-quality plan is wanted.  BOTH are TEXT models (no vision), so
# they plan queries without ever touching video frames — this respects the
# "no VLM on video" rule.  (The old "qwen3-vl:2b" fallback was removed: it is a
# vision model that is not reliably pulled locally and is slower than needed.)
DEFAULT_PLANNER_MODEL = "qwen3:0.6b"
_FALLBACK_PLANNER_MODEL = "qwen2.5:1.5b"

# Per-process cache so the same query is never planned twice (XVIII: cache
# expensive features).  Keyed on (text, model, base_url).
_PLAN_CACHE: dict[tuple[str, str, str], QueryPlan] = {}

_SYSTEM_PROMPT = """\
You are a video-retrieval query-understanding engine for a known-item-search \
benchmark (KIS / QA / TRAKE). Decompose a natural-language query into ONE \
strict JSON object that EXACTLY matches the schema below.

OUTPUT RULES (must follow or the plan is rejected):
- Respond with JSON ONLY. NO markdown fences, NO commentary, NO explanation.
- Use EXACTLY these top-level keys, no others:
  raw_text, global_query, language, entities, actions, attributes,
  constraints, events, temporal_relations, modalities
- Key order does not matter; do not add or rename keys.
- For enum fields use EXACTLY the listed values (case-sensitive, lowercase):
    language:        "en" | "vi" | "unknown"
    temporal_role:   "start" | "middle" | "end" | "first" | "last"
    constraint.kind: "count" | "attribute" | "temporal"
    constraint.operator: ">" | ">=" | "==" | "<" | "<="
    relation.kind:   "before" | "after" | "first" | "last" | "ordered"
- You MAY omit keys whose value would be empty. Set raw_text="".
- modality weights are floats in [0,1]; 0 = query gives no signal for that \
  modality, 1 = it is the primary discriminator. They need not sum to 1.

TEMPLATE (fill in, keep structure):
{
  "global_query": "<rewritten single search phrase>",
  "language": "en",
  "entities": ["glasses", "red door", "books", "table"],
  "actions": ["enters", "opens", "walks"],
  "attributes": ["red", "glasses"],
  "constraints": [
    {"kind": "attribute", "subject": "glasses", "raw": "wearing glasses"},
    {"kind": "count", "subject": "books", "operator": ">", "value": 3,
     "raw": "more than 3 books"}
  ],
  "events": [
    {"index": 0, "description": "a man wearing glasses enters a room",
     "entities": ["glasses", "person"], "actions": ["enters"],
     "attributes": ["glasses"], "temporal_role": "start"},
    {"index": 1, "description": "opens a red door and walks to a table",
     "entities": ["red door", "table"], "actions": ["opens", "walks"],
     "attributes": ["red"], "temporal_role": "middle"}
  ],
  "temporal_relations": [
    {"kind": "before", "source": 0, "target": 1}
  ],
  "modalities": {"semantic": 1.0, "temporal": 1.0, "ocr": 0.0,
                 "asr": 0.0, "object": 1.0, "count": 1.0}
}

GUIDANCE:
- entities = concrete objects/scenes; actions = verbs; attributes = colours or \
  visual properties.
- Split multi-event queries ("then", "after", "followed by", "starts with", \
  "ends with") into an ordered events[] list with temporal_role.
- If "only one X" or "exactly N X" -> count constraint with operator "==" and \
  value N. If "more/over/than N" -> operator ">". If "fewer/under/less than N" \
  -> operator "<".
- Set ocr weight 1.0 only if the query mentions visible text/signs/labels; \
  asr weight 1.0 only if spoken words/dialogue are implied. Otherwise 0.0.
"""


def _ollama_client(model: str, base_url: str, timeout: float) -> LocalLLM:
    # Small context + bounded output keeps the tiny model fast (user wants "lẹ").
    return OllamaLLM(
        model=model,
        base_url=base_url,
        temperature=0.0,
        num_predict=800,
        num_ctx=2048,
        think=False,  # keep it fast; we only need the structured plan
        timeout_seconds=timeout,
    )


def plan_query_llm(
    text: str,
    model: str = DEFAULT_PLANNER_MODEL,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 180.0,
) -> QueryPlan:
    """Build a QueryPlan with the local LLM.

    Tries ``model`` first; if that model is unavailable (not pulled) it retries
    once with ``_FALLBACK_PLANNER_MODEL``. Raises ``LLMInvocationError`` only if
    both fail.
    """
    models_to_try = [model]
    if model != _FALLBACK_PLANNER_MODEL:
        models_to_try.append(_FALLBACK_PLANNER_MODEL)
    last_err: Exception | None = None
    for m in models_to_try:
        try:
            client = _ollama_client(m, base_url, timeout_seconds)
            plan = client.structured(
                system=_SYSTEM_PROMPT,
                user=f"Query: {text}\n\nReturn the structured plan as JSON.",
                schema=QueryPlan,
            )
            if not plan.global_query:
                plan.global_query = text
            return plan
        except LLMInvocationError as exc:
            last_err = exc
            logger.warning("LLM planner model %s failed: %s", m, exc)
    raise last_err or LLMInvocationError("All planner models failed.")


def plan_query(
    text: str,
    use_llm: bool = True,
    model: str = DEFAULT_PLANNER_MODEL,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 180.0,
    fallback_on_error: bool = True,
) -> QueryPlan:
    """Plan a query. Uses the local LLM when ``use_llm`` is True, otherwise (or on
    any LLM failure) falls back to the deterministic rule-based parser.

    This is the single entry point the rest of the system should call. It is
    A/B-able: flip ``use_llm`` to compare planner quality on a dev set. Results
    are cached per (text, model, base_url) so the LLM is never called twice for
    the same query (XVIII).
    """
    cache_key = (text, model, base_url)
    if cache_key in _PLAN_CACHE:
        return _PLAN_CACHE[cache_key]
    plan: QueryPlan
    if use_llm:
        try:
            plan = plan_query_llm(text, model=model, base_url=base_url, timeout_seconds=timeout_seconds)
            _PLAN_CACHE[cache_key] = plan
            return plan
        except LLMInvocationError as exc:
            if not fallback_on_error:
                raise
            logger.warning("LLM planner failed (%s); using rule-based fallback.", exc)
    plan = parse_query(text)
    _PLAN_CACHE[cache_key] = plan
    return plan
