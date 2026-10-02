"""LLM-based query analyzer for KIS/QA (replaces deterministic expansion).

Uses a local lightweight LLM (qwen2.5:1.5b via Ollama) to decompose a query
into structured components that drive retrieval:

- ``events``: temporal segments for multi-event queries
- ``entities``: objects/things to detect (maps to detector vocabulary)
- ``actions``: verbs for temporal ordering
- ``attributes``: colors, materials, spatial relations
- ``expansion_variants``: 3-5 focused sub-queries for RRF fusion
- ``modality_hints``: which modalities (ASR/OCR/count) are relevant

Falls back to deterministic ``parse_query()`` on any LLM failure.

Design
------
- Local only (no paid API), fast (~200-400ms on CPU with 1.5B model).
- Structured JSON output via ``OllamaLLM.structured()``.
- Temperature 0 for reproducibility.
- System prompt is engineered for concise, focused output.
"""

from __future__ import annotations

import logging
from pydantic import BaseModel, Field

from aic2026.agent.local_llm import LLMInvocationError

logger = logging.getLogger(__name__)

DEFAULT_KIS_MODEL = "qwen2.5:1.5b"
_FALLBACK_KIS_MODEL = "qwen2.5:3b"

# Free LLM providers (no credit card required)
# Set env vars to enable: GEMINI_API_KEY or OPENROUTER_API_KEY
import os

# Per-process cache so the same query is never analyzed twice.
_ANALYSIS_CACHE: dict[str, QueryAnalysis] = {}


class QueryAnalysis(BaseModel):
    """Structured LLM output for KIS query understanding."""

    # Decomposed events (for multi-event / action-chain queries).
    events: list[str] = Field(default_factory=list)
    # Key objects/entities to detect (maps to detector vocabulary).
    entities: list[str] = Field(default_factory=list)
    # Action verbs for temporal ordering.
    actions: list[str] = Field(default_factory=list)
    # Colors, materials, spatial relations.
    attributes: list[str] = Field(default_factory=list)
    # Fine-grained visual details the user cares about (for VLM verification).
    # Each item is a specific check: "red shirt", "text on screen says X",
    # "person on the left", "2 dogs", "wearing glasses".
    fine_details: list[str] = Field(default_factory=list)
    # 3-5 focused sub-queries for RRF fusion (replaces deterministic expansion).
    expansion_variants: list[str] = Field(default_factory=list)
    # Relevance scores per variant (Improvement.md Task 4).  Each float in [0,1]
    # indicates how relevant the LLM considers that variant.  Used as weights
    # in weighted RRF fusion.  Defaults to [1.0, ...] when absent (backward compatible).
    variant_relevance_scores: list[float] = Field(default_factory=list)
    # Which modalities are relevant (ASR, OCR, count).
    use_asr: bool = False
    use_ocr: bool = False
    use_count: bool = False
    # Language hint.
    language: str = "unknown"


_SYSTEM_PROMPT = """\
You are a video-search query analyzer. Your job is to decompose a natural-language \
query into structured fields that drive video retrieval. Pay CLOSE attention to \
fine-grained visual details — colors, spatial positions, counts, text, clothing, \
facial expressions, objects held, background elements.

OUTPUT RULES:
- Respond with JSON ONLY. NO markdown fences, NO commentary, NO explanation.
- Use EXACTLY these keys: events, entities, actions, attributes, fine_details, \
expansion_variants, use_asr, use_ocr, use_count, language
- If a list would be empty, OMIT the key entirely (do not emit empty arrays).

GUIDANCE:
- events: Split multi-event/action-chain queries into temporal segments. \
For short single-scene queries, use the full query as ONE event.
- entities: Key objects/people/animals the query asks about. \
Use simple English nouns that a vision detector would label (e.g. "person", "car", "bottle").
- actions: Action verbs (e.g. "open", "walk", "sit").
- attributes: Colors, materials, spatial relations (e.g. "red", "outdoor", "left side").
- fine_details: SPECIFIC visual checks a VLM should verify on each frame. \
Be precise and atomic — each item is ONE checkable fact about the scene. \
Include: exact colors + objects ("red shirt", "blue car"), spatial positions \
("person on the left", "object in background"), counts ("3 people", "2 dogs"), \
clothing/accessories ("wearing glasses", "holding a phone"), text/labels \
("sign says EXIT"), expressions ("smiling", "looking surprised"). \
Extract ALL fine details the query implies, even implicit ones.
- expansion_variants: Generate 3-5 SHORT focused sub-queries for video retrieval. \
Each should be a standalone CLIP-friendly phrase describing a visual scene. \
Example: for "a person opens a red door then walks into a room", produce:
  ["person opening a red door", "person walking into a room", "red door", "person in a room"]
- variant_relevance_scores: A float in [0,1] for EACH expansion_variant, \
indicating how relevant that variant is to the original query. The first variant \
(closest to the original) should be 1.0. Others should be lower if they are \
partial/focused sub-queries. Example: [1.0, 0.8, 0.6, 0.5]. \
If all variants are equally relevant, use [1.0, 1.0, ...].
- use_asr: true if the query mentions speech, conversation, or something spoken.
- use_ocr: true if the query mentions text, signs, labels, or writing.
- use_count: true if the query asks about a specific number of objects.
- language: "en" | "vi" | "unknown"

EXAMPLES:

Query: "a red car parked near a building"
→ {"events":["a red car parked near a building"],"entities":["car","building"],"attributes":["red","parked"],"fine_details":["red car","car near building","parked car"],"expansion_variants":["red car parked near building","red car","car near building"],"variant_relevance_scores":[1.0, 0.8, 0.6],"language":"en"}

Query: "người đàn ông mặc áo trắng đứng bên trái trời mưa"
→ {"events":["người đàn ông mặc áo trắng đứng bên trái trời mưa"],"entities":["person"],"actions":["stand"],"attributes":["white","left","rainy"],"fine_details":["white shirt","person on the left side","rainy weather","wet ground"],"expansion_variants":["man in white shirt standing in rain","person on left side in rain","white shirt rainy scene"],"variant_relevance_scores":[1.0, 0.7, 0.5],"use_asr":false,"language":"vi"}

Query: "cô gái đội mũ đỏ đang cười cầm ly cà phê"
→ {"events":["cô gái đội mũ đỏ đang cười cầm ly cà phê"],"entities":["person","cup"],"actions":["hold"],"attributes":["red","smiling"],"fine_details":["red hat","girl smiling","holding coffee cup","girl face visible"],"expansion_variants":["girl with red hat smiling","person holding coffee cup","red hat girl coffee"],"variant_relevance_scores":[1.0, 0.6, 0.5],"use_asr":false,"language":"vi"}

Query: "what did the person say about the weather"
→ {"events":["person talking about weather"],"entities":["person"],"use_asr":true,"expansion_variants":["person speaking about weather","weather discussion"],"language":"en"}
"""


def _build_client(
    provider: str = "auto",
    model: str = DEFAULT_KIS_MODEL,
    base_url: str = "",
    timeout: float = 5.0,
):
    """Build Gemini LLM client (reads GEMINI_API_KEY or GOOGLE_API_KEY from env / .env)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from aic2026.agent.free_llm import GeminiFreeLLM

    api_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise LLMInvocationError(
            "GEMINI_API_KEY not found in environment or .env file. "
            "Get a free API key at https://aistudio.google.com/apikey"
        )
    return GeminiFreeLLM(api_key=api_key, model="gemini-3.6-flash", timeout_seconds=timeout)


def analyze_query_llm(
    query: str,
    model: str = DEFAULT_KIS_MODEL,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 5.0,
    provider: str = "auto",
) -> QueryAnalysis:
    """Analyze a query using LLM (auto-detect: Gemini > OpenRouter > Ollama).

    Uses ``model`` for Ollama; Gemini/OpenRouter use their default models.
    If unavailable or fails, raises LLMInvocationError so caller immediately
    falls back to deterministic parsing.
    """
    cache_key = f"{query}|{model}|{base_url}|{provider}"
    if cache_key in _ANALYSIS_CACHE:
        logger.debug("QueryAnalysis cache hit: %s...", query[:40])
        return _ANALYSIS_CACHE[cache_key]

    try:
        client = _build_client(provider, model, base_url, timeout_seconds)
        analysis = client.structured(
            system=_SYSTEM_PROMPT,
            user=f"Query: {query}",
            schema=QueryAnalysis,
        )
        # Ensure at least the original query is in expansion variants.
        if not analysis.expansion_variants:
            analysis.expansion_variants = [query]
        if not analysis.events:
            analysis.events = [query]
        _ANALYSIS_CACHE[cache_key] = analysis
        logger.info(
            "QueryAnalysis (LLM, %s): %d events, %d entities, %d variants",
            provider,
            len(analysis.events),
            len(analysis.entities),
            len(analysis.expansion_variants),
        )
        return analysis
    except LLMInvocationError as exc:
        logger.warning("KIS LLM analyzer (%s) failed: %s", provider, exc)
        raise exc


def analyze_query_fallback(query: str) -> QueryAnalysis:
    """Deterministic fallback using ``parse_query()`` (no LLM).

    Produces a QueryAnalysis that mirrors the deterministic expansion logic
    but in the LLM-compatible format.
    """
    from aic2026.query.parser import parse_query, _ACTION_CLAUSE_COMMA_RE
    from aic2026.query.expansion import expand_text

    plan = parse_query(query)

    # Events from the parser.
    events = [e.description for e in plan.events] if plan.events else [query]

    # Expansion variants from the deterministic path.
    try:
        variants = expand_text(query, max_variants=4)
    except Exception:
        variants = [query]

    # For multi-event queries, prepend clean event descriptions to variants
    # so individual visual concepts are scored directly by SigLIP2.
    if len(events) > 1:
        seen = {v.casefold() for v in variants}
        clean_events = [ev for ev in events if ev.casefold() not in seen]
        variants = clean_events + variants

    # Action-chain splitting: for multi-action queries, split on action commas
    # and add individual action phrases as expansion variants.  This ensures
    # action chains like "enters, opens, sits" get per-action recall branches.
    action_clauses = _ACTION_CLAUSE_COMMA_RE.split(query)
    if len(action_clauses) > 1:
        seen = {v.casefold() for v in variants}
        for clause in action_clauses:
            clause = clause.strip(" ,;|/\n\t-")
            if not clause:
                continue
            # Prepend "person" if the clause starts with a verb (detector-friendly).
            phrase = clause if len(clause.split()) > 1 else f"person {clause}"
            key = phrase.casefold()
            if key not in seen:
                seen.add(key)
                variants.append(phrase)

    # Detect modality hints from the parser's modality weights.
    use_asr = plan.modalities.asr > 0
    use_ocr = plan.modalities.ocr > 0
    use_count = plan.modalities.count > 0

    # Build fine_details from entities + attributes + spatial hints.
    # NOTE: For Vietnamese queries, filter out short entities that are false
    # positives from the parser's _requested_concepts — e.g. "ong"→bee,
    # "cua"→crab in "người đàn ông mở cửa". English queries are unaffected.
    from aic2026.reranking.lexical import _fold_accents
    if plan.language == "vi":
        safe_entities = [e for e in plan.entities if len(_fold_accents(e)) >= 5]
    else:
        safe_entities = list(plan.entities)
    fine_details: list[str] = []
    for ent in safe_entities:
        fine_details.append(ent)
    for attr in plan.attributes:
        fine_details.append(attr)
    # Combine entity+attribute pairs for richer checks.
    for attr in plan.attributes:
        for ent in safe_entities:
            if len(fine_details) < 8:
                fine_details.append(f"{attr} {ent}")

    return QueryAnalysis(
        events=events,
        entities=safe_entities,
        actions=plan.actions,
        attributes=plan.attributes,
        fine_details=fine_details[:8],
        expansion_variants=(variants or [query])[:4],
        use_asr=use_asr,
        use_ocr=use_ocr,
        use_count=use_count,
        language=plan.language or "unknown",
    )


def analyze_query(
    query: str,
    use_llm: bool = True,
    model: str = DEFAULT_KIS_MODEL,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 5.0,
    provider: str = "auto",
) -> QueryAnalysis:
    """Analyze a KIS query (single entry point).

    Uses the LLM when ``use_llm`` is True (default), falls back to the
    deterministic rule-based parser on any LLM failure. Results are cached.

    Provider priority (auto): Gemini > OpenRouter > Ollama.
    Set env vars GEMINI_API_KEY or OPENROUTER_API_KEY to use free APIs.
    """
    if use_llm:
        try:
            return analyze_query_llm(
                query, model=model, base_url=base_url,
                timeout_seconds=timeout_seconds, provider=provider,
            )
        except LLMInvocationError as exc:
            logger.warning(
                "KIS LLM analyzer failed (%s); using deterministic fallback.",
                exc,
            )

    return analyze_query_fallback(query)


__all__ = [
    "DEFAULT_KIS_MODEL",
    "QueryAnalysis",
    "analyze_query",
    "analyze_query_fallback",
    "analyze_query_llm",
]
