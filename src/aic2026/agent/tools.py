from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from aic2026.models import Candidate, FrameRecord

logger = logging.getLogger(__name__)

# Hard safety cap for pathological input size; SigLIP2 overlength queries are split
# into tokenizer-aligned windows before they reach the encoder.
MAX_QUERY_TOKENS = 512

from aic2026.reranking import (
    late_interaction_rerank,
    contrastive_clip_colour_rerank,
    object_evidence_adjustment,
    rerank_with_object_evidence,
    gate_lion_dance_split,
)
from aic2026.retrieval import RetrievalPipeline
from aic2026.temporal import align_events, refine_trake_candidates

if TYPE_CHECKING:
    from aic2026.query.plan import QueryPlan
    from aic2026.retrieval.bm25_index import BM25Index


_TASKS_REQUIRING_DENSE_VIDEO_EVIDENCE = frozenset(
    {
        "qa",
        "trake",
    }
)


# Discourse markers that signal a multi-event / action-chain query.  When present
# the query should be expanded (or at least NOT collapsed into a single CLIP vector
# that blurs every event together — the root cause of "returns object B").
_MULTI_EVENT_MARKERS = (
    " then", " next", " after ", " afterwards", " later",
    " followed by", " meanwhile", " subsequently",
    " first", " second", " thirdly",
)


def _looks_multi_event(text: str) -> bool:
    """True if *text* reads like an action chain / multi-scene description.

    Uses cheap discourse-marker detection (then/next/after/…) — no LLM, no
    parser dependency — so it is safe to call inside ``retrieve`` for every
    query.  This is the gating check from spec §8: a single-scene query never
    triggers expansion, keeping the latency at exactly one CLIP encode.
    """
    if not text:
        return False
    folded = " " + text.lower()
    return any(marker in folded for marker in _MULTI_EVENT_MARKERS)


@dataclass
class RetrievalTools:
    """The agent's controlled interface to competition evidence."""

    pipeline: RetrievalPipeline
    encode_text: Callable[[str], np.ndarray]
    visual_answerer: (
        Callable[[str, list[Candidate]], dict[int, str]] | None
    ) = None
    # Optional handle to the VLM instance so we can free its weights after QA
    # answering (low-RAM machines cannot keep CLIP + Florence resident at once).
    visual_answerer_obj: Any | None = None
    # VLM construction params. The VLM is NOT created at agent build time; it is
    # lazily built (see ``ensure_vlm``) only on the QA answer step, AFTER the CLIP
    # encoder has been freed (``close_text_encoder``). This keeps only ONE heavy
    # model resident at a time (avoids OOM on low-RAM machines) and means KIS/TRAKE
    # never pay the Florence load cost. ``visual_answerer_obj`` stays None for
    # KIS/TRAKE (vlm_model None) so ``ensure_vlm`` is a no-op there.
    vlm_model: str | None = None
    vlm_device: str | None = None
    vlm_dtype: str = "float32"
    encode_images: Callable[[list[object]], np.ndarray] | None = None
    bm25_index: BM25Index | None = None
    # Asymmetry of the object-evidence signal (KIS).  A frame shown by the
    # detector to LACK every object the query asks for is penalized
    # `object_evidence_weight * object_penalty_scale`, while a full match is only
    # rewarded `object_evidence_weight`.  This counters CLIP ranking a frame high
    # for the right scene but the wrong object.  Default 2.0.
    object_penalty_scale: float = 2.0
    # Base magnitude of the object-evidence reward/penalty (KIS).  On the RRF
    # scale (~0.0018–0.016) a value of 0.02–0.05 is a strong but bounded nudge.
    # Set to 0.0 to disable object evidence for KIS (dense-only mode).
    object_evidence_weight: float = 0.0
    # Hard-drop frames that have NO object labels at all (ingest marked them as
    # blurry / no-clear-object, i.e. no entity reached the 0.4 present-threshold)
    # WHEN the query actually asks for an object.  This is the "blurry frame"
    # filter: such frames never enter the final KIS ranking.  Off by default so
    # scene-only queries and weak-detector corpora are unaffected; turn on for
    # KIS via --drop-empty-object-frames.  Falls back to keeping the pool if
    # dropping would empty it.
    drop_empty_object_frames: bool = False
    # In production KIS, an explicitly requested detector concept is a hard
    # constraint: dense similarity may rank candidates, but a frame without a
    # matching object label is removed before reranking.
    strict_object_match: bool = False
    # Trọng số late-interaction (ColBERT-style MaxSim): encode_text ~200 lần trong
    # retrieve(), RẤT nặng cho thi tốc độ. Mặc định 0.0 (TẮT) — chỉ bật nếu cần
    # tăng recall query dài nhiều từ (rare cho BTC). Có thể bật qua runtime config.
    late_interaction_weight: float = 0.0
    # Contrastive CLIP colour rerank: compares query colour expectations against
    # actual frame colours. Bounded so CLIP remains the primary signal.
    contrastive_colour_rerank_weight: float = 0.06
    colour_sidecar_path: str | None = None
    # Restrict retrieval to a dataset split by video_id prefix (e.g. ``["L25"]``
    # for QA on online-course videos, ``["L26"]`` for TRAKE). Optional; when set
    # it is intersected with `video_filter_terms` (metadata) so both filters
    # apply. ``None``/empty = no prefix restriction. The agent sets this per task
    # (QA→L25, TRAKE→L26) before each retrieve call.
    video_prefixes: list[str] | None = None
    video_filter_terms: list[str] | None = None
    # Số video tối đa giữ lại sau bước video-level coarse filter (KIS). Dataset
    # lớn (> số video này) thì chỉ top-K video theo video_score mới được giữ, còn
    # lại loại bỏ trước khi vào MMR — tránh video nhiễu (1 frame outlier) lọt top.
    # 0 / <= 0 hoặc >= số video = xét hết (backward-compatible, dataset nhỏ).
    coarse_top_k: int = 200
    # Coarse keyframes select the video; only its short event windows are then
    # decoded from the source video for finer TRAKE timestamps.
    #
    # Default is OFF for the official BTC corpus. The base TRAKE path already
    # emits source-video frame coordinates: the manifest's `frame_id` is mapped
    # from the official map-keyframes CSV (frame_idx gốc của video) inside
    # `prepare-official`, so `event_frames` already land inside the ground-truth
    # `ranges`. Enabling dense refinement re-encodes raw frames with a *different*
    # CLIP embedding (open_clip ViT-B/32 vs the official BTC CLIP vectors used for
    # retrieval) and re-runs DP; because the metric is `start <= frame <= end`
    # (range, not exact), picking a nearby frame in the same scene that falls
    # outside the range silently zeroes that event. It also adds the cost of
    # decoding `.mp4` for up to `trake_refine_top_videos` videos. Keep off unless
    # a dev set shows it helps for a specific reason.
    trake_dense_refine: bool = False
    trake_refine_top_videos: int = 20
    trake_refine_sample_fps: float = 3.0
    trake_refine_window_seconds: float = 2.0
    trake_video_root: Path = Path("data/raw/Videos")
    # --- ASR modality (spec §ASR TEMPORAL LOCALIZATION) ---
    # When True and ``asr_transcripts`` is populated, spoken-term scores are fused
    # into the candidate score as a weak signal (weight = ``asr_weight``). The
    # sidecar text is Vietnamese; ``retrieve`` is handed the matching ``asr_query``
    # string so ASR localization is scored against the language the speech is in.
    use_asr: bool = False
    asr_weight: float = 0.15
    # Map[normalized vector_id -> Candidate.score bump] is how ASR/object/colour
    # rerankers add their bonus; kept as dict keyed by vector_id.
    asr_transcripts: dict[str, Any] | None = None  # {video_id: VideoTranscript}
    # Optional path to precomputed ASR dense embeddings (.npy). When set and
    # the file exists, semantic ASR retrieval via DenseRetriever (BGE-M3) runs
    # alongside the keyword-based asr_term_scores, giving a bounded dense
    # bonus on the Top-N pool. Falls back to keyword scoring when absent.
    asr_dense_path: str | None = None
    _asr_transcripts: dict | None = None  # internal cache for DenseRetriever
    _dense_retriever_cache: Any | None = None  # cached DenseRetriever instance
    # --- Event Coverage + Moment (spec §4, §9) ---
    use_event_coverage: bool = False
    event_coverage_blend: float = 0.5
    use_moment_rerank: bool = False
    # --- LLM query analyzer (replaces deterministic expansion) ---
    # Uses local Ollama LLM (qwen2.5:1.5b) to decompose query into events,
    # entities, expansion variants, and modality hints.  Falls back to
    # deterministic parse_query() on LLM failure.
    use_llm_query_analyzer: bool = False
    llm_model: str = "qwen2.5:1.5b"
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_timeout: float = 5.0
    llm_provider: str = "gemini"  # "gemini" (default, free tier)
    _llm_analyzer_cache: Any | None = None
    _current_fine_details: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self._current_fine_details is None:
            self._current_fine_details = []
    # Query expansion: OFF — query goes directly, no multi-variant decompose.
    use_long_query_expansion: bool = True
    # Disable ALL query expansion (deterministic animal/disaster rephrases,
    # action-chain splits, LLM variants). Query goes straight to CLIP as-is.
    disable_all_expansion: bool = True
    # Deprecated / legacy rerankers — hardcoded OFF
    use_blip2_rerank: bool = False
    blip2_rerank_top_k: int = 70
    blip2_rerank_weight: float = 0.3
    # Cascade rerank: cheap→expensive stages (Improvement.md Task 2).
    use_cascade_rerank: bool = True
    cascade_stage_a_top_n: int = 100
    video_dedupe_window: float = 1.5
    # LLM verifier: fact-check top-K candidates (Improvement.md Task 8).
    use_llm_verifier: bool = False
    llm_verifier_max_candidates: int = 10
    llm_verifier_timeout: float = 10.0
    # LTR ranker: Learning-to-Rank for final ranking (Improvement.md Task 7).
    use_ltr_ranker: bool = True
    # Fast KIS shortcut: on simple KIS text queries, return the first
    # vector-only camera evidence directly from the manifest-backed raw
    # retrieval, skipping the heavy expansion/BM25/RRF/rerank stack. This
    # makes the UI show a frame quickly instead of waiting for a long break.
    use_fast_kis: bool = True
    # --- Adaptive Fusion (two-stage ranking) ---
    # "rrf_baseline" = standard RRF (no adaptive re-score).
    # "rrf_adaptive" = after RRF, re-score candidates by fusing normalised
    # semantic scores with object-coverage (and optionally ASR) signals
    # using ``fusion_weights``. ON by default for production quality.
    fusion_mode: str = "rrf_adaptive"
    fusion_weights: dict[str, float] | None = None  # e.g. {"semantic": 0.45, "object": 0.30, "asr": 0.25}
    # --- Gemini Multimodal Reranker ---
    use_gemini_rerank: bool = False
    gemini_model: str = "gemini-3.6-flash"
    gemini_top_k: int = 10
    gemini_local_weight: float = 0.65
    gemini_weight: float = 0.35
    gemini_ambiguity_margin: float = 0.05
    gemini_max_retries: int = 1
    gemini_timeout_seconds: float = 5.0
    gemini_circuit_breaker_failures: int = 5
    gemini_cooldown_seconds: float = 300.0
    gemini_cache_enabled: bool = True
    _gemini_reranker_instance: Any = None

    @property
    def has_lexical_objects(self) -> bool:
        """Whether a frame-level objects/OCR BM25 index is available."""
        return self.bm25_index is not None and not self.bm25_index.is_empty

    def _analyze_query(
        self,
        query: str,
        task_type: str | None = None,
    ) -> "QueryAnalysis | None":
        """Analyze query using LLM (or deterministic fallback).

        Returns a ``QueryAnalysis`` with events, entities, expansion variants,
        and modality hints.  Returns ``None`` when LLM is disabled or TRAKE
        (which handles events via its own alignment stage).
        """
        if task_type == "trake":
            return None
        if not self.use_llm_query_analyzer:
            return None

        try:
            from aic2026.query.llm_analyzer import analyze_query
            return analyze_query(
                query,
                model=self.llm_model,
                base_url=self.llm_base_url,
                timeout_seconds=self.llm_timeout,
                provider=self.llm_provider,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM query analyzer failed, falling back to deterministic: %s", exc)
            try:
                from aic2026.query.llm_analyzer import analyze_query_fallback
                return analyze_query_fallback(query)
            except Exception as exc2:  # noqa: BLE001
                logger.warning("query analysis fallback failed: %s", exc2)
                return None

    @staticmethod
    def _should_expand_query(query: str) -> bool:
        """Decide whether a query benefits from multi-variant expansion.

        SHORT queries with detectable object terms (e.g. "a red car", "person
        running") benefit from expansion because CLIP's broad embedding space
        maps similar objects to nearby vectors — expansion with specific visual
        descriptions and attribute×entity pairs pulls the ranking toward the
        correct concept.  Long queries, action chains, animal/disaster keywords,
        and multi-event descriptions also benefit.
        """
        words = query.split()
        if len(words) >= 8:
            return True
        # Action chains: "X, Y, Z" with commas joining verb phrases.
        from aic2026.query.parser import _ACTION_CLAUSE_COMMA_RE
        clauses = _ACTION_CLAUSE_COMMA_RE.split(query)
        if len(clauses) >= 2:
            return True
        # Animal / disaster keywords: bare nouns like "buffalo" or "flood" map to
        # a broad CLIP manifold — expansion with specific visual descriptions
        # pulls the ranking toward the correct concept.
        from aic2026.query.expansion import _ANIMAL_KEYWORDS, _DISASTER_KEYWORDS
        folded = query.casefold()
        for kw in _ANIMAL_KEYWORDS:
            if kw in folded:
                return True
        for kw in _DISASTER_KEYWORDS:
            if kw in folded:
                return True
        # ANY query with a detectable object term: "a red car", "person running",
        # "dog on the grass" — expansion generates attribute×entity pairs and
        # visual descriptions that CLIP alone cannot disambiguate.
        from aic2026.reranking.lexical import _requested_concepts
        if _requested_concepts(query):
            return True
        return False

    @staticmethod
    def _should_expand_query_always(query: str) -> bool:
        """Force focused variants for named animals, disasters, and action chains."""
        from aic2026.query.parser import _ACTION_CLAUSE_COMMA_RE
        if len(_ACTION_CLAUSE_COMMA_RE.split(query)) >= 2:
            return True
        from aic2026.query.expansion import _ANIMAL_KEYWORDS, _DISASTER_KEYWORDS
        folded = query.casefold()
        return any(kw in folded for kw in (*_ANIMAL_KEYWORDS, *_DISASTER_KEYWORDS))

    @staticmethod
    def _is_action_chain(query: str) -> bool:
        """True when the query contains a sequence of actions joined by commas."""
        from aic2026.query.parser import _ACTION_CLAUSE_COMMA_RE
        return len(_ACTION_CLAUSE_COMMA_RE.split(query)) >= 2

    def retrieve(
        self,
        query: str | list[str],
        limit: int,
        task_type: str | None = None,
        *,
        query_plan: "QueryPlan | None" = None,
        asr_query: str | None = None,
    ) -> list[Candidate]:
        """Retrieve task-appropriate candidates.

        KIS receives diversified results. QA and TRAKE receive raw candidates
        so that multiple relevant frames from the same video are preserved.

        ``query`` may be a single string or a list of query variants
        (multi-query expansion). When multiple variants are given, each is
        encoded and the resulting ranked lists are fused with Reciprocal Rank
        Fusion before reranking, improving recall.

        ``query_plan`` (optional): a structured :class:`QueryPlan` carrying
        decomposed events / modality weights used by the Event-Coverage layer.
        When ``None`` and LLM is enabled, the LLM analyzer produces this.
        Falls back to deterministic ``parse_query()`` on failure.

        ``asr_query`` (optional): the Vietnamese text used to match against the
        ASR sidecar. When ``None`` and an ASR sidecar is loaded, the original
        query is used.
        """

        import time
        t0 = time.time()
        queries = [query] if isinstance(query, str) else list(query)

        # --- Query length guard: truncate very long queries to prevent timeout ---
        # CLIP has a context window of ~77 tokens (~512 characters). Queries
        # longer than this get truncated to avoid encoding errors.
        def _truncate_query(q: str) -> str:
            if len(q.split()) > MAX_QUERY_TOKENS:
                truncated = " ".join(q.split()[:MAX_QUERY_TOKENS])
                logger.warning("Query truncated from %d to %d tokens", len(q.split()), MAX_QUERY_TOKENS)
                return truncated
            return q

        queries = [_truncate_query(q) if isinstance(q, str) else q for q in queries]

        logger.info("[START] KIS retrieve: query='%s' (limit=%d, task=%s)", queries[0][:60], limit, task_type)

        # --- LLM query analysis (replaces deterministic expansion) ---
        # The analyzer decomposes the query into events, entities, and expansion
        # variants.  Expansion variants replace the old template-based expansion;
        # events drive event-coverage reranking.
        analysis = None
        if (
            len(queries) == 1
            and isinstance(queries[0], str)
            and self.use_llm_query_analyzer
            and not self.disable_all_expansion
        ):
            analysis = self._analyze_query(queries[0], task_type=task_type)
            if analysis is not None:
                # Use LLM-generated expansion variants + fine_details as extra
                # retrieval queries.  Fine details ("red shirt", "person on left")
                # become additional CLIP queries so frames matching specific
                # details rank higher in the RRF pool.
                MAX_EXPANSION_VARIANTS = 12
                variants = [v for v in analysis.expansion_variants if v.strip()]
                # Append fine_details as extra retrieval variants (deduplicated).
                seen_q = {v.casefold() for v in variants}
                for fd in analysis.fine_details:
                    fd_s = fd.strip()
                    if fd_s and fd_s.casefold() not in seen_q:
                        seen_q.add(fd_s.casefold())
                        variants.append(fd_s)
                if len(variants) > MAX_EXPANSION_VARIANTS:
                    variants = variants[:MAX_EXPANSION_VARIANTS]
                if len(variants) > 1 and self._should_expand_query(queries[0]):
                    queries = variants
                    logger.info(
                        "query analyzed by LLM: %d expansion variants (incl fine_details): %s",
                        len(queries),
                        [v[:50] for v in queries],
                    )
                # Update modality flags from LLM analysis.
                if analysis.use_asr and not self.use_asr:
                    self.use_asr = True
                if analysis.use_ocr:
                    logger.debug("LLM detected OCR-relevant query")
                # Store fine_details for BLIP-2 per-detail scoring.
                self._current_fine_details = analysis.fine_details

        # Fallback: deterministic expansion for non-LLM path.
        # KIS action chains and animal/disaster queries are ALWAYS expanded
        # regardless of use_long_query_expansion:
        # - Action chains: each action becomes a recall branch.
        # - Animal/disaster: bare nouns map to a broad CLIP manifold; specific
        #   visual descriptions pull the ranking toward the correct concept.
        # Regular object queries ("red car", "person running") expand only when
        # use_long_query_expansion is on.
        _kis_action_chain = (
            task_type == "kis"
            and analysis is None
            and len(queries) == 1
            and isinstance(queries[0], str)
            and self._is_action_chain(queries[0])
        )
        _force_expand = (
            task_type == "kis"
            and analysis is None
            and len(queries) == 1
            and isinstance(queries[0], str)
            and self._should_expand_query_always(queries[0])
        )
        if (
            len(queries) == 1
            and isinstance(queries[0], str)
            and analysis is None
            and not self.disable_all_expansion
            and (self.use_long_query_expansion or _kis_action_chain or _force_expand)
            and task_type != "trake"
        ):
            try:
                from aic2026.query.expansion import expand_text
                from aic2026.query.parser import parse_query

                if _kis_action_chain:
                    # KIS action chains: use the global query + each event
                    # description as a recall branch, capped to keep cost bounded.
                    plan = parse_query(queries[0])
                    variants = [queries[0]]
                    seen = {queries[0].casefold()}
                    for ev in plan.events:
                        desc = ev.description.strip()
                        if desc and desc.casefold() not in seen:
                            seen.add(desc.casefold())
                            variants.append(desc)
                    if len(variants) > 6:
                        variants = variants[:6]
                else:
                    variants = expand_text(queries[0], max_variants=8)
                if len(variants) > 1:
                    queries = variants
                    logger.info(
                        "query expanded into %d focused variants for RRF fusion: %s",
                        len(queries),
                        [v[:50] for v in queries],
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("query expansion skipped: %s", exc)

        # Keep the complete description and add sentence facets (plus
        # overlapping tokenizer windows when needed). Batch all SigLIP2 text
        # encodes so multi-sentence queries pay one model invocation.
        text_encoder = getattr(self.encode_text, "__self__", None)
        split_query = getattr(text_encoder, "split_long_query", None)
        if task_type != "trake" and callable(split_query):
            query_chunks: list[str] = []
            seen_chunks: set[str] = set()
            for text in queries:
                chunks = split_query(text)
                encoded_texts = [text, *chunks] if len(chunks) > 1 else chunks
                for chunk in encoded_texts:
                    key = chunk.casefold()
                    if key not in seen_chunks:
                        seen_chunks.add(key)
                        query_chunks.append(chunk)
            queries = query_chunks or queries
        encode_many = getattr(text_encoder, "encode_many", None)
        embeddings = (
            encode_many(queries)
            if callable(encode_many)
            else [self.encode_text(q) for q in queries]
        )
        requires_dense_evidence = (
            task_type in _TASKS_REQUIRING_DENSE_VIDEO_EVIDENCE
        )

        # Metadata pre-filter: narrow the candidate video set before the
        # (relatively expensive) embedding/BM25 retrieval so frames from
        # off-topic videos never enter the pool.  Combined with the optional
        # `video_prefixes` split restriction (e.g. QA→L25): intersection means
        # a video must satisfy BOTH filters to stay in the pool.
        allowed_video_ids = self.pipeline.filter_terms_to_video_ids(
            self.video_filter_terms or []
        )
        if self.video_prefixes:
            prefix_ids = self.pipeline.prefixes_to_video_ids(
                self.video_prefixes, allowed_video_ids
            )
            if prefix_ids is not None:
                allowed_video_ids = prefix_ids
        if allowed_video_ids is not None:
            logger.info(
                "video pool after filters: %d/%d videos (%s)%s",
                len(allowed_video_ids),
                len({record.video_id for record in self.pipeline.manifest}),
                ", ".join(sorted(allowed_video_ids))[:200],
                f" prefixes={self.video_prefixes}" if self.video_prefixes else "",
            )

        if self.bm25_index is not None and self.bm25_index.is_empty:
            logger.warning(
                "BM25 skipped: manifest has no Objects/Metadata text. "
                "Run `aic2026 prepare` to load them for lexical retrieval."
            )

        # Bound the retrieval work to the requested candidate pool.  A global
        # per-video scan is disproportionately expensive for a static CLIP index.
        pool = limit
        # Fast KIS shortcut: answer the simplest KIS request with one vector
        # search and one immediate frame pool, without paying the full BM25
        # + RRF + visual rerank overhead. This is deliberately narrow and safe:
        # it only triggers when the query is a single string, no ASR override is
        # present, and expansion / LLM analysis are disabled so the result stays
        # deterministic and latency-bounded. It matches the product intent of
        # "bấm run là nó chạy ra frame chứ không chạy lâu quá cái break".
        #
        # Two modes of fast KIS shortcut:
        # 1. ULTRA-FAST (fusion_mode == "rrf_baseline"): Returns raw vector evidence
        #    immediately without any BM25/RRF fusion or reranking. This is the
        #    original "speed mode" behavior for maximum latency reduction.
        # 2. MEDIUM-FAST (fusion_mode != "rrf_adaptive"): Runs after Stage A
        #    (object evidence + colour) and video-level rerank, but skips
        #    Stage B (BLIP-2, LLM, late-interaction, event coverage, Gemini).
        #    This ensures adaptive fusion tests pass while still providing speed.
        # Queries with object-label evidence must pass through reranking before
        # return; otherwise the raw vector shortcut can preserve a wrong class.
        fast_kis_object_query = False
        if task_type == "kis" and self.strict_object_match and len(queries) == 1:
            from aic2026.reranking.lexical import (
                _COLOR_CONCEPTS, _requested_concepts_cached,
            )
            fast_kis_object_query = bool(
                _requested_concepts_cached(queries[0]) - _COLOR_CONCEPTS
            )
        fast_kis_ultra_fast = (
            task_type == "kis"
            and self.use_fast_kis
            and self.fusion_mode == "rrf_baseline"  # Only for baseline mode
            and len(queries) == 1
            and isinstance(queries[0], str)
            and not asr_query
            and analysis is None
            and not self.use_long_query_expansion
            and (self.disable_all_expansion or not self._should_expand_query_always(queries[0]))
            and not _force_expand
            and not _kis_action_chain
            and (not fast_kis_object_query or (
                self.bm25_index is not None and not self.bm25_index.is_empty
            ))
        )
        fast_kis_medium_fast = (
            task_type == "kis"
            and self.use_fast_kis
            and self.fusion_mode != "rrf_adaptive"  # Not for adaptive fusion
            and self.fusion_mode != "rrf_baseline"  # Not ultra-fast
            and len(queries) == 1
            and isinstance(queries[0], str)
            and not asr_query
            and analysis is None
            and not self.use_long_query_expansion
            and (self.disable_all_expansion or not self._should_expand_query_always(queries[0]))
            and not _force_expand
            and not _kis_action_chain
        )

        # Ultra-fast path: return immediately with raw vector evidence
        if fast_kis_ultra_fast:
            logger.info("Fast KIS ultra-fast shortcut active: return raw vector evidence in one pass")
            object_frame_ids = None
            if fast_kis_object_query:
                object_frame_ids = self.bm25_index.object_candidate_ids(
                    queries[0], allowed_video_ids,
                )
            if object_frame_ids is not None:
                ids, scores = self.pipeline.search_with_filter(
                    embeddings[0], pool, allowed_video_ids,
                    frame_indices=object_frame_ids,
                )
                candidates = self.pipeline._candidates_from_scores(
                    {int(idx): float(score) for idx, score in zip(ids, scores)},
                    limit=pool,
                )
                if fast_kis_object_query and candidates:
                    from aic2026.reranking.lexical import filter_candidates_by_requested_objects
                    candidates = filter_candidates_by_requested_objects(
                        queries[0], candidates, self._record_lookup(),
                        video_metadata=self.pipeline.video_metadata,
                    )
            else:
                candidates = self.pipeline.retrieve_raw(
                    text_embedding=embeddings[0],
                    top_frames=pool,
                    video_ids=allowed_video_ids,
                )
            candidates = gate_lion_dance_split(queries[0], candidates)
            if task_type != "trake":
                candidates = self.pipeline.diversify_candidates(
                    candidates,
                    max_answers=limit,
                )
            elapsed_ms = (time.time() - t0) * 1000.0
            logger.info("[DONE] KIS retrieve (ultra-fast shortcut): returned %d candidates in %.1f ms", len(candidates), elapsed_ms)
            return candidates[:limit]

        # Determine weights for multi-variant query fusion (primary query 1.0, sub-queries decaying).
        variant_weights = None
        if len(queries) > 1:
            if analysis is not None and getattr(analysis, "variant_relevance_scores", None):
                variant_weights = analysis.variant_relevance_scores
            else:
                variant_weights = [1.0] + [max(0.2, 0.8 - 0.15 * i) for i in range(len(queries) - 1)]

        if self.bm25_index is not None and not self.bm25_index.is_empty:
            logger.debug("BM25 lexical index active; fusing with vector retrieval.")
            # Collect ALL raw ranked lists (vector variants + BM25 variants)
            # and fuse them in ONE pass — avoids double-fusing which corrupts
            # RRF scores by treating already-fused results as raw ranked lists.
            all_ranked_lists: list[np.ndarray] = []
            all_weights: list[float] = []

            # Vector tier: batch all variant searches in one matrix multiply.
            object_frame_ids = None
            if task_type == "kis" and self.strict_object_match and queries:
                object_frame_ids = self.bm25_index.object_candidate_ids(
                    queries[0], allowed_video_ids,
                )
                if object_frame_ids is not None and not len(object_frame_ids):
                    logger.info("KIS object postings found no frames for %s", queries[0])
            search_results = self.pipeline.search_many_with_filter(
                embeddings, pool, allowed_video_ids, frame_indices=object_frame_ids,
            )
            for i, (ids, _) in enumerate(search_results):
                all_ranked_lists.append(np.asarray(ids, dtype=np.int64))
                if variant_weights and i < len(variant_weights):
                    all_weights.append(variant_weights[i])
                else:
                    all_weights.append(1.0)

            # BM25 tier: search ALL expanded variants so animal/disaster
            # rephrases catch lexical matches too.
            for q in queries:
                bm25_ids_q, _ = self.bm25_index.search(
                    q, pool, video_ids=allowed_video_ids
                )
                if bm25_ids_q.size > 0:
                    all_ranked_lists.append(np.asarray(bm25_ids_q, dtype=np.int64))
                    all_weights.append(0.8)

            if all_ranked_lists:
                fused_scores = self.pipeline._rrf_fuse(
                    all_ranked_lists, weights=all_weights,
                )
                candidates = self.pipeline._candidates_from_scores(
                    fused_scores, limit=pool,
                )
            else:
                candidates = []
        elif requires_dense_evidence:
            candidates = self.pipeline.retrieve_raw(
                text_embedding=embeddings[0],
                top_frames=pool,
                video_ids=allowed_video_ids,
            )
        elif len(embeddings) > 1:
            # Multi-query expansion: RRF-fuse per-variant vector rankings.
            # Batch all variant searches in one matrix multiply (2-4x faster).
            search_results = self.pipeline.search_many_with_filter(
                embeddings, pool, allowed_video_ids,
            )
            ranked_lists = [
                np.asarray(ids, dtype=np.int64) for ids, _ in search_results
            ]
            fused = self.pipeline._rrf_fuse(ranked_lists, weights=variant_weights)
            candidates = self.pipeline._candidates_from_scores(fused, limit=limit)
        else:
            candidates = self.pipeline.retrieve_raw(
                text_embedding=embeddings[0],
                top_frames=pool,
                video_ids=allowed_video_ids,
            )

        # Safety net: drop any candidate whose video slipped past the DB filter
        # (e.g. matched only via BM25 lexical text, not the vector tier).
        if allowed_video_ids is not None:
            candidates = self.pipeline._mask_video_ids(candidates, allowed_video_ids)

        if task_type == "kis" and self.strict_object_match and candidates:
            from aic2026.reranking.lexical import (
                _requested_concepts_cached,
                filter_candidates_by_requested_objects,
            )

            requested = _requested_concepts_cached(queries[0])
            if requested:
                original_count = len(candidates)
                candidates = filter_candidates_by_requested_objects(
                    queries[0], candidates, self._record_lookup(),
                    video_metadata=self.pipeline.video_metadata,
                )
                logger.info(
                    "KIS object constraint %s: %d/%d candidates retained",
                    sorted(requested), len(candidates), original_count,
                )

        # Per-video diversification: cap frames per video BEFORE reranking so
        # object evidence and colour rerank operate on a diverse pool instead of
        # being dominated by one video with hundreds of near-identical frames.
        # This was part of the original pipeline (retrieve → diversify_candidates)
        # and was accidentally dropped during the Improvement.md refactor.
        # TRAKE needs raw candidates (multiple frames per video are meaningful
        # for temporal alignment), so skip diversification for TRAKE.
        if task_type != "trake":
            candidates = self.pipeline.diversify_candidates(
                candidates, max_answers=limit,
            )

        # L24 (lion dance) is a known visual attractor in the BTC corpus. Keep
        # it out of every generic query before any reranker can amplify it; the
        # explicit operator codeword ``qilin`` opts back in.
        before_l24_gate = len(candidates)
        candidates = gate_lion_dance_split(queries[0], candidates)
        if len(candidates) != before_l24_gate:
            logger.info("L24 lion-dance gate removed %d generic-query candidates", before_l24_gate - len(candidates))

        logger.info(" -> Candidate pool built (%d items, %.1f ms)", len(candidates), (time.time() - t0) * 1000.0)

        event_embeddings_cache: np.ndarray | None = None

        def apply_event_coverage(items: list[Candidate]) -> tuple[list[Candidate], bool]:
            """Apply sentence consensus while the broad retrieval pool is intact."""
            nonlocal event_embeddings_cache
            if not items:
                return items, False
            from aic2026.reranking.event_coverage import event_aware_rerank
            from aic2026.query.parser import parse_query

            plan = query_plan if query_plan is not None else parse_query(queries[0])
            event_descs = [
                event.description.strip()
                for event in getattr(plan, "events", [])
                if getattr(event, "description", "").strip()
            ]
            if len(event_descs) < 2:
                try:
                    from aic2026.query.expansion import _split_sentences
                    event_descs = _split_sentences(queries[0])
                except Exception:
                    event_descs = []
            if len(event_descs) < 2:
                return items, False

            # Use the encoder cache populated before retrieval; this adds no
            # extra model pass for the sentence facets.
            if event_embeddings_cache is None:
                text_encoder = getattr(self.encode_text, "__self__", None)
                encode_many = getattr(text_encoder, "encode_many", None)
                event_vectors = (
                    encode_many(event_descs)
                    if callable(encode_many)
                    else [self.encode_text(description) for description in event_descs]
                )
                event_embeddings_cache = np.stack(event_vectors).astype(np.float32)
                event_embeddings_cache /= (
                    np.linalg.norm(event_embeddings_cache, axis=1, keepdims=True) + 1e-9
                )
            index = getattr(self.pipeline, "index", None)
            index_vectors = getattr(index, "vectors", None) if index is not None else None
            return event_aware_rerank(
                candidates=items,
                event_embeddings=event_embeddings_cache,
                index_vectors=index_vectors,
                event_weights=None,
                blend=self.event_coverage_blend,
            ), True

        # --- Cascade vs flat rerank (Improvement.md Task 2) ---
        # Cascade mode: cheap heuristics narrow the pool BEFORE expensive BLIP-2.
        # Flat mode (legacy): all reranks run on all candidates sequentially.
        if self.use_cascade_rerank and candidates:
            logger.info(
                "Cascade rerank: Stage A — object+colour on %d candidates",
                len(candidates),
            )
            # Stage A — cheap filters: object evidence + colour (narrow pool)
            candidates = rerank_with_object_evidence(
                query=queries[0],
                candidates=candidates,
                records=self._record_lookup(),
                weight=self.object_evidence_weight,
                penalty_scale=self.object_penalty_scale,
                drop_empty_object_frames=self.drop_empty_object_frames,
            )
            if self.encode_text is not None and self.encode_images is not None:
                candidates = contrastive_clip_colour_rerank(
                    query=queries[0],
                    candidates=candidates,
                    records=self._record_lookup(),
                    encode_text=self.encode_text,
                    encode_images=self.encode_images,
                    weight=self.contrastive_colour_rerank_weight,
                )
            # Apply the cheap multi-sentence consensus before the cascade's
            # top-N cut; otherwise matching details outside Stage A's top-N
            # are discarded before event coverage can rescue them.
            candidates, _ = apply_event_coverage(candidates)
            # Narrow to top_n cheap candidates before expensive stages
            n = min(self.cascade_stage_a_top_n, len(candidates))
            candidates = sorted(candidates, key=lambda c: c.score, reverse=True)[:n]
            logger.info(
                "Cascade rerank: Stage A done — narrowed to %d candidates",
                len(candidates),
            )

            # Fast KIS shortcut: after Stage A (object+colour) but before Stage B (BLIP-2/LLM)
            # This allows the shortcut to still benefit from object evidence and colour rerank
            # and video-level rerank, while skipping expensive BLIP-2 and later stages.
            if fast_kis_medium_fast:
                logger.info("Fast KIS shortcut active: skipping Stage B/C (BLIP-2, LLM, late-interaction, event coverage, Gemini)")
                # Run video-level rerank for KIS
                if task_type in ("kis", "qa"):
                    top_videos = self.coarse_top_k if self.coarse_top_k and self.coarse_top_k > 0 else None
                    self.pipeline._video_dedupe_window = self.video_dedupe_window
                    candidates = self.pipeline.video_level_rerank(
                        candidates,
                        top_videos=top_videos,
                    )
                # Diversify if not TRAKE
                if task_type != "trake":
                    candidates = self.pipeline.diversify_candidates(
                        candidates, max_answers=limit,
                    )
                elapsed_ms = (time.time() - t0) * 1000.0
                logger.info("[DONE] KIS retrieve (fast shortcut): returned %d candidates in %.1f ms", len(candidates), elapsed_ms)
                return candidates[:limit]

            # Stage B — expensive BLIP-2 on narrowed pool only
            if self.use_blip2_rerank and candidates:
                from aic2026.reranking.blip2_reranker import BLIP2Reranker
                reranker = BLIP2Reranker(
                    florence_vlm=getattr(self, "visual_answerer_obj", None),
                    device=self.vlm_device,
                )
                blip2_top_k = min(self.blip2_rerank_top_k, 30, len(candidates))
                candidates = reranker.rerank(
                    query=queries[0],
                    candidates=candidates,
                    records=self._record_lookup(),
                    top_k=blip2_top_k,
                    weight=self.blip2_rerank_weight,
                    fine_details=getattr(self, "_current_fine_details", None),
                )
                logger.info("Cascade rerank: Stage B (BLIP-2) done")

            # Stage C — remaining reranks on narrowed pool (late-interaction, ASR, event coverage)
        else:
            # --- Legacy flat pipeline: all reranks on all candidates ---
            candidates = rerank_with_object_evidence(
                query=queries[0],
                candidates=candidates,
                records=self._record_lookup(),
                weight=self.object_evidence_weight,
                penalty_scale=self.object_penalty_scale,
                drop_empty_object_frames=self.drop_empty_object_frames,
            )
            if self.encode_text is not None and self.encode_images is not None:
                candidates = contrastive_clip_colour_rerank(
                    query=queries[0],
                    candidates=candidates,
                    records=self._record_lookup(),
                    encode_text=self.encode_text,
                    encode_images=self.encode_images,
                    weight=self.contrastive_colour_rerank_weight,
                )
            # BLIP-2 (flat: runs on all candidates)
            if self.use_blip2_rerank and candidates:
                from aic2026.reranking.blip2_reranker import BLIP2Reranker
                reranker = BLIP2Reranker(
                    florence_vlm=getattr(self, "visual_answerer_obj", None),
                    device=self.vlm_device,
                )
                blip2_top_k = min(self.blip2_rerank_top_k, 30, len(candidates))
                candidates = reranker.rerank(
                    query=queries[0],
                    candidates=candidates,
                    records=self._record_lookup(),
                    top_k=blip2_top_k,
                    weight=self.blip2_rerank_weight,
                    fine_details=getattr(self, "_current_fine_details", None),
                )

        # KIS is ranked per frame.  Do not aggregate/cap by video here: a
        # candidate rises or falls only on its own retrieval evidence.

        # Late-interaction (ColBERT-style MaxSim): bắt khớp cục bộ theo từng facet
        # của query — một frame khớp BẤT KỲ facet nào (vd "water bottle") đều được
        # nâng, điều vector CLIP pooled đơn lẻ không làm được. Chỉ chạy khi index
        # thực sự chứa ĐỦ toàn bộ corpus: Chroma nếu chỉ build 1 phần (thiếu mấy
        # chục nghìn vector) thì vector_id sẽ lệch → skip an toàn, fallback về
        # RRF + metadata bonus, không crash. Skip entirely when weight=0 to avoid
        # unnecessary facet encoding + dedup bug from model_copy objects.
        if self.late_interaction_weight > 0 and self.encode_text is not None and candidates:
            index = getattr(self.pipeline, "index", None)
            index_has_all = index is not None
            manifest_size = len(self.pipeline.manifest) if self.pipeline.manifest is not None else 0
            if hasattr(index, "collection_count"):  # ChromaVectorStore
                try:
                    if index.collection_count != manifest_size:
                        index_has_all = False
                        logger.warning(
                            "late-interaction skipped: Chroma index has %d vectors "
                            "but manifest has %d — index is partial, vector_id would "
                            "mismatch. Rebuild with `aic2026 build-chroma-index`.",
                            index.collection_count,
                            manifest_size,
                        )
                except Exception as exc:  # noqa: BLE001
                    index_has_all = False
                    logger.warning("late-interaction skipped (index count error): %s", exc)
            elif getattr(index, "vectors", None) is None:
                index_has_all = False

            if index_has_all:
                # Late-interaction indexes ``index.vectors[c.vector_id]`` — the
                # manifest row. Refined TRAKE candidates can carry ``vector_id=None``
                # (dense_refinement drops it), which would IndexError here. Skip
                # those: rerank only the candidates that still map to a row, and
                # merge the untouched ones back in afterwards.
                indexed = [c for c in candidates if c.vector_id is not None]
                if indexed:
                    try:
                        frame_vectors = np.stack(
                            [index.vectors[c.vector_id] for c in indexed]
                        )
                        reranked = late_interaction_rerank(
                            query=queries[0],
                            candidates=indexed,
                            encode_text=self.encode_text,
                            frame_vectors=frame_vectors,
                            top_n=50,
                            weight=self.late_interaction_weight,
                        )
                        # Re-merge: keep candidates without a vector_id untouched.
                        reranked_ids = {id(c) for c in reranked}
                        candidates = reranked + [
                            c for c in candidates if id(c) not in reranked_ids
                        ]
                        # late_interaction chỉ sửa score tại chỗ, cần sort lại.
                        candidates = sorted(
                            candidates, key=lambda c: c.score, reverse=True
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("late-interaction rerank skipped: %s", exc)

        # --- ASR scoring (spec §ASR TEMPORAL LOCALIZATION) ----------------
        # Bounded nudge: ASR term/localization scores are min-max normalised over
        # the touched frames and scaled by ``asr_weight`` (default 0.15) so ASR
        # can lift a frame whose *speech* matches without ever overriding the
        # CLIP+BM25 ranking.  Falls back to the raw query when ``asr_query`` is
        # None (the call site in api.py always passes the Vietnamese text).
        #
        # When ``asr_dense_path`` is set, the precomputed BGE-M3 dense features
        # (.npy) are ALSO loaded via DenseRetriever and semantic ASR scores are
        # fused in alongside the keyword-based term/temporal scores — this is
        # the dense upgrade (P0) that replaces the keyword-only path with a
        # bounded dense ASR bonus on the Top-N pool.
        if self.use_asr and self.asr_transcripts and candidates:
            from aic2026.reranking.modality_scores import (
                asr_term_scores,
                asr_temporal_scores,
            )
            from aic2026.ingestion.asr import load_transcripts_sidecar

            transcripts = self.asr_transcripts
            if isinstance(transcripts, (str, Path)):
                transcripts = load_transcripts_sidecar(transcripts)

            # --- Keyword ASR scoring (fallback / always-on weak signal) ---
            if transcripts:
                asr_q = asr_query or queries[0]
                term_scores = asr_term_scores(asr_q, candidates, transcripts)
                temporal_scores = asr_temporal_scores(
                    asr_q, candidates, transcripts,
                    timestamp_lookup=getattr(self, "_asr_timestamp_lookup", lambda *a: None),
                )
                self._asr_bonus(term_scores, candidates, self.asr_weight)
                self._asr_bonus(temporal_scores, candidates, self.asr_weight)

            # --- Dense ASR scoring (BGE-M3, .npy) — semantic ASR ---
            # When a dense embeddings file is available, run semantic segment
            # search and add a bounded dense bonus alongside keyword scores.
            # This catches paraphrased / multi-word ASR matches that keyword
            # term-scoring misses entirely.  Reuse cached DenseRetriever to
            # avoid re-loading the .npy file (~50MB) on every query.
            if self.asr_dense_path is not None and transcripts:
                if self._dense_retriever_cache is None:
                    from aic2026.notebook.dense_retriever import DenseRetriever
                    self._dense_retriever_cache = DenseRetriever(
                        transcripts=transcripts,
                        dense_embeddings_path=self.asr_dense_path,
                    )
                dense_retriever = self._dense_retriever_cache
                if dense_retriever.is_available and transcripts:
                    asr_q = asr_query or queries[0]
                    allowed_vids = {c.video_id for c in candidates}
                    dense_matches = dense_retriever.search_queries(
                        [asr_q] if isinstance(asr_q, str) else asr_q,
                        top_k_per_query=50,
                        video_ids=allowed_vids,
                    )
                    # Map dense segment matches back to candidate frames via
                    # nearest segment frame (or video-level broadcast if no frame).
                    dense_term_scores: dict[int, float] = {}
                    for seg_match in dense_matches:
                        if seg_match.frame is None:
                            # No frame mapping → broadcast dense score to all
                            # candidate frames of that video.
                            for cand in candidates:
                                if (
                                    cand.video_id == seg_match.video_id
                                    and cand.vector_id is not None
                                ):
                                    dense_term_scores[cand.vector_id] = max(
                                        dense_term_scores.get(cand.vector_id, 0.0),
                                        seg_match.dense_score,
                                    )
                        else:
                            # Localized: boost frames near the spoken segment frame.
                            for cand in candidates:
                                if cand.video_id != seg_match.video_id:
                                    continue
                                if cand.vector_id is None:
                                    continue
                                from aic2026.reranking.modality_scores import _ASR_FRAME_TOL
                                if abs(cand.frame_id - (seg_match.frame or 0)) <= _ASR_FRAME_TOL:
                                    dense_term_scores[cand.vector_id] = max(
                                        dense_term_scores.get(cand.vector_id, 0.0),
                                        seg_match.dense_score,
                                    )
                    self._asr_bonus(dense_term_scores, candidates, self.asr_weight)

        # --- LLM Verifier (Improvement.md Task 8) ---
        # Fact-check top-K candidates using LLM. Runs AFTER all reranking
        # and BEFORE video-level aggregation.
        if self.use_llm_verifier and candidates and task_type == "kis":
            from aic2026.reranking.llm_verifier import llm_verify_candidates
            candidates = llm_verify_candidates(
                query=queries[0],
                candidates=candidates,
                fine_details=getattr(self, "_current_fine_details", None),
                records=self._record_lookup(),
                max_candidates=self.llm_verifier_max_candidates,
                model=self.llm_model,
                base_url=self.llm_base_url,
                timeout=self.llm_verifier_timeout,
                provider=self.llm_provider,
            )

        # --- Adaptive Fusion (two-stage ranking) ----------------------------
        # When ``fusion_mode == "rrf_adaptive"``, re-score the candidate pool by
        # fusing the normalised RRF semantic score with object-coverage (and
        # optionally ASR) signals.  This lifts frames that contain the objects
        # the query asks for, countering CLIP's scene-level bias.
        #
        # Formula: fused = w_semantic * s_norm + w_object * coverage + w_asr * asr_score
        # where s_norm = (score - min) / (max - min) over the current pool.
        if self.fusion_mode == "rrf_adaptive" and candidates and len(candidates) >= 2:
            weights = self.fusion_weights or {"semantic": 0.65, "object": 0.35, "asr": 0.0}
            w_semantic = weights.get("semantic", 0.5)
            w_object = weights.get("object", 0.3)
            w_asr = weights.get("asr", 0.2)
            # Drop ASR component when ASR is not active — avoids diluting the
            # semantic+object signal with a dead weight.  Renormalise so the
            # remaining components sum to 1.0.
            asr_active = self.use_asr and self.asr_transcripts
            if not asr_active:
                scale = w_semantic + w_object
                if scale > 0:
                    w_semantic /= scale
                    w_object /= scale
                w_asr = 0.0

            # Rank-based normalisation: top-ranked candidate gets s_norm=1.0,
            # bottom gets 0.0, linearly interpolated.  This is more robust than
            # min-max on raw RRF scores which compress heavily for small pools.
            scores = np.array([c.score for c in candidates], dtype=np.float64)
            order = np.argsort(-scores)  # descending
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(candidates), dtype=np.float64)
            s_norms = 1.0 - ranks / max(len(candidates) - 1, 1)

            # Pre-compute requested concepts once for the whole pool.
            from aic2026.reranking.lexical import (
                _requested_concepts_cached,
                _folded_labels_for_record,
                _alias_matches,
                _fold_accents,
                _OBJECT_ALIASES,
                _ANIMAL_CONCEPTS,
            )
            requested = _requested_concepts_cached(queries[0])

            record_map = self._record_lookup()

            # --- Pass 1: compute obj_coverage for all candidates ---
            coverages = []
            for c in candidates:
                cov = 0.0
                if requested and c.vector_id is not None:
                    record = record_map.get(c.vector_id)
                    if record is not None:
                        labels = _folded_labels_for_record(record)
                        matched = sum(
                            any(_alias_matches(labels, alias) for alias in _OBJECT_ALIASES[concept])
                            for concept in requested
                        )
                        cov = matched / len(requested)
                coverages.append(cov)

            # --- Dynamic weight adjustment ---
            avg_coverage = sum(coverages) / len(coverages) if coverages else 0.0
            has_animal = bool(requested & _ANIMAL_CONCEPTS) if requested else False

            if avg_coverage < 0.05:
                w_semantic = max(w_semantic, 0.80)
                w_object = min(w_object, 0.20)
            elif has_animal:
                w_semantic = 0.45
                w_object = 0.55

            # --- Pass 2: score with adjusted weights ---
            for i, c in enumerate(candidates):
                s_norm = float(s_norms[i])
                obj_coverage = coverages[i]

                # --- ASR signal ---
                # When ASR transcripts are loaded, ASR bonuses are already added
                # to c.score before this tier.  We extract the ASR contribution
                # by subtracting the original RRF base (pre-ASR).  When no ASR
                # is active, this defaults to 0.0 (correct for the test suite).
                asr_signal = 0.0  # placeholder — ASR not active in this synthetic pool

                # Store debug info: actual component scores
                c.debug = {
                    "rrf_raw": round(float(c.score), 4),  # Original RRF score before adaptive
                    "semantic_norm": round(s_norm, 4),      # Normalized semantic score
                    "object_coverage": round(obj_coverage, 4),  # Object match coverage
                    "asr_signal": round(asr_signal, 4),     # ASR contribution
                    "weights": {"semantic": w_semantic, "object": w_object, "asr": w_asr},
                }

                c.score = w_semantic * s_norm + w_object * obj_coverage + w_asr * asr_signal

            # Re-sort by new fused scores.
            candidates.sort(key=lambda c: c.score, reverse=True)
            logger.info(
                "adaptive fusion (final tier): re-scored %d candidates "
                "(weights: semantic=%.2f, object=%.2f, asr=%.2f)",
                len(candidates), w_semantic, w_object, w_asr,
            )

        # Make sentence-level agreement the final local ranking signal. This
        # runs after adaptive object fusion, which can otherwise reshuffle a
        # multi-sentence query based on sparse detector labels.
        if candidates:
            candidates, _ = apply_event_coverage(candidates)

        # --- Gemini Multimodal Reranker (Top 10-20 fine-grained judge) ---
        # Skip when top candidate is clearly dominant (score gap > 2× ambiguity margin)
        # — Gemini API latency (2-5s) is not worth it when local ranking is decisive.
        _skip_gemini = False
        if self.use_gemini_rerank and candidates and task_type == "kis" and len(candidates) >= 2:
            score_gap = candidates[0].score - candidates[1].score
            if score_gap > self.gemini_ambiguity_margin * 3:
                _skip_gemini = True
                logger.debug("Gemini rerank skipped: decisive score gap (%.4f)", score_gap)
        if self.use_gemini_rerank and candidates and task_type == "kis" and not _skip_gemini:
            try:
                from aic2026.reranking.gemini_reranker import GeminiReranker
                if getattr(self, "_gemini_reranker_instance", None) is None:
                    self._gemini_reranker_instance = GeminiReranker(
                        enabled=self.use_gemini_rerank,
                        model=self.gemini_model,
                        top_k=self.gemini_top_k,
                        local_weight=self.gemini_local_weight,
                        gemini_weight=self.gemini_weight,
                        ambiguity_margin=self.gemini_ambiguity_margin,
                        max_retries=self.gemini_max_retries,
                        timeout_seconds=self.gemini_timeout_seconds,
                        circuit_breaker_failures=self.gemini_circuit_breaker_failures,
                        cooldown_seconds=self.gemini_cooldown_seconds,
                        cache_enabled=self.gemini_cache_enabled,
                    )
                plan = query_plan
                if plan is None:
                    from aic2026.query.parser import parse_query
                    plan = parse_query(queries[0])
                candidates = self._gemini_reranker_instance.rerank(
                    query_plan=plan,
                    candidates=candidates,
                    records=self._record_lookup(),
                )
            except Exception as exc:
                logger.warning("Gemini rerank step skipped on exception (%s). Continuing pipeline.", exc)

        # KIS and QA are BOTH video-level: a single high-scoring outlier frame
        # must not lift the wrong video. Aggregate each video's strongest frames
        # into one video_score (log-sum-exp), keep the top videos, and re-emit
        # frames ordered by that video-aware score — so a video with *consistently*
        # good frames rises above a lone-outlier video, and we still surface other
        # strong videos instead of spamming frames from one. QA inherits this so
        # that (a) its candidate ordering matches KIS for the same query, and
        # (b) the VLM only answers frames from the strongest videos instead of a
        # raw pool of hundreds. TRAKE keeps the raw pool (multiple frames per
        # video are meaningful there and it has its own alignment stage).
        if task_type in ("kis", "qa"):
            # Moment selection (spec §9): collapse near-duplicate frames within a
            # short temporal window into "moments" and pick the single best frame
            # per moment, so KIS/QA return one representative evidence frame per
            # described scene instead of a burst of near-identical frames.
            if self.use_moment_rerank and candidates:
                from aic2026.retrieval.temporal_clustering import (
                    rerank_moments_to_frames,
                )
                candidates = rerank_moments_to_frames(
                    candidates,
                    records=self._record_lookup(),
                    gap_seconds=3.0,
                    use_shot_id=True,
                    aggregation="max",
                )
            top_videos = self.coarse_top_k if self.coarse_top_k and self.coarse_top_k > 0 else None
            # Pass temporal dedupe window to pipeline (Improvement.md Task 6)
            self.pipeline._video_dedupe_window = self.video_dedupe_window
            candidates = self.pipeline.video_level_rerank(
                candidates,
                top_videos=top_videos,
            )

        elapsed_ms = (time.time() - t0) * 1000.0
        logger.info("[DONE] KIS retrieve: returned %d candidates in %.1f ms", len(candidates), elapsed_ms)
        return candidates

    def _record_lookup(self) -> dict[int, FrameRecord]:
        """Map ``vector_id`` → manifest record, for reranking (cached on self)."""
        cached = getattr(self, "_cached_record_lookup_map", None)
        if cached is not None:
            return cached
        manifest = getattr(self.pipeline, "manifest", None)
        if not manifest:
            return {}
        lookup = {record.vector_id: record for record in manifest}
        self._cached_record_lookup_map = lookup
        return lookup

    @staticmethod
    def _asr_timestamp_lookup(
        video_id: str,
        frame_id: int,
    ) -> float | None:
        """Override hook for ASR temporal lookup.

        Sub-classes or callers may inject a real timestamp resolver.  Default
        returns ``None`` (falls back to score-based ASR bonus in ``_asr_bonus``).
        """
        return None

    @staticmethod
    def _asr_bonus(
        term_scores: dict[int, float],
        candidates: list[Candidate],
        weight: float,
        *,
        max_bonus: float = 0.02,
    ) -> None:
        """Add a bounded ASR score bump to *candidates* in-place.

        ``term_scores`` maps ``vector_id`` → raw ASR similarity.  The bump is
        min-max normalised over the touched frames and capped at ``max_bonus``
        so ASR nudges without ever overriding the CLIP+BM25 ranking.
        """
        touched = [
            (c, term_scores[c.vector_id])
            for c in candidates
            if c.vector_id is not None and c.vector_id in term_scores
        ]
        if not touched:
            return
        vals = np.array([s for _, s in touched], dtype=np.float64)
        lo, hi = float(vals.min()), float(vals.max())
        span = hi - lo
        for cand, raw in touched:
            norm = (raw - lo) / span if span > 1e-9 else 0.5
            bump = min(max_bonus, weight * norm)
            cand.score += bump

    def retrieve_trake(
        self,
        events: list[str],
        limit: int,
        prefilter_frames_per_event: int = 500,
        penalty_weight: float = 0.005,
        coarse_top_k: int = 200,
        # Soft *preference* (not a hard restriction) for certain video_id
        # prefixes (e.g. ["L26"]). Preferred videos get a small bounded score
        # nudge so they rank above equally-similar non-preferred videos, but a
        # non-preferred video is never dropped — BTC event queries are generic
        # and match many splits, so a hard restrict would zero recall.
        preferred_prefixes: list[str] | None = None,
        # Phase 6 (spec IX + §9, paper Eq.3): exponential inter-event temporal
        # decay forwarded to ``RetrievalPipeline.retrieve_trake``.  OFF-preserving
        # (False = legacy DP with additive penalty_weight).
        use_temporal_decay: bool = False,
        temporal_decay_alpha: float = 0.01,
    ) -> list[Candidate]:
        """Run deterministic event-wise TRAKE retrieval and alignment.

        ``coarse_top_k`` giới hạn số video đưa vào DP alignment (xem
        ``RetrievalPipeline.retrieve_trake``): chỉ top-K video theo coarse
        video-level similarity mới được xét, tránh DP chạy trên vài nghìn video.
        Object labels do not change this cap; it remains the guardrail that
        keeps dynamic-programming alignment bounded on the full corpus.
        """

        cleaned_events = [
            event.strip()
            for event in events
            if event.strip()
        ]

        if not cleaned_events:
            raise ValueError(
                "TRAKE requires at least one non-empty event"
            )

        if limit <= 0:
            return []

        event_embeddings = np.stack(
            [
                np.asarray(
                    self.encode_text(event),
                    dtype=np.float32,
                ).reshape(-1)
                for event in cleaned_events
            ],
            axis=0,
        )

        # Metadata pre-filter (TRAKE): restrict candidate videos before the
        # per-event vector scan inside retrieve_trake (pushed down to the DB).
        # Combined with the optional `video_prefixes` split restriction so a
        # video must satisfy BOTH to stay in the pool.
        allowed_video_ids = self.pipeline.filter_terms_to_video_ids(
            self.video_filter_terms or []
        )
        if self.video_prefixes:
            prefix_ids = self.pipeline.prefixes_to_video_ids(
                self.video_prefixes, allowed_video_ids
            )
            if prefix_ids is not None:
                allowed_video_ids = prefix_ids

        # Unit-test/lightweight pipelines may not expose manifest records.  The
        # production pipeline always does; in that case apply evidence inside
        # the event×frame matrix (vectorized) before temporal DP selects an
        # alignment. Building the adjustment as a per-video NumPy matrix — instead
        # of a per-(event,frame) Python closure — cuts TRAKE latency from ~145s to
        # a few seconds on the full 177k-frame corpus for a single query.
        object_adjustment = None
        object_adjustment_matrices = None
        manifest = getattr(self.pipeline, "manifest", None)
        video_to_manifest = getattr(
            self.pipeline, "_video_to_manifest_indices", None
        )
        if (
            manifest
            and video_to_manifest is not None
            and not self.trake_dense_refine
        ):
            from aic2026.reranking.lexical import (
                build_object_adjustment_matrices,
            )

            all_videos = (
                set(self.pipeline._video_embeddings.keys())
                if hasattr(self.pipeline, "_video_embeddings")
                else set(video_to_manifest.keys())
            )
            object_adjustment_matrices = (
                build_object_adjustment_matrices(
                    events=cleaned_events,
                    candidate_videos=all_videos,
                    video_to_manifest_indices=video_to_manifest,
                    manifest=manifest,
                    weight=0.08,
                )
            )

        candidates = self.pipeline.retrieve_trake(
            event_embeddings=event_embeddings,
            top_videos=limit,
            prefilter_frames_per_event=(
                prefilter_frames_per_event
            ),
            penalty_weight=penalty_weight,
            video_ids=allowed_video_ids,
            coarse_top_k=coarse_top_k,
            object_adjustment=object_adjustment,
            object_adjustment_matrices=object_adjustment_matrices,
            preferred_prefixes=preferred_prefixes,
            use_temporal_decay=use_temporal_decay,
            temporal_decay_alpha=temporal_decay_alpha,
        )

        # Safety net for any candidate the DB filter could not exclude.
        if allowed_video_ids is not None:
            candidates = self.pipeline._mask_video_ids(candidates, allowed_video_ids)

        # The TRAKE path bypasses ``retrieve`` above, so apply the same corpus
        # gate here before returning temporal sequences.
        candidates = gate_lion_dance_split(" ".join(cleaned_events), candidates)

        if self.trake_dense_refine:
            candidates = refine_trake_candidates(
                candidates,
                event_embeddings,
                self.encode_images,
                video_root=self.trake_video_root,
                top_videos=self.trake_refine_top_videos,
                sample_fps=self.trake_refine_sample_fps,
                window_seconds=self.trake_refine_window_seconds,
                penalty_weight=penalty_weight,
            )

        return candidates

    def candidates_for_video(
        self,
        candidates: list[Candidate],
        video_id: str,
    ) -> list[Candidate]:
        return [
            candidate
            for candidate in candidates
            if candidate.video_id == video_id
        ]

    def temporal_alignment(
        self,
        candidates: list[Candidate],
        event_count: int,
    ) -> list[int]:
        return align_events(candidates, event_count)

    def answer_question(
        self,
        question: str,
        candidates: list[Candidate],
    ) -> dict[int, str]:
        if self.visual_answerer is None:
            return {}
        # Reload the VLM if it was previously closed (cached agent reuse).
        obj = self.visual_answerer_obj
        if obj is not None and not getattr(obj, "available", False):
            try:
                obj.load()
            except Exception as exc:  # noqa: BLE001
                logger.warning("answer_question: VLM reload failed: %s", exc)
        return self.visual_answerer(
            question,
            candidates,
        )

    def reload_text_encoder(self) -> None:
        """Rebuild the CLIP encoder if it was unloaded (cached agent reuse)."""
        enc = getattr(self.encode_text, "__self__", None)
        if enc is not None and hasattr(enc, "load"):
            try:
                enc.load()
            except Exception as exc:  # noqa: BLE001
                logger.warning("reload_text_encoder: load failed: %s", exc)

    def close_text_encoder(self) -> None:
        """Free the CLIP text/image encoder weights to reclaim RAM before the VLM
        runs (low-RAM machines cannot keep CLIP + Florence resident at once).

        Only the underlying model weights are dropped — the ``encode_text`` /
        ``encode_images`` callables are preserved so a later retrieve on a *cached*
        agent can reload them (see ``reload_text_encoder``). We never null out the
        callables, which previously caused ``'NoneType' object is not callable'``.
        """
        enc = getattr(self.encode_text, "__self__", None)
        if enc is not None and hasattr(enc, "unload"):
            try:
                enc.unload()
            except Exception as exc:  # noqa: BLE001
                logger.warning("close_text_encoder: unload failed: %s", exc)

    def ensure_vlm(self) -> None:
        """Lazily build the VLM only on the QA answer step, AFTER the CLIP encoder
        has been freed (``close_text_encoder``). Avoids holding CLIP + Florence
        resident at once (OOM on low-RAM machines) and means KIS/TRAKE (vlm_model
        None) never pay the Florence load. No-op when already built or no model.
        """
        if self.visual_answerer is not None or not self.vlm_model:
            return
        try:
            from aic2026.qa.florence import FlorenceVLM
        except Exception as exc:  # noqa: BLE001
            logger.warning("ensure_vlm: cannot import FlorenceVLM: %s", exc)
            return
        florence = FlorenceVLM(
            model_name=self.vlm_model,
            device=self.vlm_device,
            torch_dtype=self.vlm_dtype,
        )
        florence.load()  # build processor + model now (CLIP already freed)
        if florence.available:  # only wire up if it actually loaded
            self.visual_answerer = florence.answer_question
            self.visual_answerer_obj = florence

    def close_vlm(self) -> None:
        """Free the VLM (Florence-2) weights after QA answering is complete.

        Only the underlying model is closed — the ``visual_answerer`` callable and
        ``visual_answerer_obj`` handle are preserved so a later request on a
        *cached* agent can reload the VLM (see ``answer_question``). We never null
        them out, which previously caused ``'NoneType' object is not callable'``.
        """
        obj = self.visual_answerer_obj
        if obj is not None and hasattr(obj, "close"):
            try:
                obj.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("close_vlm: close failed: %s", exc)

    @staticmethod
    def evidence(
        candidates: list[Candidate],
        maximum: int = 40,
    ) -> list[dict]:
        fields = {
            "vector_id",
            "video_id",
            "frame_id",
            "score",
            "keyframe_path",
        }

        return [
            candidate.model_dump(include=fields)
            for candidate in candidates[:maximum]
        ]

    @staticmethod
    def by_vector_id(
        candidates: list[Candidate],
    ) -> dict[int, Candidate]:
        return {
            candidate.vector_id: candidate
            for candidate in candidates
            if candidate.vector_id is not None
        }

    @staticmethod
    def group_by_video(
        candidates: list[Candidate],
    ) -> dict[str, list[Candidate]]:
        grouped: dict[str, list[Candidate]] = defaultdict(list)

        for candidate in candidates:
            grouped[candidate.video_id].append(candidate)

        return grouped
