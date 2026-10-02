from __future__ import annotations

import json
import logging

from aic2026.models import Candidate, Query
from aic2026.tasks import default_registry

from .local_llm import LocalLLM
from .tools import RetrievalTools
from .types import AgentDecision, AgentPlan, AgentResult, AgentTrace
from aic2026.temporal.alignment import decay_alpha_from_wording

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """Bounded retrieval state machine with task-aware evidence selection."""

    def __init__(
        self,
        tools: RetrievalTools,
        llm: LocalLLM | None = None,
        answer_limit: int = 100,
        retrieval_pool_size: int = 500,
        # Optional per-task dataset-split *preference* (video_id prefixes). Both
        # KIS and Q&A search the FULL corpus (no restriction) — KIS is whole-video
        # by nature, and Q&A reads OCR'd text that lives across splits. TRAKE
        # *prefers* L26 (the large TRAKE-oriented set) as a soft bias, but is NOT
        # hard-restricted to it: BTC event queries are generic and match many
        # splits, so excluding L21/L22/L25/etc. would silently zero recall. Pass
        # None to disable the preference; pass a concrete list to override it.
        trake_preferred_prefixes: list[str] | None = None,
        # Query understanding (III/IV). When True, ``run`` builds a structured
        # :class:`QueryPlan` (LLM-backed local planner, rule-based fallback) and
        # attaches it to the returned plan so downstream rerank stages can read
        # entities / constraints / modality weights. Enabled by default for
        # production-quality query understanding.
        use_query_planner: bool = False,
        planner_model: str = "qwen3:0.6b",
        planner_url: str = "http://127.0.0.1:11434",
        planner_timeout: float = 8.0,
        # Phase 6 (spec IX + §9): turn on exponential inter-event temporal
        # decay for TRAKE (paper Eq.3). The decay alpha is auto-derived from the
        # query wording (immediately>then>later) unless overridden per-call. Off
        # by default so the legacy penalty-weight DP stays A/B baseline A.
        use_temporal_decay: bool = False,
        temporal_decay_alpha: float = 0.01,
        # Phase 7 (spec X, rule 12): re-rank KIS/Q&A candidates by how well their
        # object labels satisfy the structured COUNT/ATTRIBUTE constraints. Weak-
        # label-safe: videos without object_labels are no-ops (never penalised),
        # because the user's self-extracted keyframes carry no detector output.
        # Off by default so legacy CLIP+BM25 stays baseline A.
        use_constraint_verification: bool = False,
    ) -> None:
        if answer_limit <= 0:
            raise ValueError("answer_limit must be greater than zero")

        if retrieval_pool_size <= 0:
            raise ValueError(
                "retrieval_pool_size must be greater than zero"
            )

        # ``llm`` is kept for backwards compatibility (the backend may still
        # pass an OllamaLLM), but the deterministic pipeline no longer calls it.
        self.llm = llm
        self.tools = tools
        self.answer_limit = answer_limit
        self.retrieval_pool_size = retrieval_pool_size
        # Số video tối đa đưa vào DP alignment (TRAKE coarse filter). Dataset lớn
        # (~100GB, hàng chục nghìn video) cần K nhỏ (200) để DP không chạy trên
        # vài nghìn video. 0 = xét hết (backward-compatible).
        self.coarse_top_k = 200
        self.trake_preferred_prefixes = trake_preferred_prefixes
        self.use_query_planner = use_query_planner
        self.planner_model = planner_model
        self.planner_url = planner_url
        self.planner_timeout = planner_timeout
        self.use_temporal_decay = use_temporal_decay
        self.temporal_decay_alpha = temporal_decay_alpha
        # Phase 7: propagate onto tools (retrieve() reads the flag from there).
        self.tools.use_constraint_verification = use_constraint_verification
        # Bật dịch VI→EN tự động. Mặc định False (tắt) vì:
        #  (1) người dùng có thể nhập sẵn tiếng Anh;
        #  (2) dịch phụ thuộc Ollama — nếu fail thì gây ra frame sai;
        #  (3) có query đòi giữ nguyên cụm từ gốc (tên riêng, chữ trên biển báo).
        # Bật qua nút "Dịch VI→EN" trên UI hoặc tham số khi khởi tạo agent.
        self.translate: bool = False

    @staticmethod
    def _deduplicate_best(
        candidates: list[Candidate],
    ) -> list[Candidate]:
        """Keep the highest-scoring occurrence of each vector id."""

        unique: dict[int, Candidate] = {}

        for candidate in candidates:
            if candidate.vector_id is None:
                continue

            current = unique.get(candidate.vector_id)

            if current is None or candidate.score > current.score:
                unique[candidate.vector_id] = candidate

        return sorted(
            unique.values(),
            key=lambda candidate: candidate.score,
            reverse=True,
        )

    @staticmethod
    def _resolve_trake_events(
        query: Query,
        plan: AgentPlan,
    ) -> list[str]:
        """Return the authoritative ordered events for a TRAKE query."""
        # Explicit multi-line/API events win.  A single event equal to the raw
        # query is only a UI placeholder, so parse it into its ordered actions.
        explicit = list(query.events) if query.events else list(plan.events)
        raw = query.text.strip()
        events = explicit if len(explicit) > 1 or (explicit and explicit[0].strip() != raw) else []
        if not events and raw:
            from aic2026.query.parser import parse_query

            parsed = parse_query(raw)
            events = [event.description for event in parsed.events if event.description.strip()]
        if not events:
            events = [query.text]
        return events

    def run(
        self,
        query: Query,
    ) -> AgentResult:
        # Ensure the CLIP encoder is resident (a prior QA request on this cached
        # agent may have unloaded it to free RAM for the VLM). Reloading here
        # keeps every request independent and avoids 'NoneType' not callable.
        self.tools.reload_text_encoder()

        # Validate that the task type is supported.
        default_registry.handler_for(query.type)

        # --- Translate Vietnamese → English (deterministic pipeline dropped the
        # LLM planner that used to do this). Text inside brackets (…), […],
        # {…}, and quotes is preserved verbatim (e.g. names / sign text).
        #
        # CHỈ dịch khi ``self.translate=True`` (đã tick ô "Dịch VI→EN" trên UI,
        # hoặc truyền ``translate_query=True`` từ CLI). Khi tắt, dùng NGUYÊN BẢN
        # query — đúng semantics của checkbox "Mặc định TẮT — dùng query nguyên
        # bản". Không dịch ngầm dù chưa tick.
        #
        # Khi bật: dịch OFFLINE (từ điển, không cần LLM, không bao giờ lỗi) để
        # đưa tiếng Việt → tiếng Anh cho CLIP text encoder; nếu có LLM thì dịch
        # tiếp bằng LLM để sửa lỗi chính tả EN, LLM lỗi tự fallback bản offline.
        from .translator import translate_query_fields

        # Lưu query gốc (nguyên bản, trước khi dịch) để dùng làm fallback cho
        # ASR matching. Transcript (phụ đề) là tiếng Việt, nên so khớp phải dùng
        # bản VI — KHÔNG dùng bản EN đã dịch. Khi user không nhập ô "ASR query"
        # riêng, ta tự động dùng query gốc VI này thay vì query.text (đã EN).
        original_text = query.text

        if self.translate:
            # Translation sits on the critical retrieval path.  The offline
            # translator is deterministic and instant; LLM query planning stays
            # an explicit opt-in below instead of adding an Ollama round-trip to
            # every long KIS query.
            use_llm = False
            translated_text, translated_question, translated_events, translated_changed, translation_source = (
                translate_query_fields(
                    query.text,
                    query.question,
                    query.events,
                    self.llm,
                    use_llm=use_llm,
                )
            )
            if translated_changed:
                changes: dict[str, dict[str, str]] = {}
                if translated_text != query.text:
                    changes["text"] = {"from": query.text, "to": translated_text}
                if (translated_question or "") != (query.question or ""):
                    changes["question"] = {
                        "from": query.question or "",
                        "to": translated_question or "",
                    }
                for old_ev, new_ev in zip(query.events or [], translated_events):
                    if old_ev != new_ev:
                        changes.setdefault("events", []).append({"from": old_ev, "to": new_ev})

                query = Query(
                    query_id=query.query_id,
                    type=query.type,
                    text=translated_text,
                    question=translated_question,
                    events=translated_events,
                )
                trace = [
                    AgentTrace(
                        step="translate",
                        detail=json.dumps(
                            {
                                "normalized": True,
                                "source": translation_source,
                                "changes": changes,
                            },
                            ensure_ascii=False,
                        ),
                    )
                ]
            else:
                trace = []
        else:
            # Chưa bật "Dịch VI→EN" → dùng query nguyên bản (chuẩn semantics checkbox).
            trace = []

        # --- Fully deterministic pipeline (no LLM round-trips) -------------
        # Planner + judge LLM are replaced by retrieval (CLIP + BM25 RRF) and
        # an algorithmic rerank (metadata bonus + MMR diversity). This is the
        # fastest, most stable path and is the default for KIS/Q&A.
        # TRAKE keeps its deterministic event-wise retrieval + DP alignment.
        #
        # Query understanding (Task 2 / III-IV): optionally build a structured
        # QueryPlan from the raw text. This is LOCAL (Ollama) and cached per
        # query string, so it only runs once and falls back to a rule-based
        # parser on any failure. The plan is carried on ``AgentPlan.query_plan``
        # so later stages (object/count verification) can consume it; retrieval
        # itself still uses the raw query text as the CLIP/BM25 input today.
        from aic2026.query import plan_query

        query_plan_obj = None
        planner_built = False
        if self.use_query_planner and query.text:
            try:
                query_plan_obj = plan_query(
                    query.text,
                    use_llm=True,
                    model=self.planner_model,
                    base_url=self.planner_url,
                    timeout_seconds=self.planner_timeout,
                )
                planner_built = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Query planner failed; using raw query: %s", exc)
        rationale = (
            "Deterministic plan with LLM-backed query understanding (QueryPlan "
            "attached). Retrieval uses CLIP + BM25 RRF; selection uses "
            "algorithmic rerank."
            if planner_built
            else "Deterministic plan: no LLM planner. Retrieval uses CLIP + BM25 "
            "RRF; selection uses algorithmic rerank (metadata bonus + MMR)."
        ) if self.use_query_planner else (
            "Deterministic plan: no LLM planner. Retrieval uses CLIP + BM25 "
            "RRF; selection uses algorithmic rerank (metadata bonus + MMR)."
        )
        plan = AgentPlan(
            query_variants=[query.text],
            events=(
                list(query.events)
                if query.type == "trake"
                else []
            ),
            rationale=rationale,
            query_plan=(
                query_plan_obj.model_dump() if query_plan_obj is not None else None
            ),
        )
        trace.append(
            AgentTrace(
                step="plan",
                detail=(
                    f"QueryPlan: {query_plan_obj.global_query!r}"
                    if planner_built
                    else plan.rationale
                ),
            )
        )

        # TRAKE uses deterministic event-wise retrieval and monotonic DP.
        if query.type == "trake":
            events = self._resolve_trake_events(
                query=query,
                plan=plan,
            )

            # TRAKE searches the FULL corpus. We do NOT hard-restrict to L26: BTC
            # event queries are generic and match many splits, so excluding
            # L21/L22/L25/... would silently zero recall (verified: a generic
            # "enters room / sits" query returns 0 L26 candidates). Instead we
            # pass L26 as a soft *preference* so it is nudged up without being
            # the only allowed split. Override via trake_preferred_prefixes.
            self.tools.video_prefixes = None
            candidates = self.tools.retrieve_trake(
                events=events,
                limit=self.answer_limit,
                prefilter_frames_per_event=(
                    self.retrieval_pool_size
                ),
                coarse_top_k=self.coarse_top_k,
                preferred_prefixes=(
                    self.trake_preferred_prefixes
                    if self.trake_preferred_prefixes is not None
                    else ["L26"]
                ),
                use_temporal_decay=self.use_temporal_decay,
                # Auto-derive alpha from ordering wording when using decay
                # (immediately>then>later); explicit config still wins if set.
                temporal_decay_alpha=(
                    self.temporal_decay_alpha
                    if not self.use_temporal_decay
                    else decay_alpha_from_wording(query.text)
                ),
            )

            trace.append(
                AgentTrace(
                    step="trake_align",
                    detail=(
                        f"Aligned {len(events)} ordered events "
                        f"across {len(candidates)} candidate videos."
                    ),
                )
            )

            return AgentResult(
                candidates=candidates,
                plan=plan,
                trace=trace,
            )

        # KIS / Q&A: retrieve directly on the (translated) query text.
        # Multi-query expansion was tried but added a second LLM round-trip
        # (slower) for marginal recall gain on short, specific BTC queries,
        # so we keep a single translation + single retrieval for speed.
        # Both KIS and Q&A search the FULL corpus (no split restriction): KIS is
        # whole-video retrieval by nature, and Q&A reads OCR'd text that spans
        # every split. OCR itself is scoped to L25 at build time; retrieval is
        # not.
        self.tools.video_prefixes = None
        # ASR matching dùng transcript TIẾNG VIỆT.
        # CHỈ bật ASR cho NOTEBOOK track. KIS/QA/TRAKE KHÔNG dùng ASR
        # (transcript tiếng Việt không khớp query tiếng Anh → nhiễu).
        # Nếu user không nhập ô "ASR query" riêng, fallback về query gốc VI
        # (original_text) thay vì query.text đã dịch EN.
        asr_query_for_match = None
        if query.type == "notebook":
            asr_query_for_match = query.asr_query or (original_text if self.translate else None)
            if asr_query_for_match is not None and asr_query_for_match.strip():
                self.tools.use_asr = True
        else:
            # KIS/QA/TRAKE: không dùng ASR (transcript tiếng Việt ≠ query tiếng Anh)
            self.tools.use_asr = False
        found = self.tools.retrieve(
            query=query.text,
            limit=self.retrieval_pool_size,
            task_type=query.type,
            query_plan=query_plan_obj,
            asr_query=asr_query_for_match,
        )
        trace.append(
            AgentTrace(
                step="retrieve",
                detail=f"{len(found)} candidates for task={query.type}",
            )
        )

        ranked = self._deduplicate_best(found)

        # The retrieval pool already carries the final blended score (RRF fusion
        # + bounded metadata keyword bonus). We deliberately do NOT apply any
        # further rerank stage (no video-level coarse filter, no late-interaction
        # ColBERT pass, no MMR): those layers were found to flatten the fine-
        # grained RRF ordering and hurt the single-video / few-video cases. The
        # unified RRF + metadata ranking is returned as-is (capped to the answer
        # limit by the task handler in _finalize).
        trace.append(
            AgentTrace(
                step="rerank",
                detail=(
                    f"Unified RRF (CLIP + BM25) fusion + bounded metadata bonus "
                    f"over {len(ranked)} candidates; no further rerank stage."
                ),
            )
        )

        candidates = self._finalize(
            query=query,
            candidates=ranked,
            decision=AgentDecision(action="finish", rationale=plan.rationale),
            plan=plan,
        )

        return AgentResult(
            candidates=candidates[: self.answer_limit],
            plan=plan,
            trace=trace,
        )

    def _finalize(
        self,
        query: Query,
        candidates: list[Candidate],
        decision: AgentDecision,
        plan: AgentPlan,
    ) -> list[Candidate]:
        candidates = sorted(
            candidates,
            key=lambda item: item.score,
            reverse=True,
        )

        # Q&A no longer auto-generates answers with a VLM (too slow). The human
        # inspects the retrieved keyframe gallery in the Streamlit UI, picks the
        # correct frame(s), and types the answer into a single shared box. Until
        # then ``Candidate.answer`` stays None. We therefore leave answers
        # entirely to the UI and only do the (video_id, frame_id) dedup here.

        # Deduplicate to keep the highest-scoring candidate for each (video_id, frame_id)
        seen_frame = set()
        deduped_candidates = []
        for c in sorted(candidates, key=lambda x: x.score, reverse=True):
            key = (c.video_id, c.frame_id)
            if key not in seen_frame:
                seen_frame.add(key)
                deduped_candidates.append(c)

        candidates = deduped_candidates

        return default_registry.handler_for(
            query.type
        )(
            query,
            candidates,
        )

