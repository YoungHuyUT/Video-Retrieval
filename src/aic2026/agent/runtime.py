from __future__ import annotations

import json
import logging

from aic2026.models import Candidate, Query
from aic2026.tasks import default_registry

from .local_llm import LocalLLM
from .tools import RetrievalTools
from .types import AgentDecision, AgentPlan, AgentResult, AgentTrace

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
        import re
        events = list(query.events) if query.events else list(plan.events)
        if not events and query.text:
            parts = [p.strip() for p in re.split(r'[\n;.]+', query.text) if p.strip()]
            if parts:
                events = parts
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
        # --- Translate Vietnamese → English (deterministic pipeline dropped the
        # LLM planner that used to do this). Text inside brackets (…), […],
        # {…}, and quotes is preserved verbatim (e.g. names / sign text).
        from .translator import decompose_query_modalities, translate_query_fields

        if self.translate:
            use_llm = self.llm is not None
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
            trace = []

        # For QA, retrieval text combines context text and question keywords
        retrieval_text = query.text
        if query.type == "qa" and query.question:
            q_clean = query.question.strip()
            t_clean = query.text.strip()
            if t_clean and q_clean and q_clean.lower() not in t_clean.lower():
                retrieval_text = f"{t_clean} {q_clean}"
            elif not t_clean and q_clean:
                retrieval_text = q_clean

        # Modality routing & decomposition (Visual / OCR / ASR) — AAAI 2026
        modality_plan = decompose_query_modalities(
            retrieval_text,
            llm=self.llm,
            use_llm=self.translate and self.llm is not None,
        )

        plan = AgentPlan(
            query_variants=[retrieval_text],
            events=(
                list(query.events)
                if query.type == "trake"
                else []
            ),
            rationale=(
                f"Adaptive multimodal plan: w_vis={modality_plan.w_vis}, "
                f"w_ocr={modality_plan.w_ocr}, w_asr={modality_plan.w_asr}. {modality_plan.reason}"
            ),
            modality=modality_plan,
        )
        trace.append(
            AgentTrace(
                step="plan",
                detail=plan.rationale,
            )
        )

        # TRAKE uses deterministic event-wise retrieval and monotonic DP.
        if query.type == "trake":
            events = self._resolve_trake_events(
                query=query,
                plan=plan,
            )

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

        # KIS / Q&A: retrieve with Adaptive Multimodal Fusion across Visual, OCR, and ASR
        self.tools.video_prefixes = None
        found = self.tools.retrieve(
            query=retrieval_text,
            limit=self.retrieval_pool_size,
            task_type=query.type,
            modality=modality_plan,
        )
        trace.append(
            AgentTrace(
                step="retrieve",
                detail=(
                    f"{len(found)} candidates via Adaptive Multimodal Fusion "
                    f"(vis={modality_plan.w_vis}, ocr={modality_plan.w_ocr}, asr={modality_plan.w_asr})"
                ),
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

        if query.type == "qa":
            # Free the CLIP encoder (retrieval is done) so Florence-2 can load on
            # low-RAM machines without OOM-ing on the combined ~1.2GB footprint.
            self.tools.close_text_encoder()
            # Lazily build Florence-2 only now (after CLIP is freed) so the two
            # heavy models are never resident at once. KIS/TRAKE have vlm_model
            # None so this is a no-op there.
            self.tools.ensure_vlm()
            answers = self.tools.answer_question(
                query.question or query.text,
                candidates,
            )
            # Release VLM weights now that answering is finished.
            self.tools.close_vlm()

            if answers:
                for item in candidates:
                    if item.vector_id is not None and item.vector_id in answers:
                        item.answer = answers[item.vector_id]

            # Fallback for candidates missing an answer (e.g. VLM offline/failed/unavailable)
            for item in candidates:
                if not item.answer:
                    q_text = (query.question or query.text).strip()
                    item.answer = f"Visible in keyframe"

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

