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

        events = (
            list(query.events)
            if query.events
            else list(plan.events)
        )

        if not events:
            raise ValueError(
                "TRAKE requires at least one event."
            )

        return events

    def run(
        self,
        query: Query,
    ) -> AgentResult:
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
            # Chưa bật "Dịch VI→EN" → dùng query nguyên bản (chuẩn semantics checkbox).
            trace = []

        # --- Fully deterministic pipeline (no LLM round-trips) -------------
        # Planner + judge LLM are replaced by retrieval (CLIP + BM25 RRF) and
        # an algorithmic rerank (metadata bonus + MMR diversity). This is the
        # fastest, most stable path and is the default for KIS/Q&A.
        # TRAKE keeps its deterministic event-wise retrieval + DP alignment.
        plan = AgentPlan(
            query_variants=[query.text],
            events=(
                list(query.events)
                if query.type == "trake"
                else []
            ),
            rationale=(
                "Deterministic plan: no LLM planner. Retrieval uses CLIP + BM25 "
                "RRF; selection uses algorithmic rerank (metadata bonus + MMR)."
            ),
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

            candidates = self.tools.retrieve_trake(
                events=events,
                limit=self.answer_limit,
                prefilter_frames_per_event=(
                    self.retrieval_pool_size
                ),
                coarse_top_k=self.coarse_top_k,
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
        found = self.tools.retrieve(
            query=query.text,
            limit=self.retrieval_pool_size,
            task_type=query.type,
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

        if query.type == "qa":
            answers = self.tools.answer_question(
                query.question or query.text,
                candidates,
            )

            if answers:
                candidates = [
                    item
                    for item in candidates
                    if item.vector_id in answers
                ]

                for item in candidates:
                    if item.vector_id is not None:
                        item.answer = answers[item.vector_id]
            else:
                # VLM unavailable (not installed / no VRAM / offline): candidates
                # keep empty answers and the submission will be rejected later.
                # Surface it now instead of letting the user discover at submit time.
                logger.warning(
                    "QA query '%s': no answers produced (VLM unavailable?). "
                    "Output submission will be invalid until answers are attached.",
                    query.query_id,
                )

        return default_registry.handler_for(
            query.type
        )(
            query,
            candidates,
        )
