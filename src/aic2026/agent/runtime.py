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
        # {…}, and quotes is preserved verbatim (e.g. names / sign text). If the
        # LLM is unavailable or errors, we keep the original text so retrieval
        # still runs. ─────────────────────────────────────────────────────────
        from .translator import translate_query_fields

        translated_text, translated_question, translated_events, translated_changed = (
            translate_query_fields(
                query.text,
                query.question,
                query.events,
                self.llm,
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
                            "changes": changes,
                        },
                        ensure_ascii=False,
                    ),
                )
            ]
        else:
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

        selected = self._algorithmic_select(
            query=query,
            ranked=ranked,
            k=self.answer_limit,
        )
        trace.append(
            AgentTrace(
                step="rerank",
                detail=(
                    f"Algorithmic rerank selected {len(selected)} candidates "
                    f"(metadata bonus + MMR diversity, no LLM judge)."
                ),
            )
        )

        candidates = self._finalize(
            query=query,
            candidates=selected,
            decision=AgentDecision(action="finish", rationale=plan.rationale),
            plan=plan,
        )

        return AgentResult(
            candidates=candidates[: self.answer_limit],
            plan=plan,
            trace=trace,
        )

    @staticmethod
    def _algorithmic_select(
        query: Query,
        ranked: list[Candidate],
        k: int,
    ) -> list[Candidate]:
        """Select the top ``k`` candidates without an LLM.

        ``ranked`` already carries a retrieval score that ``tools.retrieve``
        blended with a metadata keyword bonus. Here we apply **Maximal Marginal
        Relevance (MMR)** to trade a little relevance for diversity, so one
        video does not dominate the submission and recall across videos
        improves.
        """
        if not ranked or k <= 0:
            return []

        lambda_mmr = 0.7  # 0.7 relevance, 0.3 diversity
        selected: list[Candidate] = []
        remaining = list(ranked)
        max_score = max((c.score for c in ranked), default=1.0) or 1.0

        def _redundancy(cand: Candidate) -> float:
            # Cheap proxy: same video ⇒ high redundancy, else low.
            return (
                max((1.0 for s in selected if s.video_id == cand.video_id), default=0.0)
                if selected
                else 0.0
            )

        while remaining and len(selected) < k:
            best = None
            best_gain = None
            for cand in remaining:
                rel = cand.score / max_score
                gain = lambda_mmr * rel - (1 - lambda_mmr) * _redundancy(cand)
                if best_gain is None or gain > best_gain:
                    best_gain = gain
                    best = cand
            selected.append(best)
            remaining.remove(best)

        return selected

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