from __future__ import annotations

import json

from aic2026.models import Candidate, Query
from aic2026.tasks import default_registry

from .local_llm import LocalLLM
from .prompts import JUDGE_SYSTEM, PLANNER_SYSTEM, task_instruction
from .tools import RetrievalTools
from .types import AgentDecision, AgentPlan, AgentResult, AgentTrace


class RetrievalAgent:
    """Bounded retrieval state machine with task-aware evidence selection."""

    def __init__(
        self,
        llm: LocalLLM,
        tools: RetrievalTools,
        max_tool_rounds: int = 4,
        answer_limit: int = 100,
        retrieval_pool_size: int = 500,
    ) -> None:
        if max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds must be greater than zero")

        if answer_limit <= 0:
            raise ValueError("answer_limit must be greater than zero")

        if retrieval_pool_size <= 0:
            raise ValueError(
                "retrieval_pool_size must be greater than zero"
            )

        self.llm = llm
        self.tools = tools
        self.max_tool_rounds = max_tool_rounds
        self.answer_limit = answer_limit
        self.retrieval_pool_size = retrieval_pool_size

    def _plan(
        self,
        query: Query,
    ) -> AgentPlan:
        try:
            return self.llm.structured(
                PLANNER_SYSTEM,
                query.model_dump_json()
                + "\n"
                + task_instruction(query),
                AgentPlan,
            )
        except Exception:
            return AgentPlan(
                query_variants=[query.text],
                rationale=(
                    "Fallback plan used because the local LLM "
                    "was unavailable or slow."
                ),
            )

    def _judge(
        self,
        query: Query,
        ranked: list[Candidate],
        plan: AgentPlan,
    ) -> AgentDecision:
        try:
            payload = {
                "query": query.model_dump(),
                "evidence": self.tools.evidence(ranked),
                "required_output": task_instruction(query),
            }

            return self.llm.structured(
                JUDGE_SYSTEM,
                json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
                AgentDecision,
            )
        except Exception:
            selected_ids = [
                item.vector_id
                for item in ranked
                if item.vector_id is not None
            ][: self.answer_limit]

            return AgentDecision(
                action="finish",
                selected_vector_ids=selected_ids,
                rationale=(
                    "Fallback judgment used because the local LLM "
                    "was unavailable or slow."
                ),
            )

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

    def run(
        self,
        query: Query,
    ) -> AgentResult:
        # Validate that the task type is supported.
        default_registry.handler_for(query.type)

        plan = self._plan(query)

        variants = list(
            dict.fromkeys(
                [
                    query.text,
                    *plan.query_variants,
                ]
            )
        )[: self.max_tool_rounds]

        pool: list[Candidate] = []

        trace = [
            AgentTrace(
                step="plan",
                detail=plan.rationale,
            )
        ]

        for variant in variants:
            found = self.tools.retrieve(
                query=variant,
                limit=self.retrieval_pool_size,
                task_type=query.type,
            )

            pool.extend(found)

            trace.append(
                AgentTrace(
                    step="retrieve",
                    detail=(
                        f"{len(found)} candidates "
                        f"for task={query.type} from: {variant}"
                    ),
                )
            )

        ranked = self._deduplicate_best(pool)

        decision = self._judge(
            query=query,
            ranked=ranked,
            plan=plan,
        )

        allowed = self.tools.by_vector_id(ranked)

        selected = [
            allowed[vector_id]
            for vector_id in decision.selected_vector_ids
            if vector_id in allowed
        ]

        finalize_input = (
            ranked
            if query.type == "trake"
            else selected or ranked
        )

        candidates = self._finalize(
            query=query,
            candidates=finalize_input,
            decision=decision,
            plan=plan,
        )

        trace.append(
            AgentTrace(
                step="judge",
                detail=decision.rationale,
            )
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

        elif query.type == "trake":
            events = plan.events or query.events
            groups = self.tools.group_by_video(candidates)

            aligned: list[Candidate] = []

            for group in groups.values():
                if len(group) < len(events) or not events:
                    continue

                group = sorted(
                    group,
                    key=lambda item: item.score,
                    reverse=True,
                )

                representative = group[0].model_copy(
                    deep=True,
                )

                representative.event_frames = (
                    self.tools.temporal_alignment(
                        candidates=group,
                        event_count=len(events),
                    )
                )

                aligned.append(representative)

            return sorted(
                aligned,
                key=lambda item: item.score,
                reverse=True,
            )

        return default_registry.handler_for(
            query.type
        )(
            query,
            candidates,
        )