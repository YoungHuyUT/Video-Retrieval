from __future__ import annotations

import json

from aic2026.models import Candidate, Query
from aic2026.tasks import default_registry
from .local_llm import LocalLLM
from .tools import RetrievalTools
from .types import AgentDecision, AgentPlan, AgentResult, AgentTrace
from .prompts import JUDGE_SYSTEM, PLANNER_SYSTEM, task_instruction


class RetrievalAgent:
    """State machine bounded by retrieval tools, not a free-form autonomous agent."""
    def __init__(self, llm: LocalLLM, tools: RetrievalTools, max_tool_rounds: int = 4, answer_limit: int = 100):
        self.llm, self.tools = llm, tools
        self.max_tool_rounds, self.answer_limit = max_tool_rounds, answer_limit

    def _plan(self, query: Query) -> AgentPlan:
        try:
            return self.llm.structured(PLANNER_SYSTEM, query.model_dump_json() + "\n" + task_instruction(query), AgentPlan)
        except Exception:
            return AgentPlan(query_variants=[query.text], rationale="Fallback plan used because local LLM was unavailable or slow.")

    def _judge(self, query: Query, ranked: list[Candidate], plan: AgentPlan) -> AgentDecision:
        try:
            return self.llm.structured(JUDGE_SYSTEM, json.dumps({"query": query.model_dump(), "evidence": self.tools.evidence(ranked), "required_output": task_instruction(query)}, ensure_ascii=False), AgentDecision)
        except Exception:
            return AgentDecision(action="finish", selected_vector_ids=[item.vector_id for item in ranked if item.vector_id is not None][:self.answer_limit], rationale="Fallback judgment used because local LLM was unavailable or slow.")

    def run(self, query: Query) -> AgentResult:
        default_registry.handler_for(query.type)
        plan = self._plan(query)
        variants = list(dict.fromkeys([query.text, *plan.query_variants]))[:self.max_tool_rounds]
        pool: list[Candidate] = []
        trace = [AgentTrace(step="plan", detail=plan.rationale)]
        for variant in variants:
            found = self.tools.retrieve(variant, self.answer_limit)
            pool.extend(found)
            trace.append(AgentTrace(step="retrieve", detail=f"{len(found)} candidates from: {variant}"))
        unique: dict[int, Candidate] = {item.vector_id: item for item in pool if item.vector_id is not None}
        ranked = sorted(unique.values(), key=lambda item: item.score, reverse=True)
        decision = self._judge(query, ranked, plan)
        allowed = self.tools.by_vector_id(ranked)
        selected = [allowed[item] for item in decision.selected_vector_ids if item in allowed]
        candidates = self._finalize(query, ranked if query.type == "trake" else (selected or ranked), decision, plan)
        trace.append(AgentTrace(step="judge", detail=decision.rationale))
        return AgentResult(candidates=candidates[:self.answer_limit], plan=plan, trace=trace)

    def _finalize(self, query: Query, candidates: list[Candidate], decision: AgentDecision, plan: AgentPlan) -> list[Candidate]:
        candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
        if query.type == "qa":
            answers = self.tools.answer_question(query.question or query.text, candidates)
            if answers:
                candidates = [item for item in candidates if item.vector_id in answers]
                for item in candidates:
                    item.answer = answers[item.vector_id]  # Answer provenance is the VLM tool, not text LLM.
        elif query.type == "trake":
            events = plan.events or query.events
            groups: dict[str, list[Candidate]] = {}
            for item in candidates:
                groups.setdefault(item.video_id, []).append(item)
            aligned: list[Candidate] = []
            for video_id, group in groups.items():
                if len(group) >= len(events) > 0:
                    representative = group[0].model_copy(deep=True)
                    representative.event_frames = self.tools.temporal_alignment(group, len(events))
                    aligned.append(representative)
            return sorted(aligned, key=lambda item: item.score, reverse=True)
        return default_registry.handler_for(query.type)(query, candidates)
