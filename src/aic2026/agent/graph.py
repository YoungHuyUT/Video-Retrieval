from __future__ import annotations

from typing import Any, TypedDict
import json

from aic2026.models import Candidate, Query
from aic2026.tasks import default_registry
from .runtime import JUDGE_SYSTEM, PLANNER_SYSTEM, RetrievalAgent
from .prompts import task_instruction
from .types import AgentDecision, AgentPlan, AgentResult, AgentTrace


class AgentState(TypedDict, total=False):
    query: Query
    plan: AgentPlan
    ranked: list[Candidate]
    decision: AgentDecision
    result: AgentResult
    trace: list[AgentTrace]


class LangGraphRetrievalAgent:
    """Graph orchestration for the bounded AIC agent; every data lookup stays deterministic."""
    def __init__(self, agent: RetrievalAgent):
        self.agent = agent

    def run(self, query: Query) -> AgentResult:
        default_registry.handler_for(query.type)
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError("Install agent extra: uv sync --extra agent") from exc

        def plan_node(state: AgentState) -> dict[str, Any]:
            plan = self.agent.llm.structured(PLANNER_SYSTEM, query.model_dump_json() + "\n" + task_instruction(query), AgentPlan)
            return {"plan": plan, "trace": [AgentTrace(step="plan", detail=plan.rationale)]}

        def retrieve_node(state: AgentState) -> dict[str, Any]:
            variants = list(dict.fromkeys([query.text, *state["plan"].query_variants]))[:self.agent.max_tool_rounds]
            pool: list[Candidate] = []
            trace = list(state["trace"])
            for variant in variants:
                found = self.agent.tools.retrieve(variant, self.agent.answer_limit)
                pool.extend(found)
                trace.append(AgentTrace(step="retrieve", detail=f"{len(found)} candidates from: {variant}"))
            unique = {item.vector_id: item for item in pool if item.vector_id is not None}
            return {"ranked": sorted(unique.values(), key=lambda item: item.score, reverse=True), "trace": trace}

        def judge_node(state: AgentState) -> dict[str, Any]:
            evidence = self.agent.tools.evidence(state["ranked"])
            decision = self.agent.llm.structured(JUDGE_SYSTEM, json.dumps({"query": query.model_dump(), "evidence": evidence, "required_output": task_instruction(query)}, ensure_ascii=False), AgentDecision)
            return {"decision": decision}

        def finalize_task(state: AgentState, task: str) -> dict[str, Any]:
            allowed = self.agent.tools.by_vector_id(state["ranked"])
            selected = [allowed[key] for key in state["decision"].selected_vector_ids if key in allowed]
            source = state["ranked"] if task == "trake" else (selected or state["ranked"])
            candidates = self.agent._finalize(query, source, state["decision"], state["plan"])
            trace = [*state["trace"], AgentTrace(step=f"{task}_finalize", detail=state["decision"].rationale)]
            return {"result": AgentResult(candidates=candidates[:self.agent.answer_limit], plan=state["plan"], trace=trace)}

        def kis_finalize(state: AgentState) -> dict[str, Any]:
            return finalize_task(state, "kis")

        def qa_finalize(state: AgentState) -> dict[str, Any]:
            return finalize_task(state, "qa")

        def trake_finalize(state: AgentState) -> dict[str, Any]:
            return finalize_task(state, "trake")

        def route_task(state: AgentState) -> str:
            return {"kis": "kis_finalize", "qa": "qa_finalize", "trake": "trake_finalize"}[query.type]

        graph = StateGraph(AgentState)
        graph.add_node("plan", plan_node)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("judge", judge_node)
        graph.add_node("kis_finalize", kis_finalize)
        graph.add_node("qa_finalize", qa_finalize)
        graph.add_node("trake_finalize", trake_finalize)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "judge")
        graph.add_conditional_edges("judge", route_task, {"kis_finalize": "kis_finalize", "qa_finalize": "qa_finalize", "trake_finalize": "trake_finalize"})
        graph.add_edge("kis_finalize", END)
        graph.add_edge("qa_finalize", END)
        graph.add_edge("trake_finalize", END)
        return graph.compile().invoke({"query": query})["result"]
