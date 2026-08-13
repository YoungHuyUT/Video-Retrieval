from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from aic2026.models import Candidate

Action = Literal["retrieve", "align_events", "answer_question", "finish"]


class AgentPlan(BaseModel):
    """Kế hoạch nhỏ, có schema; tránh agent tự sinh truy vấn/tool tùy ý."""
    query_variants: list[str] = Field(min_length=1, max_length=6)
    events: list[str] = Field(default_factory=list, max_length=8)
    rationale: str = Field(max_length=500)


class AgentDecision(BaseModel):
    action: Action
    selected_vector_ids: list[int] = Field(default_factory=list, max_length=100)
    answer: str | None = Field(default=None, max_length=160)
    rationale: str = Field(default="", max_length=500)


class AgentTrace(BaseModel):
    step: str
    detail: str


class AgentResult(BaseModel):
    candidates: list[Candidate]
    plan: AgentPlan
    trace: list[AgentTrace]
