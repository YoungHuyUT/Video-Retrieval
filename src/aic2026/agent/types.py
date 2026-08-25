from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from aic2026.models import Candidate

Action = Literal["retrieve", "align_events", "answer_question", "finish"]


class ModalityDecomposition(BaseModel):
    """Decomposition of a user query across Visual, OCR, and ASR modalities.

    Based on 'Unified Interactive Multimodal Moment Retrieval' (AAAI 2026).
    """

    visual_query: str = Field(description="Visual action, scene, objects, or colors")
    ocr_query: str = Field(default="", description="On-screen text, signs, logos, numbers")
    asr_query: str = Field(default="", description="Spoken dialogue, speech keywords, narration")
    w_vis: float = Field(default=0.6, ge=0.0, le=1.0, description="Visual modality weight")
    w_ocr: float = Field(default=0.2, ge=0.0, le=1.0, description="OCR modality weight")
    w_asr: float = Field(default=0.2, ge=0.0, le=1.0, description="ASR modality weight")
    reason: str = Field(default="", max_length=500, description="Reason for modality weight assignment")


class AgentPlan(BaseModel):
    """Kế hoạch nhỏ, có schema; tránh agent tự sinh truy vấn/tool tùy ý."""
    query_variants: list[str] = Field(min_length=1, max_length=6)
    events: list[str] = Field(default_factory=list, max_length=8)
    rationale: str = Field(max_length=500)
    modality: ModalityDecomposition | None = None


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
