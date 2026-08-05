import numpy as np

from aic2026.agent import RetrievalAgent
from aic2026.agent.tools import RetrievalTools
from aic2026.agent.types import AgentDecision, AgentPlan
from aic2026.models import Candidate, Query

class FakeLLM:
    def structured(self, system, user, schema):
        if schema is AgentPlan:
            return AgentPlan(query_variants=["red speaker"], rationale="visual rewrite")
        return AgentDecision(action="finish", selected_vector_ids=[7, 999999], rationale="evidence only")

class FakePipeline:
    def retrieve(self, embedding, top_frames, max_answers):
        return [Candidate(video_id="L01_V001", frame_id=505, score=.9, vector_id=7)]

def test_agent_can_only_return_retrieved_evidence():
    tools = RetrievalTools(pipeline=FakePipeline(), encode_text=lambda _: np.array([1.0]))
    result = RetrievalAgent(FakeLLM(), tools).run(Query(query_id="q", type="kis", text="red speaker"))
    assert [candidate.vector_id for candidate in result.candidates] == [7]
