import numpy as np
from aic2026.agent import RetrievalAgent
from aic2026.agent.tools import RetrievalTools
from aic2026.models import Candidate, FrameRecord, Query
from aic2026.retrieval import RetrievalPipeline
from aic2026.agent.types import AgentDecision, AgentPlan

class MockVectorIndex:
    def __init__(self, size: int):
        self.vectors = np.random.randn(size, 512).astype(np.float32)

def test_temporal_frame_expansion():
    # 1. Create a minimal manifest
    # Video V1: frames 0, 90, 270
    manifest = [
        FrameRecord(vector_id=0, video_id="V1", frame_id=0, keyframe_path="V1/001.jpg"),
        FrameRecord(vector_id=1, video_id="V1", frame_id=90, keyframe_path="V1/002.jpg"),
        FrameRecord(vector_id=2, video_id="V1", frame_id=270, keyframe_path="V1/003.jpg"),
    ]
    index = MockVectorIndex(len(manifest))
    pipeline = RetrievalPipeline(index, manifest, frames_per_video=3)
    
    assert "V1" in pipeline._video_to_manifest_indices
    
    tools = RetrievalTools(pipeline=pipeline, encode_text=lambda x: np.zeros(512))
    agent = RetrievalAgent(tools=tools)
    
    # 2. Run _finalize with Candidate at frame_id=90 and a duplicate
    query = Query(query_id="q1", type="kis", text="dummy")
    candidates = [
        Candidate(video_id="V1", frame_id=90, score=0.8, vector_id=1, keyframe_path="V1/002.jpg"),
        Candidate(video_id="V1", frame_id=90, score=0.5, vector_id=1, keyframe_path="V1/002.jpg"),
    ]
    decision = AgentDecision(action="finish", rationale="test")
    plan = AgentPlan(query_variants=["dummy"], rationale="test")
    
    final_candidates = agent._finalize(query, candidates, decision, plan)
    
    # Output should contain only the highest-scoring candidate for (V1, 90)
    assert len(final_candidates) == 1
    
    # Sort order checking
    assert final_candidates[0].frame_id == 90
    assert final_candidates[0].score == 0.8
