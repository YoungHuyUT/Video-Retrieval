import pytest
from aic2026.models import Candidate, Query
from aic2026.submission import competition_answer, validate_candidates

def test_submission_limits_and_orders_scores():
    q = Query(query_id="q", type="kis", text="test")
    answer = validate_candidates(q, [Candidate(video_id="a", frame_id=1, score=.2), Candidate(video_id="b", frame_id=2, score=.9)])
    assert [c.video_id for c in answer] == ["b", "a"]

def test_qa_must_have_answer():
    with pytest.raises(ValueError):
        validate_candidates(Query(query_id="q", type="qa", text="x"), [Candidate(video_id="a", frame_id=1, score=1)])

def test_competition_output_has_only_brief_fields():
    kis = Query(query_id="k", type="kis", text="x")
    assert competition_answer(kis, Candidate(video_id="v", frame_id=9, score=.8, vector_id=3)) == {"video_id": "v", "frame_id": 9}
    trake = Query(query_id="t", type="trake", text="x", events=["a", "b"])
    assert competition_answer(trake, Candidate(video_id="v", frame_id=1, score=.8, event_frames=[9, 20])) == {"video_id": "v", "frame_ids": [9, 20]}
