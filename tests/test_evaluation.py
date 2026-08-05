from aic2026.evaluation import evaluate_query
from aic2026.models import Candidate, GroundTruth, Query

def test_kis_and_rank_cutoffs():
    q = Query(query_id="q1", type="kis", text="laptop")
    gt = GroundTruth(video_id="L01_V001", ranges=[(500, 510)])
    candidates = [Candidate(video_id="wrong", frame_id=505, score=1), Candidate(video_id="L01_V001", frame_id=505, score=.9)]
    result = evaluate_query(q, candidates, gt)
    assert result["R@1"] == 0 and result["R@5"] == 1
    assert result["final_score"] == .8

def test_qa_requires_correct_answer():
    q = Query(query_id="q2", type="qa", text="cup", question="color?")
    gt = GroundTruth(video_id="L05_V005", ranges=[(800, 900)], answer="màu xanh")
    assert evaluate_query(q, [Candidate(video_id="L05_V005", frame_id=888, score=1, answer="màu trắng")], gt)["final_score"] == 0

def test_trake_wrong_video_is_zero():
    q = Query(query_id="q3", type="trake", text="jump", events=["take-off", "landing"])
    gt = GroundTruth(video_id="L10_V010", ranges=[(95, 105), (195, 205)])
    c = Candidate(video_id="other", frame_id=101, score=1, event_frames=[101, 203])
    assert evaluate_query(q, [c], gt)["final_score"] == 0
