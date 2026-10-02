from aic2026.models import Candidate, FrameRecord
from aic2026.reranking.lexical import filter_candidates_by_requested_objects


def test_explicit_dog_query_drops_frames_without_dog_label():
    records = {
        1: FrameRecord(vector_id=1, video_id="v1", frame_id=1, keyframe_path="1.jpg", object_labels=["Dog", "Person"]),
        2: FrameRecord(vector_id=2, video_id="v2", frame_id=1, keyframe_path="2.jpg", object_labels=["Lion"]),
        3: FrameRecord(vector_id=3, video_id="v3", frame_id=1, keyframe_path="3.jpg", object_labels=[]),
    }
    candidates = [
        Candidate(video_id="v1", frame_id=1, vector_id=1, score=0.4),
        Candidate(video_id="v2", frame_id=1, vector_id=2, score=0.9),
        Candidate(video_id="v3", frame_id=1, vector_id=3, score=0.8),
    ]

    kept = filter_candidates_by_requested_objects("a dog", candidates, records)

    assert [candidate.vector_id for candidate in kept] == [1]


def test_query_without_recognized_object_keeps_dense_candidates():
    candidate = Candidate(video_id="v1", frame_id=1, vector_id=1, score=0.4)

    assert filter_candidates_by_requested_objects("a red scene", [candidate], {}) == [candidate]


def test_multi_object_query_requires_all_objects_in_one_frame():
    records = {
        1: FrameRecord(vector_id=1, video_id="both", frame_id=1, keyframe_path="1.jpg", object_labels=["Woman", "Dog"]),
        2: FrameRecord(vector_id=2, video_id="person", frame_id=1, keyframe_path="2.jpg", object_labels=["Woman"]),
        3: FrameRecord(vector_id=3, video_id="dog", frame_id=1, keyframe_path="3.jpg", object_labels=["Dog"]),
    }
    candidates = [
        Candidate(video_id="both", frame_id=1, vector_id=1, score=0.4),
        Candidate(video_id="person", frame_id=1, vector_id=2, score=0.9),
        Candidate(video_id="dog", frame_id=1, vector_id=3, score=0.8),
    ]

    kept = filter_candidates_by_requested_objects("a woman next to a dog", candidates, records)

    assert [candidate.vector_id for candidate in kept] == [1]


def test_next_to_is_spatial_not_an_event_separator():
    from aic2026.query.parser import parse_query

    plan = parse_query("a woman next to a dog")

    assert len(plan.events) == 1
    assert plan.spatial_relations == ["next to"]
