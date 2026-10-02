import pytest

from aic2026.models import FrameRecord
from aic2026.retrieval.bm25_index import BM25Index


def test_object_query_uses_detector_label_postings():
    pytest.importorskip("rank_bm25")
    records = [
        FrameRecord(vector_id=0, video_id="dog", frame_id=1, keyframe_path="0.jpg", object_labels=["Dog"]),
        FrameRecord(vector_id=1, video_id="lion", frame_id=1, keyframe_path="1.jpg", object_labels=["Lion"]),
        FrameRecord(vector_id=2, video_id="empty", frame_id=1, keyframe_path="2.jpg", object_labels=[]),
    ]
    index = BM25Index(records)

    ids, scores = index.search("a dog", 10)

    assert ids.tolist() == [0]
    assert scores.tolist()[0] > 0


def test_object_postings_respect_video_filter():
    pytest.importorskip("rank_bm25")
    records = [
        FrameRecord(vector_id=0, video_id="dog1", frame_id=1, keyframe_path="0.jpg", object_labels=["Dog"]),
        FrameRecord(vector_id=1, video_id="dog2", frame_id=1, keyframe_path="1.jpg", object_labels=["Dog"]),
    ]
    index = BM25Index(records)

    ids, _ = index.search("dog", 10, video_ids={"dog2"})

    assert ids.tolist() == [1]


def test_multi_object_query_uses_frame_level_intersection():
    pytest.importorskip("rank_bm25")
    records = [
        FrameRecord(vector_id=0, video_id="both", frame_id=1, keyframe_path="0.jpg", object_labels=["Woman", "Dog"]),
        FrameRecord(vector_id=1, video_id="person", frame_id=1, keyframe_path="1.jpg", object_labels=["Woman"]),
        FrameRecord(vector_id=2, video_id="dog", frame_id=1, keyframe_path="2.jpg", object_labels=["Dog"]),
    ]
    index = BM25Index(records)

    ids, _ = index.search("a woman next to a dog", 10)

    assert ids.tolist() == [0]
