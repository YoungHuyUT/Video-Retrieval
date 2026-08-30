import numpy as np

from aic2026.models import Candidate, FrameRecord
from aic2026.reranking.lexical import rerank_with_semantic_text


def test_semantic_rerank_prioritizes_vietnamese_matches() -> None:
    records = {
        0: FrameRecord(
            vector_id=0,
            video_id="L01_V001",
            frame_id=10,
            keyframe_path="L01_V001/10.jpg",
            metadata_keywords=["bàn tay"],
            object_labels=["hand"],
        ),
        1: FrameRecord(
            vector_id=1,
            video_id="L01_V002",
            frame_id=20,
            keyframe_path="L01_V002/20.jpg",
            metadata_keywords=["nước"],
            object_labels=["bottle"],
        ),
    }

    def fake_encoder(text: str) -> np.ndarray:
        text = text.lower()
        if "bàn tay" in text or "hand" in text:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0], dtype=np.float32)

    candidates = [
        Candidate(video_id="L01_V001", frame_id=10, score=0.70, vector_id=0),
        Candidate(video_id="L01_V002", frame_id=20, score=0.72, vector_id=1),
    ]

    reranked = rerank_with_semantic_text(
        query="bàn tay",
        candidates=candidates,
        records=records,
        encode_text=fake_encoder,
        weight=1.0,
    )

    assert reranked[0].video_id == "L01_V001"
    assert reranked[0].score > reranked[1].score
