from __future__ import annotations

import numpy as np
import pytest

from aic2026.agent.tools import RetrievalTools
from aic2026.models import FrameRecord
from aic2026.retrieval.bm25_index import BM25Index
from aic2026.retrieval.index import VectorIndex
from aic2026.retrieval.pipeline import RetrievalPipeline


def test_fast_kis_dog_query_does_not_return_unicorn_only_frame():
    pytest.importorskip("rank_bm25")
    records = [
        FrameRecord(
            vector_id=0, video_id="unicorn", frame_id=1,
            keyframe_path="unicorn.jpg", object_labels=["Toy", "Horse"],
        ),
        FrameRecord(
            vector_id=1, video_id="dog", frame_id=1,
            keyframe_path="dog.jpg", object_labels=["Dog", "Person"],
        ),
    ]
    # The wrong frame wins a global dense search; the actual dog is rank 2.
    index = VectorIndex(
        np.asarray([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32),
        assume_normalized=True,
    )
    pipeline = RetrievalPipeline(index, records)
    tools = RetrievalTools(
        pipeline,
        encode_text=lambda _: np.asarray([1.0, 0.0], dtype=np.float32),
        bm25_index=BM25Index(records),
    )
    tools.use_fast_kis = True
    tools.fusion_mode = "rrf_baseline"
    tools.strict_object_match = True
    tools.use_llm_query_analyzer = False
    tools.use_long_query_expansion = False
    tools.disable_all_expansion = True

    results = tools.retrieve("a dog", limit=1, task_type="kis")

    assert results
    assert results[0].video_id == "dog"
    assert all(record.object_labels == ["Dog", "Person"] for record in records if record.video_id == results[0].video_id)
