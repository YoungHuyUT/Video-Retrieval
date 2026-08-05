import numpy as np

from aic2026.agent.tools import RetrievalTools
from aic2026.models import FrameRecord
from aic2026.retrieval import RetrievalPipeline, VectorIndex


def build_pipeline() -> RetrievalPipeline:
    vectors = np.asarray(
        [
            [1.00, 0.00],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],
            [0.00, 1.00],
        ],
        dtype=np.float32,
    )

    manifest = [
        FrameRecord(
            vector_id=0,
            video_id="L01_V001",
            frame_id=10,
            keyframe_path="L01_V001/10.jpg",
        ),
        FrameRecord(
            vector_id=1,
            video_id="L01_V001",
            frame_id=20,
            keyframe_path="L01_V001/20.jpg",
        ),
        FrameRecord(
            vector_id=2,
            video_id="L01_V001",
            frame_id=30,
            keyframe_path="L01_V001/30.jpg",
        ),
        FrameRecord(
            vector_id=3,
            video_id="L01_V001",
            frame_id=40,
            keyframe_path="L01_V001/40.jpg",
        ),
        FrameRecord(
            vector_id=4,
            video_id="L01_V002",
            frame_id=50,
            keyframe_path="L01_V002/50.jpg",
        ),
    ]

    return RetrievalPipeline(
        index=VectorIndex(vectors),
        manifest=manifest,
        frames_per_video=2,
    )


def test_raw_retrieval_preserves_same_video_frames() -> None:
    pipeline = build_pipeline()

    candidates = pipeline.retrieve_raw(
        text_embedding=np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        top_frames=5,
    )

    first_video_candidates = [
        candidate
        for candidate in candidates
        if candidate.video_id == "L01_V001"
    ]

    assert len(candidates) == 5
    assert len(first_video_candidates) == 4


def test_diversified_retrieval_applies_per_video_limit() -> None:
    pipeline = build_pipeline()

    candidates = pipeline.retrieve(
        text_embedding=np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        top_frames=5,
        max_answers=5,
    )

    first_video_candidates = [
        candidate
        for candidate in candidates
        if candidate.video_id == "L01_V001"
    ]

    assert len(first_video_candidates) == 2
    assert len(candidates) == 3


def test_tools_use_diversified_candidates_for_kis() -> None:
    tools = RetrievalTools(
        pipeline=build_pipeline(),
        encode_text=lambda _: np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )

    candidates = tools.retrieve(
        query="a person speaking",
        limit=5,
        task_type="kis",
    )

    first_video_candidates = [
        candidate
        for candidate in candidates
        if candidate.video_id == "L01_V001"
    ]

    assert len(first_video_candidates) == 2


def test_tools_use_raw_candidates_for_trake() -> None:
    tools = RetrievalTools(
        pipeline=build_pipeline(),
        encode_text=lambda _: np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )

    candidates = tools.retrieve(
        query="a person enters then sits down",
        limit=5,
        task_type="trake",
    )

    first_video_candidates = [
        candidate
        for candidate in candidates
        if candidate.video_id == "L01_V001"
    ]

    assert len(first_video_candidates) == 4


def test_tools_use_raw_candidates_for_qa() -> None:
    tools = RetrievalTools(
        pipeline=build_pipeline(),
        encode_text=lambda _: np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )

    candidates = tools.retrieve(
        query="what happens after the person enters?",
        limit=5,
        task_type="qa",
    )

    first_video_candidates = [
        candidate
        for candidate in candidates
        if candidate.video_id == "L01_V001"
    ]

    assert len(first_video_candidates) == 4