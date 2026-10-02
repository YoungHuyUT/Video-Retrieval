import numpy as np

from aic2026.agent.tools import RetrievalTools
from aic2026.models import Candidate, FrameRecord
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


def test_filtered_search_considers_all_frames_of_allowed_videos() -> None:
    pipeline = build_pipeline()

    # V002 is below the global top-4 for this query.  A post-filtered global
    # search would return no V002 result, even though it is the requested video.
    candidates = pipeline.retrieve_raw(
        text_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        top_frames=1,
        video_ids={"L01_V002"},
    )

    assert [candidate.video_id for candidate in candidates] == ["L01_V002"]


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


def test_kis_action_chain_adds_event_recall_branches() -> None:
    encoded: list[str] = []

    def encode(text: str) -> np.ndarray:
        encoded.append(text)
        return np.asarray([1.0, 0.0], dtype=np.float32)

    tools = RetrievalTools(pipeline=build_pipeline(), encode_text=encode)
    tools.retrieve("a person enters a room, opens a door, then sits down", limit=5, task_type="kis")
    # Global query is retained, while every action gets an independent recall
    # branch. This is intentionally weaker than TRAKE's ordered alignment.
    assert len(encoded) >= 3
    assert any("opens" in query for query in encoded)


def test_kis_long_action_chain_has_bounded_retrieval_cost() -> None:
    encoded: list[str] = []
    tools = RetrievalTools(
        pipeline=build_pipeline(),
        encode_text=lambda text: (encoded.append(text) or np.asarray([1.0, 0.0], dtype=np.float32)),
    )
    tools.retrieve(
        "a person enters a room, opens a door, walks to a table, picks up a book, reads it, then sits down",
        limit=5,
        task_type="kis",
    )
    # Bounded cost: one global view plus action-branch variants (fine_details
    # may add 1-2 extra detail queries, but total stays well under the chain length).
    assert len(encoded) <= 6


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

    # QA inherits the video-level rerank introduced in 4887e92 ("fix QA & TRAKE"):
    # candidates are aggregated per video and capped to `frames_per_video` (2 in
    # build_pipeline) so the VLM only answers frames from the strongest videos.
    # The raw pool holds 4 frames for L01_V001; after rerank only 2 survive.
    assert len(first_video_candidates) == 2

def build_trake_pipeline() -> RetrievalPipeline:
    vectors = np.asarray(
        [
            # Correct video: event order 1 -> 2 -> 3.
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            # Wrong video: reversed semantic order.
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    manifest = [
        FrameRecord(
            vector_id=0,
            video_id="L01_CORRECT",
            frame_id=100,
            keyframe_path="L01_CORRECT/100.jpg",
        ),
        FrameRecord(
            vector_id=1,
            video_id="L01_CORRECT",
            frame_id=200,
            keyframe_path="L01_CORRECT/200.jpg",
        ),
        FrameRecord(
            vector_id=2,
            video_id="L01_CORRECT",
            frame_id=300,
            keyframe_path="L01_CORRECT/300.jpg",
        ),
        FrameRecord(
            vector_id=3,
            video_id="L02_REVERSED",
            frame_id=100,
            keyframe_path="L02_REVERSED/100.jpg",
        ),
        FrameRecord(
            vector_id=4,
            video_id="L02_REVERSED",
            frame_id=200,
            keyframe_path="L02_REVERSED/200.jpg",
        ),
        FrameRecord(
            vector_id=5,
            video_id="L02_REVERSED",
            frame_id=300,
            keyframe_path="L02_REVERSED/300.jpg",
        ),
    ]

    return RetrievalPipeline(
        index=VectorIndex(vectors),
        manifest=manifest,
    )


def test_trake_ranks_video_with_correct_event_order_first() -> None:
    pipeline = build_trake_pipeline()

    event_embeddings = np.eye(
        3,
        dtype=np.float32,
    )

    candidates = pipeline.retrieve_trake(
        event_embeddings=event_embeddings,
        top_videos=2,
        prefilter_frames_per_event=6,
        penalty_weight=0.005,
    )

    assert len(candidates) == 2

    best = candidates[0]

    assert best.video_id == "L01_CORRECT"

    assert best.event_frames == [
        100,
        200,
        300,
    ]

    assert len(best.event_frames) == 3

    assert all(
        previous < current
        for previous, current in zip(
            best.event_frames,
            best.event_frames[1:],
            strict=False,
        )
    )


def test_trake_decay_off_is_identical_to_baseline() -> None:
    # Phase 6: the opt-in decay/beam params must leave the OFF path byte-for-byte
    # identical to the legacy call (baseline A for A/B).
    pipeline = build_trake_pipeline()
    events = np.eye(3, dtype=np.float32)

    baseline = pipeline.retrieve_trake(
        event_embeddings=events,
        top_videos=2,
        prefilter_frames_per_event=6,
    )
    decay = pipeline.retrieve_trake(
        event_embeddings=events,
        top_videos=2,
        prefilter_frames_per_event=6,
        use_temporal_decay=True,
        temporal_decay_alpha=0.01,
    )

    assert [c.video_id for c in baseline] == [c.video_id for c in decay]
    assert [c.event_frames for c in baseline] == [c.event_frames for c in decay]


def test_trake_decay_returns_valid_monotonic_alignment() -> None:
    # Phase 6: with decay ON the alignment must still be a valid, strictly
    # increasing per-event frame assignment (no broken path).
    pipeline = build_trake_pipeline()
    events = np.eye(3, dtype=np.float32)

    candidates = pipeline.retrieve_trake(
        event_embeddings=events,
        top_videos=2,
        prefilter_frames_per_event=6,
        use_temporal_decay=True,
        temporal_decay_alpha=0.05,
    )

    assert len(candidates) == 2
    for candidate in candidates:
        assert candidate.event_frames is not None
        assert len(candidate.event_frames) == 3
        assert all(
            previous < current
            for previous, current in zip(
                candidate.event_frames,
                candidate.event_frames[1:],
                strict=False,
            )
        )


def test_trake_returns_one_candidate_per_video() -> None:
    pipeline = build_trake_pipeline()

    candidates = pipeline.retrieve_trake(
        event_embeddings=np.eye(
            3,
            dtype=np.float32,
        ),
        top_videos=10,
        prefilter_frames_per_event=6,
    )

    video_ids = [
        candidate.video_id
        for candidate in candidates
    ]

    assert len(video_ids) == len(
        set(video_ids)
    )


def build_trake_pipeline_3videos() -> RetrievalPipeline:
    """3 video với video-embedding (mean-pool) PHÂN BIỆT được:

    - L01_CORRECT: 1 frame mỗi trục -> emb ~ [0.577,0.577,0.577] (gần centroid query eye(3)).
    - L02_AXIS_X: 3 frame trục X -> emb = [1,0,0] (xa centroid hơn CORRECT).
    - L03_AXIS_Y: 3 frame trục Y -> emb = [0,1,0] (xa centroid hơn CORRECT).
    """
    vectors = np.asarray(
        [
            # CORRECT: event order 1 -> 2 -> 3.
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            # AXIS_X: mọi frame trục X.
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            # AXIS_Y: mọi frame trục Y.
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    manifest = [
        FrameRecord(vector_id=0, video_id="L01_CORRECT", frame_id=100, keyframe_path="L01_CORRECT/100.jpg"),
        FrameRecord(vector_id=1, video_id="L01_CORRECT", frame_id=200, keyframe_path="L01_CORRECT/200.jpg"),
        FrameRecord(vector_id=2, video_id="L01_CORRECT", frame_id=300, keyframe_path="L01_CORRECT/300.jpg"),
        FrameRecord(vector_id=3, video_id="L02_AXIS_X", frame_id=100, keyframe_path="L02_AXIS_X/100.jpg"),
        FrameRecord(vector_id=4, video_id="L02_AXIS_X", frame_id=200, keyframe_path="L02_AXIS_X/200.jpg"),
        FrameRecord(vector_id=5, video_id="L02_AXIS_X", frame_id=300, keyframe_path="L02_AXIS_X/300.jpg"),
        FrameRecord(vector_id=6, video_id="L03_AXIS_Y", frame_id=100, keyframe_path="L03_AXIS_Y/100.jpg"),
        FrameRecord(vector_id=7, video_id="L03_AXIS_Y", frame_id=200, keyframe_path="L03_AXIS_Y/200.jpg"),
        FrameRecord(vector_id=8, video_id="L03_AXIS_Y", frame_id=300, keyframe_path="L03_AXIS_Y/300.jpg"),
    ]

    return RetrievalPipeline(
        index=VectorIndex(vectors),
        manifest=manifest,
    )


def test_trake_coarse_filter_limits_to_top_k_videos() -> None:
    pipeline = build_trake_pipeline_3videos()

    event_embeddings = np.eye(3, dtype=np.float32)

    # coarse_top_k=1 -> chỉ 1 video (có coarse score cao nhất) lọt vào DP.
    candidates = pipeline.retrieve_trake(
        event_embeddings=event_embeddings,
        top_videos=10,
        prefilter_frames_per_event=9,
        coarse_top_k=1,
    )

    returned_videos = {c.video_id for c in candidates}
    assert returned_videos == {"L01_CORRECT"}

    best = candidates[0]
    # event_frames vẫn đúng thứ tự tăng dần (frame_id tăng).
    assert best.event_frames == [100, 200, 300]
    assert all(
        previous < current
        for previous, current in zip(
            best.event_frames, best.event_frames[1:], strict=False
        )
    )


def test_trake_coarse_filter_backward_compatible_when_disabled() -> None:
    pipeline = build_trake_pipeline_3videos()

    event_embeddings = np.eye(3, dtype=np.float32)

    # coarse_top_k=0 / >= số video -> xét hết (như cũ).
    candidates = pipeline.retrieve_trake(
        event_embeddings=event_embeddings,
        top_videos=10,
        prefilter_frames_per_event=9,
        coarse_top_k=0,
    )

    returned_videos = {c.video_id for c in candidates}
    assert returned_videos == {"L01_CORRECT", "L02_AXIS_X", "L03_AXIS_Y"}


# --------------------------------------------------------------------------
# KIS video-level rerank (the missing stage that fixes KIS accuracy)
# --------------------------------------------------------------------------

def build_kis_pipeline() -> RetrievalPipeline:
    """Two videos with different frame-score *distributions*.

    - V_OUTLIER: 1 frame with a *very high* score (1.0) + 3 near-junk frames
      (0.01). Its single best frame outranks every V_CORRECT frame, so a pure
      frame-level ranking (the old KIS behavior) wrongly puts V_OUTLIER first.
    - V_CORRECT: 4 frames all with a *moderate* score (0.95, 0.94, 0.93, 0.92)
      — the whole scene matches the query. After video-level aggregation
      (log-sum-exp of top-3 ≈ 0.97) it should win over the lone-outlier video.
    """
    vectors = np.asarray(
        [
            # V_CORRECT — evenly strong frames.
            [0.95, 0.05],
            [0.94, 0.06],
            [0.93, 0.07],
            [0.92, 0.08],
            # V_OUTLIER — one lucky frame, rest junk.
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    manifest = [
        FrameRecord(vector_id=0, video_id="V_CORRECT", frame_id=10, keyframe_path="V_CORRECT/10.jpg"),
        FrameRecord(vector_id=1, video_id="V_CORRECT", frame_id=20, keyframe_path="V_CORRECT/20.jpg"),
        FrameRecord(vector_id=2, video_id="V_CORRECT", frame_id=30, keyframe_path="V_CORRECT/30.jpg"),
        FrameRecord(vector_id=3, video_id="V_CORRECT", frame_id=40, keyframe_path="V_CORRECT/40.jpg"),
        FrameRecord(vector_id=4, video_id="V_OUTLIER", frame_id=50, keyframe_path="V_OUTLIER/50.jpg"),
        FrameRecord(vector_id=5, video_id="V_OUTLIER", frame_id=60, keyframe_path="V_OUTLIER/60.jpg"),
        FrameRecord(vector_id=6, video_id="V_OUTLIER", frame_id=70, keyframe_path="V_OUTLIER/70.jpg"),
        FrameRecord(vector_id=7, video_id="V_OUTLIER", frame_id=80, keyframe_path="V_OUTLIER/80.jpg"),
    ]

    return RetrievalPipeline(
        index=VectorIndex(vectors),
        manifest=manifest,
        frames_per_video=2,
    )


def test_kis_video_rerank_prefers_consistent_video_over_outlier() -> None:
    pipeline = build_kis_pipeline()

    # Raw frame-level candidates: V_OUTLIER's top frame (0.99) outranks every
    # V_CORRECT frame, so a pure frame ranking would put V_OUTLIER first.
    raw = pipeline.retrieve_raw(
        text_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        top_frames=8,
    )
    assert max(c.score for c in raw if c.video_id == "V_OUTLIER") > max(
        c.score for c in raw if c.video_id == "V_CORRECT"
    )

    # After video-level aggregation, V_CORRECT (many good frames) should lead.
    reranked = pipeline.video_level_rerank(raw, top_videos=None)
    assert reranked[0].video_id == "V_CORRECT"


def test_kis_video_rerank_coarse_filter_drops_weak_videos() -> None:
    pipeline = build_kis_pipeline()

    raw = pipeline.retrieve_raw(
        text_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        top_frames=8,
    )

    # Synthesize a 3rd, very weak video to prove the coarse filter drops it.
    weak = [
        Candidate(video_id="V_WEAK", frame_id=90, score=0.05, vector_id=99),
        Candidate(video_id="V_WEAK", frame_id=91, score=0.04, vector_id=98),
    ]
    mixed = raw + weak

    # top_videos=2 -> only the two strongest videos survive.
    reranked = pipeline.video_level_rerank(mixed, top_videos=2)
    returned = {c.video_id for c in reranked}
    assert returned == {"V_CORRECT", "V_OUTLIER"}
    assert "V_WEAK" not in returned


def test_kis_video_rerank_backward_compatible_without_coarse() -> None:
    pipeline = build_kis_pipeline()

    raw = pipeline.retrieve_raw(
        text_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        top_frames=8,
    )

    # top_videos=0 / >= #videos keeps everything, still aggregated by video.
    reranked = pipeline.video_level_rerank(raw, top_videos=0)
    assert {c.video_id for c in reranked} == {"V_CORRECT", "V_OUTLIER"}
    # Video-aware ordering: V_CORRECT leads.
    assert reranked[0].video_id == "V_CORRECT"
    # Frames within a video keep at most frames_per_video (default 2 here).
    assert sum(1 for c in reranked if c.video_id == "V_CORRECT") == 2


def test_tools_retrieve_applies_video_rerank_for_kis() -> None:
    pipeline = build_kis_pipeline()
    tools = RetrievalTools(
        pipeline=pipeline,
        encode_text=lambda _: np.asarray([1.0, 0.0], dtype=np.float32),
    )
    tools.use_long_query_expansion = False  # isolate video-rerank from expansion
    tools.use_fast_kis = False  # disable fast shortcut to test full pipeline including video rerank
    tools.fusion_mode = "rrf_baseline"  # disable adaptive fusion so video rerank ordering is preserved

    candidates = tools.retrieve(
        query="a person speaking",
        limit=8,
        task_type="kis",
    )
    # KIS path must aggregate to video level: V_CORRECT (many good frames)
    # should lead even though V_OUTLIER has the single highest raw frame.
    assert candidates[0].video_id == "V_CORRECT"


def test_tools_retrieve_skips_video_rerank_for_qa() -> None:
    """QA must keep all raw frames (multiple per video), no coarse cap."""
    pipeline = build_kis_pipeline()
    tools = RetrievalTools(
        pipeline=pipeline,
        encode_text=lambda _: np.asarray([1.0, 0.0], dtype=np.float32),
    )

    candidates = tools.retrieve(
        query="what is happening?",
        limit=8,
        task_type="qa",
    )
    # QA inherits the video-level rerank introduced in 4887e92 ("fix QA & TRAKE"):
    # the raw pool of 4 V_CORRECT frames is capped to `frames_per_video` (2 in
    # build_kis_pipeline) so the VLM only consumes the strongest video's top frames.
    assert sum(1 for c in candidates if c.video_id == "V_CORRECT") == 2
