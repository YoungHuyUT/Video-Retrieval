from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

QueryType = str


class FrameRecord(BaseModel):
    """Per-frame record. Object labels stay here because they change per frame.

    Video-level text (title/description/keywords) used to be denormalised into
    every record, which inflated the manifest ~10x and broke BM25 IDF (the same
    video's keywords were counted 200 times). Those fields are now kept only as
    deprecated defaults so old manifests on disk still load — see
    ``retrieval.video_metadata.VideoMetadataStore`` for the canonical store.
    """

    vector_id: int
    video_id: str
    frame_id: int
    # Frame coordinates may be a frame index or milliseconds (BTC VFR videos).
    frame_unit: str | None = None
    keyframe_path: str
    object_labels: list[str] = Field(default_factory=list)
    # Deprecated: prefer VideoMetadataStore.get(video_id).metadata_keywords.
    # Kept as defaults so legacy manifests still validate; new code should not
    # read these fields.
    metadata_keywords: list[str] = Field(default_factory=list)
    title: str | None = None
    description: str | None = None
    video_path: str | None = None
    object_path: str | None = None
    metadata_path: str | None = None
    clip_feature_index: int | None = None
    # Set True once OCR text has been merged into ``object_labels`` for this
    # frame. Lets ``ocr-manifest --resume`` skip already-finished frames so an
    # interrupted (or partial) run can continue without redoing work. Defaults
    # to False so legacy manifests on disk still load.
    ocr_done: bool = False
    # --- Shot-based extraction (spec §5, §10) ---------------------------------
    # Added for the shot-adaptive pipeline; default None so the legacy official
    # manifest (uniform 1 FPS) still loads. ``shot_id`` groups frames within the
    # same detected shot; ``timestamp`` is the frame's position in seconds so the
    # query-time temporal zoom-in (spec §6) can pick neighbours by window.
    shot_id: int | None = None
    timestamp: float | None = None
    # Lightweight-extraction provenance (spec §2, §10). One of "btc" / "uniform" /
    # "motion" recording where the frame came from. Default None for legacy
    # manifests. BTC frames reuse the original ``official`` keyframe images (no
    # re-decode); uniform/motion frames are freshly decoded into the lightweight
    # keyframe dir.
    source: str | None = None
    # Embedding provenance for versioned re-builds (spec §5, §11). When None the
    # record came from the legacy BTC pipeline where versioning was not tracked.
    model_version: str | None = None
    embedding_version: str | None = None


class Candidate(BaseModel):
    video_id: str
    frame_id: int
    score: float
    vector_id: int | None = None
    keyframe_path: str | None = None
    # Playback metadata is optional for legacy manifests.  It lets the UI open
    # the source video at the exact evidence frame without exposing disk paths.
    timestamp: float | None = None
    frame_unit: str | None = None
    # Nominal/source video frame rate. For VFR material this is the stream's
    # declared/base FPS; timestamps (PTS) remain the exact playback coordinate.
    fps: float | None = None
    fps_source: str | None = None
    event_timestamps: list[float] | None = None
    answer: str | None = None
    event_frames: list[int] | None = None
    # Debug info: actual component scores for dashboard display
    debug: dict | None = None


class Query(BaseModel):
    query_id: str
    type: QueryType
    text: str
    question: str | None = None
    events: list[str] = Field(default_factory=list)
    # Query tiếng Việt riêng cho ASR scoring (khớp transcript Whisper). Khi query
    # chính là tiếng Anh mà ASR bật, transcript (tiếng Việt) = 0 khớp BM25; ô này
    # cho phép user cung cấp từ khóa tiếng Việt tách biệt. None/"" = dùng text chính.
    asr_query: str | None = None


class GroundTruth(BaseModel):
    video_id: str
    ranges: list[tuple[int, int]]
    answer: str | None = None


def project_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
