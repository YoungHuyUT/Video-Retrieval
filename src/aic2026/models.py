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


class Candidate(BaseModel):
    video_id: str
    frame_id: int
    score: float
    vector_id: int | None = None
    keyframe_path: str | None = None
    answer: str | None = None
    event_frames: list[int] | None = None


class Query(BaseModel):
    query_id: str
    type: QueryType
    text: str
    question: str | None = None
    events: list[str] = Field(default_factory=list)


class GroundTruth(BaseModel):
    video_id: str
    ranges: list[tuple[int, int]]
    answer: str | None = None


def project_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
