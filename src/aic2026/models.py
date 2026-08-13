from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

QueryType = str


class FrameRecord(BaseModel):
    vector_id: int
    video_id: str
    frame_id: int
    keyframe_path: str
    object_labels: list[str] = Field(default_factory=list)
    title: str | None = None
    description: str | None = None
    video_path: str | None = None
    object_path: str | None = None
    metadata_path: str | None = None
    clip_feature_index: int | None = None


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
