"""Types for the NOTEBOOK / Video Search pipeline (Improvement.md §53-126).

All types are Pydantic models for JSON serialisation and FastAPI compatibility.
The pipeline is:

    raw query → NotebookPlan → (ASR + visual retrieval) → VideoEvidence
    → EventCoverage + TemporalEvidence → NotebookCandidate → NotebookResult
"""

from __future__ import annotations

from pydantic import field_validator

from enum import Enum

from pydantic import BaseModel, Field


class TemporalRole(str, Enum):
    """Temporal role of an event within the query sequence."""

    START = "start"
    MIDDLE = "middle"
    END = "end"
    FIRST = "first"
    LAST = "last"


class RelationKind(str, Enum):
    """Ordering relation between two events or between an event and the clip."""

    BEFORE = "before"
    AFTER = "after"
    FIRST = "first"
    LAST = "last"
    ORDERED = "ordered"


class NotebookEvent(BaseModel):
    """One decomposed event from the planner.

    Each event carries its Vietnamese text (for ASR matching) and optional
    English translation (for visual SigLIP2 retrieval if the user provides one).
    """

    index: int
    description: str
    vi_text: str = ""
    en_text: str = ""
    key_concepts: list[str] = Field(default_factory=list)
    temporal_role: str = "middle"
    importance: float = 1.0  # per-event weight for coverage scoring


class NotebookTemporalRelation(BaseModel):
    """Ordering between two events in the query."""

    kind: str  # "before" | "after" | "ordered" | "first" | "last"
    source: int | None = None
    target: int | None = None


class NotebookPlan(BaseModel):
    """Structured understanding of a NOTEBOOK query.

    Built by the LLM planner (qwen2.5:1.5b, no-think) with a rule-based
    fallback. The plan is the SINGLE source of truth consumed by the retrieval
    and scoring stages — never the raw query string.

    Spec §4: must distinguish ASR-searchable information from visual-only clues.
    """

    raw_text: str = ""
    language: str = "unknown"  # "en" | "vi" | "unknown"
    events: list[NotebookEvent] = Field(default_factory=list)
    asr_queries: list[str] = Field(default_factory=list, max_length=12)
    visual_clues: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    temporal_relations: list[NotebookTemporalRelation] = Field(default_factory=list)
    modality_weights: dict[str, float] = Field(
        default_factory=lambda: {"visual": 1.0, "asr": 1.0, "object": 0.5}
    )

    @field_validator("modality_weights", mode="before")
    @classmethod
    def _coerce_modality_weights(cls, v):
        """Handle LLM returning array [1] instead of dict {"asr":1.0,...}."""
        if isinstance(v, list):
            return {"visual": 1.0, "asr": 1.0, "object": 0.5}
        return v

    # Whether the query expresses a strict temporal order (before/after) that
    # violations should be penalised.
    has_strict_temporal: bool = False
    # Whether to attempt optional visual retrieval (SigLIP2). OFF by default
    # per spec §10 (first version is ASR-only).
    use_visual: bool = False


class ASRSegmentMatch(BaseModel):
    """One ASR segment that matched a query concept."""

    video_id: str
    segment_text: str
    start: float
    end: float
    frame: int | None = None
    matched_query: str
    bm25_score: float
    dense_score: float = 0.0  # BGE-M3 cosine similarity (P0 upgrade)


class EventCoverage(BaseModel):
    """How many planned events a video's evidence satisfies."""

    total_events: int
    matched_events: int
    matched_event_indices: list[int] = Field(default_factory=list)

    @property
    def coverage(self) -> float:
        """matched_events / total_events in [0, 1]."""
        if self.total_events == 0:
            return 0.0
        return self.matched_events / self.total_events

    # Weighted coverage: sum(matched_importance) / sum(all_importance)
    weighted_coverage: float = 0.0


class TemporalEvidence(BaseModel):
    """Temporal consistency of evidence within a video.

    Measures whether matched ASR segments appear in the same order as the
    query's temporal relations (spec §9).
    """

    # Ordered list of matched timestamps (seconds) across the video.
    timestamps: list[float] = Field(default_factory=list)
    # Whether the temporal order matches the query's expected order.
    order_matches: bool = True
    # Number of temporal order violations (inversions).
    violations: int = 0
    # Consistency score in [0, 1]: 1.0 = perfect order, 0.0 = fully inverted.
    consistency: float = 0.0


class ModalitySignal(BaseModel):
    """Normalised score contribution from one retrieval modality."""

    name: str  # "asr" | "visual" | "object"
    raw_score: float
    normalised: float  # min-max scaled to [0,1] within the video pool
    weight: float = 1.0


class NotebookEvidence(BaseModel):
    """All evidence collected for a candidate video."""

    video_id: str
    # ASR segment matches grouped by which query they matched.
    segment_matches: list[ASRSegmentMatch] = Field(default_factory=list)
    # Which planned events have evidence (by index).
    matched_event_indices: list[int] = Field(default_factory=list)
    # Temporal analysis of the matched segments.
    temporal: TemporalEvidence = Field(default_factory=TemporalEvidence)
    # Coverage of planned events.
    coverage: EventCoverage = Field(default_factory=EventCoverage)
    # Modality-level scores.
    modality_signals: list[ModalitySignal] = Field(default_factory=list)
    # Top evidence timestamps for display (spec §15).
    evidence_timestamps: list[float] = Field(default_factory=list)

    @property
    def asr_snippets(self) -> list[str]:
        """Brief ASR text snippets for the top evidence segments."""
        return [m.segment_text[:200] for m in self.segment_matches[:5]]


class NotebookCandidate(BaseModel):
    """A ranked candidate video for NOTEBOOK output (spec §15).

    This is the final output unit — one row in the results list.
    """

    video_id: str
    score: float
    # Evidence timestamps to display (spec §15: "evidence timestamps").
    evidence_timestamps: list[float] = Field(default_factory=list)
    # Matched event indices (spec §15: "matched events").
    matched_events: list[int] = Field(default_factory=list)
    # ASR evidence snippets (spec §15: "ASR evidence snippets").
    asr_snippets: list[str] = Field(default_factory=list)
    # How many of the query's events this video explains.
    event_coverage: float = 0.0
    # Temporal consistency score in [0, 1].
    temporal_consistency: float = 0.0
    # Breakdown of modality contributions.
    modality_signals: list[ModalitySignal] = Field(default_factory=list)
    # How many ASR segments matched (diversity indicator).
    evidence_count: int = 0
    # Best evidence frame (highest-scoring ASR segment) for quick true/false check.
    frame_id: int | None = None
    keyframe_path: str | None = None
    evidence_timestamp: float | None = None  # timestamp of the best frame
    # Source video path for the [Open Video] button (spec §15).
    video_path: str | None = None


class NotebookResult(BaseModel):
    """Full NOTEBOOK result for one query (spec §11, §12, §15)."""

    query_id: str
    plan: NotebookPlan
    candidates: list[NotebookCandidate]
    # Whether a second LLM reasoning pass was triggered (confidence-based, §12).
    second_pass: bool = False
    # LLM latency in seconds (planner + optional second pass).
    llm_latency: float = 0.0
    # Total pipeline latency in seconds.
    total_latency: float = 0.0
    # Error message if the pipeline failed.
    error: str | None = None


class NotebookRequest(BaseModel):
    """API request body for the /tasks/notebook/run endpoint."""

    query_id: str = "live-query"
    text: str = Field(min_length=1, max_length=4096)
    # Vietnamese query for ASR matching (spec §3: ASR is Vietnamese).
    asr_query: str | None = None
    # Number of top videos to return (spec §1 UI: "Top N").
    top_n: int = 5
    # Override LLM model (default qwen2.5:1.5b, per user instruction).
    llm_model: str = "qwen2.5:1.5b"
    ollama_url: str = "http://127.0.0.1:11434"
    # Optional visual retrieval (off by default per spec §10).
    use_visual: bool = False
    siglip2_model: str = "google/siglip2-base-patch16-224"
    # Embedding backend for visual retrieval.
    embedding_backend: str = "official_clip"
    manifest_path: str | None = None
    features_path: str | None = None
