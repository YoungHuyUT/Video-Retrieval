"""QueryPlan — structured representation of a retrieval query (XXII, III).

A ``QueryPlan`` is the single source of truth that the rest of the pipeline
consumes instead of the raw query string.  It is built by a **deterministic,
rule-based parser** (``aic2026.query.parser``) — no LLM, no network — so every
field is inspectable and debuggable (XXIII).

The plan separates *what the query asks for* into orthogonal dimensions:

* ``entities`` / ``actions`` / ``attributes`` — the visual vocabulary.
* ``constraints`` — hard/soft numeric or logical requirements
  (``person_count > 5``, ``glasses_count == 1``, ``red_hat_count == 3``).
* ``events`` — temporal segments for long / multi-event queries, each with a
  ``temporal_role`` (START / MIDDLE / END / FIRST / LAST).
* ``temporal_relations`` — ordering between events (BEFORE / AFTER / …).
* ``modalities`` — which retrieval modalities the query actually needs, so the
  planner can skip useless sources (OCR/ASR/object/count).

These map 1:1 onto the target architecture (IMPROVEMENTS.md §III/§IV/§XVII):
raw query → Query Understanding → Query Plan → multimodal retrieval.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class TemporalRole(str, Enum):
    """Role of an event within the temporal sequence of a query."""

    START = "start"
    MIDDLE = "middle"
    END = "end"
    FIRST = "first"
    LAST = "last"


class RelationKind(str, Enum):
    """Kind of temporal relation between two events (or an event and the clip)."""

    BEFORE = "before"
    AFTER = "after"
    FIRST = "first"
    LAST = "last"
    ORDERED = "ordered"  # E1 < E2 < ... < En, strict monotonic


class ConstraintKind(str, Enum):
    """What a constraint restricts."""

    COUNT = "count"          # cardinality of an object/entity
    ATTRIBUTE = "attribute"  # presence/value of a visual attribute
    TEMPORAL = "temporal"    # first/last/before/after occurrence


class CountOp(str, Enum):
    """Comparison operator for a count constraint."""

    GT = ">"
    GE = ">="
    EQ = "=="
    LT = "<"
    LE = "<="


class Constraint(BaseModel):
    """A single structured constraint extracted from the query.

    Examples
    --------
    * ``person_count > 5``   → kind=COUNT, subject="person", op=GT, value=5
    * ``glasses_count == 1`` → kind=COUNT, subject="glasses", op=EQ, value=1
    * ``red_hat_count == 3`` → kind=COUNT, subject="red hat", op=EQ, value=3
    * ``wears glasses``      → kind=ATTRIBUTE, subject="glasses"
    """

    kind: ConstraintKind
    subject: str
    operator: CountOp | None = None
    value: int | None = None
    raw: str = ""  # verbatim substring that produced this constraint


class Event(BaseModel):
    """One temporal segment of a long / multi-event query.

    Each event can be retrieved independently (IV); the planner later aligns
    them with ``align_events_dp`` (reused for KIS per XI).
    """

    index: int
    description: str
    entities: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)
    spatial_relations: list[str] = Field(default_factory=list)
    visual_priority: float = 1.0
    asr_priority: float = 0.0
    temporal_role: TemporalRole = TemporalRole.MIDDLE


class TemporalRelation(BaseModel):
    """An ordering between two events, or a global temporal tag."""

    kind: RelationKind
    source: int | None = None  # event index
    target: int | None = None  # event index


class ModalityWeights(BaseModel):
    """Which modalities the query actually needs (XVII, XVIII).

    All default to 1.0 (active).  The planner can zero-out a modality when the
    query contains no signal for it (e.g. no text → OCR/ASR weight 0), which
    keeps retrieval cheap and avoids spurious evidence (VIII, IX).
    """

    semantic: float = 1.0
    visual: float = 1.0
    temporal: float = 1.0
    ocr: float = 1.0
    asr: float = 1.0
    object: float = 1.0
    count: float = 1.0
    metadata: float = 1.0


class QueryPlan(BaseModel):
    """Structured understanding of a single query (III, XXII)."""

    raw_text: str = ""
    global_query: str = ""
    language: Literal["en", "vi", "unknown"] = "unknown"

    entities: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)
    spatial_relations: list[str] = Field(default_factory=list)

    constraints: list[Constraint] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    temporal_relations: list[TemporalRelation] = Field(default_factory=list)

    modalities: ModalityWeights = Field(default_factory=ModalityWeights)

    @property
    def is_multi_event(self) -> bool:
        """True when the query describes an ordered sequence of events."""
        return len(self.events) > 1

    @property
    def has_count_constraint(self) -> bool:
        return any(c.kind == ConstraintKind.COUNT for c in self.constraints)

    @property
    def requested_objects(self) -> list[str]:
        """Canonical object concepts the query asks about (reuses detector vocab)."""
        objs: list[str] = []
        for e in self.entities:
            if e not in objs:
                objs.append(e)
        return objs
