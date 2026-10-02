"""Phase 7: object / count / attribute constraint verification (spec X + §10).

WHY (spec X, rule 12 "Exact count/attribute phải có structured verification")
----------------------------------------------------------------------------
The parser already extracts structured ``Constraint`` objects (COUNT /
ATTRIBUTE) from the query, but nothing *applies* them during retrieval — a
CLIP-only match can surface a video whose frames contradict the stated count
or lack the stated attribute. This module is the missing verification stage:
it re-orders candidates by how well their (per-video) object labels satisfy
the constraints.

DESIGN — weak-label-safe (rule: corpus của user THIẾU object label)
---------------------------------------------------------------
* Verification reads ``object_labels`` only. When a video's candidate frames
  have NO labels at all, the constraint is treated as *unverifiable* and is a
  NO-OP for that video (no boost, no penalty). This is essential: the user's
  self-extracted keyframes carry no detector output, so a naive "penalise
  missing attribute" rule would nuke the entire ranking. We only act when we
  actually have evidence.
* A satisfied constraint gives a bounded boost; a violated *hard* constraint
  gives a bounded penalty (spec: "Hard constraint sai -> penalty mạnh") — but
  bounded so it re-orders, never destroys, the candidate set. CLIP evidence is
  never discarded outright (mirrors the soft object-evidence rule elsewhere).
* Fully deterministic, no LLM/VLM, no new model — keeps it A/B-able and fast.
* OFF by default (``use_constraint_verification=False``) so the legacy
  CLIP+BM25 ranking stays baseline A.

The returned candidates keep their relative CLIP order within a video; only a
constraint-driven delta is added to ``score``.
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from typing import Callable

from aic2026.models import Candidate
from aic2026.query.plan import Constraint, ConstraintKind, CountOp


def _fold(value: str) -> str:
    """Accent/case-insensitive fold for substring label matching (VI+EN)."""

    value = unicodedata.normalize("NFC", value).casefold()
    value = value.translate(str.maketrans({"đ": "d"}))
    return "".join(
        c for c in unicodedata.normalize("NFD", value)
        if unicodedata.category(c) != "Mn"
    )


def _label_blob(record) -> str:
    """Accent-folded, space-joined object labels for one frame record."""

    labels = getattr(record, "object_labels", None) or []
    return _fold(" ".join(labels))


def _has_label(labels_blob: str, subject: str) -> bool:
    """True when the folded subject appears as a substring of the label blob."""

    folded = _fold(subject)
    if not folded:
        return False
    return folded in labels_blob


def verify_constraints(
    candidates: list[Candidate],
    constraints: list[Constraint],
    record_lookup: Callable[[int], object] | dict[int, object],
    # Bounded score deltas (CLIP scores ~0–1, RRF ~0.001–0.05).
    attribute_boost: float = 0.03,
    attribute_penalty: float = 0.05,
    count_penalty: float = 0.06,
    count_boost: float = 0.03,
) -> list[Candidate]:
    """Re-order ``candidates`` by how well their frames satisfy ``constraints``.

    ``record_lookup`` maps ``candidate.vector_id`` → a frame record exposing
    ``object_labels``. It may be a ``dict`` (``vector_id`` → record, as produced
    by ``tools._record_lookup``) or a callable ``vector_id → record``. Frames
    with ``vector_id=None`` or no labels are treated as unverifiable (no-op).

    For each candidate we inspect every frame of its *video* (gathered from the
    records reachable via the candidates' vector_ids) so a COUNT constraint sees
    the whole video, not just one frame. ATTRIBUTE only needs one matching
    frame; COUNT needs an aggregate over the video's labelled frames.

    Returns the same candidate objects, with ``score`` adjusted in place.
    """

    if not constraints or not candidates:
        return candidates

    # Normalise dict → callable so the rest of the code only calls.
    if not callable(record_lookup):
        lookup = record_lookup.get
    else:
        lookup = record_lookup

    # Gather, per video, the set of labelled frame records available through the
    # candidates. We only look at records that (a) have a vector_id we can look
    # up and (b) carry object labels — frames without labels are unverifiable.
    by_video: dict[str, list[object]] = defaultdict(list)
    for candidate in candidates:
        if candidate.vector_id is None:
            continue
        record = lookup(candidate.vector_id)
        if record is None:
            continue
        labels = getattr(record, "object_labels", None)
        if not labels:
            continue  # no detector output → unverifiable, skip (weak-label-safe)
        by_video[candidate.video_id].append(record)

    # Pre-resolve which constraints are usable: any non-empty subject. A video
    # with labels but no match is handled per-candidate below (penalty); a video
    # with NO labels at all is a no-op (weak-label-safe).
    verifiable = [
        constraint
        for constraint in constraints
        if _fold(constraint.subject)
    ]

    if not verifiable:
        return candidates

    # Per-video precompute: does it have ANY labelled frame matching each
    # attribute subject? And what is the max per-frame count of a counted concept?
    video_attr_match: dict[str, dict[str, bool]] = {}
    video_concept_count: dict[str, dict[str, int]] = {}
    for video_id, recs in by_video.items():
        attr_seen: dict[str, bool] = {}
        count_seen: dict[str, int] = {}
        for constraint in verifiable:
            subject = _fold(constraint.subject)
            if constraint.kind == ConstraintKind.ATTRIBUTE:
                matched = any(
                    _has_label(_label_blob(r), constraint.subject)
                    or _partial_label_hit(r, subject)
                    for r in recs
                )
                attr_seen[subject] = matched
            elif constraint.kind == ConstraintKind.COUNT:
                # Count how many labelled frames mention the concept at least
                # once — a coarse presence count across the video's labelled
                # frames (not exact instance counting, which needs a detector
                # box count; this is the available signal on the corpus).
                n = sum(
                    1
                    for r in recs
                    if _has_label(_label_blob(r), constraint.subject)
                    or _partial_label_hit(r, constraint.subject)
                )
                count_seen[subject] = n
        video_attr_match[video_id] = attr_seen
        video_concept_count[video_id] = count_seen

    for candidate in candidates:
        video_id = candidate.video_id
        attr_match = video_attr_match.get(video_id)
        count_seen = video_concept_count.get(video_id)
        if attr_match is None and count_seen is None:
            # Video has no labelled frames → unverifiable → no-op (weak-label-safe).
            continue

        delta = 0.0
        for constraint in verifiable:
            subject = _fold(constraint.subject)
            if constraint.kind == ConstraintKind.ATTRIBUTE:
                if attr_match is None:
                    continue
                if attr_match.get(subject, False):
                    delta += attribute_boost
                else:
                    # Video HAS labels but NONE mention the subject → penalty.
                    delta -= attribute_penalty
            elif constraint.kind == ConstraintKind.COUNT:
                if count_seen is None:
                    continue
                actual = count_seen.get(subject, 0)
                want = constraint.value
                op = constraint.operator
                if want is None or op is None:
                    continue
                satisfied = _count_satisfied(actual, want, op)
                if satisfied:
                    delta += count_boost
                else:
                    delta -= count_penalty

        if delta != 0.0:
            candidate.score = float(candidate.score) + delta

    return candidates


def _partial_label_hit(record, subject: str) -> bool:
    """Allow a multi-word subject to match a label that contains one of its
    tokens (e.g. subject "red hat" matches label "hat"). Avoids over-strict
    exact-substring misses when the detector emits shorter labels.
    """

    tokens = [t for t in subject.split() if len(t) >= 3]
    if not tokens:
        return False
    blob = _label_blob(record)
    return any(token in blob for token in tokens)


def _count_satisfied(actual: int, want: int, op: CountOp) -> bool:
    """Evaluate a count constraint against the counted instances."""

    if op == CountOp.EQ:
        return actual == want
    if op == CountOp.GT:
        return actual > want
    if op == CountOp.GE:
        return actual >= want
    if op == CountOp.LT:
        return actual < want
    if op == CountOp.LE:
        return actual <= want
    return False
