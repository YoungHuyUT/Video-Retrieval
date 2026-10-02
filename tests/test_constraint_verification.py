"""Phase 7: object / count / attribute constraint verification (spec X).

These tests exercise ``verify_constraints`` directly (deterministic, no model)
and confirm the weak-label-safe contract: a video WITHOUT object labels is a
NO-OP (never penalised), because the user's self-extracted keyframes carry no
detector output. Only when a video HAS labels but they contradict a constraint
do we apply a *bounded* penalty.
"""

from aic2026.models import Candidate, FrameRecord
from aic2026.query.plan import Constraint, ConstraintKind, CountOp
from aic2026.verification import verify_constraints


def _records() -> dict[int, FrameRecord]:
    """vector_id → FrameRecord. Two videos: one labelled, one not."""

    return {
        0: FrameRecord(
            vector_id=0, video_id="L01_V001", frame_id=10,
            keyframe_path="L01_V001/10.jpg", object_labels=["person", "glasses"],
        ),
        1: FrameRecord(
            vector_id=1, video_id="L01_V001", frame_id=20,
            keyframe_path="L01_V001/20.jpg", object_labels=["person"],
        ),
        2: FrameRecord(
            vector_id=2, video_id="L26_V001", frame_id=30,
            keyframe_path="L26_V001/30.jpg", object_labels=[],  # unlabelled
        ),
        3: FrameRecord(
            vector_id=3, video_id="L26_V001", frame_id=40,
            keyframe_path="L26_V001/40.jpg", object_labels=[],  # unlabelled
        ),
    }


def _candidates() -> list[Candidate]:
    return [
        Candidate(video_id="L01_V001", frame_id=10, score=0.50, vector_id=0),
        Candidate(video_id="L26_V001", frame_id=30, score=0.60, vector_id=2),
    ]


def test_no_constraints_is_noop() -> None:
    cands = _candidates()
    out = verify_constraints(cands, [], record_lookup=_records())
    assert [c.score for c in out] == [0.50, 0.60]


def test_attribute_match_boosts() -> None:
    cands = _candidates()
    constraints = [
        Constraint(kind=ConstraintKind.ATTRIBUTE, subject="glasses"),
    ]
    out = verify_constraints(cands, constraints, record_lookup=_records())
    by_vid = {c.video_id: c.score for c in out}
    # L01_V001 has a frame with "glasses" → +0.03 boost.
    assert by_vid["L01_V001"] == 0.53
    # L26_V001 has NO labels → weak-label-safe no-op (unchanged).
    assert by_vid["L26_V001"] == 0.60


def test_attribute_mismatch_penalises_labelled_video_only() -> None:
    cands = _candidates()
    constraints = [
        Constraint(kind=ConstraintKind.ATTRIBUTE, subject="cat"),  # absent everywhere
    ]
    out = verify_constraints(cands, constraints, record_lookup=_records())
    by_vid = {c.video_id: c.score for c in out}
    # L01_V001 HAS labels but none mention "cat" → -0.05 penalty.
    assert abs(by_vid["L01_V001"] - (0.50 - 0.05)) < 1e-9
    # L26_V001 has NO labels → no-op (never penalised).
    assert by_vid["L26_V001"] == 0.60


def test_count_satisfied_boosts() -> None:
    cands = _candidates()
    # L01_V001's reachable labelled frame (vector 0) mentions "person".
    # Want >= 1 → satisfied.
    constraints = [
        Constraint(
            kind=ConstraintKind.COUNT, subject="person",
            operator=CountOp.GE, value=1,
        ),
    ]
    out = verify_constraints(cands, constraints, record_lookup=_records())
    by_vid = {c.video_id: c.score for c in out}
    assert by_vid["L01_V001"] == 0.53  # +0.03 boost
    assert by_vid["L26_V001"] == 0.60  # no labels → no-op


def test_count_violated_penalises_labelled_video() -> None:
    cands = _candidates()
    # "person" appears in 2 frames. Want > 5 → violated.
    constraints = [
        Constraint(
            kind=ConstraintKind.COUNT, subject="person",
            operator=CountOp.GT, value=5,
        ),
    ]
    out = verify_constraints(cands, constraints, record_lookup=_records())
    by_vid = {c.video_id: c.score for c in out}
    assert abs(by_vid["L01_V001"] - (0.50 - 0.06)) < 1e-9  # -0.06 penalty
    assert by_vid["L26_V001"] == 0.60  # no labels → no-op


def test_unlabelled_video_untouched_when_other_violates() -> None:
    cands = [
        Candidate(video_id="L01_V001", frame_id=10, score=0.10, vector_id=0),
        Candidate(video_id="L26_V001", frame_id=30, score=0.99, vector_id=2),
    ]
    constraints = [
        Constraint(kind=ConstraintKind.ATTRIBUTE, subject="nonexistent"),
    ]
    out = verify_constraints(cands, constraints, record_lookup=_records())
    by_vid = {c.video_id: c.score for c in out}
    # Even though L01_V001 is heavily penalised, L26_V001 (no labels) is the
    # no-op top result and must NOT be pushed below the penalised video by a
    # spurious penalty it never earned.
    assert by_vid["L26_V001"] == 0.99
    assert by_vid["L01_V001"] < by_vid["L26_V001"]


def test_accent_case_insensitive_subject_match() -> None:
    cands = _candidates()
    # Record label is "glasses"; subject uses accented/cased variant.
    constraints = [
        Constraint(kind=ConstraintKind.ATTRIBUTE, subject="GLASSES"),
    ]
    out = verify_constraints(cands, constraints, record_lookup=_records())
    by_vid = {c.video_id: c.score for c in out}
    assert by_vid["L01_V001"] == 0.53  # matched despite case difference


def test_vector_id_none_is_unverifiable() -> None:
    cands = [
        Candidate(video_id="Lxx_V001", frame_id=10, score=0.40, vector_id=None),
    ]
    constraints = [
        Constraint(kind=ConstraintKind.ATTRIBUTE, subject="cat"),
    ]
    out = verify_constraints(cands, constraints, record_lookup=_records())
    assert out[0].score == 0.40  # cannot look up → no-op
