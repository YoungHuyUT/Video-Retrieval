import pytest

from aic2026.models import Candidate, Query
from aic2026.submission import (
    competition_answer,
    csv_row,
    package_submission,
    validate_candidates,
    write_submission,
)


def test_submission_limits_and_orders_scores():
    q = Query(query_id="q", type="kis", text="test")
    answer = validate_candidates(q, [Candidate(video_id="a", frame_id=1, score=.2), Candidate(video_id="b", frame_id=2, score=.9)])
    assert [c.video_id for c in answer] == ["b", "a"]

def test_qa_must_have_answer():
    with pytest.raises(ValueError):
        validate_candidates(Query(query_id="q", type="qa", text="x"), [Candidate(video_id="a", frame_id=1, score=1)])

def test_competition_output_has_only_brief_fields():
    kis = Query(query_id="k", type="kis", text="x")
    assert competition_answer(kis, Candidate(video_id="v", frame_id=9, score=.8, vector_id=3)) == {"video_id": "v", "frame_id": 9}
    trake = Query(query_id="t", type="trake", text="x", events=["a", "b"])
    assert competition_answer(trake, Candidate(video_id="v", frame_id=1, score=.8, event_frames=[9, 20])) == {"video_id": "v", "frame_ids": [9, 20]}


def test_csv_row_kis_has_no_whitespace():
    q = Query(query_id="q1", type="kis", text="t")
    assert csv_row(q, Candidate(video_id="L25_V001", frame_id=23092, score=1.0)) == "L25_V001,23092"


def test_csv_row_qa_quotes_only_when_needed():
    q = Query(query_id="q2", type="qa", text="t")
    # Answer with a comma MUST be wrapped in double quotes.
    assert csv_row(q, Candidate(video_id="L01_V028", frame_id=3450, score=1.0, answer="Co 3 nguoi, bao gom nam va nu")) == 'L01_V028,3450,"Co 3 nguoi, bao gom nam va nu"'
    # Simple answer must NOT be wrapped.
    assert csv_row(q, Candidate(video_id="L02_V011", frame_id=1200, score=1.0, answer="Nam nguoi")) == "L02_V011,1200,Nam nguoi"


def test_csv_row_qa_escapes_inner_quotes():
    q = Query(query_id="q2", type="qa", text="t")
    assert csv_row(q, Candidate(video_id="L04_V012", frame_id=4100, score=1.0, answer='Anh ay noi "Tuyet voi"')) == 'L04_V012,4100,"Anh ay noi ""Tuyet voi"""'


def test_csv_row_trake_emits_frame_sequence():
    q = Query(query_id="q3", type="trake", text="t", events=["a", "b", "c", "d"])
    assert csv_row(q, Candidate(video_id="L10_V001", frame_id=1, score=1.0, event_frames=[1200, 1850, 2100, 2450])) == "L10_V001,1200,1850,2100,2450"


def test_write_submission_and_package(tmp_path):
    k = Query(query_id="q1", type="kis", text="t")
    q = Query(query_id="q2", type="qa", text="t")
    t = Query(query_id="q3", type="trake", text="t", events=["a", "b", "c", "d"])
    items = [
        (k, [Candidate(video_id="L25_V001", frame_id=23092, score=1)]),
        (q, [Candidate(video_id="L01_V028", frame_id=3450, score=1, answer="Nam nguoi")]),
        (t, [Candidate(video_id="L10_V001", frame_id=1, score=1, event_frames=[1200, 1850, 2100, 2450])]),
    ]
    out_dir = tmp_path / "submission_src"
    written = write_submission(items, out_dir)
    assert {p.name for p in written} == {
        "query-q1-kis.csv",
        "query-q2-qa.csv",
        "query-q3-trake.csv",
    }
    # KIS file content: no header, LF terminated, no surrounding whitespace.
    assert (out_dir / "query-q1-kis.csv").read_text(encoding="utf-8") == "L25_V001,23092\n"

    zip_path = tmp_path / "team_round1.zip"
    package_submission(out_dir, zip_path)
    from zipfile import ZipFile

    assert ZipFile(zip_path).namelist() == [
        "submission/query-q1-kis.csv",
        "submission/query-q2-qa.csv",
        "submission/query-q3-trake.csv",
    ]

