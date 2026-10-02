"""Deterministic tests for the QueryPlan parser (Task 2, III/IV of IMPROVEMENTS.md).

These tests pin the *parsing* behaviour only — no retrieval, no model.  They are
the A/B baseline for the QueryPlan abstraction: any future change to the parser
must keep these green (or deliberately extend them with a new pinned case).
"""

from __future__ import annotations

import pytest

from aic2026.query import parse_query, plan_query
from aic2026.query.plan import ConstraintKind, CountOp, TemporalRole


def _constraints_by_subject(plan):
    return {c.subject: c for c in plan.constraints}


def test_count_gt_parsed():
    p = parse_query("a group of more than 5 people lining up to exercise")
    cons = _constraints_by_subject(p)
    assert "people" in cons
    assert cons["people"].operator == CountOp.GT
    assert cons["people"].value == 5
    assert p.has_count_constraint


def test_only_one_is_eq_one():
    p = parse_query("only one person wears glasses")
    cons = _constraints_by_subject(p)
    assert "person" in cons
    assert cons["person"].operator == CountOp.EQ
    assert cons["person"].value == 1
    # attribute constraint on the worn object
    attrs = [c.subject for c in p.constraints if c.kind == ConstraintKind.ATTRIBUTE]
    assert "glasses" in attrs


def test_exactly_n_count():
    p = parse_query("exactly 3 people wear red hats")
    cons = _constraints_by_subject(p)
    assert "people" in cons
    assert cons["people"].operator == CountOp.EQ
    assert cons["people"].value == 3


def test_bare_n_count():
    p = parse_query("3 red hats on the table")
    cons = _constraints_by_subject(p)
    # Colour is meaningful for a count ("3 red hats" != "3 blue hats"), so the
    # subject keeps the colour; the trailing connector "on" must be excluded.
    assert "red hats" in cons
    assert cons["red hats"].operator == CountOp.EQ
    assert cons["red hats"].value == 3


def test_bare_number_not_shadowing_gt():
    # "more than 5 people" also contains the bare "5 people"; the GT must win.
    p = parse_query("more than 5 people")
    cons = _constraints_by_subject(p)
    assert cons["people"].operator == CountOp.GT
    assert cons["people"].value == 5


def test_multi_event_segmentation():
    p = parse_query(
        "The clip begins with a map. Then it transitions to an aerial shot of a dam. "
        "Followed by a close-up of the dam in the rain."
    )
    assert p.is_multi_event
    roles = [e.temporal_role for e in p.events]
    assert roles[0] == TemporalRole.START
    # consecutive events get an ordered BEFORE chain
    kinds = [(r.kind.value, r.source, r.target) for r in p.temporal_relations]
    assert ("before", 0, 1) in kinds
    assert ("before", 1, 2) in kinds


def test_start_end_roles():
    p = parse_query(
        "The clip starts with green peas added to squid. The clip ends with a pan shaken."
    )
    assert p.events[0].temporal_role == TemporalRole.START
    assert p.events[-1].temporal_role == TemporalRole.END


def test_bare_comma_action_chain_splits():
    # A sequential action chain joined only by commas (no "then"/"after") must
    # split into ordered events with positional start/middle/end roles, so each
    # step is retrievable independently (IV).
    p = parse_query("a man enters a room, opens a red door, walks to a table")
    assert p.is_multi_event
    roles = [e.temporal_role for e in p.events]
    assert roles[0] == TemporalRole.START
    assert roles[-1] == TemporalRole.END
    # the joiner verb is preserved in an event description
    assert any("opens" in e.description for e in p.events)


def test_vietnamese_action_chain_splits_in_order():
    p = parse_query("một người vào phòng, mở cửa rồi ngồi xuống")
    assert p.is_multi_event
    assert len(p.events) >= 3
    assert any("mo cua" in e.description for e in p.events)


def test_descriptive_comma_list_not_split():
    # A comma-separated descriptive list with no action verb must stay ONE event.
    p = parse_query("a dog, a cat and a bird in a park")
    assert not p.is_multi_event
    assert len(p.events) == 1


def test_vietnamese_count_no_diacritics():
    p = parse_query("Mot nhom hon 5 nguoi dang tap the duc")
    assert p.language == "vi"
    cons = _constraints_by_subject(p)
    assert "nguoi" in cons
    assert cons["nguoi"].operator == CountOp.GT
    assert cons["nguoi"].value == 5


def test_vietnamese_only_one():
    p = parse_query("chi mot nguoi deo kinh")
    cons = _constraints_by_subject(p)
    assert "nguoi" in cons
    assert cons["nguoi"].operator == CountOp.EQ
    assert cons["nguoi"].value == 1


def test_qa_question_has_no_count():
    p = parse_query("What is the final number displayed on the scale?")
    assert not p.has_count_constraint
    assert p.modalities.count == 0.0
    assert p.modalities.object == 0.0


def test_short_query_single_event():
    p = parse_query("a dog on a beach")
    assert not p.is_multi_event
    assert len(p.events) == 1
    assert "dog" in p.entities


def test_ocr_modality_gated():
    p = parse_query("a sign that says EXIT appears")
    assert p.modalities.ocr == 1.0
    p2 = parse_query("a dog running on a beach")
    assert p2.modalities.ocr == 0.0


def test_glasses_in_aliases_enables_entity():
    # regression: glasses must be a recognised object concept
    p = parse_query("a person wears glasses")
    assert "glasses" in p.entities


# --- plan_query (LLM planner with rule-based fallback) ----------------------
# The rule-based path is deterministic and needs no model — it is the A/B
# baseline for the planner. The LLM path is exercised only when a local Ollama
# model is available (marked, skipped otherwise) so CI never needs a GPU/model.

def test_plan_query_rule_based_fallback():
    # use_llm=False forces the deterministic parser — always available.
    p = plan_query(
        "a man wearing glasses enters a room, then opens a red door",
        use_llm=False,
    )
    assert isinstance(p, object)
    attrs = [c.subject for c in p.constraints if c.kind == ConstraintKind.ATTRIBUTE]
    assert "glasses" in attrs
    assert p.is_multi_event  # "then" splits into two events


@pytest.mark.skipif(
    __import__("os").environ.get("AIC_RUN_LLM_TESTS") != "1",
    reason="requires local Ollama + AIC_RUN_LLM_TESTS=1",
)
def test_plan_query_llm_smoke():
    p = plan_query(
        "A man wearing glasses enters a room, then opens a red door "
        "and walks to a table with more than 3 books",
        use_llm=True,
        model=__import__("os").environ.get("AIC_PLANNER_MODEL", "qwen3:0.6b"),
    )
    assert len(p.events) >= 1
    # Either the LLM or the rule fallback must surface the count + attribute.
    kinds = {c.kind for c in p.constraints}
    assert ConstraintKind.COUNT in kinds or ConstraintKind.ATTRIBUTE in kinds
