"""Tests for Phase 5 deterministic query expansion + modality routing.

Expansion is pure template logic over a QueryPlan — no LLM, no model, fast.
"""

from __future__ import annotations

from aic2026.query.expansion import expand_query, expand_text
from aic2026.query.parser import parse_query
from aic2026.query.plan import ModalityWeights, QueryPlan


def _plan_with(**overrides) -> QueryPlan:
    base = dict(
        raw_text="a man wearing glasses enters a room",
        global_query="a man wearing glasses enters a room",
        language="en",
        entities=["glasses", "room"],
        actions=["enters"],
        attributes=["glasses"],
        constraints=[],
        events=[],
        temporal_relations=[],
        modalities=ModalityWeights(),
    )
    base.update(overrides)
    return QueryPlan(**base)


def test_expand_keeps_global_as_q0():
    plan = _plan_with()
    variants = expand_query(plan)
    assert variants[0] == "a man wearing glasses enters a room"


def test_expand_produces_distinct_variants():
    plan = _plan_with()
    variants = expand_query(plan)
    # Q0 global, Q1 entities, Q2 actions, Q3 attr×entity.
    assert len(variants) >= 3
    # No duplicate variants (case/space-insensitive).
    lowered = {v.casefold() for v in variants}
    assert len(lowered) == len(variants)


def test_expand_caps_at_max_variants():
    plan = _plan_with(
        attributes=["red", "blue"],
        entities=["door", "car", "table"],
    )
    variants = expand_query(plan, max_variants=4)
    assert len(variants) <= 4


def test_modality_routing_drops_zero_weight_facet():
    # object modality off; only semantic (actions) facet relevant. Q0 (global)
    # always kept, but the EXTRA entity/attribute variants must be dropped.
    mod = ModalityWeights()
    mod.object = 0.0  # entities / attribute×entity dropped
    plan = _plan_with(modalities=mod)
    variants = expand_query(plan, route_by_modality=True)
    # Beyond Q0, only the actions variant ("enters") should remain; the entity
    # variant ("glasses room") and attribute×entity pair must be gone.
    extras = [v for v in variants if v.casefold() != plan.global_query.casefold()]
    assert extras == ["enters"]


def test_no_routing_keeps_all_nonempty_facets():
    mod = ModalityWeights()
    mod.object = 0.0
    plan = _plan_with(modalities=mod)
    variants = expand_query(plan, route_by_modality=False)
    joined = " | ".join(variants).casefold()
    assert "glasses" in joined
    assert "room" in joined


def test_expand_text_fallback_parser():
    variants = expand_text("a woman opens a red door and walks to a table")
    assert variants[0]
    assert len(variants) >= 2
