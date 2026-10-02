"""Unit test cho gating §8: expand query khi có object terms / dài / multi-event.

_should_expand_query decides whether the query benefits from multi-variant
expansion.  Queries with detectable object terms ("red car", "person speaking")
are expanded because CLIP's broad embedding space maps similar objects to nearby
vectors.  Long queries, action chains, animal/disaster keywords, and
multi-event descriptions also benefit.
"""

from __future__ import annotations

from aic2026.agent.tools import RetrievalTools


def test_short_object_query_expanded():
    """Queries with detectable object terms are expanded."""
    assert RetrievalTools._should_expand_query("a red car") is True


def test_short_query_with_entity_expanded():
    """'person' is a detectable object term → expansion."""
    assert RetrievalTools._should_expand_query("a person speaking") is True


def test_long_query_expanded():
    q = "a person walks into the room then opens the refrigerator and takes a bottle"
    assert RetrievalTools._should_expand_query(q) is True


def test_action_chain_expanded():
    assert RetrievalTools._should_expand_query("a man enters, sits down") is True


def test_multi_clause_action_chain():
    assert RetrievalTools._should_expand_query(
        "a person enters a room, opens a door, then sits down"
    ) is True


def test_very_long_single_sentence():
    assert RetrievalTools._should_expand_query(
        "the quick brown fox jumps over the lazy dog near a big tree"
    ) is True


def test_looks_multi_event_helpers():
    """Action-chain detection via comma + action verb pattern."""
    assert RetrievalTools._should_expand_query("enters room, sits down") is True
    # "a red car at an intersection" has detectable objects (red, car)
    assert RetrievalTools._should_expand_query("a red car at an intersection") is True


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
