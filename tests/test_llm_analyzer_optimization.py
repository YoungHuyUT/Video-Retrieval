"""Test for LLM analyzer Gemini fallback behavior."""

import pytest
from unittest.mock import patch, MagicMock

from aic2026.agent.local_llm import LLMInvocationError
from aic2026.query.llm_analyzer import (
    analyze_query,
    analyze_query_llm,
)


def test_llm_analyzer_gemini_fallback_on_error():
    """Verify analyze_query_llm raises LLMInvocationError when Gemini fails (no retry)."""
    with patch("aic2026.query.llm_analyzer._build_client") as mock_build:
        mock_client = MagicMock()
        mock_client.structured.side_effect = LLMInvocationError("Gemini API request failed")
        mock_build.return_value = mock_client

        # Direct call should raise LLMInvocationError immediately (no retry)
        with pytest.raises(LLMInvocationError):
            analyze_query_llm("test query", provider="gemini")

        # Ensure _build_client was called exactly once
        assert mock_build.call_count == 1


def test_analyze_query_fallback_on_llm_error():
    """Verify analyze_query returns deterministic fallback when LLM fails."""
    with patch("aic2026.query.llm_analyzer.analyze_query_llm") as mock_llm:
        mock_llm.side_effect = LLMInvocationError("Gemini API request failed")

        result = analyze_query("a red car parked near a building", use_llm=True)
        assert result is not None
        assert len(result.events) > 0
        assert "car" in result.entities or "building" in result.entities or len(result.expansion_variants) > 0
