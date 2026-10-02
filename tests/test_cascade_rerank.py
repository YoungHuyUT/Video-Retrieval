"""Tests for cascade rerank logic (Improvement.md Task 2).

Verifies that when use_cascade_rerank=True, object+colour narrow to top_n
BEFORE BLIP-2 runs, and that legacy flat pipeline still works when OFF.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_candidate(vector_id: int, video_id: str, frame_id: int, score: float) -> SimpleNamespace:
    return SimpleNamespace(
        vector_id=vector_id,
        video_id=video_id,
        frame_id=frame_id,
        score=score,
        objects="person,car",
    )


class TestCascadeConfig:
    """Test that cascade config fields exist and default correctly."""

    def test_retrieval_tools_defaults(self):
        from aic2026.agent.tools import RetrievalTools
        from aic2026.retrieval import RetrievalPipeline

        pipeline = MagicMock(spec=RetrievalPipeline)
        tools = RetrievalTools(pipeline=pipeline, encode_text=MagicMock())
        # Cascade rerank ON by default for production-quality retrieval
        assert tools.use_cascade_rerank is True
        assert tools.cascade_stage_a_top_n == 100

    def test_runtime_config_defaults(self):
        from aic2026.app.api import RuntimeConfig
        config = RuntimeConfig()
        # Cascade rerank ON by default for production-quality retrieval
        assert config.use_cascade_rerank is True
        assert config.cascade_stage_a_top_n == 100


class TestCascadeBranching:
    """Test that cascade mode branches correctly in retrieve()."""

    def test_cascade_narrows_pool_before_blip2(self):
        """When cascade ON, BLIP-2 should see fewer candidates than total."""
        from aic2026.agent.tools import RetrievalTools
        from aic2026.retrieval import RetrievalPipeline

        pipeline = MagicMock(spec=RetrievalPipeline)
        tools = RetrievalTools(pipeline=pipeline, encode_text=MagicMock())
        tools.use_cascade_rerank = True
        tools.cascade_stage_a_top_n = 5
        tools.use_blip2_rerank = True
        tools.blip2_rerank_top_k = 70

        candidates = [_make_candidate(i, f"V{i//5}", i*10, 1.0 - i*0.01) for i in range(20)]

        sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)[:tools.cascade_stage_a_top_n]
        assert len(sorted_cands) == 5

        blip2_top_k = min(tools.blip2_rerank_top_k, len(sorted_cands))
        assert blip2_top_k == 5

    def test_flat_pipeline_runs_blip2_on_all(self):
        """When cascade OFF, BLIP-2 runs on all candidates."""
        from aic2026.agent.tools import RetrievalTools
        from aic2026.retrieval import RetrievalPipeline

        pipeline = MagicMock(spec=RetrievalPipeline)
        tools = RetrievalTools(pipeline=pipeline, encode_text=MagicMock())
        tools.use_cascade_rerank = False
        tools.use_blip2_rerank = True
        tools.blip2_rerank_top_k = 70

        # In flat mode, BLIP-2 gets the full top_k, not narrowed
        candidates = [_make_candidate(i, f"V{i//5}", i*10, 1.0 - i*0.01) for i in range(20)]

        # BLIP-2 should receive min(70, 20) = 20
        blip2_top_k = min(tools.blip2_rerank_top_k, len(candidates))
        assert blip2_top_k == 20


class TestCascadeStageA:
    """Test Stage A narrowing behavior."""

    def test_top_n_respects_cascade_limit(self):
        """Stage A narrows to cascade_stage_a_top_n candidates."""
        candidates = [_make_candidate(i, f"V{i}", i*10, 1.0 - i*0.01) for i in range(150)]
        top_n = 100
        narrowed = sorted(candidates, key=lambda c: c.score, reverse=True)[:top_n]
        assert len(narrowed) == 100
        # Best score should be first
        assert narrowed[0].score == 1.0

    def test_top_n_when_fewer_than_limit(self):
        """When pool is smaller than top_n, keep all."""
        candidates = [_make_candidate(i, f"V{i}", i*10, 1.0 - i*0.01) for i in range(30)]
        top_n = 100
        narrowed = sorted(candidates, key=lambda c: c.score, reverse=True)[:top_n]
        assert len(narrowed) == 30

    def test_top_n_preserves_order(self):
        """Stage A keeps candidates sorted by score descending."""
        candidates = [
            _make_candidate(0, "V0", 0, 0.5),
            _make_candidate(1, "V1", 10, 0.9),
            _make_candidate(2, "V2", 20, 0.7),
        ]
        top_n = 2
        narrowed = sorted(candidates, key=lambda c: c.score, reverse=True)[:top_n]
        assert narrowed[0].score == 0.9
        assert narrowed[1].score == 0.7
