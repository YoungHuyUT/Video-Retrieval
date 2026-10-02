"""SigLIP2 embedder tests (Phase 3, Hướng A — query-side + RRF fuse).

These verify the embedder implements the same interface as
``OpenCLIPTextEmbedder`` and produces normalized, sensible vectors.  The model
load is gated behind AIC_RUN_HEAVY_MODEL_TESTS so lightweight CI never pulls a
~400MB checkpoint.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

_HEAVY = os.environ.get("AIC_RUN_HEAVY_MODEL_TESTS") == "1"
skip_heavy = pytest.mark.skipif(not _HEAVY, reason="requires SigLIP2 checkpoint (AIC_RUN_HEAVY_MODEL_TESTS=1)")


@pytest.fixture(scope="module")
def embedder():
    from aic2026.embeddings import Siglip2TextEmbedder

    return Siglip2TextEmbedder()


@skip_heavy
def test_encode_shape_and_normalized(embedder):
    v = embedder.encode("a red door with books on a table")
    assert v.shape == (768,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-4


@skip_heavy
def test_different_queries_have_distinct_embeddings(embedder):
    a = embedder.encode("a person walking into a room")
    b = embedder.encode("a cat sitting on a windowsill")
    sim = float(a @ b)
    assert -1.0 <= sim <= 1.0
    # unrelated queries should not be near-identical
    assert sim < 0.95


@skip_heavy
def test_load_unload_roundtrip(embedder):
    embedder.unload()
    assert embedder.model is None
    embedder.load()
    v = embedder.encode("smoke test after reload")
    assert v.shape == (768)
