from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aic2026.models import FrameRecord
from aic2026.retrieval import RetrievalPipeline
from aic2026.retrieval.index import VectorIndex

_DERIVED_FEATS = Path("data/processed/derived_features.npy")
_DERIVED_MANIFEST = Path("data/processed/derived_manifest.jsonl")


def _load_derived() -> tuple[np.ndarray, list[FrameRecord]]:
    """Load the derived L21_V001 feature matrix + manifest.

    These artifacts are produced by `aic2026 prepare` and are NOT committed to
    the repo (the .npy is ~hundreds of MB).  When they are absent the test is
    skipped rather than failed — this is an environment/data issue, not a code
    regression. Run `aic2026 prepare` (see RUNNING.md) to generate them.
    """
    if not _DERIVED_FEATS.exists() or not _DERIVED_MANIFEST.exists():
        pytest.skip(
            "derived artifacts not found — run `aic2026 prepare` first "
            "(see RUNNING.md). Missing: %s / %s"
            % (_DERIVED_FEATS, _DERIVED_MANIFEST)
        )
    feats = np.load(_DERIVED_FEATS)
    recs = [
        FrameRecord(**json.loads(line))
        for line in _DERIVED_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return feats, recs


def test_kis_self_query_retrieves_same_frame() -> None:
    """Self-query: asking for a frame should return that frame at rank 0.

    This isolates whether ``video_level_rerank`` (KIS-only reranking added in
    this branch) preserves the correct frame ordering on real L21_V001 data.
    """
    feats, recs = _load_derived()
    idx = VectorIndex(feats)
    pipe = RetrievalPipeline(idx, recs, frames_per_video=20)

    q_idx = 150
    q = feats[q_idx].copy()
    q /= np.linalg.norm(q)

    # Mimic tools.retrieve KIS path: vector retrieve + diversify, then video_level_rerank.
    cands = pipe.retrieve(q, top_frames=300, max_answers=300)
    reranked = pipe.video_level_rerank(cands, top_videos=200, frames_per_video=20)

    target_frame = recs[q_idx].frame_id
    assert reranked, "no candidates returned"
    top = reranked[0]
    print(f"\n[self-query frame {target_frame}] top5 after rerank:")
    for c in reranked[:5]:
        print(f"  frame={c.frame_id} score={c.score:.4f}")
    assert top.frame_id == target_frame, (
        f"expected frame {target_frame} at rank 0, got frame {top.frame_id}"
    )
