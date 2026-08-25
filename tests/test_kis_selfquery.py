from __future__ import annotations

import json

import numpy as np

from aic2026.ingestion import resolve_feature_sources
from aic2026.models import FrameRecord
from aic2026.retrieval import RetrievalPipeline
from aic2026.retrieval.index import VectorIndex


def _load_derived() -> tuple[np.ndarray, list[FrameRecord]]:
    m_path, f_path = resolve_feature_sources()
    if not m_path.exists() or not f_path.exists():
        import pytest
        pytest.skip(f"Index files not found at {m_path} or {f_path}")
    feats = np.load(f_path, mmap_mode="r")
    recs = [
        FrameRecord(**json.loads(line))
        for line in open(m_path, encoding="utf-8")
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
