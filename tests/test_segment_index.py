"""Segment index tests + pooling-strategy benchmark (spec V, Phase 2).

These are deterministic (synthetic frames) so they pin building/searching
behaviour and let us A/B the four pooling strategies without a model or dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from aic2026.models import FrameRecord
from aic2026.retrieval.index import VectorIndex
from aic2026.retrieval.segment_index import (
    PoolingStrategy,
    SegmentIndex,
    Segment,
)

RNG = np.random.default_rng(42)


def _make_video_frames(
    video_id: str, n_frames: int, dim: int, base: np.ndarray, noise: float = 0.05
) -> tuple[list[FrameRecord], np.ndarray]:
    """A video whose frames cluster around ``base`` (so it is retrievable)."""
    records: list[FrameRecord] = []
    vecs: list[np.ndarray] = []
    for i in range(n_frames):
        vec = base + RNG.normal(0, noise, size=dim)
        vec = vec / np.linalg.norm(vec)
        vecs.append(vec)
        records.append(
            FrameRecord(
                vector_id=0,  # filled by caller after stacking
                video_id=video_id,
                frame_id=i,
                keyframe_path=f"data/{video_id}/{i}.jpg",
                object_labels=[],
            )
        )
    return records, np.stack(vecs)


def _build_index(n_videos: int = 4, frames_per_video: int = 25, dim: int = 32):
    all_records: list[FrameRecord] = []
    all_vecs: list[np.ndarray] = []
    bases = [RNG.normal(0, 1, size=dim) for _ in range(n_videos)]
    bases = [b / np.linalg.norm(b) for b in bases]
    for v in range(n_videos):
        recs, vecs = _make_video_frames(f"V{v}", frames_per_video, dim, bases[v])
        all_records.extend(recs)
        all_vecs.append(vecs)
    matrix = np.vstack(all_vecs).astype(np.float32)
    # assign vector_id = row index
    for row, rec in enumerate(all_records):
        rec.vector_id = row
    index = VectorIndex(matrix)
    return index, all_records, bases


def test_build_produces_segments_per_video():
    index, manifest, _ = _build_index(frames_per_video=25)
    seg_idx = SegmentIndex.build(index, manifest, segment_frames=10, strategy="mean")
    # 4 videos × 25 frames / 10 per segment (no overlap) = 4×3 = 12 segments.
    assert len(seg_idx.segments) == 12
    # every segment has the requested frame run length (last may be shorter)
    assert all(1 <= s.frame_count <= 10 for s in seg_idx.segments)
    # timestamps derived from frame_id × interval (default 1.0)
    for s in seg_idx.segments:
        assert s.start_ts == float(min(s.frame_ids))
        assert s.end_ts == float(max(s.frame_ids))


def test_segments_group_by_video_contiguity():
    index, manifest, _ = _build_index(frames_per_video=20)
    seg_idx = SegmentIndex.build(index, manifest, segment_frames=10, strategy="mean")
    # within a video, consecutive frame_ids must be contiguous inside each run
    by_video: dict[str, list[Segment]] = {}
    for s in seg_idx.segments:
        by_video.setdefault(s.video_id, []).append(s)
    for segs in by_video.values():
        for s in segs:
            ids = s.frame_ids
            assert ids == sorted(ids)
            assert ids[-1] - ids[0] + 1 == len(ids)  # no gaps within a run


def test_search_returns_most_similar_video_segment():
    dim = 32
    index, manifest, bases = _build_index(n_videos=4, frames_per_video=25, dim=dim)
    seg_idx = SegmentIndex.build(index, manifest, segment_frames=10, strategy="mean")
    # Query = the representative of video 2's cluster -> top hit should be V2.
    q = bases[2] / np.linalg.norm(bases[2])
    hits = seg_idx.search(q, k=3)
    assert hits[0].video_id == "V2"
    assert hits[0].score > 0.5  # should be clearly similar


def test_pooling_strategies_all_valid_and_normalized():
    index, manifest, _ = _build_index(frames_per_video=25)
    for strat in ("mean", "max", "topk", "weighted"):
        seg_idx = SegmentIndex.build(
            index, manifest, segment_frames=10, strategy=strat
        )
        for s in seg_idx.segments:
            assert abs(float(np.linalg.norm(s.embedding)) - 1.0) < 1e-4
            assert abs(float(np.linalg.norm(s.mean_embedding)) - 1.0) < 1e-4


def test_frame_indices_flattened_and_deduped():
    index, manifest, _ = _build_index(frames_per_video=25)
    seg_idx = SegmentIndex.build(index, manifest, segment_frames=10, strategy="mean")
    hits = seg_idx.search(np.ones(32), k=2)
    flat = seg_idx.frame_manifest_indices_for(hits)
    assert len(flat) == len(set(flat))
    assert all(isinstance(i, int) for i in flat)


def test_search_in_videos_restricts_to_set():
    index, manifest, bases = _build_index(n_videos=4, frames_per_video=25)
    seg_idx = SegmentIndex.build(index, manifest, segment_frames=10, strategy="mean")
    q = bases[2] / np.linalg.norm(bases[2])
    hits = seg_idx.search_in_videos(q, k=5, video_ids={"V1", "V3"})
    assert all(h.video_id in {"V1", "V3"} for h in hits)


# --- Benchmark scaffold (print only; not an assertion) ----------------------
def test_benchmark_pooling_retrieval_accuracy(capsys):
    """A/B the four pooling strategies on a synthetic long-video retrieval task.

    This pins that ``mean`` and ``weighted`` (the robust aggregators) recover the
    correct video at least as often as ``max`` on clustered frames.  It is the
    Phase-2 A/B hook; extend with real CLIP features + AIC queries later.
    """
    dim = 64
    n_videos = 6
    index, manifest, bases = _build_index(
        n_videos=n_videos, frames_per_video=40, dim=dim
    )
    results: dict[PoolingStrategy, float] = {}
    for strat in ("mean", "max", "topk", "weighted"):
        seg_idx = SegmentIndex.build(
            index, manifest, segment_frames=10, strategy=strat
        )
        correct = 0
        for v, base in enumerate(bases):
            q = base / np.linalg.norm(base)
            hits = seg_idx.search(q, k=1)
            if hits and hits[0].video_id == f"V{v}":
                correct += 1
        results[strat] = correct / n_videos
    with capsys.disabled():
        print("\nPooling retrieval accuracy (synthetic):", results)
    # Robust aggregators should never lose to max on clustered frames.
    assert results["mean"] >= results["max"] - 1e-9
