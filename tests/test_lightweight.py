"""Unit tests for the lightweight extraction pipeline (spec §2, §5, §6, §10)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aic2026.lightweight.catalog import LightweightCatalog
from aic2026.lightweight.dedup import DedupConfig, dedup_frames
from aic2026.lightweight.merge_btc import MergeBtcConfig, load_manifest
from aic2026.lightweight.sampler import MotionChangeDetector, UniformSampler
from aic2026.models import FrameRecord


# --- sampler ---------------------------------------------------------------
def test_uniform_sampler_grid():
    # Make a tiny fake mp4? We instead test the math via a real short clip if
    # present; otherwise verify the sampler skips non-existent gracefully.
    vid = Path("data/raw/Videos_test/video/L24_V045.mp4")
    if not vid.exists():
        pytest.skip("short test video not available")
    s = UniformSampler(fps=1.0).sample(vid, duration_s=32.5)
    # ~1 frame/sec over 32.5s -> ~32-33 grid points.
    assert 30 <= len(s) <= 34
    ts = [t for t, _ in s]
    assert ts == sorted(ts)
    assert ts[0] >= 0.0


def test_motion_detector_returns_peaks():
    vid = Path("data/raw/Videos_test/video/L24_V045.mp4")
    if not vid.exists():
        pytest.skip("short test video not available")
    import cv2

    cap = cv2.VideoCapture(str(vid))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    peaks = MotionChangeDetector(threshold=0.35).peaks(vid, fc / fps)
    assert 0 <= len(peaks) <= 30
    for ts, _ in peaks:
        assert 0.0 <= ts <= fc / fps


# --- merge_btc -------------------------------------------------------------
def test_merge_btc_writes_manifest(tmp_path):
    # Point at the real official manifest but cap via a tiny custom merge using
    # the project's actual data so we don't fabricate paths.
    real_manifest = Path("data/processed/official_manifest.jsonl")
    if not real_manifest.exists():
        pytest.skip("official manifest not present")
    cfg = MergeBtcConfig(root=tmp_path, official_manifest=real_manifest)
    n = cfg.merge(limit=200)
    assert n == 200
    recs = load_manifest(cfg.manifest_path)
    assert len(recs) == 200
    assert all(r.source == "btc" for r in recs)
    # Timestamps recovered from map CSV (L21_V001 first frame == 0.0).
    assert recs[0].timestamp == 0.0
    # clip_feature_index preserved for reuse.
    assert recs[0].clip_feature_index == 0
    # frame_id counter sidecar written.
    assert (tmp_path / "frame_id_counter.json").exists()


# --- dedup -----------------------------------------------------------------
def test_dedup_keeps_temporal_coverage():
    recs = [
        FrameRecord(vector_id=0, video_id="V", frame_id=0, keyframe_path="a",
                    timestamp=0.0, source="btc"),
        FrameRecord(vector_id=1, video_id="V", frame_id=1, keyframe_path="b",
                    timestamp=0.1, source="uniform"),  # near-dup in time
        FrameRecord(vector_id=2, video_id="V", frame_id=2, keyframe_path="c",
                    timestamp=5.0, source="uniform"),  # far -> keep
    ]
    # All identical vectors -> the 0.1s uniform one near the BTC should drop,
    # but the 5.0s one is far enough to keep.
    vectors = np.ones((3, 8), dtype=np.float32)
    kept, _, idx = dedup_frames(recs, vectors, DedupConfig(sim_threshold=0.98, min_gap_s=0.5))
    assert len(kept) == 2
    assert {r.frame_id for r in kept} == {0, 2}
    assert all(i in idx for i in (0, 2))


def test_dedup_protects_btc():
    recs = [
        FrameRecord(vector_id=0, video_id="V", frame_id=0, keyframe_path="a",
                    timestamp=1.0, source="btc"),
        FrameRecord(vector_id=1, video_id="V", frame_id=1, keyframe_path="b",
                    timestamp=1.0, source="uniform"),  # exact same ts + vec
    ]
    vectors = np.stack([np.ones(8), np.ones(8)]).astype(np.float32)
    kept, _, _ = dedup_frames(recs, vectors, DedupConfig(protect_btc=True))
    # BTC must survive; the duplicate uniform is dropped.
    assert len(kept) == 1
    assert kept[0].source == "btc"


def test_dedup_never_compares_different_videos():
    recs = [
        FrameRecord(vector_id=0, video_id="A", frame_id=0, keyframe_path="a", timestamp=0.0, source="btc"),
        FrameRecord(vector_id=1, video_id="B", frame_id=0, keyframe_path="b", timestamp=0.0, source="uniform"),
    ]
    kept, _, _ = dedup_frames(recs, np.ones((2, 8), dtype=np.float32), DedupConfig(sim_threshold=0.98))
    assert len(kept) == 2


# --- catalog ---------------------------------------------------------------
def test_catalog_roundtrip_and_window(tmp_path):
    db = tmp_path / "catalog.db"
    cat = LightweightCatalog(db)
    cat.upsert_video("V", 30.0, 1000, "hash1")
    cat.add_frame("V", 0, 0.0, "btc", "p0.jpg", clip_index=0)
    cat.add_frame("V", 1, 2.0, "uniform", "p1.jpg", clip_index=None)
    cat.add_frame("V", 2, 10.0, "motion", "p2.jpg")
    cat.commit_frames()
    # A video is "built" only after mark_video_built (not merely on frame presence).
    assert not cat.is_video_done("V")
    cat.mark_video_built("V")
    assert cat.is_video_done("V")
    rows = cat.frames_in_window("V", 1.0, 4.0)
    assert {r["frame_id"] for r in rows} == {0, 1}  # 10.0 outside window
    assert cat.source_counts() == {"btc": 1, "uniform": 1, "motion": 1}
    cat.close()


# --- config A/B/C/D retrieval + fusion -----------------------------------
def _fake_index(rows, dim=16):
    from aic2026.retrieval.index import VectorIndex

    # Deterministic pseudo-embeddings: BTC frames near cluster 0, NEW near 1.
    mat = np.zeros((len(rows), dim), dtype=np.float32)
    for i, r in enumerate(rows):
        if r.source == "btc":
            mat[i, 0] = 1.0
        else:
            mat[i, 1] = 1.0
        mat[i] /= np.linalg.norm(mat[i]) + 1e-9
    return VectorIndex(mat)


def test_retrieve_config_fusion():
    rows = [
        FrameRecord(vector_id=0, video_id="V", frame_id=0, keyframe_path="a",
                    timestamp=0.0, source="btc"),
        FrameRecord(vector_id=1, video_id="V", frame_id=1, keyframe_path="b",
                    timestamp=1.0, source="uniform"),
        FrameRecord(vector_id=2, video_id="V", frame_id=2, keyframe_path="c",
                    timestamp=2.0, source="motion"),
    ]
    clip = _fake_index(rows)
    sig = _fake_index(rows)  # BTC rows zeroed in real embed; here just reuse shape
    q = np.zeros(16, dtype=np.float32)
    q[1] = 1.0  # query matches NEW frames, not BTC
    weights = {"clip": 1.0, "siglip2": 1.0}
    # A (BTC CLIP only) -> BTC frame 0 should win (clip matches btc cluster).
    a = __import__("aic2026.lightweight.bench", fromlist=["retrieve_config"]).retrieve_config(
        rows, clip, sig, "A", q, weights, k=3
    )
    # C (BTC CLIP + NEW SigLIP2): query hits NEW SigLIP2 -> NEW frames rank high.
    c = __import__("aic2026.lightweight.bench", fromlist=["retrieve_config"]).retrieve_config(
        rows, clip, sig, "C", q, weights, k=3
    )
    assert 0 in a  # BTC present in A
    # In C, NEW frames (1,2) must be retrievable (SigLIP2 path exists).
    assert any(i in c for i in (1, 2))


def test_minmax_normalization_no_raw_mix():
    # adaptive_modality_fusion normalizes each modality before summing.
    from aic2026.reranking.lexical import adaptive_modality_fusion

    ids = {"clip": [0, 1], "siglip2": [0, 1]}
    scores = {"clip": [0.2, 0.9], "siglip2": [5.0, 100.0]}  # very different scales
    fused = adaptive_modality_fusion(ids, scores, {"clip": 1.0, "siglip2": 1.0})
    # Normalized scores land in [0,1] per modality, so fused S in [0,2].
    assert all(0.0 <= v <= 2.0 + 1e-6 for v in fused.values())
    # Both modalities agree frame 1 > frame 0 -> frame 1 wins.
    assert fused[1] > fused[0]
