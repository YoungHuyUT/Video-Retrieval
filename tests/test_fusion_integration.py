"""End-to-end integration test cho hai-stage ranking refactor.

Mục tiêu: CHỨNG MINH tầng Adaptive Fusion (fusion_mode="rrf_adaptive") thực sự
đảo THỨ TỰ ứng viên cuối cùng so với RRF baseline (fusion_mode="rrf_baseline"),
chứ không chỉ passthrough.

Thiết kế synthetic (không cần CLIP/BM25 thật, chỉ numpy + index in-RAM):
  * 1 video V1, 3 frames (vector_id 0,1,2), dim=4.
  * Query "chai nước"  -> concept "bottle" được yêu cầu.
  * Frame 0: CLIP cao NHẤT (cos=1.0) nhưng label=["person"] (không có bottle).
  * Frame 1: CLIP TRUNG BÌNH (cos=0.5) nhưng label=["bottle"] (có object được hỏi).
  * Frame 2: CLIP THẤP NHẤT (cos=0.0), label=["car"].

Kỳ vọng:
  * RRF baseline -> thứ tự theo CLIP: [f0, f1, f2]  (top = f0).
  * Adaptive Fusion -> s_norm(semantic) đưa f0=1.0, f1=0.5; object coverage chỉ
    f1 có (1.0). Fused: f1 = 0.45*0.5 + 0.30*1.0 = 0.525 > f0 = 0.45*1.0 = 0.45.
    -> thứ tự [f1, f0, f2]  (top = f1).  Thứ tự BỊ ĐẢO so với baseline.

Các reranker phụ (metadata / object-evidence / late-interaction / colour / siglip2
/ dinov2) được tắt (weight=0 / off) để cô lập hoàn toàn hiệu ứng của fusion tier.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from aic2026.agent.tools import RetrievalTools
from aic2026.models import FrameRecord
from aic2026.retrieval import BM25Index, RetrievalPipeline, VectorIndex


def _make_pipeline() -> RetrievalPipeline:
    rng = np.random.RandomState(0)
    # 3 frames, dim 4. Vector của frame 0,1,2 khớp query theo cos 1.0/0.5/0.0.
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    f0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # cos 1.0
    f1 = np.array([0.5, 0.8660254, 0.0, 0.0], dtype=np.float32)  # cos 0.5
    f2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # cos 0.0
    vectors = np.stack([f0, f1, f2], axis=0)

    manifest = [
        FrameRecord(
            video_id="V1",
            frame_id=0,
            vector_id=0,
            keyframe_path="V1/0.jpg",
            timestamp=0.0,
            object_labels=["person"],
        ),
        FrameRecord(
            video_id="V1",
            frame_id=1,
            vector_id=1,
            keyframe_path="V1/1.jpg",
            timestamp=1.0,
            object_labels=["bottle"],
        ),
        FrameRecord(
            video_id="V1",
            frame_id=2,
            vector_id=2,
            keyframe_path="V1/2.jpg",
            timestamp=2.0,
            object_labels=["car"],
        ),
    ]

    index = VectorIndex(vectors)
    bm25 = BM25Index(manifest)  # object_labels -> corpus không rỗng
    pipeline = RetrievalPipeline(index=index, manifest=manifest)
    return pipeline, bm25


def _make_tools(pipeline: RetrievalPipeline, bm25: BM25Index) -> RetrievalTools:
    def encode_text(text: str) -> np.ndarray:  # deterministic, không quan trọng
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    tools = RetrievalTools(
        pipeline=pipeline,
        encode_text=encode_text,
        bm25_index=bm25,
    )
    # Cô lập: tắt mọi reranker phụ để chỉ đo hiệu ứng của fusion tier.
    tools.object_evidence_weight = 0.0
    tools.drop_empty_object_frames = False
    tools.late_interaction_weight = 0.0
    tools.contrastive_colour_rerank_weight = 0.0
    tools.use_siglip2 = False
    tools.use_dinov2 = False
    tools.use_long_query_expansion = False  # isolate fusion tier from expansion
    tools.fusion_weights = {
        "semantic": 0.45,
        "object": 0.30,
        "asr": 0.25,
    }
    return tools


def _order(cands) -> list[int]:
    return [c.vector_id for c in cands]


def test_adaptive_fusion_reranks_against_rrf_baseline(caplog) -> None:
    pipeline, bm25 = _make_pipeline()
    tools = _make_tools(pipeline, bm25)

    query = "chai nước"  # yêu cầu concept "bottle"

    with caplog.at_level(logging.INFO, logger="aic2026.agent.tools"):
        tools.fusion_mode = "rrf_baseline"
        baseline = tools.retrieve(query, limit=10, task_type="kis")

        tools.fusion_mode = "rrf_adaptive"
        adaptive = tools.retrieve(query, limit=10, task_type="kis")

    base_order = _order(baseline)
    adap_order = _order(adaptive)

    # Cả hai mode phải trả về kết quả hợp lệ.
    assert base_order, "RRF baseline returned no candidates"
    assert adap_order, "Adaptive Fusion returned no candidates"

    # Baseline phải rank frame CLIP-cao-nhất (f0) lên đầu.
    assert base_order[0] == 0, f"RRF baseline top should be f0, got {base_order}"

    # Adaptive Fusion phải nhấc frame CÓ object (f1) lên đầu nhờ coverage.
    assert adap_order[0] == 1, (
        f"Adaptive Fusion top should be f1 (bottle), got {adap_order}"
    )

    # Thứ tự cuối cùng PHẢI khác biệt -> fusion tier thực sự re-rank.
    assert adap_order != base_order, (
        f"Fusion tier did not change ordering: both={base_order}"
    )

    # Xác nhận tầng fusion thực sự chạy (log line re-scored).
    assert any(
        "adaptive fusion (final tier): re-scored" in rec.message
        for rec in caplog.records
    ), "Adaptive Fusion final tier log line not emitted"


def test_adaptive_fusion_changes_scores_not_just_order(caplog) -> None:
    """Fusion tier phải gán score mới (khác RRF) cho pool, không phải passthrough."""
    pipeline, bm25 = _make_pipeline()
    tools = _make_tools(pipeline, bm25)

    query = "chai nước"

    tools.fusion_mode = "rrf_baseline"
    baseline = tools.retrieve(query, limit=10, task_type="kis")
    base_by_vid = {c.vector_id: c.score for c in baseline}

    tools.fusion_mode = "rrf_adaptive"
    adaptive = tools.retrieve(query, limit=10, task_type="kis")
    adap_by_vid = {c.vector_id: c.score for c in adaptive}

    # f0 (CLIP cao) phải có score RRF CAO HƠN f1 trong baseline,
    # nhưng THẤP HƠN f1 sau fusion (vì object coverage của f1).
    assert base_by_vid[0] > base_by_vid[1], "baseline f0 should outscore f1"
    assert adap_by_vid[1] > adap_by_vid[0], "fusion f1 should outscore f0"
