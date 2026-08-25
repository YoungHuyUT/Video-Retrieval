import numpy as np
import pytest

from aic2026.agent.runtime import RetrievalAgent
from aic2026.agent.tools import RetrievalTools
from aic2026.agent.translator import (
    decompose_query_modalities,
    offline_decompose_modalities,
)
from aic2026.agent.types import ModalityDecomposition
from aic2026.models import FrameRecord, Query
from aic2026.retrieval.bm25_index import BM25Index
from aic2026.retrieval.index import VectorIndex
from aic2026.retrieval.pipeline import RetrievalPipeline


def test_offline_modality_decomposition_ocr() -> None:
    # Test OCR-heavy query with quote and keyword
    query_ocr = 'Tấm bảng hiệu trên đường có chữ "BENVENUTI"'
    decomp = offline_decompose_modalities(query_ocr)
    assert decomp.w_ocr > decomp.w_vis
    assert decomp.w_ocr >= 0.5
    assert "BENVENUTI" in decomp.ocr_query or "BENVENUTI" in decomp.visual_query


def test_offline_modality_decomposition_asr() -> None:
    # Test ASR-heavy query
    query_asr = "Bản tin thời sự phát thanh viên thông báo về tình hình bão lũ"
    decomp = offline_decompose_modalities(query_asr)
    assert decomp.w_asr > decomp.w_ocr
    assert decomp.w_asr >= 0.5
    assert decomp.asr_query == query_asr


def test_offline_modality_decomposition_visual() -> None:
    # Test visual-dominant query
    query_vis = "Một chiếc ô tô màu đỏ đang chạy trên đường cao tốc"
    decomp = offline_decompose_modalities(query_vis)
    assert decomp.w_vis >= 0.6
    assert decomp.w_ocr <= 0.3
    assert decomp.w_asr <= 0.3


def test_bm25_modality_specific_search() -> None:
    manifest = [
        FrameRecord(
            vector_id=0,
            video_id="L01_V001",
            frame_id=10,
            keyframe_path="L01_V001/010.jpg",
            object_labels=["sign", "BENVENUTI", "billboard"],
            asr_text=[],
        ),
        FrameRecord(
            vector_id=1,
            video_id="L01_V001",
            frame_id=20,
            keyframe_path="L01_V001/020.jpg",
            object_labels=["person", "microphone"],
            asr_text=["Chào mừng quý vị đến với bản tin thời sự sạt lở sông Cửu Long"],
        ),
    ]

    bm25 = BM25Index(manifest)
    assert not bm25.is_empty

    # OCR search matches frame 0 only
    ocr_ids, ocr_scores = bm25.search_ocr("BENVENUTI", k=5)
    assert len(ocr_ids) > 0
    assert ocr_ids[0] == 0

    # ASR search matches frame 1 only
    asr_ids, asr_scores = bm25.search_asr("sạt lở sông Cửu Long", k=5)
    assert len(asr_ids) > 0
    assert asr_ids[0] == 1


def test_adaptive_multimodal_retrieve_pipeline() -> None:
    dim = 8
    vectors = np.zeros((3, dim), dtype=np.float32)
    vectors[0, 0] = 1.0  # matches visual vector [1, 0, ...]
    vectors[1, 1] = 1.0
    vectors[2, 2] = 1.0

    manifest = [
        FrameRecord(
            vector_id=0,
            video_id="L01_V001",
            frame_id=100,
            keyframe_path="L01_V001/100.jpg",
            object_labels=["car", "red"],
            asr_text=[],
        ),
        FrameRecord(
            vector_id=1,
            video_id="L01_V002",
            frame_id=200,
            keyframe_path="L01_V002/200.jpg",
            object_labels=["store", "banner", "COFFEE_SHOP"],
            asr_text=[],
        ),
        FrameRecord(
            vector_id=2,
            video_id="L01_V003",
            frame_id=300,
            keyframe_path="L01_V003/300.jpg",
            object_labels=["anchor"],
            asr_text=["Dự báo thời tiết hôm nay trời mưa rất to"],
        ),
    ]

    index = VectorIndex(vectors)
    bm25 = BM25Index(manifest)
    pipeline = RetrievalPipeline(index=index, manifest=manifest)

    # Query seeking the coffee shop sign (OCR heavy)
    query_emb = np.zeros(dim, dtype=np.float32)
    query_emb[0] = 0.5  # slight visual match to car

    ocr_decomp = ModalityDecomposition(
        visual_query="store",
        ocr_query="COFFEE_SHOP",
        asr_query="",
        w_vis=0.2,
        w_ocr=0.8,
        w_asr=0.0,
    )

    candidates = pipeline.adaptive_multimodal_retrieve_raw(
        visual_query="store",
        text_embedding=query_emb,
        bm25_index=bm25,
        ocr_query=ocr_decomp.ocr_query,
        asr_query=ocr_decomp.asr_query,
        w_vis=ocr_decomp.w_vis,
        w_ocr=ocr_decomp.w_ocr,
        w_asr=ocr_decomp.w_asr,
        top_frames=3,
    )

    # Coffee shop should be top candidate due to w_ocr=0.8
    assert len(candidates) > 0
    assert candidates[0].video_id == "L01_V002"
    assert candidates[0].frame_id == 200


def test_retrieval_agent_with_modality_routing() -> None:
    dim = 8
    vectors = np.eye(2, dim, dtype=np.float32)
    manifest = [
        FrameRecord(
            vector_id=0,
            video_id="L21_V001",
            frame_id=1,
            keyframe_path="L21_V001/001.jpg",
            object_labels=["sign", "HOSPITAL"],
            asr_text=["Bác sĩ đang thực hiện ca phẫu thuật ghép tim"],
        ),
        FrameRecord(
            vector_id=1,
            video_id="L21_V002",
            frame_id=2,
            keyframe_path="L21_V002/002.jpg",
            object_labels=["river", "boat"],
            asr_text=["Cảnh sông nước miền Tây"],
        ),
    ]
    index = VectorIndex(vectors)
    bm25 = BM25Index(manifest)
    pipeline = RetrievalPipeline(index=index, manifest=manifest)
    tools = RetrievalTools(pipeline, encode_text=lambda q: np.zeros(dim, dtype=np.float32), bm25_index=bm25)

    agent = RetrievalAgent(tools=tools)

    # Run KIS query for speech keywords
    q = Query(query_id="q1", type="kis", text="phát thanh viên nói về ghép tim cho bệnh nhân")
    res = agent.run(q)

    assert len(res.candidates) > 0
    assert res.plan.modality is not None
    assert res.plan.modality.w_asr > res.plan.modality.w_ocr
    assert res.candidates[0].video_id == "L21_V001"
