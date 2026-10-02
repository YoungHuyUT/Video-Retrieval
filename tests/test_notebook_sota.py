"""Unit tests for SOTA enhancements in Notebook retrieval: Dynamic RRF, PRF, Gaussian Temporal Kernel."""

from __future__ import annotations

import math
import numpy as np
import pytest

from aic2026.notebook.asr_retriever import ASRRetriever, _asr_lexical_tokens, fuse_bm25_dense
from aic2026.notebook.dense_encoder import DenseTextEncoder
from aic2026.notebook.evidence import _compute_gap_penalty, compute_temporal_evidence
from aic2026.notebook.agent import NotebookAgent
from aic2026.notebook.types import ASRSegmentMatch, NotebookPlan, NotebookEvent
from aic2026.ingestion.asr import ASRSegment, VideoTranscript


def test_dynamic_rrf_scaling() -> None:
    bm25 = [
        ASRSegmentMatch(video_id="v1", segment_text="hello world", matched_query="query", start=1.0, end=5.0, bm25_score=0.9),
        ASRSegmentMatch(video_id="v2", segment_text="good morning", matched_query="query", start=10.0, end=15.0, bm25_score=0.7),
    ]
    dense = [
        ASRSegmentMatch(video_id="v2", segment_text="good morning", matched_query="query", start=10.0, end=15.0, bm25_score=0.85),
        ASRSegmentMatch(video_id="v3", segment_text="good evening", matched_query="query", start=20.0, end=25.0, bm25_score=0.6),
    ]

    fused_static = fuse_bm25_dense(bm25, dense, k=60, dynamic_k=False)
    fused_dynamic = fuse_bm25_dense(bm25, dense, dynamic_k=True)

    assert len(fused_static) == 3
    assert len(fused_dynamic) == 3
    # v2 matched both BM25 and Dense -> should rank 1st in both
    assert fused_dynamic[0].video_id == "v2"


def test_gaussian_temporal_gap_kernel() -> None:
    # Smooth gap (5s avg gap) -> high penalty score ~1.0
    p1 = _compute_gap_penalty([10.0, 15.0, 20.0], sigma=30.0)
    assert p1 > 0.95

    # Large gap (60s avg gap) -> smooth Gaussian decay
    p2 = _compute_gap_penalty([10.0, 70.0, 130.0], sigma=30.0)
    assert p2 < p1
    assert p2 > 0.0


def test_kendall_tau_and_temporal_evidence() -> None:
    matches = [
        ASRSegmentMatch(video_id="v1", segment_text="a", matched_query="query", start=10.0, end=15.0, bm25_score=0.9),
        ASRSegmentMatch(video_id="v1", segment_text="b", matched_query="query", start=20.0, end=25.0, bm25_score=0.8),
        ASRSegmentMatch(video_id="v1", segment_text="c", matched_query="query", start=30.0, end=35.0, bm25_score=0.7),
    ]

    ev = compute_temporal_evidence(matches, has_strict_temporal=True)
    assert ev.order_matches is True
    assert ev.violations == 0
    assert ev.consistency > 0.8


def test_dense_text_encoder_reuses_cached_query_embeddings() -> None:
    """Regression: an identical text query must reuse the dense vector instead
    of round-tripping through the encoder again for the same process.
    """

    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, text, normalize_embeddings=True, show_progress_bar=False):
            self.calls += 1
            return np.ones(4, dtype=np.float32)

    encoder = DenseTextEncoder.__new__(DenseTextEncoder)
    encoder._model = FakeModel()
    encoder._query_cache = {}

    first = encoder.encode("một câu hỏi lặp lại")
    second = encoder.encode("một câu hỏi lặp lại")

    assert encoder._model.calls == 1
    assert np.allclose(first, second)


def test_rocchio_prf_fallback_on_empty() -> None:
    retriever = ASRRetriever.empty()
    plan = NotebookPlan(raw_query="test", asr_queries=["từ khóa tiếng việt"])
    res = retriever.retrieve(plan, use_prf=True)
    assert res == []


def test_notebook_uses_rule_based_lexical_plan_without_loading_llm(monkeypatch) -> None:
    """NOTEBOOK is keyword-only: its planner must never invoke Ollama/Qwen."""
    import aic2026.notebook.agent as agent_module

    observed: dict[str, object] = {}

    def fake_plan(text: str, **kwargs: object) -> NotebookPlan:
        observed.update(kwargs)
        return NotebookPlan(raw_query=text, asr_queries=[text])

    monkeypatch.setattr(agent_module, "plan_notebook", fake_plan)
    agent = NotebookAgent(asr_sidecar_path="")
    monkeypatch.setattr(agent, "_get_retriever", ASRRetriever.empty)

    agent.run("cầu Phước An", top_n=5)

    assert observed["use_llm"] is False


def test_asr_canonicalises_luan_don_whisper_variant_and_returns_long_context() -> None:
    """M03 says ``Luân Đôm``; a user searching ``Luân Đôn`` must still hit it."""
    transcripts = {
        "M03_V001": VideoTranscript(
            video_id="M03_V001",
            segments=[
                ASRSegment(text="Mở đầu bản tin nghệ thuật.", start=0, end=2),
                ASRSegment(text="Bức tranh được giới thiệu.", start=2, end=4),
                ASRSegment(text="Tác phẩm từng xuất hiện ở Luân Đôm năm 2002.", start=4, end=7),
                ASRSegment(text="Nó gây nhiều chú ý.", start=7, end=9),
                ASRSegment(text="Phóng sự kết thúc.", start=9, end=11),
            ],
        ),
    }
    # Keep the corpus large enough that rank_bm25 assigns a positive IDF to
    # the London token, as happens with the real 180k-segment sidecar.
    for number in range(3):
        video_id = f"M01_V{number:03d}"
        transcripts[video_id] = VideoTranscript(
            video_id=video_id,
            segments=[
                ASRSegment(text=f"Bản tin khác số {number} phần {part}.", start=part * 2, end=part * 2 + 1)
                for part in range(3)
            ],
        )

    retriever = ASRRetriever(
        transcripts=transcripts,
    )

    assert _asr_lexical_tokens("Luân Đôn") == ["london"]
    assert _asr_lexical_tokens("Luân Đôm") == ["london"]
    matches = retriever.search_segments("Luân Đôn")

    assert matches
    m03_matches = [match for match in matches if match.video_id == "M03_V001"]
    assert m03_matches
    assert any("Mở đầu bản tin nghệ thuật" in match.segment_text for match in m03_matches)
    assert any("Phóng sự kết thúc" in match.segment_text for match in m03_matches)
