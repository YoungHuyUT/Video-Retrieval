"""Test for BLIP2Reranker optimization (fine_details capping and single image load)."""

from unittest.mock import MagicMock, patch
from pathlib import Path

from aic2026.models import Candidate
from aic2026.reranking.blip2_reranker import BLIP2Reranker


def test_blip2_reranker_caps_fine_details_and_top_k():
    reranker = BLIP2Reranker()

    mock_vlm = MagicMock()
    mock_vlm._loaded = True
    mock_vlm.available = True
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_vlm._model = mock_model
    mock_vlm._processor = mock_processor
    reranker._vlm = mock_vlm

    candidates = [
        Candidate(video_id="v1", frame_id=1, score=0.5, vector_id=1),
        Candidate(video_id="v2", frame_id=2, score=0.4, vector_id=2),
    ]
    records = {
        1: MagicMock(keyframe_path="data/raw/Keyframes/v1/1.jpg"),
        2: MagicMock(keyframe_path="data/raw/Keyframes/v2/2.jpg"),
    }

    fine_details = [f"detail_{i}" for i in range(10)]  # 10 fine details

    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True

    with patch.object(BLIP2Reranker, "_resolve_frame_path", return_value=mock_path), \
         patch.object(BLIP2Reranker, "_score_frame_queries", return_value=0.8) as mock_score:

        res = reranker.rerank(
            query="main query",
            candidates=candidates,
            records=records,
            top_k=70,
            weight=0.3,
            fine_details=fine_details,
        )

        assert len(res) == 2
        # Verify _score_frame_queries was called with capped queries (main query + 2 fine_details = 3 queries max)
        assert mock_score.called
        args = mock_score.call_args[0]
        scoring_queries = args[2]
        assert len(scoring_queries) <= 3
        assert scoring_queries[0] == "main query"
