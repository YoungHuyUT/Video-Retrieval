"""Tests for KIS eval harness (scripts/eval_kis.py).

Tests run on fake dev set and mock candidates - no real model loading needed.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path

import pytest

# Add project root to path so we can import scripts/
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.eval_kis import (
    EvalResult,
    QueryResult,
    _compute_ap,
    _is_hit,
    evaluate_single_query,
    load_dev_set,
)


def _make_candidate(video_id: str, frame_id: int, score: float) -> SimpleNamespace:
    return SimpleNamespace(video_id=video_id, frame_id=frame_id, score=score)


def _write_dev_set(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "kis_dev.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


class TestIsHit:
    def test_exact_match(self):
        assert _is_hit("V1", 100, "V1", 100, 50) is True

    def test_within_tolerance(self):
        assert _is_hit("V1", 105, "V1", 100, 50) is True

    def test_outside_tolerance(self):
        assert _is_hit("V1", 200, "V1", 100, 50) is False

    def test_wrong_video(self):
        assert _is_hit("V2", 100, "V1", 100, 50) is False

    def test_video_only_match(self):
        assert _is_hit("V1", 999, "V1", None, 50) is True

    def test_video_only_wrong_video(self):
        assert _is_hit("V2", 999, "V1", None, 50) is False


class TestComputeAP:
    def test_perfect_precision(self):
        assert abs(_compute_ap([True, True, True]) - 1.0) < 1e-9

    def test_no_relevant(self):
        assert _compute_ap([False, False, False]) == 0.0

    def test_single_relevant_at_top(self):
        assert abs(_compute_ap([True, False, False]) - 1.0) < 1e-9

    def test_single_relevant_at_bottom(self):
        assert abs(_compute_ap([False, False, True]) - 1.0 / 3.0) < 1e-9

    def test_two_relevant(self):
        expected = (1 / 2 + 2 / 3) / 2
        assert abs(_compute_ap([False, True, True]) - expected) < 1e-9

    def test_empty_list(self):
        assert _compute_ap([]) == 0.0


class TestLoadDevSet:
    def test_loads_valid_jsonl(self, tmp_path):
        entries = [
            {"query": "test query", "gt_video_id": "V1", "gt_frame_id": 100},
            {"query": "another query", "gt_video_id": "V2", "gt_frame_id": None},
        ]
        dev_set = load_dev_set(_write_dev_set(tmp_path, entries))
        assert len(dev_set) == 2
        assert dev_set[0]["query"] == "test query"
        assert dev_set[1]["gt_frame_id"] is None

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        with open(path, "w") as f:
            f.write('{"query": "ok", "gt_video_id": "V1", "gt_frame_id": 1}\n')
            f.write('NOT JSON\n')
            f.write('{"query": "also ok", "gt_video_id": "V2"}\n')
        assert len(load_dev_set(path)) == 2

    def test_skips_missing_fields(self, tmp_path):
        path = tmp_path / "missing.jsonl"
        path.write_text('{"query": "test"}\n')
        assert len(load_dev_set(path)) == 0

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert len(load_dev_set(path)) == 0


class TestEvaluateSingleQuery:
    def test_recall_1_when_gt_in_top1(self):
        candidates = [
            _make_candidate("V1", 100, 1.0),
            _make_candidate("V2", 200, 0.5),
        ]
        qr = evaluate_single_query(candidates, "V1", 100, 50)
        assert qr.recall_at_1 == 1.0
        assert qr.recall_at_5 == 1.0
        assert qr.recall_at_10 == 1.0
        assert qr.average_precision == 1.0

    def test_recall_0_when_gt_not_in_top10(self):
        candidates = [_make_candidate("V2", i, 1.0 - i * 0.01) for i in range(20)]
        qr = evaluate_single_query(candidates, "V1", 100, 50)
        assert qr.recall_at_1 == 0.0
        assert qr.recall_at_5 == 0.0
        assert qr.recall_at_10 == 0.0
        assert qr.average_precision == 0.0

    def test_recall_1_at_position_5(self):
        candidates = [_make_candidate("V2", i, 1.0 - i * 0.1) for i in range(4)]
        candidates.append(_make_candidate("V1", 100, 0.5))
        qr = evaluate_single_query(candidates, "V1", 100, 50)
        assert qr.recall_at_1 == 0.0
        assert qr.recall_at_5 == 1.0
        assert qr.recall_at_10 == 1.0
        assert abs(qr.average_precision - 1 / 5) < 1e-9

    def test_empty_candidates(self):
        qr = evaluate_single_query([], "V1", 100, 50)
        assert qr.recall_at_1 == 0.0
        assert qr.recall_at_5 == 0.0
        assert qr.recall_at_10 == 0.0
        assert qr.average_precision == 0.0
        assert qr.top1_video is None

    def test_gt_frame_none_video_only(self):
        candidates = [_make_candidate("V1", 500, 1.0)]
        qr = evaluate_single_query(candidates, "V1", None, 50)
        assert qr.recall_at_1 == 1.0

    def test_top1_info_populated(self):
        candidates = [_make_candidate("V99", 42, 0.99)]
        qr = evaluate_single_query(candidates, "V1", 100, 50)
        assert qr.top1_video == "V99"
        assert qr.top1_frame == 42
        assert qr.top1_score == 0.99


class TestRunEval:
    def test_returns_correct_schema(self, tmp_path):
        entries = [
            {"query": "q1", "gt_video_id": "V1", "gt_frame_id": 100},
            {"query": "q2", "gt_video_id": "V2", "gt_frame_id": 200},
        ]
        dev_path = _write_dev_set(tmp_path, entries)
        fake_candidates = [
            _make_candidate("V1", 100, 1.0),
            _make_candidate("V2", 200, 0.8),
        ]
        mock_orchestrator = MagicMock()
        mock_result = MagicMock()
        mock_result.candidates = fake_candidates
        mock_orchestrator.run.return_value = mock_result

        with patch("aic2026.app.api.load_orchestrator", return_value=mock_orchestrator):
            from scripts.eval_kis import run_eval
            from aic2026.app.api import RuntimeConfig
            config = RuntimeConfig()
            result = run_eval(config, str(dev_path))

        assert isinstance(result, EvalResult)
        assert result.total_queries == 2
        assert 0.0 <= result.avg_recall_at_1 <= 1.0
        assert 0.0 <= result.avg_mAP <= 1.0
        assert len(result.query_results) == 2

    def test_handles_recall_zero_gracefully(self, tmp_path):
        entries = [{"query": "q1", "gt_video_id": "MISSING", "gt_frame_id": 999}]
        dev_path = _write_dev_set(tmp_path, entries)
        fake_candidates = [_make_candidate("WRONG", i, 1.0 - i * 0.01) for i in range(10)]
        mock_orchestrator = MagicMock()
        mock_result = MagicMock()
        mock_result.candidates = fake_candidates
        mock_orchestrator.run.return_value = mock_result

        with patch("aic2026.app.api.load_orchestrator", return_value=mock_orchestrator):
            from scripts.eval_kis import run_eval
            from aic2026.app.api import RuntimeConfig
            config = RuntimeConfig()
            result = run_eval(config, str(dev_path))

        assert result.total_queries == 1
        assert result.avg_recall_at_1 == 0.0
        assert result.avg_mAP == 0.0

    def test_empty_dev_set(self, tmp_path):
        dev_path = tmp_path / "empty.jsonl"
        dev_path.write_text("")
        with patch("aic2026.app.api.load_orchestrator"):
            from scripts.eval_kis import run_eval
            from aic2026.app.api import RuntimeConfig
            config = RuntimeConfig()
            result = run_eval(config, str(dev_path))

        assert result.total_queries == 0
        assert result.avg_recall_at_1 == 0.0
