from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from aic2026.models import Candidate
from aic2026.qa.vlm import QwenVLM, _looks_video_level


def _candidates(tmp_path: Path) -> list[Candidate]:
    # Create two tiny real image files so PIL open/convert works.
    for video in ("V1", "V2"):
        folder = tmp_path / video
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "001.jpg").write_bytes(b"not-a-real-image")
    return [
        Candidate(video_id="V1", frame_id=1, score=0.9, vector_id=0, keyframe_path=str(tmp_path / "V1" / "001.jpg")),
        Candidate(video_id="V1", frame_id=2, score=0.8, vector_id=1, keyframe_path=str(tmp_path / "V1" / "001.jpg")),
        Candidate(video_id="V2", frame_id=3, score=0.7, vector_id=2, keyframe_path=str(tmp_path / "V2" / "001.jpg")),
    ]


def test_answer_question_returns_empty_without_load(tmp_path: Path) -> None:
    vlm = QwenVLM()
    assert vlm.available is False
    assert vlm.answer_question("Có bao nhiêu người?", _candidates(tmp_path)) == {}


def test_import_error_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    vlm = QwenVLM()
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("transformers", "torch"):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    vlm.load()
    assert vlm.available is False


def test_missing_keyframe_skipped_not_crash(tmp_path: Path) -> None:
    """A missing keyframe must be skipped without raising."""
    vlm = QwenVLM()
    vlm._loaded = True
    candidates = [
        Candidate(video_id="V1", frame_id=1, score=1.0, vector_id=0, keyframe_path="nonexistent.jpg"),
    ]
    answers = vlm._answer_video_level("Màu gì?", candidates)
    assert answers == {}


def test_answers_keyed_by_vector_id(tmp_path: Path) -> None:
    """video-level propagation: same answer for all candidates of a video."""
    vlm = QwenVLM()
    vlm._loaded = True

    def fake_top_k(question, candidates, top_k):
        return "5"

    vlm._answer_top_k = fake_top_k  # type: ignore[method-assign]
    answers = vlm._answer_video_level("How many people?", _candidates(tmp_path))
    assert answers == {0: "5", 1: "5", 2: "5"}


def test_looks_video_level_heuristics() -> None:
    assert _looks_video_level("Có bao nhiêu người lên sân khấu?")
    assert _looks_video_level("How many cars?")
    assert _looks_video_level("Chiếc ly màu gì?")
    assert not _looks_video_level("Mô tả người phụ nữ đang làm gì?")


def test_prompt_english_short() -> None:
    prompt = QwenVLM._build_prompt("What color is the cup?")
    assert "English" in prompt
    assert "<|image_pat|>" not in prompt  # processor chèn token ảnh tự động


def test_translate_vqa_question() -> None:
    from aic2026.qa.answers import translate_vqa_question
    assert "how many" in translate_vqa_question("Có bao nhiêu người trong phòng?")
    assert "what color" in translate_vqa_question("Chiếc xe ô tô màu gì?")
    assert translate_vqa_question("what color is the shirt?") == "what color is the shirt?"


def test_clean_vqa_answer() -> None:
    from aic2026.qa.answers import clean_vqa_answer
    assert clean_vqa_answer("<VQA>What color?<s>red</s>") == "Red"
    assert clean_vqa_answer("QA> 2 people <loc_123>") == "2 people"
    assert clean_vqa_answer(None) is None
