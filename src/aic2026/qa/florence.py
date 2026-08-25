from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from aic2026.models import Candidate

logger = logging.getLogger(__name__)

# Florence-2 là vision foundation model của Microsoft (232M params, base-ft).
# Model: microsoft/Florence-2-base-ft (đã fine-tune VQA — bắt buộc để VQA trả lời
# được; bản base thường chỉ làm caption/OCR tốt, VQA trả rác). Cùng kích thước với
# base nhưng có trọng số VQA.
# Thay thế Qwen2.5-VL / Ollama vision vì:
#   * Nhẹ (~600MB, chạy CPU float32 ổn định) — không cần GPU hay Ollama server.
#   * 1 model duy nhất xử lý caption + OCR + VQA qua task prompt (không ghép pipeline).
# Prompt format: "<VQA>{question}" — yêu cầu question bằng TIẾNG ANH (model train chủ
# yếu EN; query Tiếng Việt sẽ trả rác, nên pipeline dự án chuyển query sang EN trước).
from aic2026.qa.answers import clean_vqa_answer, resolve_keyframe_path, translate_vqa_question

_MODEL_ID = "microsoft/Florence-2-base-ft"


class FlorenceVLM:
    """Lazy-loading Florence-2 VLM wrapper for QA over candidate keyframes.

    Drop-in replacement for :class:`QwenVLM` / :class:`OllamaVisionModel`: same
    ``answer_question(question, candidates) -> {vector_id: answer}`` interface, so
    the agent degrades gracefully (returns ``{}`` when the model is unavailable).

    Strategy: test the top representative frames of each distinct video,
    and propagate that answer to every candidate of the same video.
    """

    def __init__(
        self,
        model_name: str = _MODEL_ID,
        device: str | None = None,
        torch_dtype: str = "float32",
        max_new_tokens: int = 128,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens

        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load processor + model once. Safe to call repeatedly."""
        if self._loaded:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            logger.warning(
                "FlorenceVLM: transformers/torch unavailable (%s). "
                "Install with: uv sync --extra models",
                exc,
            )
            return

        dtype = getattr(torch, self.torch_dtype, torch.float32)
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).eval()
            if self.device:
                self._model = self._model.to(self.device)
            self._torch = torch
            self._loaded = True
        except Exception as exc:  # noqa: BLE001 — model load must degrade
            logger.warning("FlorenceVLM: failed to load %s: %s", self.model_name, exc)
            self._loaded = False

    @property
    def available(self) -> bool:
        return self._loaded

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._torch = None
        self._loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Public answering
    # ------------------------------------------------------------------

    def answer_question(
        self,
        question: str,
        candidates: list[Candidate],
        strategy: str = "auto",
    ) -> dict[int, str]:
        """Answer ``question`` over ``candidates``; return ``{vector_id: answer}``.

        Returns ``{}`` when the model is unavailable so the agent degrades to the
        previous (empty-answer) behavior instead of crashing.
        """
        if not question or not candidates:
            return {}
        if not self._loaded:
            self.load()
        if not self._loaded:
            return {}

        # Normalize/translate Vietnamese question to English for Florence-2 VQA
        en_question = translate_vqa_question(question)

        # Group candidates by video; pick the highest-scoring frames per video.
        by_video: dict[str, list[Candidate]] = defaultdict(list)
        for cand in candidates:
            if cand.vector_id is not None:
                by_video[cand.video_id].append(cand)

        answers: dict[int, str] = {}
        for video_candidates in by_video.values():
            ordered = sorted(video_candidates, key=lambda c: c.score, reverse=True)
            # Try top-3 candidate frames of this video until a valid answer is generated
            answer = None
            for rep in ordered[:3]:
                answer = self._answer_one(en_question, rep)
                if answer:
                    break
            if answer is None:
                continue
            for cand in video_candidates:
                if cand.vector_id is not None:
                    answers[cand.vector_id] = answer
        return answers

    # ------------------------------------------------------------------
    # Ollama-compatible alias (so api.py can call either name)
    # ------------------------------------------------------------------

    def answer_question_parallel(
        self,
        question: str,
        candidates: list[Candidate],
        max_workers: int = 1,
    ) -> dict[int, str]:
        return self.answer_question(question, candidates)

    # ------------------------------------------------------------------
    # Single-frame VQA
    # ------------------------------------------------------------------

    def _answer_one(self, question: str, candidate: Candidate) -> str | None:
        if self._processor is None or self._model is None:
            return None
        path = resolve_keyframe_path(candidate.keyframe_path)
        if path is None or not path.exists():
            logger.debug(
                "FlorenceVLM: missing keyframe %s for %s",
                candidate.keyframe_path,
                candidate.video_id,
            )
            return None
        try:
            from PIL import Image

            image = Image.open(path).convert("RGB")
            prompt = f"<VQA>{question}"
            inputs = self._processor(text=prompt, images=image, return_tensors="pt")
            if self.device:
                inputs = inputs.to(self.device)
            with self._torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            decoded = self._processor.batch_decode(
                generated, skip_special_tokens=True
            )[0]
            answer = decoded[len(prompt):] if decoded.startswith(prompt) else decoded
            return clean_vqa_answer(answer)
        except Exception as exc:  # noqa: BLE001 — per-frame failure must not abort
            logger.debug("FlorenceVLM: failed on %s: %s", candidate.frame_id, exc)
            return None
