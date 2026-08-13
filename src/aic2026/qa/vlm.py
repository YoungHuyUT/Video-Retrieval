from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from aic2026.models import Candidate

logger = logging.getLogger(__name__)

# Đánh dấu trong template chat của Qwen2.5-VL; AutoProcessor thay thế nó bằng token ảnh.
_IMAGE_TOKEN = "<|image_pat|>"

_SYSTEM_PROMPT = (
    "You are a video keyframe understanding assistant. "
    "Answer based ONLY on what is visible in the image(s); do not invent details. "
    "Be concise and directly address the question."
)

# Nếu câu hỏi thuộc nhóm này, answer thường giống nhau cho cả video (đếm, màu sắc...).
_VIDEO_LEVEL_KEYWORDS = (
    "bao nhiêu",
    "how many",
    "có mấy",
    "số lượng",
    "đếm",
    "có bao nhiêu",
    "màu gì",
    "màu sắc",
    "what color",
    "what colour",
)


def _looks_video_level(question: str) -> bool:
    folded = (question or "").lower()
    return any(keyword in folded for keyword in _VIDEO_LEVEL_KEYWORDS)


def _select_temporal_diverse(
    candidates: list[Candidate],
    top_k: int,
) -> list[Candidate]:
    """Chọn ``top_k`` frame phủ đều thời gian, mỗi đoạn lấy frame relevance cao nhất.

    Inspired by QCA (Peng et al., 2026, arXiv:2607.00983): partition candidates into
    ``top_k`` temporal bins by ``frame_id`` and pick the highest-scoring frame per bin
    (anchor on relevance), yielding diversity across the timeline instead of K
    near-duplicate top frames. Falls back to top-score when ``frame_id`` is missing or
    there are fewer candidates than ``top_k``.

    Diversity guard: never return two frames with the same ``frame_id``; advance to the
    next best in the bin so the selection stays diverse while keeping relevance — matching
    QCA's "maximize diversity while maintaining query relevance".
    """
    if not candidates:
        return []
    k = max(top_k, 1)
    if len(candidates) <= k:
        return list(candidates)

    if any(c.frame_id is None for c in candidates):
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:k]

    lo = min(c.frame_id for c in candidates)
    hi = max(c.frame_id for c in candidates)
    span = max(hi - lo, 1)

    bins: list[list[Candidate]] = [[] for _ in range(k)]
    for cand in candidates:
        idx = min(k - 1, (cand.frame_id - lo) * k // span)
        bins[idx].append(cand)
    for bin_ in bins:
        bin_.sort(key=lambda c: c.score, reverse=True)

    chosen: list[Candidate] = []
    seen_ids: set[int] = set()
    for bin_ in bins:
        for cand in bin_:
            if cand.frame_id not in seen_ids:
                chosen.append(cand)
                seen_ids.add(cand.frame_id)
                break

    if len(chosen) < k:
        for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
            if cand.frame_id not in seen_ids:
                chosen.append(cand)
                seen_ids.add(cand.frame_id)
                if len(chosen) >= k:
                    break
    return chosen


def _load_model_class() -> tuple[Any, Any] | None:
    """Return ``(processor_cls, model_cls)`` for Qwen2.5-VL, or None if unavailable."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return None
    try:
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
        )
        return AutoProcessor, AutoModelForImageTextToText
    except ImportError:
        pass
    # transformers cũ hơn: AutoModelForVision2Seq vẫn tồn tại.
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        return AutoProcessor, AutoModelForVision2Seq
    except ImportError:
        return None


def _resolve_frame_path(keyframe_path: str | None) -> Path | None:
    """Resolve a manifest-relative keyframe path against the project root."""
    if not keyframe_path:
        return None
    candidate = Path(keyframe_path)
    if candidate.exists():
        return candidate
    root = Path.cwd()
    for probe in (root / keyframe_path, root / "data" / "raw" / keyframe_path):
        if probe.exists():
            return probe
    return None


class QwenVLM:
    """Lazy-loading Qwen2.5-VL wrapper over candidate keyframes.

    Answers a QA question over candidate frames with answer propagation:
    each distinct video is answered once from its highest-scoring frame, and the
    answer is propagated to every candidate of that video. This keeps VLM calls
    low (~number of videos) while still producing an answer for all candidates,
    which maximizes R@k under the 100-answer cap.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str | None = None,
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

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

        loader = _load_model_class()
        if loader is None:
            logger.warning(
                "QwenVLM: transformers/torch unavailable. "
                "Install with: uv sync --extra models"
            )
            return

        processor_cls, model_cls = loader
        import torch

        dtype = getattr(torch, self.torch_dtype, torch.bfloat16)
        try:
            self._processor = processor_cls.from_pretrained(self.model_name)
            if self.device:
                self._model = model_cls.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map=self.device,
                )
            else:
                self._model = model_cls.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map="auto",
                )
            self._model.eval()
        except Exception as exc:  # noqa: BLE001 — model load must degrade, never crash
            logger.warning(
                "QwenVLM: failed to load model %s: %s", self.model_name, exc
            )
            self._loaded = False
            return

        self._torch = torch
        self._loaded = True

    @property
    def available(self) -> bool:
        return self._loaded

    def close(self) -> None:
        """Free the model if loaded (GC + clear cuda cache when present)."""
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

        Strategy:
        - ``auto`` / ``video``: one VLM call per distinct video (representative
          frame = highest score), answer propagated to all candidates of the video.
        - ``frame``: answer each candidate's top ``max_frames`` individually, then
          propagate each video's highest-confidence answer to remaining candidates
          so every candidate still gets an answer.

        Returns ``{}`` when the model is unavailable so the agent degrades to
        the previous (empty-answer) behavior instead of crashing.
        """
        if not question or not candidates:
            return {}

        if not self._loaded:
            self.load()
        if not self._loaded:
            return {}

        if strategy == "frame":
            return self._answer_frame_level(question, candidates)

        return self._answer_video_level(question, candidates)

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _answer_video_level(
        self,
        question: str,
        candidates: list[Candidate],
        top_k_frames: int = 4,
    ) -> dict[int, str]:
        by_video: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in candidates:
            if candidate.vector_id is not None:
                by_video[candidate.video_id].append(candidate)

        answers: dict[int, str] = {}
        for video_candidates in by_video.values():
            # Representative frames: top-K by score (not just the highest one).
            ordered = sorted(
                video_candidates,
                key=lambda item: item.score,
                reverse=True,
            )
            answer = self._answer_top_k(question, ordered, top_k_frames)
            if answer is None:
                continue
            for candidate in video_candidates:
                if candidate.vector_id is not None:
                    answers[candidate.vector_id] = answer
        return answers

    def _answer_frame_level(
        self,
        question: str,
        candidates: list[Candidate],
        max_frames: int = 20,
    ) -> dict[int, str]:
        ordered = sorted(
            candidates,
            key=lambda item: item.score,
            reverse=True,
        )
        # Lần lượt trả lời top-N frame có score cao nhất.
        answers: dict[int, str] = {}
        for candidate in ordered[:max_frames]:
            if candidate.vector_id is None:
                continue
            answer = self._answer_one(question, candidate)
            if answer:
                answers[candidate.vector_id] = answer

        # Propagate: các candidate còn lại của cùng video nhận answer của frame
        # có score cao nhất trong video đó (nếu đã trả lời được).
        per_video_rep: dict[str, str] = {}
        for candidate in ordered:
            if candidate.vector_id in answers:
                per_video_rep.setdefault(candidate.video_id, answers[candidate.vector_id])

        for candidate in ordered:
            if candidate.vector_id is None or candidate.vector_id in answers:
                continue
            propagated = per_video_rep.get(candidate.video_id)
            if propagated:
                answers[candidate.vector_id] = propagated
        return answers

    def _answer_top_k(
        self,
        question: str,
        candidates: list[Candidate],
        top_k: int,
    ) -> str | None:
        """Tổng hợp top-K frame của 1 video trong 1 lần gọi Qwen2.5-VL.

        Chọn frame theo **temporal-diverse sampling** (inspired by QCA, Peng et al.
        2026, arXiv:2607.00983): chia candidate theo ``frame_id`` thành ``top_k`` bin,
        mỗi bin lấy frame score cao nhất (anchor on relevance) — phủ đều thời gian thay
        vì K frame gần trùng nhau. Qwen2.5-VL chấp nhận nhiều ``{"type": "image"}``
        trong 1 message, nên truyền K ảnh vào 1 prompt tổng hợp. Fallback 1 ảnh nếu
        chỉ lấy được 1 frame hợp lệ.
        """
        from PIL import Image

        top = _select_temporal_diverse(candidates, top_k)
        images: list[Image.Image] = []
        for candidate in top:
            path = _resolve_frame_path(candidate.keyframe_path)
            if path is not None and path.exists():
                try:
                    images.append(Image.open(path).convert("RGB"))
                except OSError:
                    continue
        if not images:
            return None

        multi = len(images) > 1
        prompt = (
            self._build_prompt_multi(question)
            if multi
            else self._build_prompt(question)
        )
        content: list[dict[str, str]] = []
        for image in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        try:
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._processor(
                text=[text],
                images=images,
                return_tensors="pt",
            ).to(self._model.device)
            with self._torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature,
                )
            answer = self._processor.batch_decode(
                output[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )[0].strip()
            return answer or None
        except Exception as exc:  # noqa: BLE001 — per-video failure must not abort
            logger.debug("QwenVLM: failed on video %s: %s", candidates[0].video_id, exc)
            return None

    # ------------------------------------------------------------------
    # Single-frame generation
    # ------------------------------------------------------------------

    def _answer_one(self, question: str, candidate: Candidate) -> str | None:
        if not self._loaded or self._processor is None or self._model is None:
            return None

        path = _resolve_frame_path(candidate.keyframe_path)
        if path is None or not path.exists():
            logger.debug(
                "QwenVLM: missing keyframe %s for %s (frame %s)",
                candidate.keyframe_path,
                candidate.video_id,
                candidate.frame_id,
            )
            return None

        try:
            from PIL import Image

            image = Image.open(path).convert("RGB")
            prompt = self._build_prompt(question)
            messages = [
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            ).to(self._model.device)

            with self._torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature,
                )
            answer = self._processor.batch_decode(
                output[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )[0].strip()
            return answer or None
        except Exception as exc:  # noqa: BLE001 — per-frame failure must not abort the batch
            logger.debug("QwenVLM: failed on frame %s: %s", candidate.frame_id, exc)
            return None

    @staticmethod
    def _build_prompt(question: str) -> str:
        # KHÔNG nhúng <|image_pat|> vào đây: processor chèn token ảnh tự động qua
        # message {"type": "image"} trong apply_chat_template (Qwen2.5-VL). Chèn
        # thủ công sẽ tạo 2 placeholder ảnh nhưng chỉ 1 ảnh được truyền -> lỗi.
        return (
            f"Question (about a video keyframe): {question}\n"
            "Answer briefly in English based ONLY on what is visible in this image."
        )

    @staticmethod
    def _build_prompt_multi(question: str) -> str:
        return (
            "The images below are different keyframes from the SAME video, ordered "
            f"by decreasing relevance. Question: {question}\n"
            "Synthesize information from ALL images to answer. Reply briefly in English."
        )
