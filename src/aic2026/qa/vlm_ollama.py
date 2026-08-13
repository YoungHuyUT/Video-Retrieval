from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from aic2026.models import Candidate

logger = logging.getLogger(__name__)


def _resolve_frame_path(keyframe_path: str | None) -> Path | None:
    """Resolve a manifest-relative keyframe path against common project roots."""
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


def _encode_image_base64(path: Path) -> str:
    """Read an image and return a data-URI suitable for Ollama's 'images' field."""
    import base64 as _b64

    with path.open("rb") as handle:
        return _b64.b64encode(handle.read()).decode("ascii")


def _select_temporal_diverse(
    candidates: list[Candidate],
    top_k: int,
) -> list[Candidate]:
    """Chọn ``top_k`` frame phủ đều thời gian, mỗi đoạn lấy frame relevance cao nhất.

    Inspired by QCA (Peng et al., 2026, arXiv:2607.00983): partition candidates into
    ``top_k`` temporal bins by ``frame_id`` and pick the highest-scoring frame per bin
    (anchor on relevance), which yields diversity across the video timeline instead of
    K near-duplicate top frames. Falls back to top-score when frames lack ``frame_id``
    or there are fewer candidates than ``top_k``.

    Diversity guard: if two chosen frames share the same ``frame_id`` (or a candidate
    was already picked), advance to the next best in that bin so we never return
    duplicates — mirroring QCA's "maximize diversity while keeping relevance".
    """
    if not candidates:
        return []
    k = max(top_k, 1)
    if len(candidates) <= k:
        return list(candidates)

    # Candidates phải có frame_id hợp lệ mới chia bin được.
    if any(c.frame_id is None for c in candidates):
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:k]

    lo = min(c.frame_id for c in candidates)
    hi = max(c.frame_id for c in candidates)
    span = max(hi - lo, 1)

    # Bin candidates by frame_id; each bin keeps its members sorted by score desc.
    bins: list[list[Candidate]] = [[] for _ in range(k)]
    for cand in candidates:
        idx = min(k - 1, (cand.frame_id - lo) * k // span)
        bins[idx].append(cand)
    for bin_ in bins:
        bin_.sort(key=lambda c: c.score, reverse=True)

    chosen: list[Candidate] = []
    seen_ids: set[int] = set()
    # Quét bin theo thứ tự thời gian; mỗi bin lấy frame score cao nhất chưa trùng.
    for bin_ in bins:
        for cand in bin_:
            if cand.frame_id not in seen_ids:
                chosen.append(cand)
                seen_ids.add(cand.frame_id)
                break

    # Nếu thiếu (bin rỗng hoặc trùng hết),补全 bằng frame score cao nhất chưa chọn.
    if len(chosen) < k:
        for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
            if cand.frame_id not in seen_ids:
                chosen.append(cand)
                seen_ids.add(cand.frame_id)
                if len(chosen) >= k:
                    break
    return chosen


# Mô tả ngắn gọn về cách Ollama diễn giải khung hình video.
_SYSTEM_PROMPT = (
    "You are a video keyframe understanding assistant. "
    "Answer based ONLY on what is visible in the image(s); do not invent details. "
    "Be concise and directly address the question."
)

# Prompt dùng khi chỉ có 1 ảnh (frame đại diện).
_PROMPT_TEMPLATE = (
    "Question (about a video keyframe): {question}\n"
    "Answer briefly in English based ONLY on what is visible in this image."
)

# Prompt dùng khi có nhiều ảnh cùng lúc (tổng hợp trên top-K frame của 1 video).
_PROMPT_TEMPLATE_MULTI = (
    "The images below are different keyframes from the SAME video, ordered by "
    "decreasing relevance. Question: {question}\n"
    "Synthesize information from ALL images to answer. Reply briefly in English."
)


class OllamaVisionModel:
    """Câu trả lời Q&A qua model vision chạy trên Ollama.

    Khác với :class:`QwenVLM` (cần torch + transformers trên máy), backend này
    gửi ảnh base64 tới server Ollama `127.0.0.1:11434` — nơi model vision như
    ``qwen2.5vl:3b`` (quantized GGUF) chạy tốt trên CPU/4-8GB VRAM. Vì vậy:
      * không phải cài thêm torch/transformers cho bước Q&A;
      * model nhìn được ảnh thật (multimodal) chứ không chỉ text;
      * tương thích với lệnh `ollama serve` bạn đã chạy cho planner/judge.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5vl:3b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 120,
        temperature: float = 0.0,
        num_predict: int = 128,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.num_predict = num_predict

    def available_models(self) -> list[str]:
        try:
            import httpx
        except ImportError:
            return []
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return [model["name"] for model in response.json().get("models", [])]
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return []

    def answer_question(
        self,
        question: str,
        candidates: list[Candidate],
        top_k_frames: int = 4,
    ) -> dict[int, str]:
        """Trả ``{vector_id: answer}``; mỗi video tổng hợp top-K frame rồi propagate.

        Khác bản cũ (chỉ dùng 1 frame điểm cao nhất làm đại diện): ở đây lấy
        ``top_k_frames`` frame liên quan nhất của video, ghép vào **1 lần gọi**
        Ollama để VLM tổng hợp — giúp trả lời các câu cần nhiều khung hình
        (vd "có bao nhiêu người", "ai làm X rồi Y") chính xác hơn, mà vẫn giữ
        số lần gọi VLM = số video (không tăng chi phí).
        """
        if not question or not candidates:
            return {}

        by_video: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in candidates:
            if candidate.vector_id is not None:
                by_video[candidate.video_id].append(candidate)

        answers: dict[int, str] = {}
        for video_candidates in by_video.values():
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

    def _answer_top_k(
        self,
        question: str,
        candidates: list[Candidate],
        top_k: int,
    ) -> str | None:
        """Tổng hợp top-K frame của 1 video trong 1 lần gọi Ollama.

        Chọn frame theo **temporal-diverse sampling** (inspired by QCA, Peng et al.
        2026): chia candidate theo trục thời gian (frame_id) thành ``top_k`` bin,
        mỗi bin lấy frame có score cao nhất. Khác với lấy top-K liền kề (dễ trùng
        nhau cùng 1 cảnh, bổ sung info = 0), cách này phủ đều thời gian nên VLM thấy
        được các giai đoạn khác nhau của sự kiện. Nếu chỉ lấy được 1 frame hợp lệ thì
        fallback về prompt 1 ảnh. Trả ``None`` khi không ảnh nào đọc được.
        """
        top = _select_temporal_diverse(candidates, top_k)
        images: list[str] = []
        for candidate in top:
            path = _resolve_frame_path(candidate.keyframe_path)
            if path is not None and path.exists():
                images.append(_encode_image_base64(path))
        if not images:
            return None

        multi = len(images) > 1
        prompt = (
            _PROMPT_TEMPLATE_MULTI if multi else _PROMPT_TEMPLATE
        ).format(question=question)

        try:
            import httpx
        except ImportError:
            logger.warning("OllamaVisionModel: httpx chưa được cài.")
            return None

        payload: dict[str, Any] = {
            "model": self.model_name,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": prompt,
                    "images": images,
                },
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }

        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
        except httpx.HTTPError as exc:
            logger.warning(
                "OllamaVisionModel: thất bại %s: %s",
                candidates[0].video_id if candidates else "?",
                exc,
            )
            return None
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "OllamaVisionModel: phản hồi không hợp lệ từ %s: %s",
                self.model_name,
                exc,
            )
            return None

        return content.strip() or None


def list_ollama_vision_models(
    base_url: str = "http://127.0.0.1:11434",
) -> list[str]:
    """Tiện ích CLI: liệt kê model đang có trên Ollama."""
    return OllamaVisionModel(base_url=base_url).available_models()