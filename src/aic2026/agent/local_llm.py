from __future__ import annotations

import re
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# Các model reasoning (qwen3/qwen3.5, deepseek-r1, ...) thường bọc kết quả trong
# thẻ <think>…</think>; một số model khác lại bọc JSON trong markdown ```json … ```.
# Ollama đôi khi cũng thêm text tản mạn trước/ Sau JSON. Hàm dưới bóc các lớp đó
# để lấy được chuỗi JSON hợp lệ đưa vào pydantic.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_to_json(content: str) -> str:
    """Trích xuất chuỗi JSON từ nội dung trả về của LLM, bất kể có nhiễu.

    Xử lý lần lượt: bỏ thẻ <think>, bỏ markdown code fence, rồi tìm cặp dấu
    ngoặc nhọn ``{…}`` đầu tiên. Nếu vẫn không được, trả nguyên bản để pydantic
    báo lỗi rõ ràng.
    """
    text = content.strip()

    text = _THINK_RE.sub("", text)

    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    # Tìm object JSON đầu tiên theo dấu ngoặc cân bằng.
    start = text.find("{")
    if start != -1:
        depth = 0
        in_str = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1]
    return text.strip()


class LLMInvocationError(RuntimeError):
    """Raised when an external LLM call or response cannot be used."""


class LocalLLM(Protocol):
    def structured(
        self,
        system: str,
        user: str,
        schema: type[T],
    ) -> T: ...


class OllamaLLM:
    """Minimal Ollama client with normalized integration errors."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 600,
        temperature: float = 0.0,
        num_predict: int = 1500,
        num_ctx: int = 8192,
        think: bool = False,
        keep_alive: str = "-1",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.think = think
        self.keep_alive = keep_alive

    def structured(
        self,
        system: str,
        user: str,
        schema: type[T],
    ) -> T:
        try:
            import httpx
        except ImportError as exc:
            raise LLMInvocationError(
                "httpx is required to call the local LLM."
            ) from exc

        payload = {
            "model": self.model,
            "stream": False,
            "think": self.think,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
            },
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
        }

        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()

            response_payload = response.json()
            message = response_payload["message"]
            # Some models (e.g. reasoning-tuned qwen3 variants configured with
            # think=False) emit the answer inside ``thinking``/``reasoning_content``
            # while leaving ``content`` empty.  Fall back to those fields so a valid
            # plan is not discarded as an empty payload.
            content = message.get("content") or ""
            if not content.strip():
                content = message.get("reasoning_content") or message.get("thinking") or ""
            if not content.strip():
                raise KeyError("empty content")

        except httpx.HTTPError as exc:
            raise LLMInvocationError(
                "The local LLM request failed."
            ) from exc

        except (KeyError, TypeError, ValueError) as exc:
            raise LLMInvocationError(
                "The local LLM returned an invalid response payload."
            ) from exc

        try:
            parsed = _strip_to_json(content)
            return schema.model_validate_json(parsed)

        except ValidationError as exc:
            raise LLMInvocationError(
                f"The local LLM returned invalid "
                f"{schema.__name__} JSON: {content!r}"
            ) from exc