from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


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
            "format": schema.model_json_schema(),
            "think": self.think,
            "keep_alive": self.keep_alive,
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
            content = response_payload["message"]["content"]

        except httpx.HTTPError as exc:
            raise LLMInvocationError(
                "The local LLM request failed."
            ) from exc

        except (KeyError, TypeError, ValueError) as exc:
            raise LLMInvocationError(
                "The local LLM returned an invalid response payload."
            ) from exc

        try:
            return schema.model_validate_json(content)

        except ValidationError as exc:
            raise LLMInvocationError(
                f"The local LLM returned invalid "
                f"{schema.__name__} JSON."
            ) from exc