from __future__ import annotations

import json
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LocalLLM(Protocol):
    def structured(self, system: str, user: str, schema: type[T]) -> T: ...


class OllamaLLM:
    """Minimal Ollama client; no cloud key and no agent framework lock-in."""
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", timeout_seconds: float = 45, temperature: float = 0.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install project dependencies with: uv sync") from exc
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema.model_json_schema(),
            "options": {"temperature": self.temperature},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        content = response.json()["message"]["content"]
        try:
            return schema.model_validate_json(content)
        except ValidationError as exc:
            raise ValueError(f"Local LLM returned invalid {schema.__name__} JSON") from exc
