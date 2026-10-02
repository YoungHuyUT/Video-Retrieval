"""Free LLM API client — Google Gemini only.

Free tier: 15 RPM, 1M tokens/day (no credit card required).
Get API key: https://aistudio.google.com/apikey
Models: gemini-2.5-flash, gemini-2.0-flash (deprecated)
"""
from __future__ import annotations

import logging
from typing import TypeVar

from pydantic import BaseModel

from aic2026.agent.local_llm import LLMInvocationError, _strip_to_json

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GeminiFreeLLM:
    """Google Gemini API free tier client (no credit card required).

    Free limits: 15 RPM, 1M tokens/day for gemini-2.0-flash.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.6-flash",
        temperature: float = 0.0,
        max_output_tokens: int = 300,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds

    def structured(
        self,
        system: str,
        user: str,
        schema: type[T],
    ) -> T:
        try:
            import httpx
        except ImportError as exc:
            raise LLMInvocationError("httpx is required for Gemini API.") from exc

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": f"{system}\n\n{user}"}]}
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "responseMimeType": "application/json",
                "responseSchema": _pydantic_to_gemini_schema(schema),
            },
        }

        try:
            response = httpx.post(url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            content = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            if not content.strip():
                raise LLMInvocationError("Gemini returned empty response.")
        except httpx.HTTPError as exc:
            raise LLMInvocationError(f"Gemini API request failed: {exc}") from exc
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMInvocationError(
                f"Gemini returned invalid response: {exc}"
            ) from exc

        try:
            parsed = _strip_to_json(content)
            return schema.model_validate_json(parsed)
        except Exception as exc:
            raise LLMInvocationError(
                f"Gemini returned invalid {schema.__name__} JSON: {content!r}"
            ) from exc


def _pydantic_to_gemini_schema(model: type[BaseModel]) -> dict:
    """Convert a Pydantic model to Gemini's responseSchema format."""
    schema = model.model_json_schema()

    def _clean(obj: dict) -> dict:
        out = {}
        for k, v in obj.items():
            if k in ("title", "$schema", "additionalProperties"):
                continue
            if isinstance(v, dict):
                out[k] = _clean(v)
            elif isinstance(v, list):
                out[k] = [_clean(i) if isinstance(i, dict) else i for i in v]
            else:
                out[k] = v
        return out

    return _clean(schema)
