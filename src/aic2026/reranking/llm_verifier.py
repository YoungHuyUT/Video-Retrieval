"""LLM Verifier — Gemini-only fact-checking for top-K candidates.

Uses Google Gemini free API (15 RPM, 1M tokens/day).  If GEMINI_API_KEY
is not set, the verifier is silently skipped (no crash).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def llm_verify_candidates(
    query: str,
    candidates: list[Any],
    fine_details: list[str] | None = None,
    records: dict[int, Any] | None = None,
    max_candidates: int = 20,
    timeout: float = 10.0,
    **_kwargs: Any,
) -> list[Any]:
    """Fact-check top-K candidates using Gemini free API.

    Returns candidates unchanged if GEMINI_API_KEY is not set or the
    API call fails — the layer degrades gracefully.
    """
    if not candidates:
        return candidates

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.debug("LLM verifier skipped: GEMINI_API_KEY not set")
        return candidates

    to_verify = candidates[:max_candidates]
    details_text = "; ".join(fine_details[:5]) if fine_details else "general scene match"
    summaries = []
    for c in to_verify:
        rec = records.get(c.vector_id) if records and c.vector_id else None
        objects = "none"
        if rec and hasattr(rec, "object_labels"):
            objects = ", ".join(rec.object_labels[:5]) if rec.object_labels else "none"
        summaries.append(
            f"video={c.video_id} frame={c.frame_id} score={c.score:.3f} objects=[{objects}]"
        )
    prompt = (
        f"Query: {query}\nKey details: {details_text}\n\n"
        "Candidates:\n" + "\n".join(summaries)
        + "\n\nReply with JSON array: [{index, failed_details, confidence}]"
    )

    try:
        reply = _call_gemini(prompt, api_key, timeout)
        if not reply:
            return to_verify
        if "```json" in reply:
            reply = reply.split("```json")[1].split("```")[0].strip()
        results = json.loads(reply)
        if not isinstance(results, list):
            return to_verify
        for item in results:
            idx = item.get("index", -1)
            failed = item.get("failed_details", [])
            if 0 <= idx < len(to_verify) and failed:
                to_verify[idx].score *= 0.3 if len(failed) >= 2 else 0.7
        return to_verify
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM verifier failed (Gemini): %s", exc)
        return to_verify


def _call_gemini(prompt: str, api_key: str, timeout: float) -> str:
    """Call Google Gemini free API."""
    import httpx

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-3.6-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 2048,
        },
    }
    response = httpx.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    # Extract text from response — Gemini 3.6-flash may return thought parts
    candidate = data.get("candidates", [{}])[0]
    content = candidate.get("content", {})
    parts = content.get("parts", [])
    for part in parts:
        if "text" in part and not part.get("thought", False):
            return part["text"]
    # Fallback: return any text part
    for part in parts:
        if "text" in part:
            return part["text"]
    return ""
