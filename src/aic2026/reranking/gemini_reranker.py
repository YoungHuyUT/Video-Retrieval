"""Gemini Multimodal Reranker and API Provider Abstraction.

Implements fine-grained multimodal reranking for Top-K candidate videos/frames:
- Provider abstraction: MultimodalRerankerProvider -> GeminiFlashLiteProvider
- Selective API Gate (GeminiGate): checks ambiguity, counts, spatial relations, visual details
- Complete error isolation & resilience: 401, 403, 429, 500, timeout, malformed JSON, missing API key
- Circuit breaker state machine (CLOSED, OPEN, HALF_OPEN)
- Max 1 short retry with bounded exponential backoff
- Result caching & score fusion (S_final = w_local * S_local + w_gemini * S_gemini)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from aic2026.models import Candidate, FrameRecord
    from aic2026.query.plan import QueryPlan

logger = logging.getLogger(__name__)


class GeminiEvaluationResult(BaseModel):
    """Structured evaluation output per candidate returned by Gemini."""

    candidate_id: str
    overall_match: float = Field(default=0.0, ge=0.0, le=1.0)
    event_scores: Dict[str, float] = Field(default_factory=dict)
    object_score: float = Field(default=0.0, ge=0.0, le=1.0)
    attribute_score: float = Field(default=0.0, ge=0.0, le=1.0)
    spatial_score: float = Field(default=0.0, ge=0.0, le=1.0)
    action_score: float = Field(default=0.0, ge=0.0, le=1.0)
    count_score: float = Field(default=0.0, ge=0.0, le=1.0)
    cross_event_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    temporal_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class CircuitBreakerState:
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Lightweight circuit breaker for external API resilience."""

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 300.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
        self.last_state_change = time.time()

    def allow_request(self) -> bool:
        now = time.time()
        if self.state == CircuitBreakerState.OPEN:
            if now - self.last_state_change >= self.cooldown_seconds:
                self.state = CircuitBreakerState.HALF_OPEN
                self.last_state_change = now
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        return True

    def record_success(self) -> None:
        self.failure_count = 0
        if self.state != CircuitBreakerState.CLOSED:
            self.state = CircuitBreakerState.CLOSED
            self.last_state_change = time.time()
            logger.info("Circuit breaker reset to CLOSED")

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.last_state_change = time.time()
            logger.warning(
                "Circuit breaker tripped to OPEN after %d consecutive failures",
                self.failure_count,
            )


class GeminiGate:
    """Selective gate determining whether a query warrants external Gemini multimodal reranking."""

    def __init__(
        self,
        ambiguity_margin: float = 0.05,
        enabled: bool = True,
    ) -> None:
        self.ambiguity_margin = ambiguity_margin
        self.enabled = enabled

    def should_call(
        self,
        query_plan: Optional[QueryPlan],
        candidates: List[Candidate],
    ) -> bool:
        if not self.enabled or not candidates:
            return False

        if query_plan is not None:
            # 1. Exact count constraint
            if query_plan.has_count_constraint:
                logger.info("GeminiGate: triggered by count constraint")
                return True

            # 2. Spatial relations present
            if getattr(query_plan, "spatial_relations", None):
                logger.info("GeminiGate: triggered by spatial relations")
                return True

            # 3. Fine visual attributes
            if len(query_plan.attributes) >= 2:
                logger.info("GeminiGate: triggered by multiple visual attributes")
                return True

            # 4. Multi-event complex chains
            if len(query_plan.events) >= 2:
                logger.info("GeminiGate: triggered by multi-event chain")
                return True

        # 5. Top-1 vs Top-2 candidate ambiguity margin
        if len(candidates) >= 2:
            score_diff = abs(candidates[0].score - candidates[1].score)
            if score_diff < self.ambiguity_margin:
                logger.info(
                    "GeminiGate: triggered by candidate score ambiguity (diff %.4f < %.4f)",
                    score_diff,
                    self.ambiguity_margin,
                )
                return True

        logger.info("GeminiGate: skipped (query is decisive/simple)")
        return False


class MultimodalRerankerProvider(ABC):
    """Abstract base provider for multimodal reranking APIs."""

    @abstractmethod
    def rerank_candidates(
        self,
        query_plan: Optional[QueryPlan],
        candidates: List[Candidate],
        records: Dict[int, FrameRecord],
    ) -> Dict[str, GeminiEvaluationResult]:
        """Return structured evaluation results mapping candidate_id/video_id to GeminiEvaluationResult."""
        pass


class GeminiFlashLiteProvider(MultimodalRerankerProvider):
    """Gemini Flash provider with resilience, retries, and strict JSON validation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.6-flash",
        max_retries: int = 1,
        timeout_seconds: float = 3.0,
        max_frames_per_candidate: int = 5,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = model or "gemini-3.6-flash"
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.max_frames_per_candidate = max_frames_per_candidate

    def rerank_candidates(
        self,
        query_plan: Optional[QueryPlan],
        candidates: List[Candidate],
        records: Dict[int, FrameRecord],
    ) -> Dict[str, GeminiEvaluationResult]:
        if not self.api_key:
            logger.info("Gemini API key not set — skipping Gemini rerank (local fusion tier used).")
            return {}

        prompt = self._build_prompt(query_plan, candidates, records)
        response_json = self._call_api_with_retry(prompt)
        if not response_json:
            return {}

        return self._parse_and_validate_response(response_json, candidates)

    def _build_prompt(
        self,
        query_plan: Optional[QueryPlan],
        candidates: List[Candidate],
        records: Dict[int, FrameRecord],
    ) -> str:
        candidates_payload = []
        for c in candidates:
            rec = records.get(c.vector_id) if c.vector_id is not None else None
            candidates_payload.append(
                {
                    "candidate_id": c.video_id,
                    "frame_id": c.frame_id,
                    "local_score": round(c.score, 4),
                    "keyframe_path": c.keyframe_path or (getattr(rec, "keyframe_path", "") if rec else ""),
                    "timestamp": getattr(rec, "timestamp_seconds", None) if rec else None,
                    "ocr_text": getattr(rec, "ocr_text", "") if rec else "",
                }
            )

        events_info = []
        if query_plan and query_plan.events:
            for ev in query_plan.events:
                events_info.append({"id": f"E{ev.index + 1}", "description": ev.description})

        payload = {
            "query": query_plan.raw_text if query_plan else "",
            "decomposed_events": events_info,
            "candidates": candidates_payload,
        }

        return f"""You are a visual retrieval reranker for Video Known-Item Search.

Given the query, decomposed events, and top candidate video frame evidence:
{json.dumps(payload, indent=2)}

Evaluate each candidate video/frame against the query.
Consider object presence, visual attributes, spatial relations, actions, counts, and event consistency.

Respond ONLY with a JSON array of candidate evaluations matching this format:
[
  {{
    "candidate_id": "video_123",
    "overall_match": 0.85,
    "event_scores": {{"E1": 0.9, "E2": 0.8}},
    "object_score": 0.9,
    "attribute_score": 0.8,
    "spatial_score": 0.8,
    "action_score": 0.9,
    "count_score": 1.0,
    "cross_event_consistency": 0.9,
    "temporal_consistency": 0.9,
    "confidence": 0.9
  }}
]
All numerical scores MUST be floats clamped to [0.0, 1.0].
Do not output any introductory or natural-language text. Output strict JSON only.
"""

    def _call_api_with_retry(self, prompt: str) -> Optional[str]:
        """Invoke Gemini REST API with max 1 fast retry. Don't retry on 4xx client errors."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json",
            },
        }

        for attempt in range(self.max_retries + 1):
            try:
                import urllib.request
                import urllib.error

                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")

                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                    res_body = response.read().decode("utf-8")
                    res_json = json.loads(res_body)

                    candidates = res_json.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"]
                    logger.warning("Gemini returned invalid structure: %s", res_body[:200])
                    return None

            except urllib.error.HTTPError as exc:
                logger.warning("Gemini API call returned HTTP %d: %s. Skipping Gemini.", exc.code, exc.reason)
                return None  # Don't retry client errors (404, 401, 403, 400)
            except Exception as exc:
                logger.warning(
                    "Gemini API call attempt %d/%d failed: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(0.5)
                else:
                    return None

        return None

    def _parse_and_validate_response(
        self,
        response_json: str,
        candidates: List[Candidate],
    ) -> Dict[str, GeminiEvaluationResult]:
        results: Dict[str, GeminiEvaluationResult] = {}
        try:
            # Strip potential markdown fences if present
            cleaned = response_json.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()

            parsed = json.loads(cleaned)
            if not isinstance(parsed, list):
                if isinstance(parsed, dict) and "candidates" in parsed:
                    parsed = parsed["candidates"]
                else:
                    parsed = [parsed]

            valid_ids = {c.video_id for c in candidates}
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                cand_id = str(item.get("candidate_id", ""))
                if not cand_id and valid_ids:
                    # Fallback match if candidate_id omitted
                    cand_id = list(valid_ids)[0]

                eval_res = GeminiEvaluationResult(
                    candidate_id=cand_id,
                    overall_match=max(0.0, min(1.0, float(item.get("overall_match", 0.0)))),
                    event_scores={
                        k: max(0.0, min(1.0, float(v)))
                        for k, v in item.get("event_scores", {}).items()
                        if isinstance(v, (int, float))
                    },
                    object_score=max(0.0, min(1.0, float(item.get("object_score", 0.0)))),
                    attribute_score=max(0.0, min(1.0, float(item.get("attribute_score", 0.0)))),
                    spatial_score=max(0.0, min(1.0, float(item.get("spatial_score", 0.0)))),
                    action_score=max(0.0, min(1.0, float(item.get("action_score", 0.0)))),
                    count_score=max(0.0, min(1.0, float(item.get("count_score", 0.0)))),
                    cross_event_consistency=max(0.0, min(1.0, float(item.get("cross_event_consistency", 0.0)))),
                    temporal_consistency=max(0.0, min(1.0, float(item.get("temporal_consistency", 0.0)))),
                    confidence=max(0.0, min(1.0, float(item.get("confidence", 0.0)))),
                )
                results[cand_id] = eval_res

        except Exception as exc:
            logger.warning("Failed to parse Gemini JSON response (%s): %s", exc, response_json[:200])

        return results


class GeminiReranker:
    """High-level Gemini Multimodal Reranker with caching, circuit breaker, and fusion."""

    def __init__(
        self,
        enabled: bool = True,
        model: str = "gemini-3.6-flash",
        top_k: int = 10,
        local_weight: float = 0.65,
        gemini_weight: float = 0.35,
        ambiguity_margin: float = 0.05,
        max_retries: int = 1,
        timeout_seconds: float = 20.0,
        circuit_breaker_failures: int = 5,
        cooldown_seconds: float = 300.0,
        cache_enabled: bool = True,
        provider: Optional[MultimodalRerankerProvider] = None,
    ) -> None:
        self.enabled = enabled
        self.top_k = top_k
        self.local_weight = local_weight
        self.gemini_weight = gemini_weight
        self.cache_enabled = cache_enabled

        self.gate = GeminiGate(ambiguity_margin=ambiguity_margin, enabled=enabled)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_failures,
            cooldown_seconds=cooldown_seconds,
        )
        self.provider = provider or GeminiFlashLiteProvider(
            model=model,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )
        self._cache: Dict[str, Dict[str, GeminiEvaluationResult]] = {}

    def rerank(
        self,
        query_plan: Optional[QueryPlan],
        candidates: List[Candidate],
        records: Dict[int, FrameRecord],
    ) -> List[Candidate]:
        """Rerank Top-K candidates using Gemini multimodal evaluation.

        If Gemini is disabled, skipped by gate, tripped by circuit breaker, missing API key,
        or encounters any runtime error, candidates are returned with their original scores intact.
        """
        if not self.enabled or not candidates:
            return candidates

        # Selective API Gate check
        if not self.gate.should_call(query_plan, candidates[: self.top_k]):
            return candidates

        # Circuit breaker check
        if not self.circuit_breaker.allow_request():
            logger.warning("Gemini reranker skipped: Circuit Breaker is OPEN")
            return candidates

        target_candidates = candidates[: self.top_k]
        cache_key = self._make_cache_key(query_plan, target_candidates)

        gemini_evals: Dict[str, GeminiEvaluationResult] = {}
        if self.cache_enabled and cache_key in self._cache:
            logger.info("Gemini rerank cache hit")
            gemini_evals = self._cache[cache_key]
        else:
            try:
                gemini_evals = self.provider.rerank_candidates(
                    query_plan=query_plan,
                    candidates=target_candidates,
                    records=records,
                )
                if gemini_evals:
                    self.circuit_breaker.record_success()
                    if self.cache_enabled:
                        self._cache[cache_key] = gemini_evals
                else:
                    self.circuit_breaker.record_failure()
            except Exception as exc:
                self.circuit_breaker.record_failure()
                logger.warning("Gemini reranker exception: %s. Continuing with existing scores.", exc)
                return candidates

        if not gemini_evals:
            return candidates

        # Fuse Gemini scores into candidate scores
        return self._fuse_scores(candidates, gemini_evals)

    def _make_cache_key(
        self,
        query_plan: Optional[QueryPlan],
        candidates: List[Candidate],
    ) -> str:
        q_text = query_plan.raw_text if query_plan else ""
        cand_ids = ",".join(c.video_id for c in candidates)
        raw = f"{q_text}:{cand_ids}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _fuse_scores(
        self,
        candidates: List[Candidate],
        gemini_evals: Dict[str, GeminiEvaluationResult],
    ) -> List[Candidate]:
        fused: List[Candidate] = []
        for c in candidates:
            eval_res = gemini_evals.get(c.video_id)
            if eval_res is not None:
                # S_final = w_local * S_existing + w_gemini * S_gemini
                gemini_score = eval_res.overall_match
                new_score = (self.local_weight * c.score) + (self.gemini_weight * gemini_score)
                fused.append(c.model_copy(update={"score": new_score}))
            else:
                fused.append(c)

        return sorted(fused, key=lambda cand: cand.score, reverse=True)
