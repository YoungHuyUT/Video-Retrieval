"""BLIP-2 / Florence-2 cross-encoder reranker for retrieval (spec §ASR + §X).

Reuses the already-loaded Florence-2 model (``microsoft/Florence-2-base-ft``)
as a lightweight cross-encoder to rerank the top-K retrieved candidates.
Instead of loading a separate heavy BLIP-2 model (~6GB), we reuse Florence-2
which is already resident for QA answering on the QA path.

Scoring strategy: Florence-2 generates a *caption-style* continuation from the
query, and we take the **negative log-likelihood** of the query tokens given the
image context.  This is a standard cross-encoder signal that requires no extra
model and runs in ~5-15s on CPU for a 300-frame top-K pool.

The reranker is **optional** — it degrades to a no-op (returns candidates
unchanged) if Florence-2 is unavailable, keeping the pipeline robust on
CPU-only or low-RAM machines.
"""

from __future__ import annotations

import logging
import re
import hashlib
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from aic2026.data_platform.keyframe_resolver import resolve_keyframe_path

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aic2026.models import Candidate
    from aic2026.retrieval.pipeline import RetrievalPipeline


# Global cache for BLIP-2 scores: (frame_id, query_hash, fine_details_hash) -> normalized_ce_score
_BLIP2_SCORE_CACHE: dict[tuple[int, str, str], float] = {}


class BLIP2Reranker:
    """Cross-encoder reranker that reuses Florence-2 (already loaded for QA).

    Instead of a full BLIP-2 (which would be ~6 GB), we reuse the
    ``Florence-2-base-ft`` checkpoint that is already lazy-loaded by
    ``FlorenceVLM``.  The reranker computes a per-image *contrastive score*
    between the query text and the keyframe using Florence-2's language-model
    head: the probability of the query continuation given the image context.

    Inserted into the retrieval pipeline AFTER RRF fusion + heuristic bonuses,
    BEFORE event coverage / adaptive fusion.  This is where dense evidence and
    keyword/nudge signals converge, so the cross-encoder can override CLIP-only
    mistakes at the final ranking step.

    The reranker is **lazy** — it borrows the FlorenceVLM instance from
    ``RetrievalTools`` (passed in at construction) so no double model load.
    If the VLM is unavailable it becomes a transparent no-op.
    """

    def __init__(
        self,
        florence_vlm: Any | None = None,
        device: str | None = None,
    ) -> None:
        """Accept an existing FlorenceVLM instance or build one lazily.

        Args:
            florence_vlm: An already-initialised FlorenceVLM object (preferred —
                avoids double-loading the ~600 MB model).
            device: Override device for FlorenceVLM (only used if
                ``florence_vlm`` is None and a new one must be built).
        """
        self._vlm = florence_vlm
        self._device = device
        self._model_name = "microsoft/Florence-2-base-ft"
        self._loaded = False

    def _ensure_loaded(self) -> bool:
        """Ensure Florence-2 is loaded; return True if ready to rerank."""
        if self._vlm is not None and getattr(self._vlm, "_loaded", False):
            self._loaded = True
            return True
        if self._vlm is None:
            try:
                from aic2026.qa.florence import FlorenceVLM

                self._vlm = FlorenceVLM(
                    model_name=self._model_name,
                    device=self._device,
                )
                self._vlm.load()
            except Exception as exc:  # noqa: BLE001
                logger.warning("BLIP2Reranker: Florence-2 load failed: %s", exc)
                return False
        self._loaded = self._vlm.available
        return self._loaded

    @staticmethod
    def _compute_query_hash(query: str, fine_details: list[str] | None) -> str:
        """Compute a hash for the query + fine_details combination."""
        content = query
        if fine_details:
            content += "|" + "|".join(sorted(fine_details))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        records: dict[int, Any],
        top_k: int = 100,
        weight: float = 0.3,
        fine_details: list[str] | None = None,
    ) -> list[Candidate]:
        """Rerank top-K candidates using Florence-2 cross-encoder scoring.

        For each candidate frame, Florence-2 encodes ``<VQA>{query}`` on the
        keyframe image and we measure the **likelihood** of the query text
        under the image-conditioned LM head.  Higher likelihood → the query
        describes what is *actually* in the frame → the candidate is up-ranked.

        When ``fine_details`` is provided, each detail is scored separately
        and the final score is the MEAN of detail scores (plus the global
        query score).  This catches fine-grained visual requirements that
        a single pooled query would miss.

        Args:
            query: The retrieval query (already translated to EN for Florence).
            candidates: Top-K candidates from the RRF + bonus stage.
            records: Manifest records keyed by ``vector_id`` (for keyframe_path).
            top_k: How many top candidates to rerank (cap for latency).
            weight: Blend weight for the cross-encoder score (0 = no-op).
            fine_details: Optional list of specific visual checks (from LLM analyzer).

        Returns:
            The same candidate list, re-sorted by blended score.
        """
        if not candidates or weight <= 0:
            return candidates

        if not self._ensure_loaded():
            logger.debug("BLIP2Reranker: Florence-2 not available — skipping rerank.")
            return candidates

        # Determine scoring queries: global query + fine_details (capped to 2 max for speed).
        scoring_queries = [query]
        if fine_details:
            scoring_queries.extend(fine_details[:2])
            top_k = min(top_k, 30)

        to_rerank = candidates[:top_k]
        scores: list[tuple[int, float]] = []  # (candidate_index, ce_score)

        vlm = self._vlm
        if vlm is None or vlm._processor is None or vlm._model is None:
            return candidates

        import torch

        torch_model = vlm._model
        processor = vlm._processor
        dev = getattr(torch_model, "device", "cpu")
        dtype = getattr(torch_model, "dtype", torch.float32) if hasattr(torch, "float32") else None

        # Compute query hash for caching
        query_hash = self._compute_query_hash(query, fine_details)

        for idx, cand in enumerate(to_rerank):
            record = records.get(cand.vector_id) if cand.vector_id is not None else None
            if record is None:
                continue
            keyframe_path = getattr(record, "keyframe_path", None) or getattr(record, "image_path", None)
            if not keyframe_path:
                continue
            resolved = self._resolve_frame_path(keyframe_path)
            if resolved is None or not resolved.exists():
                continue

            # Create cache key
            cache_key = (cand.vector_id or 0, query_hash, str(cand.frame_id))

            # Check cache first
            if cache_key in _BLIP2_SCORE_CACHE:
                ce_min_max = _BLIP2_SCORE_CACHE[cache_key]
            else:
                ce_score = self._score_frame_queries(torch_model, processor, scoring_queries, resolved, dev, dtype)
                if ce_score is not None:
                    # Min-max normalize will be done after collecting all scores
                    scores.append((idx, float(ce_score)))
                continue

            # Apply cached score directly (already normalized)
            to_rerank[idx].score += weight * ce_min_max
            logger.debug("BLIP2Reranker: cache hit for frame_id=%d", cand.frame_id)

        # Process uncached candidates
        if scores:
            # Min-max normalise the cross-encoder scores over the touched pool.
            vals = np.array([s for _, s in scores], dtype=np.float64)
            lo, hi = float(vals.min()), float(vals.max())
            span = hi - lo

            for idx, raw_score in scores:
                ce_min_max = (raw_score - lo) / span if span > 1e-9 else 0.5
                # Cache the normalized score
                cand = to_rerank[idx]
                cache_key = (cand.vector_id or 0, query_hash, str(cand.frame_id))
                _BLIP2_SCORE_CACHE[cache_key] = ce_min_max
                # Apply blended score
                cand.score += weight * ce_min_max

        if not _BLIP2_SCORE_CACHE and not scores:
            logger.debug("BLIP2Reranker: no scorable candidates, returning unchanged.")
            return candidates

        # Re-sort the whole candidate list (reranked top-K + untouched tail).
        candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        logger.info(
            "BLIP2Reranker: reranked %d/%d candidates (weight=%.2f, cache_size=%d)",
            len(scores) + sum(1 for c in to_rerank if (c.vector_id or 0, query_hash, str(c.frame_id)) in _BLIP2_SCORE_CACHE),
            len(to_rerank), weight, len(_BLIP2_SCORE_CACHE),
        )
        return candidates

    @staticmethod
    def _resolve_frame_path(keyframe_path: str | None) -> Path | None:
        """Resolve local paths and lazily materialize keyframes stored in ZIPs."""
        return resolve_keyframe_path(keyframe_path)

    @staticmethod
    def _score_frame_queries(
        model: Any,
        processor: Any,
        queries: list[str],
        image_path: Path,
        device: Any,
        dtype: Any,
    ) -> float | None:
        """Score multiple queries against a single keyframe image, opening PIL Image once.

        We feed ``<VQA>{query}`` as the text prefix and the image, then measure
        the log-likelihood of the query continuation under the image-conditioned LM head.
        """
        try:
            from PIL import Image
            import torch
        except ImportError:
            return None

        try:
            image = Image.open(str(image_path)).convert("RGB")
            prompts = [f"<VQA>{q}?" for q in queries]
            images = [image] * len(prompts)

            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            if device is not None and hasattr(inputs, "to"):
                inputs = inputs.to(device)
            input_ids = inputs["input_ids"]
            pixel_values = inputs.get("pixel_values")

            with torch.inference_mode():
                kwargs: dict[str, Any] = {"input_ids": input_ids}
                if pixel_values is not None:
                    kwargs["pixel_values"] = pixel_values
                if dtype is not None:
                    kwargs["return_dict"] = True
                # Use half-precision on CUDA for ~2x speedup.
                use_amp = (str(device) != "cpu" and torch.cuda.is_available())
                with torch.autocast("cuda", enabled=use_amp):
                    outputs = model(**kwargs)
                logits = outputs.logits  # (B, T, V)
                if logits is None:
                    return None

                shift_logits = logits[:, :-1, :].float()
                shift_labels = input_ids[:, 1:]
                vocab_size = shift_logits.shape[-1]

                detail_scores: list[float] = []
                for b in range(shift_logits.shape[0]):
                    flat_logits = shift_logits[b].reshape(-1, vocab_size)
                    flat_labels = shift_labels[b].reshape(-1)
                    valid = flat_labels >= 0
                    if not valid.any():
                        continue
                    loss = torch.nn.functional.cross_entropy(
                        flat_logits[valid],
                        flat_labels[valid],
                        reduction="mean",
                    )
                    detail_scores.append(-float(loss.item()))

            if detail_scores:
                return float(np.max(detail_scores))
            return None
        except Exception as exc:  # noqa: BLE001
            logger.debug("BLIP2Reranker: _score_frame_queries failed for %s: %s", image_path, exc)
            return None


__all__ = ["BLIP2Reranker"]
