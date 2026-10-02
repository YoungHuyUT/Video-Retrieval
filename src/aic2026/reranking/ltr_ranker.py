"""Learning-to-Rank ranker (Improvement.md Task 7).

Replaces the linear weight blend with a trained GradientBoosting model
from scikit-learn.  Features: [object_score, colour_score, blip2_score,
late_interaction_score, event_coverage_score, asr_score].
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_PATH = Path("models/ltr_ranker.joblib")

FEATURE_NAMES = [
    "object_score",
    "colour_score",
    "blip2_score",
    "late_interaction_score",
    "event_coverage_score",
    "asr_score",
]
NUM_FEATURES = len(FEATURE_NAMES)


class LTRRanker:
    """GradientBoosting LTR ranker replacing linear weight blending."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._available = False

    def load(self, model_path: str | Path | None = None) -> bool:
        path = Path(model_path) if model_path else _MODEL_PATH
        if not path.exists():
            logger.info("LTR model not found at %s — using linear blend", path)
            return False
        try:
            import joblib
            self._model = joblib.load(path)
            self._available = True
            logger.info("LTR model loaded from %s", path)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load LTR model: %s", exc)
            return False

    def is_available(self) -> bool:
        return self._available and self._model is not None

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.is_available():
            weights = np.array([0.04, 0.08, 0.30, 0.25, 0.40, 0.12])
            return features @ weights
        return self._model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not self.is_available():
            weights = np.array([0.04, 0.08, 0.30, 0.25, 0.40, 0.12])
            raw = features @ weights
            return 1.0 / (1.0 + np.exp(-raw))
        proba = self._model.predict_proba(features)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()


def build_feature_matrix(candidates: list[Any]) -> np.ndarray:
    """Extract reranker feature scores from candidates."""
    features = np.zeros((len(candidates), NUM_FEATURES), dtype=np.float32)
    attrs = [
        "_object_score", "_colour_score", "_blip2_score",
        "_late_interaction_score", "_event_coverage_score", "_asr_score",
    ]
    for i, cand in enumerate(candidates):
        for j, attr in enumerate(attrs):
            features[i, j] = getattr(cand, attr, 0.0)
    return features
