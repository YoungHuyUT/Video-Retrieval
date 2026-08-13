from __future__ import annotations

from typing import Protocol

from aic2026.qa.answers import normalize_answer

# ---------------------------------------------------------------------------
# Semantic answer matching for Q&A evaluation.
#
# The AIC 2026 PDF states that a Q&A answer "matches" the gold answer
# semantically, not as an exact string. This module provides a deterministic
# default (exact + normalized + synonyms) and optional plug-ins (embedding
# cosine threshold, or an LLM judge) that can be enabled per evaluation run.
# ---------------------------------------------------------------------------


class SemanticEmbedder(Protocol):
    """Encodes text into a dense vector; used for cosine semantic matching."""

    def encode(self, text: str) -> object: ...


class SemanticJudge(Protocol):
    """Decides whether two answers are semantically equivalent."""

    def judge(self, predicted: str, gold: str) -> bool: ...


# Synonym groups for common Vietnamese answer wording. Both entries in a
# group are treated as equivalent after normalization.
_SYNONYM_GROUPS: list[tuple[str, ...]] = [
    ("xanh", "màu xanh", "xanh dương", "màu xanh dương"),
    ("đỏ", "màu đỏ"),
    ("trắng", "màu trắng"),
    ("đen", "màu đen"),
    ("vàng", "màu vàng"),
    ("nâu", "màu nâu"),
    ("tím", "màu tím"),
    ("hồng", "màu hồng"),
    ("cam", "màu cam"),
    ("xanh lá", "xanh lá cây", "màu xanh lá", "màu xanh lá cây"),
    ("đứng", "đứng dậy", "đứng lên"),
    ("ngồi", "ngồi xuống"),
    ("chạy", "đang chạy", "đang chạy bộ"),
    ("đi", "đang đi", "đi bộ", "đang đi bộ"),
    ("nam", "người nam", "đàn ông", "người đàn ông"),
    ("nữ", "người nữ", "phụ nữ", "người phụ nữ"),
    ("có", "có mặt", "xuất hiện"),
]


def _normalize(value: str) -> str:
    return " ".join(normalize_answer(value).split())


def _synonym_match(predicted: str, gold: str) -> bool:
    """Check whether both strings belong to the same synonym group."""
    pred_norm = _normalize(predicted)
    gold_norm = _normalize(gold)
    if pred_norm == gold_norm:
        return True
    for group in _SYNONYM_GROUPS:
        if pred_norm in group and gold_norm in group:
            return True
    return False


def _strip_color_prefix(value: str) -> str:
    """Drop a leading 'màu X' qualifier for light synonym matching."""
    text = _normalize(value)
    for prefix in ("màu", "màu sắc"):
        if text.startswith(prefix + " "):
            rest = text[len(prefix) + 1 :].strip()
            if rest:
                return rest
    return text


class SemanticMatcher:
    """Deterministic-by-default semantic matcher with optional plug-ins.

    Matching order (first hit wins):
      1. exact / normalized equality
      2. synonym-group membership
      3. optional embedding cosine threshold (if an embedder is provided)
      4. optional LLM judge (if a judge is provided)
    If none of the plug-ins are configured and no rule matched, return False.
    """

    def __init__(
        self,
        embedder: SemanticEmbedder | None = None,
        threshold: float = 0.72,
        judge: SemanticJudge | None = None,
    ) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.judge = judge

    def match(self, predicted: str, gold: str) -> bool:
        if not predicted or not gold:
            return False

        if _synonym_match(predicted, gold):
            return True

        # Color answers commonly use "màu X" vs "X" interchangeably.
        if _strip_color_prefix(predicted) and _strip_color_prefix(predicted) == _strip_color_prefix(gold):
            return True

        if self.embedder is not None:
            pred_vec = self.embedder.encode(predicted)
            gold_vec = self.embedder.encode(gold)
            similarity = self._cosine(pred_vec, gold_vec)
            if similarity >= self.threshold:
                return True

        return self.judge is not None and self.judge.judge(predicted, gold)

    @staticmethod
    def _cosine(left: object, right: object) -> float:
        import numpy as np

        left_arr = np.asarray(left, dtype=np.float32).reshape(-1)
        right_arr = np.asarray(right, dtype=np.float32).reshape(-1)
        norm = float(
            np.linalg.norm(left_arr) * np.linalg.norm(right_arr)
        )
        if norm == 0:
            return 0.0
        return float(np.dot(left_arr, right_arr) / norm)
