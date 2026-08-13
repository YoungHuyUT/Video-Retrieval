from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

from aic2026.models import FrameRecord

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Unicode-aware tokenizer for Vietnamese/English text.

    Lowercases, strips diacritics-safe (keeps accents), removes punctuation,
    and splits on whitespace. Works well for BM25 without an external library.
    """
    text = unicodedata.normalize("NFC", text).lower()
    # Remove punctuation but keep Unicode letters, digits and spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if t]


def _record_tokens(record: FrameRecord) -> list[str]:
    """Concatenate all text fields of a FrameRecord into a token list."""
    parts: list[str] = list(record.object_labels)
    if record.title:
        parts.append(record.title)
    if record.description:
        parts.append(record.description)
    return _tokenize(" ".join(parts))


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

@dataclass
class BM25Index:
    """BM25Okapi index over frame-level text metadata.

    Build once from the manifest; call `search` at query time.
    Frames with no text metadata return score=0 for all queries, which is
    correct — they survive only through the vector index.
    """

    _bm25: object = field(init=False, repr=False)
    _size: int = field(init=False, repr=False)
    _empty: bool = field(init=False, repr=False)

    def __init__(self, manifest: list[FrameRecord]) -> None:
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "rank-bm25 is required: uv sync  (or pip install rank-bm25)"
            ) from exc

        corpus = [_record_tokens(record) for record in manifest]
        # BM25Okapi raises ZeroDivisionError when EVERY document is empty
        # (no Objects/Metadata text), because self.idf ends up empty. In that
        # case there is nothing to search lexically: fall back to a degenerate
        # index that scores every document 0 (survives only via the vector index).
        if not any(corpus):
            self._bm25 = None
            self._size = len(corpus)
            self._empty = True
            return
        # BM25Okapi handles individual empty token lists gracefully (scores them 0)
        self._bm25 = BM25Okapi(corpus)
        self._size = len(corpus)
        self._empty = False

    @property
    def is_empty(self) -> bool:
        """True when no frame has any lexical text (Objects/Metadata empty).

        Callers use this to skip the hybrid/BM25 path entirely when the manifest
        carries no text — so retrieval falls back to the vector index without the
        (useless) BM25 round-trip. Once objects/metadata are loaded, this flips to
        False and lexical retrieval is used automatically.
        """
        return self._empty

    # ------------------------------------------------------------------
    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return top-k (indices, bm25_scores) for *query*.

        Indices correspond to positions in the manifest list passed to __init__.
        Scores are raw BM25 values (always >= 0); they are NOT normalised to [0,1]
        because RRF fusion only uses ranks, not raw score magnitudes.
        """
        tokens = _tokenize(query)
        if not tokens or self._size == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)

        # Degenerate index (empty corpus): no lexical evidence, score everything 0.
        if self._bm25 is None:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)

        scores: np.ndarray = np.asarray(
            self._bm25.get_scores(tokens), dtype=np.float32
        )
        k = min(k, self._size)
        if k <= 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)

        # Partial sort for efficiency (same pattern as VectorIndex)
        top_ids = np.argpartition(-scores, k - 1)[:k]
        top_ids = top_ids[np.argsort(-scores[top_ids])]
        return top_ids, scores[top_ids]

    @classmethod
    def from_manifest(cls, manifest: list[FrameRecord]) -> BM25Index:
        """Convenience constructor; mirrors VectorIndex.from_npy naming style."""
        return cls(manifest)
