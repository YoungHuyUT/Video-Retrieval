from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from aic2026.models import FrameRecord

if TYPE_CHECKING:
    from .video_metadata import VideoMetadataStore

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


def _record_tokens(
    record: FrameRecord,
    video_meta_text: str = "",
) -> list[str]:
    """Concatenate text fields of a FrameRecord into a token list.

    Bao gồm ``object_labels`` (entity tiếng Anh từ Faster R-CNN) + video-level
    metadata keywords/title/description (đã được tách ra ngoài manifest, truyền
    qua ``video_meta_text`` để giữ nguyên behavior cũ mà không nhân bản keyword
    trên từng frame — chỉnh IDF BM25 đúng nghĩa).

    Tokenizer giữ nguyên unicode để match cả query tiếng Việt lẫn tiếng Anh.
    """
    parts: list[str] = list(record.object_labels)
    if getattr(record, "asr_text", None):
        parts.extend(record.asr_text)
    if video_meta_text:
        parts.append(video_meta_text)
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

    Optional ``video_metadata`` carries the per-video title/description/keywords
    text once (not 200x duplicated) so BM25 sees the canonical document for
    each video. Each frame in the manifest inherits its video's text via the
    precomputed ``_video_text_by_id`` cache.
    """

    _bm25: object = field(init=False, repr=False)
    _size: int = field(init=False, repr=False)
    _empty: bool = field(init=False, repr=False)
    _video_text_by_id: dict[str, str] = field(init=False, repr=False, default_factory=dict)

    def __init__(
        self,
        manifest: list[FrameRecord],
        video_metadata: "VideoMetadataStore | None" = None,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "rank-bm25 is required: uv sync  (or pip install rank-bm25)"
            ) from exc

        # Precompute per-video text once (NOT per frame). This is the fix:
        # previously the same video's keywords were tokenized 200 times, which
        # inflated term frequency and depressed IDF for video-level vocabulary.
        store = video_metadata
        video_text: dict[str, str] = {}
        if store is not None:
            for vm in store.all():
                chunks = [vm.title or "", vm.description or "", *vm.metadata_keywords]
                text = " ".join(chunks).strip()
                if text:
                    video_text[vm.video_id] = text
        # Backward-compat: legacy manifests carry keywords per-frame. If the
        # store is empty or absent for a video, fall back to the per-frame
        # metadata_keywords so we don't silently lose all lexical evidence.
        self._video_text_by_id = video_text

        corpus: list[list[str]] = []
        for record in manifest:
            text = video_text.get(record.video_id, "")
            if not text and record.metadata_keywords:
                # Legacy fallback — keyword was copied per frame in old builds.
                text = " ".join(record.metadata_keywords)
            corpus.append(_record_tokens(record, video_meta_text=text))
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
        # Ensure positive IDF floor so terms matching in small test corpora or
        # 50% frequency documents still yield positive scores (> 0).
        if hasattr(self._bm25, "idf") and isinstance(self._bm25.idf, dict):
            for word, val in self._bm25.idf.items():
                if val <= 0:
                    self._bm25.idf[word] = 0.01
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
        # Do not return arbitrary zero-score documents.  When a query has no
        # lexical match, ``argpartition`` otherwise picks implementation/order
        # dependent rows and RRF incorrectly boosts them over vector results.
        positive_ids = np.flatnonzero(scores > 0)
        if positive_ids.size == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)

        k = min(k, positive_ids.size)
        if k <= 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)

        # Partial sort for efficiency (same pattern as VectorIndex)
        candidate_scores = scores[positive_ids]
        top_positions = np.argpartition(-candidate_scores, k - 1)[:k]
        top_ids = positive_ids[top_positions]
        top_ids = top_ids[np.argsort(-scores[top_ids])]
        return top_ids, scores[top_ids]

    @classmethod
    def from_manifest(
        cls,
        manifest: list[FrameRecord],
        video_metadata: "VideoMetadataStore | None" = None,
    ) -> BM25Index:
        """Convenience constructor; mirrors VectorIndex.from_npy naming style."""
        return cls(manifest, video_metadata=video_metadata)
