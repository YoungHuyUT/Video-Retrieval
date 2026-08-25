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
    _bm25_ocr: object = field(init=False, repr=False, default=None)
    _bm25_asr: object = field(init=False, repr=False, default=None)
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

        # Precompute per-video text once (NOT per frame).
        store = video_metadata
        video_text: dict[str, str] = {}
        if store is not None:
            for vm in store.all():
                chunks = [vm.title or "", vm.description or "", *vm.metadata_keywords]
                text = " ".join(chunks).strip()
                if text:
                    video_text[vm.video_id] = text
        self._video_text_by_id = video_text

        corpus: list[list[str]] = []
        corpus_ocr: list[list[str]] = []
        corpus_asr: list[list[str]] = []

        for record in manifest:
            text = video_text.get(record.video_id, "")
            if not text and record.metadata_keywords:
                text = " ".join(record.metadata_keywords)
            corpus.append(_record_tokens(record, video_meta_text=text))
            # Modality-specific token pools
            corpus_ocr.append(_tokenize(" ".join(record.object_labels or [])))
            asr_texts = getattr(record, "asr_text", None) or []
            corpus_asr.append(_tokenize(" ".join(asr_texts)))

        self._size = len(corpus)
        if not any(corpus):
            self._bm25 = None
            self._bm25_ocr = None
            self._bm25_asr = None
            self._empty = True
            return

        self._bm25 = BM25Okapi(corpus)
        self._apply_idf_floor(self._bm25)

        self._bm25_ocr = BM25Okapi(corpus_ocr) if any(corpus_ocr) else None
        if self._bm25_ocr is not None:
            self._apply_idf_floor(self._bm25_ocr)

        self._bm25_asr = BM25Okapi(corpus_asr) if any(corpus_asr) else None
        if self._bm25_asr is not None:
            self._apply_idf_floor(self._bm25_asr)

        self._empty = False

    @staticmethod
    def _apply_idf_floor(bm25_obj: object, floor: float = 0.01) -> None:
        """Ensure positive IDF floor so matching terms still yield positive scores (> 0)."""
        if hasattr(bm25_obj, "idf") and isinstance(bm25_obj.idf, dict):
            for word, val in bm25_obj.idf.items():
                if val <= 0:
                    bm25_obj.idf[word] = floor

    @property
    def is_empty(self) -> bool:
        """True when no frame has any lexical text (Objects/Metadata empty)."""
        return self._empty

    # ------------------------------------------------------------------
    def _search_engine(self, engine: object | None, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        if engine is None or self._size == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)
        tokens = _tokenize(query)
        if not tokens:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)

        scores: np.ndarray = np.asarray(
            engine.get_scores(tokens), dtype=np.float32  # type: ignore
        )
        positive_ids = np.flatnonzero(scores > 0)
        if positive_ids.size == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)

        k = min(k, positive_ids.size)
        if k <= 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)

        candidate_scores = scores[positive_ids]
        top_positions = np.argpartition(-candidate_scores, k - 1)[:k]
        top_ids = positive_ids[top_positions]
        top_ids = top_ids[np.argsort(-scores[top_ids])]
        return top_ids, scores[top_ids]

    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return top-k (indices, bm25_scores) across full lexical corpus."""
        return self._search_engine(self._bm25, query, k)

    def search_ocr(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return top-k (indices, bm25_scores) specifically matching on-screen OCR text & object labels."""
        if self._bm25_ocr is None:
            return self.search(query, k)
        res = self._search_engine(self._bm25_ocr, query, k)
        return res if len(res[0]) > 0 else self.search(query, k)

    def search_asr(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return top-k (indices, bm25_scores) specifically matching speech transcript (ASR) text."""
        if self._bm25_asr is None:
            return self.search(query, k)
        res = self._search_engine(self._bm25_asr, query, k)
        return res if len(res[0]) > 0 else self.search(query, k)

    @classmethod
    def from_manifest(
        cls,
        manifest: list[FrameRecord],
        video_metadata: "VideoMetadataStore | None" = None,
    ) -> BM25Index:
        """Convenience constructor; mirrors VectorIndex.from_npy naming style."""
        return cls(manifest, video_metadata=video_metadata)

