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

# Pre-compile the punctuation-stripping regex once (module load), instead of
# re-compiling it 177k times inside the build loop.
_TOKEN_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

# Cache of already-tokenized object-label sets keyed by their joined string.
# Object labels are a small fixed vocabulary from Faster R-CNN, so this avoids
# re-running NFC/lower/regex on the same short phrase thousands of times.
_OBJ_TOK_CACHE: dict[tuple[str, ...], list[str]] = {}


def _tokenize(text: str) -> list[str]:
    """Unicode-aware tokenizer for Vietnamese/English text.

    Lowercases, strips diacritics-safe (keeps accents), removes punctuation,
    and splits on whitespace. Works well for BM25 without an external library.
    """
    text = unicodedata.normalize("NFC", text).lower()
    # Remove punctuation but keep Unicode letters, digits and spaces
    text = _TOKEN_RE.sub(" ", text)
    return [t for t in text.split() if t]


def _tokenize_object_labels(labels: list[str]) -> list[str]:
    """Tokenize a frame's object-label list, cached by the exact label tuple."""
    key = tuple(labels)
    cached = _OBJ_TOK_CACHE.get(key)
    if cached is None:
        cached = _tokenize(" ".join(labels))
        _OBJ_TOK_CACHE[key] = cached
    return cached


def _apply_idf_zero_floor(bm25: object) -> None:
    """Patch rank_bm25's idf table to floor *negative* values to epsilon.

    rank_bm25 0.2.2 computes ``idf = log(N - freq + 0.5) - log(freq + 0.5)`` and
    only replaces NEGATIVE idf with ``epsilon * average_idf``.  However, the
    default replacement sometimes leaves *very small negative* values that
    ``np.flatnonzero(scores > 0)`` filters out, silently destroying recall on
    small corpora.

    This mirrors rank_bm25's own negative-idf handling but ensures ALL negative
    idf values are replaced with ``epsilon * average_idf`` (with a 1e-9 absolute
    floor).  Terms with idf == 0 (appearing in exactly N/2 documents) are left
    untouched — they correctly contribute zero weight, which is the expected
    behaviour on tiny corpora where the term is not discriminative.
    """
    idf: dict = getattr(bm25, "idf", None)
    average_idf: float = getattr(bm25, "average_idf", 0.0)
    epsilon: float = getattr(bm25, "epsilon", 0.25)
    if not idf:
        return
    eps = epsilon * average_idf
    if eps <= 0 or eps != eps:  # NaN guard
        eps = 1e-9
    eps = max(eps, 1e-9)
    for word, value in list(idf.items()):
        if value < 0:
            idf[word] = eps


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

@dataclass
class BM25Index:
    """BM25Okapi index over frame-level text metadata + OCR/ASR transcripts.

    Build once from the manifest; call `search` at query time.
    Frames with no text metadata return score=0 for all queries, which is
    correct — they survive only through the vector index.

    Optional ``video_metadata`` carries the per-video title/description/keywords
    text once (not 200x duplicated) so BM25 sees the canonical document for
    each video. Each frame in the manifest inherits its video's text via the
    precomputed ``_video_text_by_id`` cache.

    Phase 4 (spec §6 + paper arxiv 2512.12935v1): ``*_text_by_video`` add the
    **OCR** and **ASR** lexical branches.  Both arrive as precomputed sidecars
    (PaddleOCR on keyframes; local Whisper on source video) — never recomputed
    at query time.  They are merged into each video's BM25 document so a lexical
    query (e.g. *"the sign reads STOP"*, *'a woman says "remember"'*) can match
    through RRF fusion even when CLIP/objects carry no signal.  Both are optional
    and default to empty, so legacy builds (object_labels + metadata only) keep
    working unchanged.
    """

    _bm25: object = field(init=False, repr=False)
    _size: int = field(init=False, repr=False)
    _empty: bool = field(init=False, repr=False)
    _video_text_by_id: dict[str, str] = field(init=False, repr=False, default_factory=dict)

    _manifest_video_ids: list[str] = field(init=False, repr=False, default_factory=list)
    _object_concept_to_rows: dict[str, np.ndarray] = field(init=False, repr=False, default_factory=dict)
    _object_token_to_rows: dict[str, np.ndarray] = field(init=False, repr=False, default_factory=dict)

    def __init__(
        self,
        manifest: list[FrameRecord],
        video_metadata: "VideoMetadataStore | None" = None,
        # Phase 4: OCR/ASR lexical branches (optional, precomputed sidecars).
        ocr_text_by_video: "dict[str, str] | None" = None,
        asr_text_by_video: "dict[str, str] | None" = None,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "rank-bm25 is required: uv sync  (or pip install rank-bm25)"
            ) from exc

        self._manifest_video_ids = [record.video_id for record in manifest]

        # Build a compact inverted index for the detector vocabulary. Object
        # queries can then run BM25 only over matching frame rows instead of
        # allocating/scanning a score vector for the whole 397k corpus.
        try:
            from aic2026.reranking.lexical import (
                _OBJECT_ALIASES, _alias_matches, _fold_accents,
            )
            label_to_concepts: dict[str, set[str]] = {}
            observed_labels = {
                label for record in manifest for label in (record.object_labels or [])
            }
            for label in observed_labels:
                folded_label = _fold_accents(label)
                label_to_concepts[label] = {
                    concept
                    for concept, aliases in _OBJECT_ALIASES.items()
                    if any(_alias_matches(folded_label, alias) for alias in aliases)
                }
            postings: dict[str, list[int]] = {}
            token_postings: dict[str, list[int]] = {}
            for row_id, record in enumerate(manifest):
                label_tokens = set(_tokenize_object_labels(record.object_labels or []))
                for token in label_tokens:
                    token_postings.setdefault(token, []).append(row_id)
                for label in record.object_labels or []:
                    for concept in label_to_concepts.get(label, ()):
                        postings.setdefault(concept, []).append(row_id)
            self._object_concept_to_rows = {
                concept: np.asarray(ids, dtype=np.intp)
                for concept, ids in postings.items()
            }
            self._object_token_to_rows = {
                token: np.asarray(ids, dtype=np.intp)
                for token, ids in token_postings.items()
            }
        except Exception:
            self._object_concept_to_rows = {}
            self._object_token_to_rows = {}

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

        # Phase 4: merge OCR + ASR sidecar text into each video's lexical
        # document.  Order: base metadata < OCR < ASR (later sources append so a
        # term appearing in ASR also counts).  Content is de-duplicated by the
        # tokenizer/BM25 naturally; we just concatenate space-separated.
        ocr_map = ocr_text_by_video or {}
        asr_map = asr_text_by_video or {}
        for vid in set(list(video_text) + list(ocr_map) + list(asr_map)):
            chunks = []
            if video_text.get(vid):
                chunks.append(video_text[vid])
            if ocr_map.get(vid):
                chunks.append(ocr_map[vid])
            if asr_map.get(vid):
                chunks.append(asr_map[vid])
            merged = " ".join(c for c in chunks if c).strip()
            if merged:
                video_text[vid] = merged
        # Backward-compat: legacy manifests carry keywords per-frame. If the
        # store is empty or absent for a video, fall back to the per-frame
        # metadata_keywords so we don't silently lose all lexical evidence.
        self._video_text_by_id = video_text

        # Per-video lexical text is identical across all frames of the same
        # video.  Tokenize it ONCE per video (not once per frame) so a 873-video
        # corpus doesn't re-run NFC + regex over the same multi-hundred-word
        # string ~200 times.  Frames just concatenate their already-cached
        # object-label tokens with the cached per-video token list.  This is the
        # hot-path fix for the 177k-record cold build (was >120s, now seconds).
        video_tokens: dict[str, list[str]] = {}
        for vid, txt in video_text.items():
            video_tokens[vid] = _tokenize(txt)
        legacy_tokens: dict[str, list[str]] = {}

        corpus: list[list[str]] = []
        for record in manifest:
            vtoks = video_tokens.get(record.video_id)
            if vtoks is None:
                # Legacy fallback — keyword was copied per frame in old builds.
                if record.metadata_keywords:
                    ltoks = legacy_tokens.get(record.video_id)
                    if ltoks is None:
                        ltoks = _tokenize(" ".join(record.metadata_keywords))
                        legacy_tokens[record.video_id] = ltoks
                    vtoks = ltoks
                else:
                    vtoks = []
            # Frame object labels are served by the compact concept postings
            # above. Keeping hundreds of thousands of per-frame object token
            # lists inside BM25 duplicates that index and makes startup slow.
            corpus.append(vtoks or [])
        # BM25Okapi raises ZeroDivisionError when EVERY document is empty
        # (no Objects/Metadata text), because self.idf ends up empty. In that
        # case there is nothing to search lexically: fall back to a degenerate
        # index that scores every document 0 (survives only via the vector index).
        if not any(corpus):
            self._bm25 = None
            self._size = len(corpus)
            self._empty = not bool(self._object_concept_to_rows or self._object_token_to_rows)
            return
        # BM25Okapi handles individual empty token lists gracefully (scores them 0)
        self._bm25 = BM25Okapi(corpus)
        # rank_bm25 0.2.2 epsilon floor only replaces NEGATIVE idf (idf < 0).
        # Terms appearing in exactly N/2 of the corpus get idf == 0 (not negative),
        # so they escape the floor and contribute zero score — silently killing
        # recall on tiny corpora (e.g. "turtle" in 2 of 4 docs scores every doc 0).
        # Apply the epsilon floor to zero-idf terms too, mirroring the negative-idf
        # handling rank_bm25 omits. The floor value matches rank_bm25's own formula:
        #   eps = epsilon * average_idf  (with a tiny 1e-9 floor to stay > 0).
        _apply_idf_zero_floor(self._bm25)
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

    def object_candidate_ids(
        self, query: str, video_ids: set[str] | None = None
    ) -> np.ndarray | None:
        """Return frame postings satisfying object concepts, or None if absent.

        All object concepts in one scene must co-occur in a frame. Ordered
        multi-event queries may match any event's object set, leaving temporal
        coverage to the event-aware reranker.
        """
        from aic2026.reranking.lexical import (
            _COLOR_CONCEPTS, _requested_concepts_cached,
        )
        concepts = _requested_concepts_cached(query) - _COLOR_CONCEPTS
        if not concepts:
            return None
        groups = [concepts]
        try:
            from aic2026.query.parser import parse_query
            plan = parse_query(query)
            if len(plan.events) > 1:
                groups = [
                    set(event.entities) - _COLOR_CONCEPTS
                    for event in plan.events
                    if set(event.entities) - _COLOR_CONCEPTS
                ] or groups
        except Exception:
            pass
        group_rows: list[np.ndarray] = []
        for group in groups:
            if any(concept not in self._object_concept_to_rows for concept in group):
                continue
            rows = [self._object_concept_to_rows[concept] for concept in group]
            if not rows:
                continue
            matching = rows[0]
            for other in rows[1:]:
                matching = np.intersect1d(matching, other, assume_unique=True)
            group_rows.append(matching)
        if not group_rows:
            return np.array([], dtype=np.intp)
        candidate_ids = np.unique(np.concatenate(group_rows))
        if video_ids is not None and self._manifest_video_ids:
            allowed = set(video_ids)
            mask = np.fromiter(
                (self._manifest_video_ids[int(idx)] in allowed for idx in candidate_ids),
                dtype=bool, count=len(candidate_ids),
            )
            candidate_ids = candidate_ids[mask]
        return candidate_ids

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int,
        video_ids: set[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return top-k (indices, bm25_scores) for *query*, optionally filtered by video_ids.

        Indices correspond to positions in the manifest list passed to __init__.
        Scores are raw BM25 values (always >= 0); they are NOT normalised to [0,1]
        because RRF fusion only uses ranks, not raw score magnitudes.
        """
        tokens = _tokenize(query)
        if not tokens or self._size == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)

        # Degenerate index (empty corpus): no lexical evidence, score everything 0.
        # Fast path: for explicit object queries, BM25 ranks only frames whose
        # detector labels contain one of those concepts. This is still a true
        # BM25 score on the matching subset, but avoids a full-corpus pass.
        try:
            from aic2026.reranking.lexical import (
                _COLOR_CONCEPTS, _requested_concepts_cached,
            )
            concepts = _requested_concepts_cached(query) - _COLOR_CONCEPTS
            concept_groups = [concepts] if concepts else []
            from aic2026.query.parser import parse_query
            plan = parse_query(query)
            if len(plan.events) > 1:
                event_groups = [
                    set(event.entities) - _COLOR_CONCEPTS
                    for event in plan.events
                ]
                concept_groups = [group for group in event_groups if group]
        except Exception:
            concept_groups = []
        object_rows = []
        for group in concept_groups:
            if any(concept not in self._object_concept_to_rows for concept in group):
                continue
            group_rows = [self._object_concept_to_rows[concept] for concept in group]
            if not group_rows:
                continue
            matching = group_rows[0]
            for rows in group_rows[1:]:
                matching = np.intersect1d(matching, rows, assume_unique=True)
            object_rows.append(matching)
        if object_rows:
            # Within one scene all requested objects co-occur; across ordered
            # events a frame may match any event's object set.
            candidate_ids = np.unique(np.concatenate(object_rows))
            if video_ids is not None and self._manifest_video_ids:
                allowed = set(video_ids)
                mask = np.fromiter(
                    (self._manifest_video_ids[int(idx)] in allowed for idx in candidate_ids),
                    dtype=bool, count=len(candidate_ids),
                )
                candidate_ids = candidate_ids[mask]
            if candidate_ids.size == 0:
                return np.array([], dtype=np.intp), np.array([], dtype=np.float32)
            subset_scores = (
                np.asarray(
                    self._bm25.get_batch_scores(tokens, candidate_ids.tolist()),
                    dtype=np.float32,
                )
                if self._bm25 is not None
                else np.ones(len(candidate_ids), dtype=np.float32)
            )
            if not np.any(subset_scores > 0):
                # Vietnamese aliases (e.g. "chó" → detector label "Dog") may
                # be a valid object-label match without shared BM25 tokens.
                subset_scores.fill(1.0)
            positive = np.flatnonzero(subset_scores > 0)
            take = min(int(k), len(positive))
            if take <= 0:
                return np.array([], dtype=np.intp), np.array([], dtype=np.float32)
            best = np.argpartition(-subset_scores[positive], take - 1)[:take]
            best = best[np.argsort(-subset_scores[positive][best])]
            selected = positive[best]
            return candidate_ids[selected], subset_scores[selected]

        # Generic object-label words that do not map to a canonical detector
        # concept (e.g. "beach") still get a fast lexical posting lookup.
        token_rows = [
            self._object_token_to_rows[token]
            for token in tokens
            if token in self._object_token_to_rows
        ]
        if token_rows:
            candidate_ids = np.unique(np.concatenate(token_rows))
            if video_ids is not None and self._manifest_video_ids:
                allowed = set(video_ids)
                mask = np.fromiter(
                    (self._manifest_video_ids[int(idx)] in allowed for idx in candidate_ids),
                    dtype=bool, count=len(candidate_ids),
                )
                candidate_ids = candidate_ids[mask]
            if candidate_ids.size == 0:
                return np.array([], dtype=np.intp), np.array([], dtype=np.float32)
            subset_scores = (
                np.asarray(self._bm25.get_batch_scores(tokens, candidate_ids.tolist()), dtype=np.float32)
                if self._bm25 is not None
                else np.ones(len(candidate_ids), dtype=np.float32)
            )
            if not np.any(subset_scores > 0):
                subset_scores.fill(1.0)
            take = min(int(k), len(candidate_ids))
            best = np.argpartition(-subset_scores, take - 1)[:take]
            best = best[np.argsort(-subset_scores[best])]
            return candidate_ids[best], subset_scores[best]

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

        # Apply video_ids filter early if specified
        if video_ids is not None and self._manifest_video_ids:
            allowed_set = set(video_ids)
            filtered_mask = np.fromiter(
                (self._manifest_video_ids[int(idx)] in allowed_set for idx in positive_ids),
                dtype=bool,
                count=positive_ids.size,
            )
            positive_ids = positive_ids[filtered_mask]
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
        ocr_text_by_video: "dict[str, str] | None" = None,
        asr_text_by_video: "dict[str, str] | None" = None,
    ) -> BM25Index:
        """Convenience constructor; mirrors VectorIndex.from_npy naming style."""
        return cls(
            manifest,
            video_metadata=video_metadata,
            ocr_text_by_video=ocr_text_by_video,
            asr_text_by_video=asr_text_by_video,
        )

    @classmethod
    def from_asr_sidecar(
        cls,
        manifest: list[FrameRecord],
        asr_sidecar: "str | Path",
        video_metadata: "VideoMetadataStore | None" = None,
        ocr_text_by_video: "dict[str, str] | None" = None,
    ) -> BM25Index:
        """Build a BM25 index that includes ASR transcript text (Phase 4).

        Reads a precomputed ``.jsonl`` ASR sidecar (one line per video, produced
        by :func:`aic2026.ingestion.asr.save_transcripts_sidecar`) and merges each
        video's transcript into its BM25 document.  Transcription is offline
        ingestion — this method only *reads* the sidecar, never runs Whisper.
        """
        from aic2026.ingestion.asr import load_transcripts_sidecar

        transcripts = load_transcripts_sidecar(asr_sidecar)
        asr_text_by_video = {
            vid: t.full_text for vid, t in transcripts.items() if t.full_text
        }
        return cls(
            manifest,
            video_metadata=video_metadata,
            ocr_text_by_video=ocr_text_by_video,
            asr_text_by_video=asr_text_by_video,
        )
