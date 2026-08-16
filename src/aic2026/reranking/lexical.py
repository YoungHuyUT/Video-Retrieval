"""Reranking lexical — RRF fusion, score normalization, and metadata-aware reranking.

This module implements the lexical/heuristic reranking layer of the retrieval
pipeline.  Its core primitive is **Reciprocal Rank Fusion (RRF)**, a scale-free
method for combining multiple ranked lists (vector search, BM25, late-interaction,
…) into a single consensus ranking.

Design notes
------------
* RRF is intentionally lightweight — no learned model, no LLM — so it can run
  over thousands of candidates in milliseconds.
* Scores from different retrievers live on different scales (RRF 0.0018–0.016,
  CLIP cosine 0.2–0.35, …).  Before blending, ``normalize_scores`` maps every
  candidate onto [0, 1] via **max-division** so magnitudes are comparable and
  no single signal silently dominates.
* ``rerank_with_metadata`` adds a *bounded* keyword-overlap bonus (match ratio
  × weight) on top of the normalized retrieval score, lifting frames whose
  labels or title literally mention query terms.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np

from aic2026.models import Candidate, FrameRecord


def _fold_accents(value: str) -> str:
    """Fold Unicode accents and case so Vietnamese matches are accent-insensitive.

    Applies NFC normalization, casefolds, transliterates ``đ/Đ`` (which Unicode
    NFD does not decompose), then strips combining marks.  This mirrors the
    behavior already used by ``SearchPipeline.filter_videos_by_metadata`` so
    that keyword-overlap matching works for accented Vietnamese queries.
    """
    value = unicodedata.normalize("NFC", value).casefold()
    value = value.translate(str.maketrans({"đ": "d"}))
    return "".join(
        c for c in unicodedata.normalize("NFD", value)
        if unicodedata.category(c) != "Mn"
    )

__all__ = [
    "RRF_K",
    "rrf_fuse",
    "rrf_rank",
    "normalize_scores",
    "object_evidence_adjustment",
    "rerank_with_object_evidence",
    "rerank_with_metadata",
]


# Canonical object concepts used by the OpenImages detector.  The aliases make
# the query vocabulary forgiving ("people" must match the detector's
# "Human"), without asking the user to configure object terms by hand.
_OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "person": ("person", "people", "human", "man", "woman", "boy", "girl", "child"),
    "turtle": ("turtle", "sea turtle", "tortoise"),
    "fish": ("fish",),
    "dog": ("dog", "puppy"),
    "cat": ("cat", "kitten"),
    "car": ("car", "automobile", "vehicle"),
    "bicycle": ("bicycle", "bike"),
    "motorcycle": ("motorcycle", "motorbike"),
    "bird": ("bird",),
    "horse": ("horse",),
    "elephant": ("elephant",),
    "cow": ("cow", "cattle"),
    "bus": ("bus",),
    "train": ("train",),
    "boat": ("boat", "ship"),
}


def _has_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?<!\\w)" + re.escape(phrase) + r"(?!\\w)", text))


def object_evidence_adjustment(
    query: str,
    record: FrameRecord,
    weight: float = 0.03,
) -> float | None:
    """Return a signed object-evidence adjustment, or ``None`` if no object is asked.

    A full object match gets ``+weight`` and a frame missing every requested
    object gets ``-weight``.  This is intentionally a soft penalty: Faster
    R-CNN can miss small/occluded objects, so vector evidence is never discarded.
    """
    query_text = _fold_accents(query)
    requested = {
        concept
        for concept, aliases in _OBJECT_ALIASES.items()
        if any(_has_phrase(query_text, alias) for alias in aliases)
    }
    if not requested:
        return None

    labels = _fold_accents(" ".join(record.object_labels or []))
    matched = sum(
        any(_has_phrase(labels, alias) for alias in _OBJECT_ALIASES[concept])
        for concept in requested
    )
    coverage = matched / len(requested)
    return weight * (2.0 * coverage - 1.0)


def rerank_with_object_evidence(
    query: str,
    candidates: list[Candidate],
    records: dict[int, FrameRecord],
    weight: float = 0.03,
) -> list[Candidate]:
    """Apply signed Object-detection evidence after vector/BM25 retrieval."""
    reranked: list[Candidate] = []
    for item in candidates:
        adjustment = None
        if item.vector_id is not None:
            record = records.get(item.vector_id)
            if record is not None:
                adjustment = object_evidence_adjustment(query, record, weight)
        score = float(item.score) if adjustment is None else float(item.score) + adjustment
        reranked.append(item.model_copy(update={"score": score}))
    return sorted(reranked, key=lambda candidate: candidate.score, reverse=True)

# --- RRF (Reciprocal Rank Fusion) -------------------------------------------------

_RRF_K = 60  # standard constant from reciprocal-rank-fusion literature
RRF_K = _RRF_K  # public alias re-exported via ``__all__``


def rrf_fuse(
    ranked_lists: list[list[int]],
    k: int = _RRF_K,
) -> dict[int, float]:
    """Fuse multiple ranked lists using **Reciprocal Rank Fusion**.

    Given several ranked lists of manifest indices (each list from a different
    retrieval method — e.g. vector search, BM25, late interaction), RRF combines
    them by summing reciprocal ranks:

        score(idx) = Σᵢ  1 / (k + rankᵢ(idx) + 1)

    where *rankᵢ(idx)* is the 0-based position of manifest index ``idx`` in
    list *i* (indices absent from a list contribute 0 to that list's term).

    Higher scores indicate better combined ranking.  This is **scale-free** and
    works well for fusing heterogeneous retrievers (vector + BM25, vector +
    late-interaction, …) because it only cares about *rank position*, not raw
    similarity magnitudes.

    Parameters
    ----------
    ranked_lists
        A list of ranked manifest-index lists.  Each inner list should be
        ordered from most-relevant to least-relevant for its own retriever.
    k
        The RRF constant that dampens the influence of rank position.  The
        default ``60`` is the value commonly used in the CLIR / IR literature.

    Returns
    -------
    dict[int, float]
        Mapping from manifest index to its fused RRF score, sorted implicitly
        by the caller via ``rrf_rank`` or a simple ``sorted(..., key=…)``.
    """
    fused_scores: dict[int, float] = {}

    for ids in ranked_lists:
        for rank, idx in enumerate(ids):
            contribution = 1.0 / (k + rank + 1)
            fused_scores[idx] = fused_scores.get(idx, 0.0) + contribution

    return fused_scores


def rrf_rank(
    fused_scores: dict[int, float],
    manifest: list[FrameRecord],
    top_n: int = 100,
) -> list[Candidate]:
    """Convert fused RRF scores into a globally ranked :class:`Candidate` list.

    Parameters
    ----------
    fused_scores
        Output from :func:`rrf_fuse`.
    manifest
        The full manifest of :class:`FrameRecord` so we can look up
        ``video_id``, ``frame_id``, ``keyframe_path``, etc.
    top_n
        How many top candidates to return.

    Returns
    -------
    list[Candidate]
        Globally ranked candidates sorted by RRF score (descending).  Each
        ``Candidate`` carries the RRF score as its ``score`` field.
    """
    if not fused_scores:
        return []

    # Rank manifest indices by fused RRF score (highest first).
    sorted_indices = sorted(
        fused_scores, key=fused_scores.get, reverse=True  # type: ignore[arg-type]
    )[:top_n]

    candidates: list[Candidate] = []
    for manifest_idx in sorted_indices:
        record = manifest[manifest_idx]
        candidates.append(
            Candidate(
                video_id=record.video_id,
                frame_id=record.frame_id,
                score=float(fused_scores[manifest_idx]),
                vector_id=record.vector_id,
                keyframe_path=record.keyframe_path,
            )
        )

    return candidates


# --- Metadata-aware reranking ----------------------------------------------------


def normalize_scores(candidates: list[Candidate]) -> list[float]:
    """Normalize candidate scores into [0, 1] via **max-division**.

    RRF (≈0.0018–0.016), CLIP cosine (≈0.2–0.35), metadata bonus, and video-level
    boosts all live on very different scales.  Before blending any of them we
    rescale to a common [0, 1] range so no single signal silently dominates the
    others.

    We use **max-division** (not min-max) deliberately:

    * It preserves the *magnitude* of the retrieval signal — a strong CLIP
      match (0.34) stays meaningfully higher than a weak one (0.20).
    * Min-max would amplify a tiny 9 % RRF lead into a 100 % gap, letting a
      strong CLIP match dominate even when metadata clearly matches a better
      frame.  Max-division keeps relative magnitudes, so a metadata match can
      nudge mid-ranked frames up without overriding genuinely strong retrieval
      results.

    Parameters
    ----------
    candidates
        List of candidates whose ``score`` attribute holds the raw retrieval
        score (RRF, CLIP, or fused).

    Returns
    -------
    list[float]
        Normalized scores in [0, 1], one per candidate, preserving input order.
    """
    if not candidates:
        return []

    raw = np.asarray([float(c.score) for c in candidates], dtype=np.float64)
    hi = float(raw.max())

    if hi < 1e-9:
        # Degenerate pool: all scores are ~0 — return zeros.
        return [0.0 for _ in candidates]

    return list(raw / hi)


def rerank_with_metadata(
    query: str,
    candidates: list[Candidate],
    records: dict[int, FrameRecord],
    weight: float = 0.2,
) -> list[Candidate]:
    """Add a bounded keyword-overlap bonus to candidate scores.

    The previous version added a fixed ``0.05`` per matched token directly onto
    the raw retrieval score.  Because RRF scores are only ~0.0018–0.016, a single
    matched object label (0.05) could outweigh the *entire* RRF ranking and
    catapult low-quality frames to the top.  This version instead:

    1. Expresses the bonus as a **match ratio** (matched_query_terms /
       total_query_terms) so it is bounded to [0, 1] and proportional to how
       much of the query the metadata actually covers.
    2. **Normalizes** incoming retrieval scores into [0, 1] (via max-division)
       first, so the metadata bonus is added on the same scale as the rest of
       the pipeline (RRF / CLIP / video-level).
    3. Sorts by the blended score, lifting frames whose labels/titles literally
       mention the query terms above pure vector matches — but now without
       drowning out the retrieval signal.

    Each candidate is **copied** (``model_copy``) before its score is updated so
    the original retrieval scores are preserved for downstream stages.

    Parameters
    ----------
    query
        The user query string.  Tokenized on whitespace; each token is matched
        as a substring against the frame's metadata.
    candidates
        Retrieval candidates (will NOT be mutated in place).
    records
        Mapping of ``vector_id → FrameRecord`` providing ``object_labels`` (entity
        EN) and ``metadata_keywords`` (keyword VN) for keyword matching.
    weight
        Weight of the metadata bonus on the normalized [0, 1] scale.
        Default ``0.2`` keeps the bonus bounded and subdominant.

    Returns
    -------
    list[Candidate]
        New list of candidates sorted by blended score (descending).  The
        original ``candidates`` list and its elements are left untouched.
    """
    # Degenerate cases: no candidates or bonus disabled → keep RRF order.
    if not candidates or weight <= 0:
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    # Tokenize the query once; match ratio = matched_terms / total_terms.
    query_terms = [t for t in query.lower().split() if t]
    term_set = set(query_terms)
    folded_terms = set(_fold_accents(t) for t in query_terms)

    # Direction B: do NOT normalize scores to [0,1]. Normalizing (dividing by the
    # pool max) collapsed the RRF dynamic range to ~1.0 for the top frame of every
    # video, which washed out the genuine RRF ordering and let the metadata bonus
    # (a fixed absolute add) dominate — breaking retrieval vs plain RRF. Instead we
    # keep the RRF score as the base and add only a SMALL bounded bonus on top, so
    # the metadata overlap acts as a gentle tie-breaker, not a re-ranking.
    reranked: list[Candidate] = []
    for item in candidates:
        blended = float(item.score)

        # Add bounded keyword-overlap bonus if metadata is available.
        if term_set and item.vector_id is not None:
            record = records.get(item.vector_id)
            if record is not None:
                # Aggregate all metadata text to search: object_labels (EN entity)
                # + metadata_keywords (VN keyword tóm tắt video). title/description
                # đã bỏ (dư thừa — keywords đã summarize).
                parts: list[str] = list(record.object_labels or [])
                if record.metadata_keywords:
                    parts.extend(record.metadata_keywords)
                haystack = _fold_accents(" ".join(parts))

                # Count how many query terms appear in the metadata.
                matched = sum(1 for term in folded_terms if term in haystack)
                ratio = matched / len(folded_terms) if folded_terms else 0.0
                blended += weight * ratio

        # model_copy preserves all fields; only score is updated.
        reranked.append(item.model_copy(update={"score": blended}))

    return sorted(reranked, key=lambda c: c.score, reverse=True)
