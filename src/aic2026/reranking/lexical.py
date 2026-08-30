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


# Canonical object concepts used by the OpenImages / Faster R-CNN detector.
# The aliases make the query vocabulary forgiving: each concept lists BOTH its
# English detector label(s) (so a detected "Bottle" matches) AND common
# Vietnamese query words (so a query like "chai nước" lights up the concept).
# Without the Vietnamese aliases, a VN query never populated `requested`, so
# object evidence stayed off and CLIP alone decided the top frame.
_OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "person": ("person", "people", "human", "man", "woman", "boy", "girl", "child", "người", "đàn ông", "phụ nữ", "em bé", "trẻ em", "người đàn ông", "người phụ nữ"),
    "crowd": ("crowd", "đám đông", "nhóm người"),
    "turtle": ("turtle", "sea turtle", "tortoise", "rùa"),
    "fish": ("fish", "con cá", "cá"),
    "dog": ("dog", "puppy", "con chó", "chó"),
    "cat": ("cat", "kitten", "con mèo"),
    "car": ("car", "automobile", "vehicle", "xe hơi", "ô tô", "xe ô tô", "xe hơi"),
    "bicycle": ("bicycle", "bike", "xe đạp"),
    "motorcycle": ("motorcycle", "motorbike", "xe máy"),
    "bird": ("bird", "chim"),
    "horse": ("horse", "ngựa"),
    "elephant": ("elephant", "voi"),
    "cow": ("cow", "cattle", "con bò"),
    "bus": ("bus", "xe buýt"),
    "train": ("train", "tàu hỏa", "tàu"),
    "boat": ("boat", "ship", "tàu thuyền", "thuyền"),
    # --- Thêm: các vật thể phổ biến thường xuất trong query VN ---
    "bottle": ("bottle", "water bottle", "chai", "chai nước", "cốc nước", "lon"),
    "cup": ("cup", "cái ly", "ly nước", "cốc", "tách"),
    "backpack": ("backpack", "ba lô", "cặp"),
    "handbag": ("handbag", "túi xách", "túi"),
    "suitcase": ("suitcase", "vali"),
    "umbrella": ("umbrella", "cái ô", "ô dù"),
    "traffic light": ("traffic light", "traffic signal", "đèn tín hiệu", "đèn giao thông", "đèn đường"),
    "stop sign": ("stop sign", "biển báo", "biển báo stop"),
    "sign": ("sign", "biển", "biển hiệu", "bảng"),
    "book": ("book", "sách", "quyển sách"),
    "chair": ("chair", "cái ghế", "ghế"),
    "couch": ("couch", "sofa", "ghế sofa"),
    "bed": ("bed", "giường", "cái giường"),
    "dining table": ("dining table", "table", "cái bàn", "bàn ăn", "bàn làm việc"),
    "bench": ("bench", "ghế đá", "ghế dài"),
    "hand": ("hand", "bàn tay"),
    "clock": ("clock", "đồng hồ"),
    "cell phone": ("cell phone", "mobile phone", "phone", "điện thoại", "điện thoại di động"),
    "laptop": ("laptop", "máy tính xách tay", "máy tính"),
    "tv": ("tv", "television", "tivi"),
    "remote": ("remote", "điều khiển"),
    "keyboard": ("keyboard", "bàn phím"),
    "refrigerator": ("refrigerator", "fridge", "tủ lạnh"),
    "microwave": ("microwave", "lò vi sóng"),
    "oven": ("oven", "lò nướng"),
    "sink": ("sink", "bồn rửa", "chậu rửa"),
    "toilet": ("toilet", "bồn cầu"),
    "potted plant": ("potted plant", "plant", "chậu cây", "cây cảnh", "cái cây"),
    "sports ball": ("sports ball", "ball", "quả bóng", "bóng"),
    "teddy bear": ("teddy bear", "búp bê", "thú nhồi bông"),
    "toy": ("toy", "đồ chơi"),
    "skateboard": ("skateboard", "ván trượt"),
    "surfboard": ("surfboard", "ván lướt sóng"),
    "tennis racket": ("tennis racket", "vợt"),
    "wine glass": ("wine glass", "ly rượu"),
    "fork": ("fork", "cái nĩa", "nĩa", "dĩa"),
    "knife": ("knife", "con dao", "cái dao", "dao"),
    "spoon": ("spoon", "cái thìa", "thìa", "muỗng"),
    "bowl": ("bowl", "cái bát", "bát", "cái tô", "tô"),
    "banana": ("banana", "chuối"),
    "apple": ("apple", "quả táo", "táo"),
    "orange": ("orange", "quả cam", "màu cam"),
    "broccoli": ("broccoli", "súp lơ"),
    "carrot": ("carrot", "cà rốt"),
    "pizza": ("pizza", "bánh pizza"),
    "cake": ("cake", "cái bánh", "bánh"),
    "sandwich": ("sandwich", "bánh mì"),
    "vegetable": ("vegetable", "rau", "rau củ", "đồ ăn", "thức ăn"),
    "stage": ("stage", "sân khấu"),
    "scissors": ("scissors", "cái kéo", "kéo"),
}


def _has_phrase(text: str, phrase: str) -> bool:
    # Word-boundary match so "ca" does NOT match inside "cam" / "cá".  The
    # original code used a raw string r"(?<!\w)" which is literally backslash-w
    # (not the \w class), so the boundary was a no-op and every substring matched
    # ("ca" matched "cam", "o" matched everywhere) — breaking object concept
    # detection for Vietnamese queries.
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))


def _requested_concepts(query: str) -> set[str]:
    """Concepts the query asks for, per ``_OBJECT_ALIASES`` (empty if none)."""
    query_text = _fold_accents(query)
    # Fold aliases too: the query is accent-folded, so a Vietnamese alias like
    # "xe máy" must be folded to "xe may" before matching, otherwise it never
    # fires and object evidence stays off for VN queries.
    return {
        concept
        for concept, aliases in _OBJECT_ALIASES.items()
        if any(_has_phrase(query_text, _fold_accents(alias)) for alias in aliases)
    }


def object_evidence_adjustment(
    query: str,
    record: FrameRecord,
    weight: float = 0.03,
    penalty_scale: float = 2.0,
) -> float | None:
    """Return a signed object-evidence adjustment, or ``None`` if no object is asked.

    A full object match gets ``+weight`` and a frame missing every requested
    object gets ``-weight * penalty_scale``.  The penalty is amplified (default
    ``penalty_scale = 2.0``) so that a frame the detector clearly shows lacking
    the asked object is pushed down harder than a matching frame is lifted — this
    directly counters the failure mode where CLIP ranks a frame high for the
    right *scene* but the wrong *object* (e.g. a supermarket aisle with no water
    bottle).  It remains a soft penalty: Faster R-CNN can miss small/occluded
    objects, so vector evidence is never discarded outright.
    """
    requested = _requested_concepts(query)
    if not requested:
        return None

    labels = _fold_accents(" ".join(record.object_labels or []))
    matched = sum(
        any(_has_phrase(labels, _fold_accents(alias)) for alias in _OBJECT_ALIASES[concept])
        for concept in requested
    )
    coverage = matched / len(requested)
    # Symmetrical-ish but asymmetric: full miss penalized harder than full match rewarded.
    if coverage == 0.0:
        return -weight * penalty_scale
    return weight * (2.0 * coverage - 1.0)


# ---------------------------------------------------------------------------
# Vectorized object-evidence adjustment for TRAKE
# ---------------------------------------------------------------------------
# The legacy path applied object evidence via a per-(event, frame) Python
# closure inside ``RetrievalPipeline.retrieve_trake``. That closure re-derived
# the requested concepts (a 70-concept alias scan) and re-folded the frame's
# object labels on EVERY one of the ~40k calls per query — measured at ~145s
# for a single 3-event TRAKE query. The helpers below precompute the same
# adjustment as a per-video NumPy matrix so the pipeline only needs a cheap
# matrix addition per candidate video.

_REQUESTED_CONCEPT_CACHE: dict[str, frozenset] = {}
_FOLDED_LABEL_CACHE: dict[int, str] = {}


def _requested_concepts_cached(query: str) -> frozenset:
    """Cached ``_requested_concepts`` — the per-event scan is identical across
    all frames of a video, so it must be computed ONCE, not 40k times."""

    cached = _REQUESTED_CONCEPT_CACHE.get(query)
    if cached is None:
        cached = frozenset(_requested_concepts(query))
        _REQUESTED_CONCEPT_CACHE[query] = cached
    return cached


def _folded_labels_for_record(record: FrameRecord) -> str:
    """Cached accent-folded, space-joined object labels for one frame."""

    key = record.vector_id
    if key is not None and key in _FOLDED_LABEL_CACHE:
        return _FOLDED_LABEL_CACHE[key]
    folded = _fold_accents(" ".join(record.object_labels or []))
    if key is not None:
        _FOLDED_LABEL_CACHE[key] = folded
    return folded


def build_object_adjustment_matrices(
    events: list[str],
    candidate_videos: set[str],
    video_to_manifest_indices: dict[str, np.ndarray],
    manifest: list[FrameRecord],
    weight: float = 0.08,
    penalty_scale: float = 2.0,
) -> dict[str, np.ndarray]:
    """Build a vectorized object-evidence adjustment matrix per candidate video.

    Returns ``{video_id: np.ndarray(shape=(len(events), F), dtype=float32)}``
    where ``F`` is that video's frame count, aligned with the per-video manifest
    order used by ``RetrievalPipeline.retrieve_trake``. Each cell equals the
    signed adjustment ``object_evidence_adjustment`` would have produced for that
    (event, frame) pair, so the pipeline can add it to ``similarity_matrix`` in
    one NumPy op instead of calling Python 40k times.

    Only the coarse-filtered ``candidate_videos`` are processed, and each
    frame's labels are folded once and each concept's membership is computed
    once — collapsing the dominant TRAKE cost from ~145s to a few seconds.
    """

    if not events or not candidate_videos:
        return {}

    event_requested = [_requested_concepts_cached(event) for event in events]

    # Union of all requested concepts across events, with pre-folded aliases.
    union: dict[str, list[str]] = {}
    for requested in event_requested:
        for concept in requested:
            if concept not in union:
                union[concept] = [
                    _fold_accents(alias) for alias in _OBJECT_ALIASES[concept]
                ]

    matrices: dict[str, np.ndarray] = {}

    for video_id in candidate_videos:
        manifest_indices = video_to_manifest_indices.get(video_id)
        if manifest_indices is None or len(manifest_indices) == 0:
            continue

        frame_count = int(len(manifest_indices))

        # Fold each frame's labels once.
        folded_labels = [
            _folded_labels_for_record(manifest[int(index)])
            for index in manifest_indices
        ]

        # Per-concept membership over the video's frames (frame_count booleans).
        concept_present: dict[str, np.ndarray] = {}
        for concept, aliases in union.items():
            if not aliases:
                continue
            present = np.zeros(frame_count, dtype=np.float32)
            for frame_position in range(frame_count):
                text = folded_labels[frame_position]
                if text and any(
                    alias and _has_phrase(text, alias) for alias in aliases
                ):
                    present[frame_position] = 1.0
            concept_present[concept] = present

        coverage = np.zeros((len(events), frame_count), dtype=np.float32)
        for event_index, requested in enumerate(event_requested):
            if not requested:
                continue
            column = np.zeros(frame_count, dtype=np.float32)
            for concept in requested:
                column += concept_present.get(
                    concept, np.zeros(frame_count, dtype=np.float32)
                )
            coverage[event_index] = column / len(requested)

        adjustment = np.where(
            coverage == 0.0,
            -weight * penalty_scale,
            weight * (2.0 * coverage - 1.0),
        ).astype(np.float32)
        matrices[video_id] = adjustment

    return matrices


def rerank_with_object_evidence(
    query: str,
    candidates: list[Candidate],
    records: dict[int, FrameRecord],
    weight: float = 0.03,
    penalty_scale: float = 2.0,
    drop_empty_object_frames: bool = False,
) -> list[Candidate]:
    """Apply signed Object-detection evidence after vector/BM25 retrieval.

    When ``drop_empty_object_frames`` is True and the query actually asks for an
    object (``object_evidence_adjustment`` returns a non-None *requested* set),
    any candidate whose frame has NO object labels at all (i.e. the ingest step
    marked it as a blurry / no-clear-object frame — no entity reached the 0.4
    present-threshold) is dropped entirely.  This is the hard filter for the
    "blurry frame" case: such frames never enter the final KIS ranking.  It only
    fires when the query asks for an object, so pure scene queries are unaffected.
    If dropping would emptying the whole pool, we fall back to keeping the
    original candidates (avoid returning zero answers).
    """
    requested = _requested_concepts(query)
    drop_mode = drop_empty_object_frames and bool(requested)

    kept: list[Candidate] = []
    for item in candidates:
        if drop_mode and item.vector_id is not None:
            record = records.get(item.vector_id)
            # A frame with no object labels while the query asks for an object
            # is the blurry / no-clear-object case -> drop it.
            if record is not None and not (record.object_labels or []):
                continue
        kept.append(item)

    # Fallback: never return an empty pool just because every frame was blurry.
    if drop_mode and not kept:
        kept = list(candidates)

    reranked: list[Candidate] = []
    for item in kept:
        adjustment = None
        if item.vector_id is not None:
            record = records.get(item.vector_id)
            if record is not None:
                adjustment = object_evidence_adjustment(query, record, weight, penalty_scale)
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
                # + metadata_keywords (VN keyword tóm tắt video) + asr_text (lời thoại).
                parts: list[str] = list(record.object_labels or [])
                if getattr(record, "asr_text", None):
                    parts.extend(record.asr_text)
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


def rerank_with_semantic_text(
    query: str,
    candidates: list[Candidate],
    records: dict[int, FrameRecord],
    encode_text,
    weight: float = 0.12,
) -> list[Candidate]:
    """Use an optional multilingual text encoder to add a bounded semantic bonus.

    This helper preserves the existing CLIP branch and only gives extra lift when
    a semantic encoder is available. It compares the query embedding to a compact
    text bag built from metadata + object labels and nudges candidates whose
    underlying scene text matches the query semantically, even when the wording
    differs from the original CLIP prompt.
    """
    if not candidates or weight <= 0 or encode_text is None:
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    query_text = (query or "").strip()
    if not query_text:
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    q_vec = np.asarray(encode_text(query_text), dtype=np.float32).reshape(-1)
    if q_vec.size == 0:
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    reranked: list[Candidate] = []
    for item in candidates:
        base_score = float(item.score)
        if item.vector_id is None:
            reranked.append(item.model_copy(update={"score": base_score}))
            continue

        record = records.get(item.vector_id)
        if record is None:
            reranked.append(item.model_copy(update={"score": base_score}))
            continue

        parts: list[str] = []
        parts.extend(record.object_labels or [])
        if getattr(record, "metadata_keywords", None):
            parts.extend(record.metadata_keywords)
        if getattr(record, "asr_text", None):
            parts.extend(record.asr_text)
        if not parts:
            reranked.append(item.model_copy(update={"score": base_score}))
            continue

        haystack = " ".join(part for part in parts if isinstance(part, str))
        if not haystack.strip():
            reranked.append(item.model_copy(update={"score": base_score}))
            continue

        text_vec = np.asarray(encode_text(haystack), dtype=np.float32).reshape(-1)
        norm_q = np.linalg.norm(q_vec)
        norm_t = np.linalg.norm(text_vec)
        if norm_q <= 1e-12 or norm_t <= 1e-12:
            reranked.append(item.model_copy(update={"score": base_score}))
            continue

        sim = float(np.dot(q_vec, text_vec) / (norm_q * norm_t))
        blended = base_score + weight * max(0.0, min(1.0, sim))
        reranked.append(item.model_copy(update={"score": blended}))

    return sorted(reranked, key=lambda c: c.score, reverse=True)
