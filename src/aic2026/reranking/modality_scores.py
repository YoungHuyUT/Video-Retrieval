"""Per-modality score extraction for Adaptive Fusion (paper Eq.1–2).

This module computes *independent* raw scores for each retrieval modality
(``semantic`` = CLIP cosine, ``object`` = detected-object coverage,
``asr`` = spoken-term overlap) over the **Top-N candidate pool** produced by
the RRF recall stage.  It deliberately does NOT build a new BM25 index (the
existing lexical index stays merged object+OCR+ASR for RRF recall); instead it
scores the already-retrieved frames, which is cheap and keeps the architecture
additive (no rewrite of the retrieval backends, per the refactor constraints).

Each helper returns ``dict[int, float]`` keyed by **manifest/vector id** so the
caller can feed them straight into :func:`adaptive_modality_fusion`.  Scores are
left on their native scale — the fusion stage min–max normalizes them.
"""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aic2026.ingestion.asr import VideoTranscript
    from aic2026.models import Candidate, FrameRecord


def _fold(value: str) -> str:
    """Accent/case-fold text for Vietnamese-insensitive matching.

    Mirrors :func:`aic2026.reranking.lexical._fold_accents` so object/ASR
    matching behaves identically to the metadata reranker.
    """
    value = unicodedata.normalize("NFC", value).casefold()
    value = value.translate(str.maketrans({"đ": "d"}))
    return "".join(
        c for c in unicodedata.normalize("NFD", value)
        if unicodedata.category(c) != "Mn"
    )


def _query_terms(query: str) -> list[str]:
    return [t for t in _fold(query).split() if t]


# ---------------------------------------------------------------------------
# Object-evidence coverage (frame-level)
# ---------------------------------------------------------------------------

def object_coverage_scores(
    query: str,
    candidates: list["Candidate"],
    records: dict[int, "FrameRecord"],
) -> dict[int, float]:
    """Per-frame object-coverage score in [0, 1].

    For each candidate frame we look up its detected ``object_labels`` and
    measure how many of the query's *requested* object concepts (resolved via
    the detector alias table in ``reranking.lexical``) are actually present.
    A pure scene query (no requested object) yields an empty score map, so that
    modality is simply dropped from the fusion (its weight is renormalized away).

    Returns ``{vector_id: coverage}`` for frames that mention at least one label.
    """
    from aic2026.reranking.lexical import _requested_concepts

    requested = _requested_concepts(query)
    if not requested:
        return {}

    # Pre-fold the request so each alias comparison is a plain substring test.
    requested_aliases: dict[str, list[str]] = {
        concept: [_fold(a) for a in aliases]
        for concept, aliases in _OBJECT_ALIASES_ITEMS()
        if concept in requested
    }
    if not requested_aliases:
        return {}

    from aic2026.reranking.lexical import _has_phrase

    scores: dict[int, float] = {}
    for cand in candidates:
        if cand.vector_id is None:
            continue
        record = records.get(cand.vector_id)
        if record is None:
            continue
        labels = _fold(" ".join(record.object_labels or []))
        if not labels:
            continue
        matched = sum(
            1
            for aliases in requested_aliases.values()
            if any(alias and _has_phrase(labels, alias) for alias in aliases)
        )
        coverage = matched / len(requested_aliases)
        if coverage > 0.0:
            scores[cand.vector_id] = coverage
    return scores


# Lazily expose the alias table from the lexical module without re-importing the
# giant dict at module load (it is only needed when an object is actually asked).
def _OBJECT_ALIASES_ITEMS():
    from aic2026.reranking.lexical import _OBJECT_ALIASES

    return list(_OBJECT_ALIASES.items())


# ---------------------------------------------------------------------------
# ASR spoken-term overlap (segment-level, localized to the spoken frame)
# ---------------------------------------------------------------------------

# Frames within this distance of the ASR segment that *said* a query term still
# earn partial credit (decaying).  ~5 s @ 30 fps — wide enough to catch the
# nearest keyframe when no keyframe sits exactly on the utterance.
_ASR_FRAME_TOL = 150

# Query dài (chuỗi hành động) có quá nhiều từ để nằm gọn trong 1 cửa sổ 5s,
# nên localize frame sẽ luôn ra ratio rất thấp (vd 3/18 từ). Với query có số từ
# >= ngưỡng này, chuyển sang chấm điểm MỨC VIDEO (coverage): bao nhiêu phần của
# chuỗi được nói trong TOÀN BỘ video → broadcast cho mọi frame của video đó.
_ASR_LONG_QUERY_WORDS = 8


def asr_term_scores(
    query: str,
    candidates: list["Candidate"],
    transcripts_by_video: "dict[str, VideoTranscript] | None",
) -> dict[int, float]:
    """Per-frame ASR spoken-term score in [0, 1], localized to the spoken frame.

    Unlike the old video-level scoring (which folded the whole transcript into one
    string and broadcast the same score onto every frame of the video), this uses
    the per-segment ``frame``/``start``/``end`` fields the sidecar already carries:
    a query term matches *only* the candidate frames near the segment that
    actually said it.  So *"a woman says remember"* now boosts the frame where the
    word was spoken, not an arbitrary frame from the same video.

    Score per candidate = (fraction of query terms said within ``_ASR_FRAME_TOL``
    of the frame) × proximity, where proximity decays from 1.0 at the utterance
    frame to a 0.3 floor at the edge of the window.  Videos whose sidecar has no
    segment-level ``frame`` data fall back to the original whole-video broadcast so
    recall is never lost.  Returns ``{}`` when no ASR sidecar was loaded (the
    ``asr`` weight is then renormalized away by fusion).

    Only the Top-N candidate pool is scored — no new BM25, no per-query decode.
    """
    if not transcripts_by_video:
        return {}

    terms = _query_terms(query)
    if not terms:
        return {}

    # Query dài (chuỗi hành động) → chấm điểm MỨC VIDEO rồi broadcast cho mọi
    # frame của video đó (xem giải thích _ASR_LONG_QUERY_WORDS). Localize frame
    # chỉ hợp lý với query ngắn (1 hành động / 1 câu) nằm gọn trong 1 cửa sổ 5s.
    is_long_query = len(terms) >= _ASR_LONG_QUERY_WORDS

    # Pre-fold each candidate video's segments once.  ``frames`` = list of
    # (seg_frame, folded_text); ``has_frame`` = any segment carries a frame coord
    # (older sidecars may not — those degrade to whole-video broadcast).
    seg_cache: dict[str, tuple[list[tuple[int, str]], bool]] = {}
    for cand in candidates:
        vid = cand.video_id
        if vid in seg_cache or vid not in transcripts_by_video:
            continue
        transcript = transcripts_by_video[vid]
        segs: list[tuple[int, str]] = []
        for seg in transcript.segments:
            if seg.text:
                segs.append((seg.frame if seg.frame is not None else -1, _fold(seg.text)))
        seg_cache[vid] = (segs, any(f >= 0 for f, _ in segs))

    # Với query dài: tính coverage 1 lần / video (không phụ thuộc frame).
    # coverage[video] = (số từ query xuất hiện trong TOÀN transcript) / tổng từ.
    long_coverage: dict[str, float] = {}
    if is_long_query:
        for vid, (segs, _) in seg_cache.items():
            haystack = " ".join(t for _, t in segs)
            if not haystack:
                continue
            matched = sum(1 for term in terms if term in haystack)
            cov = matched / len(terms)
            if cov > 0.0:
                long_coverage[vid] = cov

    scores: dict[int, float] = {}
    for cand in candidates:
        if cand.vector_id is None:
            continue
        cached = seg_cache.get(cand.video_id)
        if not cached:
            continue
        segs, has_frame = cached

        if is_long_query:
            # Broadcast video-level coverage — action chain trải rộng khắp video,
            # không thể gắn vào 1 frame. Mọi frame của video có transcript khớp
            # đều được điểm coverage này.
            cov = long_coverage.get(cand.video_id)
            if cov is None:
                continue
            scores[cand.vector_id] = cov
            continue

        if not has_frame:
            # Fallback: no per-segment frame data → old whole-video behaviour.
            haystack = " ".join(t for _, t in segs)
            matched = sum(1 for term in terms if term in haystack)
            ratio = matched / len(terms)
            if ratio > 0.0:
                scores[cand.vector_id] = ratio
            continue

        # Localized scoring: only segments within tolerance of this frame count.
        near = [
            (f, t) for f, t in segs
            if f >= 0 and abs(cand.frame_id - f) <= _ASR_FRAME_TOL
        ]
        matched = sum(
            1 for term in terms if any(term in t for _, t in near)
        )
        if matched == 0:
            continue
        ratio = matched / len(terms)
        # Proximity to the nearest matched segment (1.0 on the utterance frame,
        # decaying to the 0.3 floor at the window edge).
        min_dist = min(
            abs(cand.frame_id - f)
            for f, t in near
            if any(term in t for term in terms)
        )
        prox = max(0.3, 1.0 - min_dist / _ASR_FRAME_TOL)
        scores[cand.vector_id] = ratio * prox
    return scores


# ---------------------------------------------------------------------------
# ASR temporal localization (spec §ASR TEMPORAL LOCALIZATION appendix)
# ---------------------------------------------------------------------------
#
# The sidecar already carries per-segment ``start`` / ``end`` / ``frame``.  This
# lets ASR act as *temporal evidence*, not just a lexical bag: for a candidate
# frame at timestamp ``t_f`` we find the nearest spoken segment whose text
# matches the query and apply a temporal-decay gate:
#
#     S_asr(f) = max_s [ sim(query, segment_s) · exp(-alpha · dist(t_f, center_s)) ]
#
# where ``center_s = (start_s + end_s)/2`` and ``dist`` is the gap in seconds.
# A temporal window is widened by ``margin`` seconds around each segment so a
# keyframe a little before/after the exact utterance still earns credit.  This
# replaces the old *video-level broadcast* (same score on every frame) which the
# spec explicitly forbids for timestamp-localizable evidence (§16, appendix).
#
# Low ASR confidence (few/weak matches) does NOT drop candidates — ASR stays a
# weak fusion signal; only when a segment is genuinely localizable does it
# narrow the temporal search region.  Falls back to the existing
# :func:`asr_term_scores` (frame-local / video-level) when the sidecar has no
# usable timestamps, so recall is never lost.

# Temporal-decay alpha for ASR localization (spec §13: alpha configurable; 0.01
# default; tighter when the query is an action chain).
_ASR_TEMPORAL_ALPHA = 0.01
# Widen the spoken-segment window by this many seconds on each side.
_ASR_TEMPORAL_MARGIN = 2.0
# Floor for the temporal-decay weight so a slightly-off frame is not zeroed.
_ASR_TEMPORAL_FLOOR = 0.1


def asr_temporal_scores(
    query: str,
    candidates: list["Candidate"],
    transcripts_by_video: "dict[str, VideoTranscript] | None",
    *,
    alpha: float = _ASR_TEMPORAL_ALPHA,
    margin_seconds: float = _ASR_TEMPORAL_MARGIN,
    timestamp_lookup: "Callable[[str, int], float | None] | None" = None,
) -> dict[int, float]:
    """Temporal-localized ASR score per candidate frame (spec §ASR TEMPORAL).

    For each candidate frame at timestamp ``t_f`` (resolved via
    ``timestamp_lookup(video_id, frame_id)`` — falls back to ``frame_id`` scaled
    by ``margin_seconds`` when unavailable), score the best matching spoken
    segment weighted by a temporal-decay gate.  Returns ``{}`` (ASR weight
    renormalized away) when no sidecar is loaded or no segment is localizable.

    Only the small Top-N pool is scored — no new BM25, no decode.
    """
    if not transcripts_by_video:
        return {}
    from aic2026.reranking.lexical import _fold_accents

    terms = _query_terms(query)
    if not terms:
        return {}

    # Resolve each candidate's timestamp (seconds).  Prefer the candidate's own
    # ``timestamp`` field (already populated from the manifest), then the explicit
    # lookup, falling back to ``frame_id`` when neither is present.
    def _ts(cand: "Candidate") -> float:
        ts = getattr(cand, "timestamp", None)
        if ts is not None:
            return float(ts)
        if timestamp_lookup is not None:
            try:
                t = timestamp_lookup(cand.video_id, cand.frame_id)
                if t is not None:
                    return float(t)
            except Exception:  # noqa: BLE001
                pass
        return float(cand.frame_id)

    # Pre-index segments with usable (start,end,center,folded_text) per video.
    seg_cache: dict[str, list[tuple[float, float, str]]] = {}
    for cand in candidates:
        vid = cand.video_id
        if vid in seg_cache or vid not in transcripts_by_video:
            continue
        transcript = transcripts_by_video[vid]
        segs: list[tuple[float, float, str]] = []
        for seg in transcript.segments:
            if seg.text and seg.start is not None and seg.end is not None:
                center = (float(seg.start) + float(seg.end)) / 2.0
                segs.append((center, float(seg.start), _fold(seg.text)))
        seg_cache[vid] = segs

    scores: dict[int, float] = {}

    def _match_ratio(text: str) -> float:
        if not text:
            return 0.0
        matched = sum(1 for term in terms if term in text)
        return matched / len(terms)

    for cand in candidates:
        if cand.vector_id is None:
            continue
        segs = seg_cache.get(cand.video_id)
        if not segs:
            continue
        t_f = _ts(cand)
        best = 0.0
        for center, start, text in segs:
            ratio = _match_ratio(text)
            if ratio <= 0.0:
                continue
            # Temporal window around the spoken segment [start-margin, end+margin].
            lo = start - margin_seconds
            hi = (start + (center - start) * 2) + margin_seconds  # end + margin
            if t_f < lo or t_f > hi:
                # Outside the window: only a weak decayed whisper survives.
                decay = max(
                    _ASR_TEMPORAL_FLOOR,
                    np.exp(-alpha * abs(t_f - center)),
                )
                score = ratio * decay * 0.3
            else:
                # Inside the localized window: full ratio × temporal proximity.
                decay = np.exp(-alpha * abs(t_f - center))
                score = ratio * max(_ASR_TEMPORAL_FLOOR, decay)
            if score > best:
                best = score
        if best > 0.0:
            scores[cand.vector_id] = float(best)
    return scores

