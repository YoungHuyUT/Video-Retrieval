"""Event Coverage scoring for multi-event / action-chain queries (spec §3, §4, §9).

HIGH RECALL (RRF) → PRECISION (Adaptive Fusion) → EVENT COVERAGE (§4)
→ TEMPORAL CONSISTENCY → BEST EVIDENCE FRAME.

Motivation
----------
A single query frame must not only match ONE clause strongly; it should
*explain as many of the query's events as possible*.  Example from the spec:

    Frame A:  E1=.95  E2=.10  E3=.05     → coverage 0.37
    Frame B:  E1=.88  E2=.82  E3=.79     → coverage 0.83   (preferred)

So instead of ranking frames by their single best event match, we average the
per-event similarities with query-aware weights:

    S_event(f) = Σ_i  w_i · S(E_i, f)  /  Σ_i w_i

over the **Top-N candidate pool** (cheap; no full-corpus work).  ``S(E_i, f)``
is the cosine between event ``i``'s CLIP embedding and frame ``f``'s CLIP
embedding.  This is a *precision* signal layered on top of the high-recall RRF
pool — it never loses recall because frames outside the pool are untouched.

Implementation note — reuse, do not duplicate
------------------------------------------------
The frame CLIP embeddings already live in ``pipeline.index.vectors`` (the same
177k–184k matrix used by FAISS).  We reuse that matrix directly and do ONE
matrix multiply (events × frames) for the whole pool, exactly like
``pipeline.retrieve_trake`` does for its similarity matrix.  No new index, no
re-encoding.  This keeps the stage in the milliseconds on the Top-N pool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aic2026.models import Candidate

logger = logging.getLogger(__name__)


@dataclass
class EventCoverageResult:
    """What :func:`event_coverage_scores` computed, for debuggability."""

    # {vector_id: S_event coverage in [0,1]} (min-max normalized over the pool).
    coverage: dict[int, float]
    # {vector_id: raw un-normalized Σ w_i·S(E_i,f) / Σ w_i} for inspection.
    raw: dict[int, float]
    # Per-event similarity matrix [event, frame-pool] (rows share pool order).
    per_event: dict[int, list[float]]
    n_events: int


def event_coverage_scores(
    event_embeddings: np.ndarray,  # [n_events, dim], L2-normalized
    candidate_vectors: np.ndarray,  # [n_frames, dim], L2-normalized
    candidate_vector_ids: list[int],  # [n_frames] manifest/vector ids
    event_weights: "list[float] | None" = None,  # [n_events] optional (>=0)
) -> EventCoverageResult:
    """Compute ``S_event(f)`` for each frame in the candidate pool.

    Parameters
    ----------
    event_embeddings:
        CLIP embeddings of the decomposed events (rows).  MUST be L2-normalized.
    candidate_vectors:
        CLIP embeddings of the Top-N candidate frames (rows).  MUST be
        L2-normalized so the dot product is cosine.
    candidate_vector_ids:
        Manifest/vector id for each row of ``candidate_vectors`` (same order).
    event_weights:
        Optional per-event weights ``w_i`` (≥0).  Defaults to all-equal ``1.0``.
        A weight of 0 removes that event from the average.  If every weight is 0
        the result is empty (no coverage signal).

    Returns
    -------
    EventCoverageResult with ``coverage`` min-max normalized onto [0,1] across the
    pool (Eq.1 of the paper) so it can feed Adaptive Fusion or be blended directly.
    """
    if event_embeddings is None or candidate_vectors is None or not candidate_vector_ids:
        return EventCoverageResult({}, {}, {}, 0)
    events = np.asarray(event_embeddings, dtype=np.float32)
    frames = np.asarray(candidate_vectors, dtype=np.float32)
    if events.ndim != 2 or frames.ndim != 2 or events.shape[1] != frames.shape[1]:
        return EventCoverageResult({}, {}, {}, 0)
    n_events = events.shape[0]
    if n_events == 0:
        return EventCoverageResult({}, {}, {}, 0)

    if event_weights is None:
        weights = np.ones(n_events, dtype=np.float32)
    else:
        weights = np.asarray(event_weights, dtype=np.float32)
        if weights.shape[0] != n_events:
            # Mismatched weights → fall back to equal weights rather than crash.
            weights = np.ones(n_events, dtype=np.float32)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return EventCoverageResult({}, {}, {}, n_events)

    # [n_events, n_frames] cosine similarity (no copy of the big frame matrix).
    sim = events @ frames.T  # type: ignore[operator]

    # S_event(f) = Σ_i w_i·sim[i,f] / Σ_i w_i   —— per column (frame).
    col_scores = (weights[:, None] * sim).sum(axis=0) / weight_sum  # [n_frames]

    # Per-event similarity per frame for debug/inspection (only the pool rows).
    per_event: dict[int, list[float]] = {
        i: sim[i].astype(float).tolist() for i in range(n_events)
    }

    raw = {
        vid: float(col_scores[idx])
        for idx, vid in enumerate(candidate_vector_ids)
    }

    # Min-max normalize onto [0,1] (paper Eq.1) so the coverage sits on the same
    # scale as the other Adaptive-Fusion modalities.
    lo = float(col_scores.min())
    hi = float(col_scores.max())
    span = hi - lo
    if span <= 1e-9:
        # All frames scored identically → give them a neutral middle score so the
        # modality neither helps nor hurts.
        coverage = {vid: 0.5 for vid in candidate_vector_ids}
    else:
        norm = (col_scores - lo) / span
        coverage = {
            vid: float(norm[idx])
            for idx, vid in enumerate(candidate_vector_ids)
        }

    return EventCoverageResult(
        coverage=coverage,
        raw=raw,
        per_event=per_event,
        n_events=n_events,
    )


def event_aware_rerank(
    candidates: list["Candidate"],
    event_embeddings: np.ndarray,
    index_vectors: np.ndarray,
    event_weights: "list[float] | None" = None,
    blend: float = 0.5,
) -> list["Candidate"]:
    """Re-rank the RRF/Adaptive-Fusion candidate pool by Event Coverage.

    This is the **precision** layer for query-dài / action-chain queries (spec
    §4).  It keeps the incoming order but *blends* in the event-coverage score so
    a frame explaining many events rises above one matching only a single clause.

    Strategy — additive blend (no recall loss):
        new_score = (1 - blend) · incoming_score_norm + blend · S_event(f)
    where ``incoming_score_norm`` is the min-max normalization of the candidates'
    current ``score`` (so the two scales are comparable) and ``S_event(f)`` is the
    min-max-normalized event coverage.  Frames outside the event-coverage pool
    keep their incoming score.  Any failure degrades gracefully to ``candidates``.

    ``blend`` controls how much Event Coverage dominates: 0.0 = ignore coverage
    (pure incoming ranking), 1.0 = pure coverage.  Default 0.5 balances the two.
    """
    if not candidates or event_embeddings is None or index_vectors is None:
        return candidates
    try:
        from aic2026.reranking.lexical import minmax_normalize

        pool = [c for c in candidates if c.vector_id is not None]
        if not pool:
            return candidates

        vids = np.asarray([c.vector_id for c in pool], dtype=np.int64)
        frame_vecs = np.stack([index_vectors[v] for v in vids]).astype(np.float32)
        # L2-normalize defensively (the official matrix is already normalized).
        norms = np.maximum(np.linalg.norm(frame_vecs, axis=1, keepdims=True), 1e-12)
        frame_vecs = frame_vecs / norms

        result = event_coverage_scores(
            event_embeddings=event_embeddings,
            candidate_vectors=frame_vecs,
            candidate_vector_ids=[int(v) for v in vids],
            event_weights=event_weights,
        )
        if not result.coverage:
            return candidates

        # Min-max normalize incoming scores over the pool for a comparable scale.
        incoming = np.asarray(
            [float(c.score) for c in pool], dtype=np.float32
        )
        incoming_norm = minmax_normalize(incoming)

        blended = {}
        for idx, c in enumerate(pool):
            cov = result.coverage.get(c.vector_id)
            if cov is None:
                blended[c.vector_id] = float(c.score)
                continue
            b = (1.0 - blend) * float(incoming_norm[idx]) + blend * cov
            blended[c.vector_id] = b

        re_scored = []
        for c in candidates:
            if c.vector_id is not None and c.vector_id in blended:
                re_scored.append(c.model_copy(update={"score": blended[c.vector_id]}))
            else:
                re_scored.append(c)
        re_scored.sort(key=lambda c: c.score, reverse=True)
        logger.info(
            "event-coverage rerank: blended %d/%d candidates (n_events=%d, blend=%.2f)",
            len(result.coverage), len(candidates), result.n_events, blend,
        )
        return re_scored
    except Exception as exc:  # noqa: BLE001
        logger.warning("event-coverage rerank skipped: %s", exc)
        return candidates
