from __future__ import annotations

import re

import numpy as np

from aic2026.models import Candidate

from .lexical import minmax_normalize, rrf_fuse as srrf_fuse


# --- Query faceting -----------------------------------------------------------
# Late interaction (ColBERT-style) scores a query against a document by taking the
# MAX similarity over query tokens, so a frame that matches *any* facet of the
# query ranks high. A single pooled CLIP vector cannot do this — it blends all
# facets into one vector and washes out partial matches. We approximate the query
# side of late interaction cheaply: split the query into facets (the full query
# plus each meaningful phrase/token group) and encode each into its own vector.
# The late-interaction score for a frame is then max_i(cosine(facet_i, frame)).

_PHRASE_SPLIT_RE = re.compile(r"[;|/,\n]+")

# Facets whose text query exactly matches one of these entity markers get extra
# late-interaction weight (object-centric queries benefit most from facet match).
_ENTITY_BOOST_TOKENS = (
    "person", "people", "man", "woman", "child", "dog", "cat", "car", "vehicle",
    "bottle", "cup", "phone", "book", "bag", "chair", "table", "sign", "logo",
)


def facet_queries(query: str) -> list[str]:
    """Split a query into late-interaction facets.

    Facets = the full query (keeps global context) plus each phrase split on
    common delimiters and each individual content word. Deduplicated, order kept.
    Stopwords are dropped from single-word facets so "a", "the", "of" do not
    become noise facets. This is the query side of ColBERT-style MaxSim.
    """
    facets: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        t = text.strip().lower()
        if t and t not in seen:
            seen.add(t)
            facets.append(t)

    _add(query)  # full query first
    for phrase in _PHRASE_SPLIT_RE.split(query):
        phrase = phrase.strip()
        if not phrase:
            continue
        _add(phrase)
        # also offer individual words as atomic facets
        for w in re.findall(r"\b[\wÀ-ỹ]+\b", phrase, flags=re.UNICODE):
            if len(w) > 2 and w.lower() not in ("the", "and", "for", "with") and w.lower() not in seen:
                seen.add(w.lower())
                facets.append(w.lower())
        if len(facets) >= 10:
            break
    return facets[:10]


def _facet_importance(facet: str) -> float:
    """Per-facet weight for the late-interaction MaxSim (Phase 9).

    The full query and entity-bearing single facets matter more than a lone
    generic word; this nudges the MaxSim aggregation toward the facets that
    carry retrieval signal. Returns a multiplier in (0, ~1.5].
    """
    f = facet.strip().lower()
    if f in _ENTITY_BOOST_TOKENS:
        return 1.4
    # full query / multi-word phrase: keep strong
    if " " in f or len(f) > 12:
        return 1.2
    return 1.0


def late_interaction_rerank(
    query: str,
    candidates: list[Candidate],
    encode_text: "Callable[[str], np.ndarray]",
    frame_vectors: np.ndarray,
    top_n: int = 200,
    weight: float = 0.4,
    k: int = 60,
) -> list[Candidate]:
    """Two-stage rerank: RRF already produced ``candidates``; this stage refines
    their order using a ColBERT-style late-interaction score, fused back with the
    incoming retrieval score.

    Stage 1 (RRF, fast, already done) keeps only the cheap top-``top_n`` frames.
    Stage 2 (this function, also fast) computes, for each surviving frame, the
    max cosine between the frame vector and any query facet vector. Frames that
    match *any* facet of the query (not just the blended whole) get boosted, which
    recovers partial matches CLIP's single pooled vector misses.

    Phase 9 change: the late-interaction signal is **SRRF-fused** with CLIP
    (paper Eq.1-2) instead of a naive additive nudge. The old code did
    ``cand.score += weight * li``, which mixed incompatible scales and flattened
    the CLIP ordering (the same bug class as the old video_level_rerank). Fusion
    blends by rank + min-max-normalized score, so CLIP still dominates globally
    while facet matches reorder *within* the list. Per-facet importance weights
    (entity-centric facets count more) sharpen the MaxSim.

    ``frame_vectors`` must be L2-normalized and ordered to match ``candidates``
    (i.e. ``frame_vectors[i]`` is the embedding of ``candidates[i]``).

    Speed: only ``len(facets)`` text encodes (a handful) plus matrix dot products
    over at most ``top_n`` frames — milliseconds, no LLM, no training.
    """
    if not candidates:
        return []
    if frame_vectors is None or len(frame_vectors) == 0:
        return candidates

    facets = facet_queries(query)
    # Encode each facet once; normalize. Keep per-facet importance for MaxSim.
    facet_vecs: list[np.ndarray] = []
    facet_weights: list[float] = []
    for f in facets:
        v = np.asarray(encode_text(f), dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n > 0:
            facet_vecs.append(v / n)
            facet_weights.append(_facet_importance(f))
    if not facet_vecs:
        return candidates
    facet_matrix = np.stack(facet_vecs)  # [F, D]
    facet_weight_vec = np.asarray(facet_weights, dtype=np.float32)  # [F]

    # Take only the top_n candidates for the (cheap) late-interaction pass.
    considered = candidates[:top_n]
    considered_vecs = frame_vectors[:top_n]

    # MaxSim: for each frame, max over (importance-weighted) facets of cosine
    # similarity. cosine = facet_matrix @ frame_vec (both normalized) -> [F, N];
    # weight each facet's similarity by its importance, then max over F.
    sims = facet_matrix @ considered_vecs.T  # [F, N]
    weighted_sims = sims * facet_weight_vec[:, None]
    max_sim = weighted_sims.max(axis=0)  # [N]

    # Build two ranked lists for SRRF fusion:
    #   * CLIP retrieval: incoming candidate order (by .score).
    #   * Late interaction: by MaxSim score.
    li_scores = [float(max_sim[i]) for i in range(len(considered))]
    clip_ids = np.asarray(
        [c.vector_id for c in considered if c.vector_id is not None], dtype=np.int64
    )
    # Only rank candidates that have a stable vector_id key for fusion.
    clip_order = [c.vector_id for c in considered if c.vector_id is not None]
    clip_scores = [float(c.score) for c in considered if c.vector_id is not None]
    li_order = [
        considered[i].vector_id
        for i in sorted(range(len(considered)), key=lambda j: li_scores[j], reverse=True)
        if considered[i].vector_id is not None
    ]
    li_scores_ranked = sorted(li_scores, reverse=True)

    if not clip_order or not li_order:
        # Fallback: apply the bounded late-interaction nudge on candidates that
        # have a vector_id (keeps the legacy behaviour for the degenerate case).
        out: list[Candidate] = []
        for i, cand in enumerate(candidates):
            if i < len(considered):
                new_score = float(cand.score) + weight * float(max_sim[i])
                out.append(cand.model_copy(update={"score": new_score}))
            else:
                out.append(cand)
        return out

    try:
        fused = srrf_fuse(
            [
                np.asarray(clip_order, dtype=np.int64),
                np.asarray(li_order, dtype=np.int64),
            ],
            score_lists=[clip_scores, li_scores_ranked],
            k=k,
        )
    except Exception:  # noqa: BLE001 — fall back to the bounded nudge
        out = []
        for i, cand in enumerate(candidates):
            if i < len(considered):
                new_score = float(cand.score) + weight * float(max_sim[i])
                out.append(cand.model_copy(update={"score": new_score}))
            else:
                out.append(cand)
        return sorted(out, key=lambda c: c.score, reverse=True)

    # Blend the fused late-interaction score with the original CLIP retrieval
    # score by ``weight`` (0 = CLIP only, 1 = full fusion). This keeps ``weight``
    # semantically meaningful and avoids both the old flatten bug and an
    # all-or-nothing swap. Candidates outside the considered window keep their
    # CLIP score untouched (no recall loss).
    clip_by_id = {c.vector_id: float(c.score) for c in considered if c.vector_id is not None}
    reranked_candidates: list[Candidate] = []
    for i, cand in enumerate(candidates):
        if i < len(considered) and cand.vector_id is not None and cand.vector_id in fused:
            new_score = (1.0 - weight) * clip_by_id[cand.vector_id] + weight * fused[cand.vector_id]
            reranked_candidates.append(cand.model_copy(update={"score": new_score}))
        else:
            reranked_candidates.append(cand)
    # Re-sort: scores were reassigned, so the candidate order no longer matches
    # the score order.
    return sorted(reranked_candidates, key=lambda c: c.score, reverse=True)


# Re-export minmax_normalize so callers importing from this module keep working.
__all__ = ["facet_queries", "late_interaction_rerank", "minmax_normalize"]
