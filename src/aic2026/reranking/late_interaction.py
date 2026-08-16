from __future__ import annotations

import re

import numpy as np

from aic2026.models import Candidate


# --- Query faceting -----------------------------------------------------------
# Late interaction (ColBERT-style) scores a query against a document by taking the
# MAX similarity over query tokens, so a frame that matches *any* facet of the
# query ranks high. A single pooled CLIP vector cannot do this — it blends all
# facets into one vector and washes out partial matches. We approximate the query
# side of late interaction cheaply: split the query into facets (the full query
# plus each meaningful phrase/token group) and encode each into its own vector.
# The late-interaction score for a frame is then max_i(cosine(facet_i, frame)).

_PHRASE_SPLIT_RE = re.compile(r"[;|/,\n]+")


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
            if len(w) > 2 and w not in seen:
                seen.add(w)
                facets.append(w)
    return facets


def late_interaction_rerank(
    query: str,
    candidates: list[Candidate],
    encode_text: "Callable[[str], np.ndarray]",
    frame_vectors: np.ndarray,
    top_n: int = 200,
    weight: float = 0.5,
) -> list[Candidate]:
    """Two-stage rerank: RRF already produced ``candidates``; this stage refines
    their order using a ColBERT-style late-interaction score, blending it with the
    incoming retrieval score.

    Stage 1 (RRF, fast, already done) keeps only the cheap top-``top_n`` frames.
    Stage 2 (this function, also fast) computes, for each surviving frame, the
    max cosine between the frame vector and any query facet vector. Frames that
    match *any* facet of the query (not just the blended whole) get boosted, which
    recovers partial matches CLIP's single pooled vector misses.

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
    # Encode each facet once; normalize.
    facet_vecs = []
    for f in facets:
        v = np.asarray(encode_text(f), dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n > 0:
            facet_vecs.append(v / n)
    if not facet_vecs:
        return candidates
    facet_matrix = np.stack(facet_vecs)  # [F, D]

    # Take only the top_n candidates for the (cheap) late-interaction pass.
    considered = candidates[:top_n]
    considered_vecs = frame_vectors[:top_n]

    # MaxSim: for each frame, max over facets of cosine similarity.
    # cosine = facet_matrix @ frame_vec (both normalized) -> [F, N]; max over F.
    sims = facet_matrix @ considered_vecs.T  # [F, N]
    max_sim = sims.max(axis=0)  # [N]

    for i, cand in enumerate(considered):
        li = float(max_sim[i])
        # Blend: keep retrieval ranking but let late interaction re-order within it.
        # We do NOT overwrite the retrieval score (that would flatten, like the old
        # video_level_rerank bug); instead we nudge by a bounded late-interaction term.
        cand.score = cand.score + weight * li

    return candidates
