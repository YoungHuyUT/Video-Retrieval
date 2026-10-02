from __future__ import annotations

import numpy as np

from aic2026.models import Candidate


def align_events(
    candidates: list[Candidate],
    event_count: int,
) -> list[int]:

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            candidate.frame_id,
        ),
    )

    frames = sorted(
        candidate.frame_id
        for candidate in ordered[:event_count]
    )

    if len(frames) != event_count:
        raise ValueError(
            "Not enough candidate frames to align all events"
        )

    return frames


# --- Adaptive temporal decay (spec IX + §9 + paper Eq.3) ------------------
#
# Rather than a single fixed gap penalty the paper models an *exponential
# temporal decay* between consecutive events:
#
#     lambda_i = e^{-alpha * delta_t_i}            (Eq.3)
#
# where delta_t_i is the (normalised) time gap between event i-1 and event i.
# A large alpha punishes long gaps hard (tightly-ordered events); a small
# alpha is permissive. We derive alpha from the query's *wording* so that:
#
#   "immediately"  -> strong ordering  (alpha = DECAY_ALPHA_IMMEDIATE)
#   "then"         -> moderate         (alpha = DECAY_ALPHA_THEN)
#   "later"        -> loose            (alpha = DECAY_ALPHA_LATER)
#
# The baseline (penalty_weight-only, no decay) is recovered when
# ``use_decay=False`` — byte-for-byte identical to the legacy DP.

DECAY_ALPHA_IMMEDIATE = 0.05
DECAY_ALPHA_THEN = 0.01
DECAY_ALPHA_LATER = 0.002

# Switch from exact DP to beam search once the event count reaches this.
# DP is O(E*F^2) so it becomes expensive for many events on long videos;
# beam search keeps the monotonic alignment bounded at O(E*F*B) (paper B=8).
BEAM_THRESHOLD = 8
BEAM_WIDTH = 8


def decay_alpha_from_wording(
    text: str,
    default: float = DECAY_ALPHA_THEN,
) -> float:
    """Pick the temporal-decay alpha from ordering cues in *text*.

    Scans the lower-cased query for immediacy / lateness markers and returns
    the matching alpha. Falls back to ``default`` when no marker is present.
    The most *intense* marker wins (immediately > then > later).
    """

    lowered = (text or "").lower()

    # Strongest first.
    immediate_markers = (
        "immediately", "right away", "ngay lập tức", "ngay sau đó",
    )
    then_markers = (
        "then", "next", "after that", "afterwards", "followed by",
        "subsequently", "sau đó", "rồi", "tiếp theo", "kế tiếp",
    )
    later_markers = (
        "later", "eventually", "sau cùng", "cuối cùng", "một lúc sau",
    )

    if any(marker in lowered for marker in immediate_markers):
        return DECAY_ALPHA_IMMEDIATE
    if any(marker in lowered for marker in later_markers):
        return DECAY_ALPHA_LATER
    if any(marker in lowered for marker in then_markers):
        return DECAY_ALPHA_THEN
    return default


# --- Final temporal reranking: multiplicative gating (paper Eq.6-7) ----------
#
# After the additive beam/DP search (paper Eq.4-5) selects a sequence, the paper
# applies a *fine-grained* validation that multiplies three per-event components
# into a hard multi-faceted gate:
#
#     S_i(final) = s_i · λ_i · b_i            (Eq.6)
#     SS(final)  = Σ_{i=1}^K s_i · λ_i · b_i  (Eq.7)
#
#   s_i  : semantic score of event i (the similarity at the aligned frame)
#   λ_i  : temporal-decay weight of event i (exp(-alpha · Δt_i), Eq.3)
#   b_i  : fine-grained reranker score of event i (e.g. late-interaction MaxSim
#          or SigLIP2 grading at the aligned frame)
#
# Unlike the additive search score (which tolerates one weak link), this gate
# SUPPRESSES the whole sequence when any single facet is low — "if any single
# component is low, the overall score is suppressed, ensuring a strict
# multi-faceted quality constraint".  b_i defaults to 1.0 when no fine reranker
# is available, reducing Eq.7 to the standard additive decay score.

TAU = 1e-9  # floor so a missing b_i never produces a negative/NaN gate


def temporal_lambdas(
    aligned_frame_positions: list[int],
    use_decay: bool = False,
    decay_alpha: float = 0.01,
) -> list[float]:
    """Per-event temporal-decay weights λ_i = exp(-alpha · Δt_i) (paper Eq.3).

    ``aligned_frame_positions`` are the *ordered* integer frame ids backing the
    aligned columns (event 1 → position 0, …).  Event 0 (the first) has no
    predecessor, so its λ is 1.0 (anchored).  Subsequent events use the
    *normalised* time gap between consecutive aligned frames (so long videos do
    not dominate), exactly like the decay term inside :func:`align_events_dp`.

    With ``use_decay=False`` every λ_i = 1.0 (no temporal suppression), matching
    the legacy additive score when decay is off.
    """

    positions = np.asarray(aligned_frame_positions, dtype=np.float32)
    if not use_decay or len(positions) == 0:
        return [1.0 for _ in positions]
    norm = _normalised_positions(positions)
    lambdas: list[float] = [1.0]
    for i in range(1, len(norm)):
        delta_t = float(norm[i] - norm[i - 1])
        lambdas.append(float(np.exp(-decay_alpha * delta_t)))
    return lambdas


def final_trake_gating(
    event_scores: list[float],
    lambdas: list[float],
    fine_rerank: list[float] | None = None,
) -> float:
    """Paper Eq.6-7: SS(final) = Σ_i s_i · λ_i · b_i.

    Multiplicative gate over the three per-event facets (semantic × temporal ×
    fine-grained).  ``fine_rerank`` (b_i) defaults to all-1.0 when no fine reranker
    is supplied, so the gate degrades to the additive decay score.  Floors each
    b_i at :data:`TAU` so a single missing rerank score never collapses the
    entire sequence to zero.  Returns a single scalar — the sequence's gated
    score — to be assigned as the candidate's final ``score``.
    """

    if not event_scores:
        return 0.0
    if fine_rerank is None:
        fine_rerank = [1.0 for _ in event_scores]
    # Tolerate length mismatch defensively (e.g. extra rerank scores): zip stops
    # at the shortest, and anything beyond event_count is ignored.
    if len(fine_rerank) < len(event_scores):
        fine_rerank = list(fine_rerank) + [1.0] * (len(event_scores) - len(fine_rerank))
    if len(lambdas) < len(event_scores):
        lambdas = list(lambdas) + [1.0] * (len(event_scores) - len(lambdas))

    total = 0.0
    for s_i, lam_i, b_i in zip(event_scores, lambdas, fine_rerank):
        b_safe = b_i if (b_i is not None and b_i > 0) else TAU
        total += float(s_i) * float(lam_i) * b_safe
    return total


def _normalised_positions(
    positions: np.ndarray,
) -> np.ndarray:
    """Map integer frame positions to [0, 1] by their rank spread.

    A pure integer gap overweights long videos; normalising by the span keeps
    the decay comparable across videos of different lengths (Eq.3 uses a
    normalised time delta). Falls back to raw integers when the span is 0.
    """

    positions = np.asarray(positions, dtype=np.float32)
    span = float(positions.max() - positions.min())
    if span <= 0:
        return positions
    return (positions - float(positions.min())) / span


def align_events_dp(
    similarity_matrix: np.ndarray,
    penalty_weight: float = 0.005,
    # --- Phase 6 opt-in extensions (all OFF by default) -------------------
    # ``use_decay`` enables exponential inter-event temporal decay (Eq.3).
    use_decay: bool = False,
    decay_alpha: float = DECAY_ALPHA_THEN,
    # ``frame_positions`` are the *ordered* integer frame ids backing the
    # columns of ``similarity_matrix`` (same order/length). Required only when
    # ``use_decay`` is True; ignored otherwise. When omitted with decay on,
    # integer column indices are used (legacy behaviour).
    frame_positions: np.ndarray | None = None,
    # First/last event semantics: if an event carries one of these roles its
    # selected frame is constrained to be the *earliest* (first) or highest
    # score (last) satisfying frame rather than the column with the top score.
    event_roles: list[str] | None = None,
    # Switch DP -> beam search for many events (paper B=8).
    use_beam: bool = False,
    beam_width: int = BEAM_WIDTH,
) -> tuple[float, list[int]]:
    """Find the highest-scoring monotonic event-to-frame alignment.

    The returned positions are column indices inside similarity_matrix.
    Each event receives exactly one frame and consecutive events must
    select strictly increasing frame positions.

    The optimised objective (legacy, ``use_decay=False``) is:

        sum(event-to-frame similarities)
        - penalty_weight * sum(position gaps)

    When ``use_decay=True`` the gap term becomes the paper's exponential
    temporal decay (Eq.3):

        sum(event-to-frame similarities)
        - penalty_weight * sum(position gaps)
        - decay_alpha   * sum(delta_t_i)            # normalised time gap

    i.e. long gaps between *consecutive events* are penalised exponentially,
    with the strength set by ``decay_alpha`` (wording-derived). The two terms
    coexist so coarse gap-compaction and fine temporal-decay both apply.

    ``event_roles`` lets a caller pin semantics: ``"first"`` forces the event
    onto the earliest timestamp whose similarity passes the column maximum's
    threshold (not just the argmax column), and ``"last"`` onto the most
    similar frame overall. This implements "first moment = earliest ts that
    satisfies the condition" rather than "highest frame score".
    """

    scores = np.asarray(
        similarity_matrix,
        dtype=np.float32,
    )

    if scores.ndim != 2:
        raise ValueError(
            "similarity_matrix must be a 2D matrix"
        )

    event_count, frame_count = scores.shape

    if event_count == 0:
        raise ValueError(
            "At least one event is required"
        )

    if frame_count == 0:
        raise ValueError(
            "At least one frame is required"
        )

    if frame_count < event_count:
        raise ValueError(
            "There are fewer frames than events"
        )

    if penalty_weight < 0:
        raise ValueError(
            "penalty_weight must not be negative"
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "similarity_matrix must contain finite values"
        )

    if use_beam and event_count >= BEAM_THRESHOLD:
        return _align_events_beam(
            scores=scores,
            penalty_weight=penalty_weight,
            use_decay=use_decay,
            decay_alpha=decay_alpha,
            frame_positions=frame_positions,
            event_roles=event_roles,
            beam_width=beam_width,
        )

    return _align_events_dp_core(
        scores=scores,
        penalty_weight=penalty_weight,
        use_decay=use_decay,
        decay_alpha=decay_alpha,
        frame_positions=frame_positions,
        event_roles=event_roles,
    )


def _align_events_dp_core(
    scores: np.ndarray,
    penalty_weight: float,
    use_decay: bool,
    decay_alpha: float,
    frame_positions: np.ndarray | None,
    event_roles: list[str] | None,
) -> tuple[float, list[int]]:
    """Exact dynamic program over the monotonic alignment."""

    event_count, frame_count = scores.shape

    norm_positions: np.ndarray | None = None
    if use_decay:
        if frame_positions is not None:
            norm_positions = _normalised_positions(frame_positions)
        else:
            norm_positions = _normalised_positions(
                np.arange(frame_count, dtype=np.float32)
            )

    dp = np.full(
        (event_count, frame_count),
        -np.inf,
        dtype=np.float32,
    )

    predecessor = np.full(
        (event_count, frame_count),
        -1,
        dtype=np.int32,
    )

    # Event đầu tiên có thể chọn bất kỳ frame nào.
    dp[0, :] = scores[0, :]

    for event_index in range(
        1,
        event_count,
    ):
        running_best_score = -np.inf
        running_best_index = -1

        # Ít nhất phải chừa event_index frame ở phía trước.
        for frame_index in range(
            event_index,
            frame_count,
        ):
            previous_index = frame_index - 1

            previous_score = (
                dp[
                    event_index - 1,
                    previous_index,
                ]
                + penalty_weight * previous_index
            )

            if previous_score > running_best_score:
                running_best_score = previous_score
                running_best_index = previous_index

            if running_best_index < 0:
                continue

            if not np.isfinite(
                running_best_score
            ):
                continue

            edge = scores[event_index, frame_index]
            if use_decay and norm_positions is not None:
                # Eq.3: longer inter-event gaps are penalised (smaller
                # effective weight on the later event), so SUBTRACT. The gap is
                # measured from the *actual* chosen predecessor (running_best_
                # index), NOT the immediately preceding frame index — the DP
                # running-max can pick a non-adjacent predecessor.
                delta_t = (
                    norm_positions[frame_index]
                    - norm_positions[running_best_index]
                )
                edge -= decay_alpha * float(delta_t)

            dp[
                event_index,
                frame_index,
            ] = (
                edge
                + running_best_score
                - penalty_weight * frame_index
            )

            predecessor[
                event_index,
                frame_index,
            ] = running_best_index

    final_position = int(
        np.argmax(
            dp[-1, :]
        )
    )

    final_score = float(
        dp[-1, final_position]
    )

    if not np.isfinite(final_score):
        raise ValueError(
            "No valid monotonic alignment exists"
        )

    path = [0] * event_count
    current_position = final_position

    for event_index in range(
        event_count - 1,
        -1,
        -1,
    ):
        path[event_index] = current_position

        if event_index == 0:
            continue

        current_position = int(
            predecessor[
                event_index,
                current_position,
            ]
        )

        if current_position < 0:
            raise RuntimeError(
                "Broken DP predecessor chain"
            )

    return final_score, _apply_event_roles(
        path,
        scores,
        event_roles,
    )


def _align_events_beam(
    scores: np.ndarray,
    penalty_weight: float,
    use_decay: bool,
    decay_alpha: float,
    frame_positions: np.ndarray | None,
    event_roles: list[str] | None,
    beam_width: int,
) -> tuple[float, list[int]]:
    """Beam-search approximation of the monotonic alignment (paper B=8).

    Keeps only the top-``beam_width`` partial alignments at each event step,
    so cost is O(E*F*B) instead of O(E*F^2). Exact when ``beam_width >=
    frame_count``. Used for many-event queries where exact DP would be slow.
    """

    event_count, frame_count = scores.shape
    beam_width = max(1, min(beam_width, frame_count))

    norm_positions: np.ndarray | None = None
    if use_decay:
        if frame_positions is not None:
            norm_positions = _normalised_positions(frame_positions)
        else:
            norm_positions = _normalised_positions(
                np.arange(frame_count, dtype=np.float32)
            )

    # Beam state: (carry, path). ``carry`` mirrors the DP recurrence: it is the
    # accumulated dp-cell value PLUS ``penalty_weight * last_frame`` (the exact
    # quantity the next event's running-best adds before subtracting the new
    # frame's penalty). Storing the carry (not the raw partial score) makes the
    # beam reproduce the DP math exactly when ``beam_width >= frame_count``, and
    # stays a faithful approximation when pruned.
    beam: list[tuple[float, list[int]]] = [
        (
            float(scores[0, f]) + penalty_weight * f,
            [int(f)],
        )
        for f in range(frame_count)
    ]
    beam.sort(key=lambda item: item[0], reverse=True)
    beam = beam[:beam_width]

    for event_index in range(1, event_count):
        # Keep the best partial score that reaches each candidate *next* frame
        # (dedup by the new last_frame). This guarantees we never prune a
        # reachable frame, so a valid path survives if one exists — while still
        # bounding width when many frames compete for the same position.
        best_to_frame: dict[int, tuple[float, list[int]]] = {}
        for carry, beam_path in beam:
            last_frame = beam_path[-1]
            for frame_index in range(
                last_frame + 1,
                frame_count,
            ):
                edge = float(scores[event_index, frame_index])
                if use_decay and norm_positions is not None:
                    delta_t = (
                        norm_positions[frame_index]
                        - norm_positions[last_frame]
                    )
                    edge -= decay_alpha * float(delta_t)
                cell = edge + carry - penalty_weight * frame_index
                new_carry = cell + penalty_weight * frame_index
                prev = best_to_frame.get(frame_index)
                if prev is None or new_carry > prev[0]:
                    best_to_frame[frame_index] = (
                        new_carry,
                        beam_path + [int(frame_index)],
                    )
        beam = sorted(
            best_to_frame.values(),
            key=lambda item: item[0],
            reverse=True,
        )[:beam_width]

    if not beam:
        raise ValueError(
            "No valid monotonic alignment exists"
        )

    best_score, best_path = beam[0]
    # ``best_score`` is the stored *carry* (cell value + penalty*last_frame);
    # strip the carry back to the true dp-cell value so it matches the DP's
    # reported ``final_score`` exactly.
    best_score = best_score - penalty_weight * best_path[-1]
    return best_score, _apply_event_roles(
        best_path,
        scores,
        event_roles,
    )


def _apply_event_roles(
    path: list[int],
    scores: np.ndarray,
    event_roles: list[str] | None,
) -> list[int]:
    """Rewrite first/last events to earliest-ts / top-score semantics.

    ``"first"`` -> earliest frame whose similarity is within the column's
    achieved max (the "first moment that satisfies the condition"), not the
    argmax column. ``"last"`` -> the most similar frame in the column.
    Only meaningful when the caller has supplied ordered ``event_roles``.
    """

    if not event_roles:
        return path

    event_count, frame_count = scores.shape
    out = list(path)

    for event_index, role in enumerate(event_roles[:event_count]):
        if role not in ("first", "last"):
            continue
        column = scores[event_index, :]
        col_max = float(column.max())
        if role == "last":
            out[event_index] = int(np.argmax(column))
            continue
        # first: earliest frame whose similarity reaches the column max.
        # argmax already returns the first occurrence of the max, which is the
        # earliest position with the highest score — equivalent to "earliest ts
        # satisfying the condition".
        matches = np.where(column >= col_max - 1e-6)[0]
        if len(matches):
            out[event_index] = int(matches[0])

    return out
