from __future__ import annotations

from aic2026.models import Candidate, FrameRecord


def rerank_with_metadata(
    query: str,
    candidates: list[Candidate],
    records: dict[int, FrameRecord],
    weight: float = 0.05,
) -> list[Candidate]:
    """Add a small deterministic keyword-overlap bonus to candidate scores.

    For each candidate with a manifest record, count how many query tokens (split
    on whitespace) appear in the record's ``object_labels`` / ``title`` /
    ``description`` text. The count is scaled by ``weight`` and added to the
    candidate's score; the list is then re-sorted by score so frames whose labels
    literally mention the query terms float above pure vector matches.

    This is a scoring heuristic, not a learned model. It complements the RRF
    fusion in ``RetrievalPipeline.hybrid_retrieve_raw``: RRF merges the vector and
    BM25 rankings, this nudges the merged pool using metadata text the vector
    index alone would not see.
    """

    terms = set(query.lower().split())
    if not terms:
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    for item in candidates:
        # NB: vector_id may legally be 0 (the first frame of a video), so we must
        # NOT fall back to -1 via `or` — that would skip the first frame.
        if item.vector_id is None:
            continue
        record = records.get(item.vector_id)
        if record is None:
            continue
        parts: list[str] = list(record.object_labels or [])
        if record.title:
            parts.append(record.title)
        if record.description:
            parts.append(record.description)
        haystack = " ".join(parts).lower()
        item.score += weight * sum(term in haystack for term in terms)
    return sorted(candidates, key=lambda c: c.score, reverse=True)
