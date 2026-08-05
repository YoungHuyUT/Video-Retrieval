from __future__ import annotations

from aic2026.models import Candidate, FrameRecord


def rerank_with_metadata(query: str, candidates: list[Candidate], records: dict[int, FrameRecord], weight: float = 0.05) -> list[Candidate]:
    terms = set(query.lower().split())
    for item in candidates:
        record = records.get(item.vector_id or -1)
        if record:
            haystack = " ".join(record.object_labels + [record.title or "", record.description or ""]).lower()
            item.score += weight * sum(term in haystack for term in terms)
    return sorted(candidates, key=lambda c: c.score, reverse=True)
