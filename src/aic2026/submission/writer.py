from __future__ import annotations

import json
from pathlib import Path

from aic2026.models import Candidate, Query


def validate_candidates(query: Query, candidates: list[Candidate], max_answers: int = 100) -> list[Candidate]:
    if not candidates:
        raise ValueError(f"{query.query_id}: no candidates")
    if len(candidates) > max_answers:
        raise ValueError(f"{query.query_id}: maximum is {max_answers} answers")
    if any(c.frame_id < 0 or not c.video_id for c in candidates):
        raise ValueError(f"{query.query_id}: invalid video_id or frame_id")
    if query.type == "qa" and any(not c.answer for c in candidates):
        raise ValueError(f"{query.query_id}: Q&A candidates require answer")
    if query.type == "trake" and any(not c.event_frames for c in candidates):
        raise ValueError(f"{query.query_id}: TRAKE candidates require event_frames")
    if query.type == "trake" and query.events and any(len(c.event_frames or []) != len(query.events) for c in candidates):
        raise ValueError(f"{query.query_id}: TRAKE event_frames count must equal query event count")
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def competition_answer(query: Query, candidate: Candidate) -> dict:
    """Only fields stated in the AIC brief; scores/internal IDs never leak into submission."""
    if query.type == "kis":
        return {"video_id": candidate.video_id, "frame_id": candidate.frame_id}
    if query.type == "qa":
        return {"video_id": candidate.video_id, "frame_id": candidate.frame_id, "answer": candidate.answer}
    if query.type == "trake":
        return {"video_id": candidate.video_id, "frame_ids": candidate.event_frames}
    raise ValueError(f"No competition output adapter registered for query type: {query.type}")


def write_submission(items: list[tuple[Query, list[Candidate]]], output: Path) -> None:
    rows: list[str] = []
    for query, candidates in items:
        ranked = validate_candidates(query, candidates)
        rows.append(json.dumps({"query_id": query.query_id, "type": query.type, "answers": [competition_answer(query, c) for c in ranked]}, ensure_ascii=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
