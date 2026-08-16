from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import typer

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


def _quote(value: str) -> str:
    """CSV-quote a Q&A answer only when it contains comma, quote or newline.

    Per the AIC brief, simple answers (letters, digits, plain spaces) are NOT
    wrapped in quotes; quotes are mandatory when the value holds a delimiter,
    a quote, or a line break. Quotes are escaped by doubling them.
    """
    if value and any(ch in value for ch in (",", '"', "\n", "\r")):
        return '"' + value.replace('"', '""') + '"'
    return value


def csv_row(query: Query, candidate: Candidate) -> str:
    """One CSV line per the AIC 2026 submission brief.

    - KIS:   ``<video_id>,<frame_id>``
    - Q&A:   ``<video_id>,<frame_id>,<answer>`` (answer quoted only if needed)
    - TRAKE: ``<video_id>,<frame_1>,<frame_2>,...,<frame_N>``
    """
    if query.type == "kis":
        return f"{candidate.video_id},{candidate.frame_id}"
    if query.type == "qa":
        return f"{candidate.video_id},{candidate.frame_id},{_quote(candidate.answer or '')}"
    if query.type == "trake":
        frames = candidate.event_frames or []
        return f"{candidate.video_id}," + ",".join(str(f) for f in frames)
    raise ValueError(f"No CSV adapter registered for query type: {query.type}")


def write_submission(items: list[tuple[Query, list[Candidate]]], output_dir: Path) -> list[Path]:
    """Write one CSV file per query into ``output_dir`` (no header row, UTF-8, LF).

    Returns the list of written CSV paths. Filenames follow the BTC convention
    ``query-<query_id>-<type>.csv`` so they slot straight into the submission
    folder. Q&A answers longer than 100 chars are warned about (per the brief
    limit) but NOT truncated, to avoid silently dropping meaning.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for query, candidates in items:
        ranked = validate_candidates(query, candidates)
        if query.type == "qa":
            for c in ranked:
                if c.answer and len(c.answer) > 100:
                    typer.echo(
                        f"⚠️ {query.query_id}: answer dài {len(c.answer)} ký tự "
                        f"(vượt giới hạn 100 của BTC): {c.answer[:40]}..."
                    )
        lines = [csv_row(query, c) for c in ranked]
        path = output_dir / f"query-{query.query_id}-{query.type}.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)
    return written


def package_submission(output_dir: Path, zip_path: Path) -> Path:
    """Zip ``output_dir`` so the archive root contains ``submission/``.

    The brief requires the zip to contain a ``submission`` folder holding all
    the CSV files (do not zip the CSVs loosely).
    """
    output_dir = Path(output_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w") as zf:
        for path in sorted(output_dir.glob("*.csv")):
            zf.write(path, arcname=f"submission/{path.name}")
    return zip_path
