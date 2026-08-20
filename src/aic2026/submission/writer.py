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


# Field separator mandated by the BTC submission brief: a single comma, no
# surrounding space (plain RFC 4180). All three streams (KIS / Q&A / TRAKE)
# join EVERY field with this. video_id / frame_id are never quoted.
CSV_FIELD_SEP = ","

# Characters that, if present in a Q&A answer, force RFC-4180 quoting so the
# BTC evaluator parses the field correctly. Mirrors the brief's six rules.
_REQUIRED_QUOTE_CHARS = (",", '"', "\n", "\r")


def _answer_needs_quoting(answer: str) -> bool:
    """Whether a Q&A ``answer`` MUST be wrapped in double quotes to survive CSV.

    Encodes the BTC brief's six rules verbatim:

    * comma  in answer -> MUST quote (splits the field otherwise)
    * quote  in answer -> MUST quote + double the inner quotes
    * newline in answer -> MUST quote
    * leading/trailing whitespace -> MUST quote so it is preserved verbatim
      (a strict CSV reader trims an unquoted field, so quoting is the only way
      to honour the "không tự động trim" requirement)
    * a plain answer with none of the above -> LEFT UNQUOTED (e.g. ``5``,
      ``Năm người``, ``Màu đỏ``)
    """
    if not answer:
        return False
    if any(ch in answer for ch in _REQUIRED_QUOTE_CHARS):
        return True
    # Leading/trailing whitespace must be quoted or a strict reader trims it.
    if answer != answer.strip():
        return True
    return False


def _format_answer(answer: str) -> str:
    """Format a Q&A answer per BTC quoting rules. Never trimmed/stripped."""
    if _answer_needs_quoting(answer):
        escaped = answer.replace('"', '""')
        return '"' + escaped + '"'
    return answer


def csv_row(query: Query, candidate: Candidate) -> str:
    """One CSV submission line covering EVERY BTC format rule.

    Separator rule: all fields are joined by a single comma (no surrounding
    space), per the BTC brief.

    Quoting rule: applies ONLY to the Q&A answer, and ONLY when it contains a
    special character or leading/trailing whitespace. video_id / frame_id are
    never quoted (they carry no special characters and must not be altered).

    Output shapes:
      KIS   : ``<video_id>,<frame_id>``
      Q&A   : ``<video_id>,<frame_id>,<answer>``            (answer may be quoted)
      TRAKE : ``<video_id>,<f1>,<f2>,...,<fN>``

    This is the single source of truth for submission formatting; both
    ``write_submission`` and ``export-submission`` route through it.
    """
    if query.type == "kis":
        return CSV_FIELD_SEP.join([
            candidate.video_id,
            str(candidate.frame_id),
        ])
    if query.type == "qa":
        return CSV_FIELD_SEP.join([
            candidate.video_id,
            str(candidate.frame_id),
            _format_answer(candidate.answer or ""),
        ])
    if query.type == "trake":
        frames = candidate.event_frames or []
        return CSV_FIELD_SEP.join(
            [candidate.video_id, *[str(f) for f in frames]]
        )
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
