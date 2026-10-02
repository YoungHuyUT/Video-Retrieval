#!/usr/bin/env python3
"""Merge ASR JSONL batches with conservative Vietnamese typo correction.

This utility is intentionally *not* a generative rewriter.  ASR is evidence:
when a phrase cannot be corrected with high confidence it remains untouched.
Every applied replacement is written to an audit JSON file, and the original
sidecar is preserved before it is replaced.

Usage:
    uv run python scripts/merge_asr_sidecars.py
    uv run python scripts/merge_asr_sidecars.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "data" / "processed" / "asr_sidecar.jsonl"
DEFAULT_BATCH = ROOT / "data" / "processed" / "asr_batch2.jsonl"
DEFAULT_AUDIT = ROOT / "data" / "processed" / "asr_correction_audit.json"

# Only unambiguous transcription/spelling mistakes.  Do not add guessed person,
# place, or programme names here: those must stay as ASR output unless verified
# against the source video.
# These were reviewed against the surrounding ASR sentence.  They deliberately
# contain only deterministic corrections (names, standard Vietnamese words and
# fixed phrases), never a guessed rewrite of a whole sentence.
SAFE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("rơi dãi", "rơi vãi"),
    ("tìm ẩn", "tiềm ẩn"),
    ("khuyến cá", "khuyến cáo"),
    ("gây gắt", "gay gắt"),
    ("ưu đại", "ưu đãi"),
    ("lỗ rồng", "lỗ ròng"),
    ("dường thú", "vườn thú"),
    ("EUCN", "IUCN"),
    ("bợp dịch tuyệt chủng", "bờ vực tuyệt chủng"),
    ("trái điểm tiếp giáp", "tại điểm tiếp giáp"),
    ("Luân Đôm", "Luân Đôn"),
    ("nồng độ cụ", "nồng độ cồn"),
    ("nhỗn", "Nhổn"),
    ("Độ Thị", "Đô thị"),
    ("phiên tói", "phiền toái"),
    ("lĩnh tỉnh", "lỉnh kỉnh"),
    ("lại gì", "lạ gì"),
    ("trước thèm nọt mới", "trước thềm năm học mới"),
    ("dấn nạn", "vấn nạn"),
    ("chiếm đạt", "chiếm đoạt"),
    ("chiếm đặt", "chiếm đoạt"),
    ("chủng mực", "chuẩn mực"),
    ("hư hóng", "hư hỏng"),
    ("niêm miết", "niêm yết"),
    ("điều khóe mạnh", "đều khỏe mạnh"),
    ("thúy sáng", "thủy sản"),
    ("ném phà ứng cứu", "điều phà ứng cứu"),
)


def _clean_text(text: object, corrections: Counter[str]) -> str:
    """Normalise whitespace and apply only auditable, safe replacements."""
    value = unicodedata.normalize("NFC", str(text or ""))
    value = re.sub(r"\s+", " ", value).strip()
    for wrong, right in SAFE_REPLACEMENTS:
        if wrong == right:
            continue
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)

        def preserve_case(match: re.Match[str]) -> str:
            found = match.group(0)
            if found.islower():
                return right.lower()
            if found.isupper():
                return right.upper()
            return right

        value, count = pattern.subn(preserve_case, value)
        if count:
            corrections[f"{wrong} → {right}"] += count
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict) or not str(row.get("video_id", "")).strip():
                raise ValueError(f"{path}:{line_number}: missing video_id")
            rows.append(row)
    return rows


def _normalise_record(row: dict[str, Any], corrections: Counter[str]) -> dict[str, Any]:
    video_id = str(row["video_id"]).strip()
    segments: list[dict[str, Any]] = []
    for segment in row.get("segments", []):
        if not isinstance(segment, dict):
            continue
        text = _clean_text(segment.get("text"), corrections)
        if not text:
            continue
        try:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
        except (TypeError, ValueError):
            continue
        if start < 0 or end < start:
            continue
        clean_segment: dict[str, Any] = {"text": text, "start": start, "end": end}
        if segment.get("frame") is not None:
            try:
                clean_segment["frame"] = int(segment["frame"])
            except (TypeError, ValueError):
                pass
        segments.append(clean_segment)

    # The ASR reader indexes this field.  Rebuild it from validated segments so
    # it exactly matches the corrected evidence and never carries stale text.
    full_text = " ".join(segment["text"] for segment in segments)
    if not full_text:
        full_text = _clean_text(row.get("text"), corrections)
    return {"video_id": video_id, "text": full_text, "segments": segments}


def _quality(row: dict[str, Any]) -> tuple[int, int]:
    return (len(row["segments"]), len(row["text"]))


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        tmp = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    corrections: Counter[str] = Counter()
    combined: dict[str, dict[str, Any]] = {}
    replaced_duplicates = 0
    for source_index, source in enumerate((args.base, args.batch)):
        if not source.is_file():
            raise FileNotFoundError(source)
        for raw in _read_jsonl(source):
            record = _normalise_record(raw, corrections)
            old = combined.get(record["video_id"])
            # The refreshed batch is authoritative for a duplicate video ID.
            # Keep the old record only when the refreshed row has no usable
            # segments, which protects against an interrupted ASR export.
            batch_has_evidence = source_index == 1 and bool(record["segments"])
            if old is None or batch_has_evidence or _quality(record) > _quality(old):
                if old is not None:
                    replaced_duplicates += 1
                combined[record["video_id"]] = record

    records = [combined[video_id] for video_id in sorted(combined)]
    audit = {
        "inputs": [str(args.base), str(args.batch)],
        "output": str(args.output),
        "video_count": len(records),
        "segment_count": sum(len(row["segments"]) for row in records),
        "duplicate_records_replaced": replaced_duplicates,
        "corrections": dict(sorted(corrections.items())),
        "correction_count": sum(corrections.values()),
        "policy": "Only deterministic high-confidence typo fixes; uncertain ASR text is preserved.",
    }
    # Windows consoles can still use cp1252; keep terminal reporting portable.
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    if args.dry_run:
        return 0

    if args.output.resolve() == args.base.resolve():
        batch1_backup = args.output.with_suffix(args.output.suffix + ".bak_batch1")
        if not batch1_backup.exists():
            shutil.copy2(args.base, batch1_backup)
        refresh_backup = args.output.with_suffix(args.output.suffix + ".bak_before_batch2_refresh")
        if not refresh_backup.exists():
            shutil.copy2(args.base, refresh_backup)
    _write_jsonl_atomic(args.output, records)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
