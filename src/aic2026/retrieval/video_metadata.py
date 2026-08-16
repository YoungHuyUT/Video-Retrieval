"""Per-video metadata store.

The BTC metadata JSON lives once per *video* (title, description, keywords),
not per frame. The original code copied those fields into every FrameRecord,
which inflated the manifest ~10x and broke BM25 IDF (term frequency of
"cửa hàng" was counted 200 times for the same video).

This module owns a tiny ``video_metadata.jsonl`` (one row per video) and gives
the rest of the codebase a small read API: lookup by ``video_id``,
``all()``, ``save()``. Loading is O(V) — 873 docs vs the 200k FrameRecord
list, so metadata-filter cost drops from O(N) to O(V) for free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class VideoMetaRecord:
    """One video's metadata. Fields mirror the BTC Metadata JSON."""

    video_id: str
    title: str | None = None
    description: str | None = None
    metadata_keywords: list[str] = field(default_factory=list)
    metadata_path: str | None = None


class VideoMetadataStore:
    """In-memory store mapping ``video_id`` → :class:`VideoMetaRecord`.

    Backed by a JSONL file (``<one record per line>``). Built once at startup
    (~873 lines, < 1 ms on local SSD) and consulted in tight retrieval loops.
    """

    def __init__(self, records: list[VideoMetaRecord]) -> None:
        self._by_id: dict[str, VideoMetaRecord] = {r.video_id: r for r in records}
        # Preserve insertion order so ``all()`` is deterministic.
        self._ordered: list[VideoMetaRecord] = list(records)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get(self, video_id: str) -> VideoMetaRecord | None:
        return self._by_id.get(video_id)

    def all(self) -> list[VideoMetaRecord]:
        return list(self._ordered)

    def __len__(self) -> int:
        return len(self._ordered)

    def __contains__(self, video_id: object) -> bool:
        return video_id in self._by_id

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> VideoMetadataStore:
        """Read a JSONL file written by :meth:`save`."""
        path = Path(path)
        records: list[VideoMetaRecord] = []
        if not path.exists():
            return cls([])
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(
                VideoMetaRecord(
                    video_id=str(data["video_id"]),
                    title=data.get("title"),
                    description=data.get("description"),
                    metadata_keywords=list(data.get("metadata_keywords") or []),
                    metadata_path=data.get("metadata_path"),
                )
            )
        return cls(records)

    def save(self, path: str | Path) -> None:
        """Persist the store as one JSONL record per video."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for record in self._ordered:
            lines.append(
                json.dumps(
                    {
                        "video_id": record.video_id,
                        "title": record.title,
                        "description": record.description,
                        "metadata_keywords": list(record.metadata_keywords),
                        "metadata_path": record.metadata_path,
                    },
                    ensure_ascii=False,
                )
            )
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    @classmethod
    def empty(cls) -> VideoMetadataStore:
        return cls([])
