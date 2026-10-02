"""SQLite catalog for the lightweight extraction pipeline (spec §2, §6, §10).

Stores provenance for the three frame sources (btc / uniform / motion) without
re-opening every video. The ``frames`` table is what the query-time temporal
zoom-in reads: given a candidate timestamp ``t`` for ``video_id`` we SELECT rows
inside ``[t-WINDOW, t+WINDOW]`` (spec §6 — no realtime video decode).

Schema mirrors :class:`aic2026.extraction.catalog.ShotCatalog` but drops the shot
concept (lightweight has no shot detector) and adds a ``source`` column.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class CatalogLock:
    """Cross-interpreter file lock so only ONE build mutates catalog.db at once.

    Windows-only (uses ``msvcrt`` stdlib). A second ``lw-build`` / ``lw-merge-btc``
    invocation will retry briefly then raise a clear error instead of corrupting
    the SQLite catalog through concurrent writes (the earlier failure mode).
    """

    def __init__(self, root: Path, timeout_s: float = 30.0, poll_s: float = 0.5):
        self.lock_path = Path(root) / ".build.lock"
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self._fh = None

    def __enter__(self) -> "CatalogLock":
        import msvcrt

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        # "w+b" creates the file if missing; binary mode is required by msvcrt.
        self._fh = open(self.lock_path, "w+b")
        waited = 0.0
        while True:
            try:
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                return self
            except OSError:
                if waited >= self.timeout_s:
                    self._fh.close()
                    self._fh = None
                    raise RuntimeError(
                        f"Another lightweight build holds {self.lock_path} — "
                        f"wait for it to finish or delete the stale lock file."
                    )
                time.sleep(self.poll_s)
                waited += self.poll_s

    def __exit__(self, *exc) -> None:
        import msvcrt

        if self._fh is not None:
            try:
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self._fh.close()
            self._fh = None


class LightweightCatalog:
    """Thin SQLite wrapper. One DB per pipeline root (``catalog.db``)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        # Retry contended writes instead of aborting with a locked-db error.
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.execute("PRAGMA journal_mode=WAL")
        # FK-style consistency is checked by ``validate_artifacts``; we do not
        # enforce FK at runtime so ad-hoc re-adds never abort a long run.
        self.conn.execute("PRAGMA foreign_keys=OFF")
        self._init_schema()
        self._migrate_schema()

    # -- schema -----------------------------------------------------------
    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS videos (
                video_id    TEXT PRIMARY KEY,
                fps         REAL,
                frame_count INTEGER,
                source_hash TEXT,
                created_at  REAL,
                is_built    INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS frames (
                frame_id          INTEGER,
                video_id          TEXT NOT NULL,
                timestamp         REAL NOT NULL,
                source            TEXT NOT NULL,
                path              TEXT NOT NULL,
                clip_index        INTEGER,
                model_version     TEXT,
                embedding_version TEXT,
                object_labels     TEXT,
                PRIMARY KEY (video_id, frame_id),
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                name         TEXT PRIMARY KEY,
                path         TEXT NOT NULL,
                model_name   TEXT,
                model_version TEXT,
                dim          INTEGER,
                dtype        TEXT,
                created_at   REAL
            );
            CREATE TABLE IF NOT EXISTS model_versions (
                model    TEXT NOT NULL,
                version  TEXT NOT NULL,
                dim      INTEGER,
                dtype    TEXT,
                PRIMARY KEY (model, version)
            );
            CREATE INDEX IF NOT EXISTS idx_frames_video_ts
                ON frames(video_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_frames_source
                ON frames(source);
            """
        )
        self.conn.commit()

    # -- migrations -------------------------------------------------------
    def _migrate_schema(self) -> None:
        """Add columns that post-date the original catalog, idempotently.

        ``CREATE TABLE IF NOT EXISTS`` won't add a column to an existing table,
        so a lightweight catalog built before ``object_labels`` existed would
        silently drop object evidence.  We ALTER only when the column is absent
        (guarded by the schema check), so re-running on a fresh catalog is a
        no-op.
        """
        cols = {
            r["name"] for r in self.conn.execute("PRAGMA table_info(frames)").fetchall()
        }
        if "object_labels" not in cols:
            self.conn.execute("ALTER TABLE frames ADD COLUMN object_labels TEXT")
        self.conn.commit()

    # -- writes ------------------------------------------------------------
    def upsert_video(
        self, video_id: str, fps: float, frame_count: int, source_hash: str | None = None
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO videos (video_id, fps, frame_count, source_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                fps=excluded.fps,
                frame_count=excluded.frame_count,
                source_hash=COALESCE(excluded.source_hash, videos.source_hash),
                created_at=excluded.created_at
            """,
            (video_id, float(fps), int(frame_count), source_hash, time.time()),
        )
        self.conn.commit()

    def add_frame(
        self,
        video_id: str,
        frame_id: int,
        timestamp: float,
        source: str,
        path: str,
        clip_index: int | None = None,
        model_version: str | None = None,
        embedding_version: str | None = None,
        object_labels: list[str] | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO frames
                (video_id, frame_id, timestamp, source, path, clip_index, model_version, embedding_version, object_labels)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id, frame_id) DO UPDATE SET
                timestamp=excluded.timestamp,
                source=excluded.source,
                path=excluded.path,
                clip_index=excluded.clip_index,
                model_version=excluded.model_version,
                embedding_version=excluded.embedding_version,
                object_labels=COALESCE(excluded.object_labels, frames.object_labels)
            """,
            (
                video_id,
                int(frame_id),
                float(timestamp),
                source,
                path,
                clip_index,
                model_version,
                embedding_version,
                json.dumps(sorted(set(object_labels or [])), ensure_ascii=False)
                if object_labels
                else None,
            ),
        )

    def commit_frames(self) -> None:
        self.conn.commit()

    def set_frame_object_labels(
        self, video_id: str, frame_id: int, object_labels: list[str]
    ) -> None:
        """Write detector-derived object labels for one frame (idempotent)."""
        self.conn.execute(
            """
            UPDATE frames SET object_labels=?
            WHERE video_id=? AND frame_id=?
            """,
            (
                json.dumps(sorted(set(object_labels or [])), ensure_ascii=False)
                if object_labels
                else None,
                video_id,
                int(frame_id),
            ),
        )

    def register_artifact(
        self,
        name: str,
        path: str,
        model_name: str | None = None,
        model_version: str | None = None,
        dim: int | None = None,
        dtype: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO artifacts (name, path, model_name, model_version, dim, dtype, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                path=excluded.path,
                model_name=excluded.model_name,
                model_version=excluded.model_version,
                dim=excluded.dim,
                dtype=excluded.dtype,
                created_at=excluded.created_at
            """,
            (name, str(path), model_name, model_version, dim, dtype, time.time()),
        )
        self.conn.commit()

    def register_model_version(
        self, model: str, version: str, dim: int | None = None, dtype: str | None = None
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO model_versions (model, version, dim, dtype)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(model, version) DO UPDATE SET
                dim=COALESCE(excluded.dim, model_versions.dim),
                dtype=COALESCE(excluded.dtype, model_versions.dtype)
            """,
            (model, version, dim, dtype),
        )
        self.conn.commit()

    def is_video_done(self, video_id: str) -> bool:
        # A video counts as "built" only after lw-build has committed its
        # uniform/motion frames (mark_video_built). A video with only its BTC
        # base, or one interrupted mid-build, is NOT done and will be resumed.
        cur = self.conn.execute(
            "SELECT 1 FROM videos WHERE video_id=? AND is_built=1 LIMIT 1",
            (video_id,),
        )
        return cur.fetchone() is not None

    def mark_video_built(self, video_id: str) -> None:
        """Mark a video complete AFTER its frames have been committed."""
        self.conn.execute(
            "UPDATE videos SET is_built=1 WHERE video_id=?", (video_id,)
        )
        self.conn.commit()

    def source_counts(self) -> dict[str, int]:
        cur = self.conn.execute(
            "SELECT source, COUNT(*) AS n FROM frames GROUP BY source"
        )
        return {r["source"]: int(r["n"]) for r in cur.fetchall()}

    # -- reads (query-time) ------------------------------------------------
    def frames_in_window(
        self, video_id: str, center_s: float, window_s: float
    ) -> list[sqlite3.Row]:
        """Spec §6 zoom-in: neighbour frames inside ``[center-w, center+w]``."""
        lo, hi = center_s - window_s, center_s + window_s
        cur = self.conn.execute(
            """
            SELECT frame_id, video_id, timestamp, source, path, clip_index
            FROM frames
            WHERE video_id=? AND timestamp BETWEEN ? AND ?
            ORDER BY ABS(timestamp - ?) ASC
            """,
            (video_id, lo, hi, center_s),
        )
        return cur.fetchall()

    def all_frame_paths(self) -> list[str]:
        cur = self.conn.execute("SELECT path FROM frames ORDER BY video_id, timestamp")
        return [r["path"] for r in cur.fetchall()]

    def frame_ids(self, video_id: str) -> list[int]:
        """All frame_ids already committed for ``video_id`` (used to dedupe NEW
        frames against BTC rows so the real frame index is never renumbered)."""
        cur = self.conn.execute(
            "SELECT frame_id FROM frames WHERE video_id = ?", (video_id,)
        )
        return [int(r["frame_id"]) for r in cur.fetchall()]

    def all_frames(self) -> list[sqlite3.Row]:
        """All frame rows ordered by ``(video_id, frame_id)``.

        Grouped per-video (NOT a global frame_id sort) so the regenerated
        manifest follows the same per-video order the build used, keeping it
        row-aligned with the feature matrices assembled in
        ``lw_embed``/``assemble_features``. frame_id itself is the TRUE frame
        index within each video (submission-correct, Ctrl+G-able).
        """
        cur = self.conn.execute(
            """
            SELECT frame_id, video_id, timestamp, source, path,
                   clip_index, model_version, embedding_version, object_labels
            FROM frames ORDER BY video_id ASC, frame_id ASC
            """
        )
        return cur.fetchall()

    @staticmethod
    def parse_object_labels(raw) -> list[str]:
        """Decode the stored JSON-string ``object_labels`` column (or ''/None)."""
        if raw is None or raw == "":
            return []
        if isinstance(raw, (list, tuple)):
            return list(raw)
        try:
            parsed = json.loads(raw)
            return list(parsed) if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    def frame_count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) AS n FROM frames")
        return int(cur.fetchone()["n"])

    def video_count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) AS n FROM videos")
        return int(cur.fetchone()["n"])

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()
