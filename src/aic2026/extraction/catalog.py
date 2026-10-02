"""SQLite catalog for the shot-adaptive pipeline (spec §10, §11, §8).

Stores the provenance graph needed for query-time temporal zoom-in and
resume/cache without re-opening every video:

    videos(video_id, fps, frame_count, source_hash, created_at)
    shots(shot_id, video_id, start_s, end_s, n_frames)
    frames(frame_id, video_id, shot_id, timestamp, path, model_version, embedding_version)
    artifacts(name, path, model_name, model_version, dim, dtype, created_at)
    model_versions(model, version, dim, dtype)

The ``frames`` table is the index the retrieval layer queries: given a candidate
timestamp ``t`` for ``video_id`` we SELECT rows inside ``[t-WINDOW, t+WINDOW]``
(spec §6, lightweight frame-index lookup — no realtime video decode).
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class ShotCatalog:
    """Thin SQLite wrapper. One DB per pipeline root (``catalog.db``)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        # FK constraints are declared in the schema for documentation/tooling,
        # but we do NOT enforce them at runtime: the pipeline always writes
        # videos -> shots -> frames in order, and a single out-of-order ad-hoc
        # write (e.g. re-adding a frame) should not abort the whole run.
        # Consistency is verified by ``validate_artifacts`` instead.
        self.conn.execute("PRAGMA foreign_keys=OFF")
        self._init_schema()

    # -- schema -----------------------------------------------------------
    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS videos (
                video_id    TEXT PRIMARY KEY,
                fps         REAL,
                frame_count INTEGER,
                source_hash TEXT,
                created_at  REAL
            );
            CREATE TABLE IF NOT EXISTS shots (
                shot_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id  TEXT NOT NULL,
                start_s   REAL NOT NULL,
                end_s     REAL NOT NULL,
                n_frames  INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            );
            CREATE TABLE IF NOT EXISTS frames (
                frame_id          INTEGER,
                video_id          TEXT NOT NULL,
                shot_id           INTEGER,
                timestamp         REAL NOT NULL,
                path              TEXT NOT NULL,
                model_version     TEXT,
                embedding_version TEXT,
                PRIMARY KEY (video_id, frame_id),
                FOREIGN KEY (shot_id) REFERENCES shots(shot_id),
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
            CREATE INDEX IF NOT EXISTS idx_shots_video
                ON shots(video_id);
            """
        )
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

    def add_shot(
        self, video_id: str, start_s: float, end_s: float, n_frames: int = 0
    ) -> int:
        # Ensure the parent video row exists (pipeline always upserts first, but
        # this keeps ad-hoc/catalog-only writes from tripping the FK constraint).
        self.conn.execute(
            "INSERT OR IGNORE INTO videos (video_id, fps, frame_count, created_at) VALUES (?, 0, 0, ?)",
            (video_id, time.time()),
        )
        cur = self.conn.execute(
            """
            INSERT INTO shots (video_id, start_s, end_s, n_frames)
            VALUES (?, ?, ?, ?)
            """,
            (video_id, float(start_s), float(end_s), int(n_frames)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_frame(
        self,
        video_id: str,
        frame_id: int,
        shot_id: int | None,
        timestamp: float,
        path: str,
        model_version: str | None = None,
        embedding_version: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO frames (video_id, frame_id, shot_id, timestamp, path, model_version, embedding_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id, frame_id) DO UPDATE SET
                shot_id=excluded.shot_id,
                timestamp=excluded.timestamp,
                path=excluded.path,
                model_version=excluded.model_version,
                embedding_version=excluded.embedding_version
            """,
            (
                video_id,
                int(frame_id),
                shot_id,
                float(timestamp),
                path,
                model_version,
                embedding_version,
            ),
        )

    def commit_frames(self) -> None:
        """Flush batched frame inserts (call after a video's frames are added)."""
        self.conn.commit()

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
        cur = self.conn.execute(
            "SELECT 1 FROM frames WHERE video_id=? LIMIT 1", (video_id,)
        )
        return cur.fetchone() is not None

    # -- reads (query-time) ------------------------------------------------
    def frames_in_window(
        self, video_id: str, center_s: float, window_s: float
    ) -> list[sqlite3.Row]:
        """Spec §6 zoom-in: neighbour frames inside ``[center-w, center+w]``."""
        lo, hi = center_s - window_s, center_s + window_s
        cur = self.conn.execute(
            """
            SELECT frame_id, video_id, shot_id, timestamp, path
            FROM frames
            WHERE video_id=? AND timestamp BETWEEN ? AND ?
            ORDER BY ABS(timestamp - ?) ASC
            """,
            (video_id, lo, hi, center_s),
        )
        return cur.fetchall()

    def shot_for_timestamp(self, video_id: str, timestamp: float) -> sqlite3.Row | None:
        cur = self.conn.execute(
            """
            SELECT * FROM shots
            WHERE video_id=? AND start_s <= ? AND end_s >= ?
            ORDER BY (end_s - start_s) ASC
            LIMIT 1
            """,
            (video_id, timestamp, timestamp),
        )
        return cur.fetchone()

    def all_frame_paths(self) -> list[str]:
        cur = self.conn.execute("SELECT path FROM frames ORDER BY video_id, timestamp")
        return [r["path"] for r in cur.fetchall()]

    def frame_count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) AS n FROM frames")
        return int(cur.fetchone()["n"])

    def shot_count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) AS n FROM shots")
        return int(cur.fetchone()["n"])

    def video_count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) AS n FROM videos")
        return int(cur.fetchone()["n"])

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()
