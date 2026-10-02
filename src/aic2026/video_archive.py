"""On-demand extraction of a single BTC video from its remote ZIP archive.

The browser cannot play a member inside a ZIP file.  This module reads only the
ZIP central directory and the selected compressed member via HTTP Range, writes
that one video to a local cache, then lets FastAPI stream it normally.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import zipfile
from collections import OrderedDict
from pathlib import Path, PurePosixPath

import requests

ARCHIVE_BASE_URL = "https://aic-data.ledo.io.vn"
ARCHIVES: dict[str, str] = {
    "S01": "Video_S01.zip",
    **{f"N{i:03d}-N{i + 9:03d}": f"Video_N{i:03d}-N{i + 9:03d}.zip" for i in range(1, 100, 10)},
    **{f"M{i:02d}": f"Videos_M{i:02d}.zip" for i in range(1, 11)},
}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts", ".m4v", ".flv"}
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def archive_key_for_video(video_id: str) -> str | None:
    """Return the known archive key for a safe BTC video id."""
    if not re.fullmatch(r"(?:S\d{2}-V\d+|N\d{3}[-_]V\d+|M\d{2}_V\d+)", video_id, re.I):
        return None
    if re.match(r"^S01-V\d+$", video_id, re.I):
        return "S01"
    match = re.match(r"^N(\d{3})[-_]V\d+$", video_id, re.I)
    if match:
        start = ((int(match.group(1)) - 1) // 10) * 10 + 1
        return f"N{start:03d}-N{start + 9:03d}"
    match = re.match(r"^M(\d{2})_V\d+$", video_id, re.I)
    return f"M{int(match.group(1)):02d}" if match else None


class _HttpRangeReader:
    """Small seekable reader over HTTP Range, sufficient for ``zipfile``."""

    def __init__(self, url: str, block_bytes: int = 8 * 1024 * 1024) -> None:
        self.url, self.block_bytes, self.position = url, block_bytes, 0
        self.blocks: OrderedDict[int, bytes] = OrderedDict()
        self.session = requests.Session()
        self.session.headers.update({"Accept-Encoding": "identity", "User-Agent": "AIC2026-video-player/1.0"})
        response = self.session.head(url, allow_redirects=True, timeout=(10, 30))
        response.raise_for_status()
        try:
            self.size = int(response.headers["Content-Length"])
        finally:
            response.close()
        if len(self._range(0, 1)) != 1:
            raise RuntimeError("Remote archive does not support HTTP Range")

    def _range(self, start: int, end: int) -> bytes:
        response = self.session.get(
            self.url, headers={"Range": f"bytes={start}-{end - 1}"}, stream=True,
            allow_redirects=True, timeout=(15, 120),
        )
        try:
            if response.status_code != 206:
                raise RuntimeError(f"Remote archive did not honor Range (HTTP {response.status_code})")
            data = response.raw.read(end - start)
            if len(data) != end - start:
                raise IOError("Short HTTP Range read")
            return data
        finally:
            response.close()

    def _block(self, index: int) -> bytes:
        cached = self.blocks.get(index)
        if cached is not None:
            self.blocks.move_to_end(index)
            return cached
        start = index * self.block_bytes
        value = self._range(start, min(start + self.block_bytes, self.size))
        self.blocks[index] = value
        if len(self.blocks) > 2:
            self.blocks.popitem(last=False)
        return value

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        remaining = min(size, max(0, self.size - self.position))
        chunks = bytearray()
        while remaining:
            block = self._block(self.position // self.block_bytes)
            offset = self.position % self.block_bytes
            take = min(remaining, len(block) - offset)
            chunks.extend(block[offset:offset + take])
            self.position += take
            remaining -= take
        return bytes(chunks)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        self.position = offset if whence == os.SEEK_SET else self.position + offset if whence == os.SEEK_CUR else self.size + offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def tell(self) -> int:
        return self.position

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        self.session.close()
        self.blocks.clear()


def fetch_video_to_cache(video_id: str, cache_dir: str | Path = "data/processed/video_cache") -> Path | None:
    """Extract one remote archive member atomically, returning its cached path."""
    archive_key = archive_key_for_video(video_id)
    archive_name = ARCHIVES.get(archive_key or "")
    if not archive_name:
        return None
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    lock_key = video_id.upper()
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(lock_key, threading.Lock())
    with lock:
        ready = next((p for p in cache.glob(f"{video_id}.*") if p.suffix.lower() in VIDEO_SUFFIXES), None)
        if ready is not None and ready.stat().st_size > 0:
            return ready
        remote = _HttpRangeReader(f"{ARCHIVE_BASE_URL}/{archive_name}")
        try:
            with zipfile.ZipFile(remote) as archive:
                info = next((item for item in archive.infolist() if not item.is_dir() and PurePosixPath(item.filename).stem == video_id and PurePosixPath(item.filename).suffix.lower() in VIDEO_SUFFIXES), None)
                if info is None:
                    return None
                suffix = PurePosixPath(info.filename).suffix.lower() or ".mp4"
                output = cache / f"{video_id}{suffix}"
                temporary = cache / f".{video_id}{suffix}.part"
                with archive.open(info) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
                temporary.replace(output)
                return output
        finally:
            remote.close()


__all__ = ["archive_key_for_video", "fetch_video_to_cache"]
