#!/usr/bin/env python3
"""Transcribe Batch 2 videos from remote ZIPs without downloading whole archives.

The data host supports HTTP Range.  This script gives ``zipfile.ZipFile`` a
seekable, block-cached HTTP reader, extracts one missing video at a time, runs
faster-whisper, appends that video's transcript to the production Notebook
sidecar, and removes the temporary video.  Re-running skips video IDs already
present in the sidecar.

Install the optional ASR dependency first:
    python -m pip install "faster-whisper>=1.0"

Preview the videos / approximate transfer size without writing transcripts:
    python scripts/transcribe_batch2_asr.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections import OrderedDict
from pathlib import Path, PurePosixPath

import requests

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/processed/siglip2/manifest_siglip2.jsonl"
SIDECAR = ROOT / "data/processed/asr_sidecar.jsonl"
TEMP_ROOT = ROOT / "data/processed/asr_tmp"
BASE_URL = "https://aic-data.ledo.io.vn"
RANGE_BLOCK_BYTES = 8 * 1024 * 1024
RANGE_CACHE_BLOCKS = 2
VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts", ".m4v", ".flv"}

ARCHIVES: dict[str, str] = {
    "S01": "Video_S01.zip",
    **{f"N{i:03d}-N{i+9:03d}": f"Video_N{i:03d}-N{i+9:03d}.zip" for i in range(1, 100, 10)},
    **{f"M{i:02d}": f"Videos_M{i:02d}.zip" for i in range(1, 11)},
}


class HttpRangeReader:
    """Minimal seekable file object backed by cached HTTP Range reads."""

    def __init__(self, url: str, *, block_bytes: int = RANGE_BLOCK_BYTES) -> None:
        self.url = url
        self.block_bytes = block_bytes
        self._position = 0
        self._blocks: OrderedDict[int, bytes] = OrderedDict()
        self._session = requests.Session()
        self._session.headers.update({"Accept-Encoding": "identity", "User-Agent": "AIC2026-ASR/1.0"})
        response = self._session.head(url, allow_redirects=True, timeout=(15, 60))
        response.raise_for_status()
        try:
            self.size = int(response.headers["Content-Length"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError(f"Server omitted archive size for {url}") from exc
        finally:
            response.close()
        # Verify range support using exactly one byte before asking zipfile to
        # seek around a potentially 50+ GB archive.
        probe = self._request_range(0, 1)
        if len(probe) != 1:
            raise RuntimeError(f"HTTP Range probe returned {len(probe)} bytes for {url}")

    def _request_range(self, start: int, end: int) -> bytes:
        last_error: Exception | None = None
        for attempt in range(3):
            response = None
            try:
                response = self._session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end - 1}"},
                    stream=True,
                    allow_redirects=True,
                    timeout=(15, 120),
                )
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Server does not honor Range for {self.url} (HTTP {response.status_code}); "
                        "refusing to download the full ZIP"
                    )
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {start}-"):
                    raise RuntimeError(f"Unexpected Content-Range {content_range!r} for {self.url}")
                data = response.raw.read(end - start)
                if len(data) != end - start:
                    raise IOError(f"Short HTTP range read: expected {end-start}, got {len(data)}")
                return data
            except (requests.RequestException, OSError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.0 + attempt)
            finally:
                if response is not None:
                    response.close()
        raise IOError(f"Failed reading bytes {start}-{end} from {self.url}: {last_error}")

    def _block(self, block_number: int) -> bytes:
        data = self._blocks.get(block_number)
        if data is not None:
            self._blocks.move_to_end(block_number)
            return data
        start = block_number * self.block_bytes
        end = min(start + self.block_bytes, self.size)
        data = self._request_range(start, end)
        self._blocks[block_number] = data
        self._blocks.move_to_end(block_number)
        while len(self._blocks) > RANGE_CACHE_BLOCKS:
            self._blocks.popitem(last=False)
        return data

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            new_position = offset
        elif whence == os.SEEK_CUR:
            new_position = self._position + offset
        elif whence == os.SEEK_END:
            new_position = self.size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        if new_position < 0:
            raise ValueError("Negative seek position")
        self._position = new_position
        return self._position

    def tell(self) -> int:
        return self._position

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = max(0, self.size - self._position)
        size = min(size, max(0, self.size - self._position))
        if size == 0:
            return b""
        result = bytearray()
        while size:
            block_no = self._position // self.block_bytes
            block_offset = self._position % self.block_bytes
            block = self._block(block_no)
            take = min(size, len(block) - block_offset)
            if take <= 0:
                break
            result.extend(block[block_offset:block_offset + take])
            self._position += take
            size -= take
        return bytes(result)

    def close(self) -> None:
        self._session.close()
        self._blocks.clear()


def _batch2_video_ids() -> set[str]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    video_ids: set[str] = set()
    with MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("source") == "batch2":
                video_ids.add(str(row["video_id"]))
    return video_ids


def _existing_transcript_ids() -> set[str]:
    if not SIDECAR.is_file():
        return set()
    found: set[str] = set()
    with SIDECAR.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                found.add(str(json.loads(line)["video_id"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return found


def _archive_key_for_video(video_id: str) -> str | None:
    if video_id.startswith("S01-") or video_id.startswith("S01_"):
        return "S01"
    match = re.match(r"N(\d{3})[-_]", video_id, re.IGNORECASE)
    if match:
        number = int(match.group(1))
        start = ((number - 1) // 10) * 10 + 1
        return f"N{start:03d}-N{start + 9:03d}"
    match = re.match(r"M(\d{2})_", video_id, re.IGNORECASE)
    if match:
        return f"M{int(match.group(1)):02d}"
    return None


def _append_transcript(transcript: object) -> None:
    SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    with SIDECAR.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(transcript.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _transcribe(args: argparse.Namespace) -> int:
    # Lazy import lets --dry-run work before faster-whisper is installed.
    from aic2026.ingestion.asr import ASRTranscriber

    transcriber = ASRTranscriber(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=None if args.language == "auto" else args.language,
    )
    # Batch 2 includes long, mostly silent traffic footage; VAD prevents
    # running ASR repeatedly over silence and reduces hallucinated segments.
    transcriber.vad_filter = True
    if not transcriber.load():
        raise RuntimeError("faster-whisper did not load; install it with python -m pip install faster-whisper")

    todo = _batch2_video_ids() - _existing_transcript_ids()
    counts = {"done": 0, "skipped": 0, "empty": 0}
    temp_root = Path(args.temp_dir).resolve() if args.temp_dir else TEMP_ROOT
    temp_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(temp_root).free
    reserve_bytes = 2 * 1024**3
    unsupported_ranges: list[str] = []
    logging.info("%d Batch 2 videos lack ASR; temp disk free %.1f GiB", len(todo), free / 1024**3)

    if args.split:
        selected = {archive_key for archive_key in args.split if archive_key in ARCHIVES}
        todo = {vid for vid in todo if _archive_key_for_video(vid) in selected}
    if args.limit is not None:
        todo = set(sorted(todo)[:args.limit])

    for archive_key, archive_name in ARCHIVES.items():
        archive_todo = {vid for vid in todo if _archive_key_for_video(vid) == archive_key}
        if not archive_todo:
            continue
        url = f"{BASE_URL}/{archive_name}"
        logging.info("Reading %s via HTTP Range (%d target videos)", archive_name, len(archive_todo))
        try:
            remote = HttpRangeReader(url)
        except (RuntimeError, requests.RequestException) as exc:
            unsupported_ranges.append(archive_name)
            logging.error("Skipping %s safely: %s", archive_name, exc)
            continue
        try:
            with zipfile.ZipFile(remote) as archive:
                entries = {
                    PurePosixPath(info.filename).stem: info
                    for info in archive.infolist()
                    if not info.is_dir()
                    and PurePosixPath(info.filename).suffix.lower() in VIDEO_SUFFIXES
                    and PurePosixPath(info.filename).stem in archive_todo
                }
                missing = archive_todo - entries.keys()
                if missing:
                    logging.warning("%s: archive has no matching entries for %d IDs (sample=%s)", archive_name, len(missing), sorted(missing)[:5])
                for video_id, info in entries.items():
                    if info.file_size + reserve_bytes > shutil.disk_usage(temp_root).free:
                        raise OSError(
                            f"Insufficient temporary disk for {video_id}: needs about "
                            f"{(info.file_size + reserve_bytes)/1024**3:.1f} GiB free"
                        )
                    suffix = PurePosixPath(info.filename).suffix.lower() or ".mp4"
                    temp_path = temp_root / f"{video_id}{suffix}"
                    logging.info("Extracting %s (%.2f GiB uncompressed)", video_id, info.file_size / 1024**3)
                    try:
                        with archive.open(info, "r") as source, temp_path.open("wb") as target:
                            shutil.copyfileobj(source, target, length=4 * 1024**2)
                        transcript = transcriber.transcribe_video(temp_path)
                        transcript.video_id = video_id
                        _append_transcript(transcript)
                        counts["done"] += 1
                        if not transcript.segments:
                            counts["empty"] += 1
                            logging.warning("No speech segments emitted for %s", video_id)
                        logging.info("Saved ASR for %s (%d segments)", video_id, len(transcript.segments))
                    finally:
                        temp_path.unlink(missing_ok=True)
        finally:
            remote.close()
    transcriber.unload()
    logging.info(
        "Finished this run: %s; archives without HTTP Range=%s; sidecar=%s",
        counts, unsupported_ranges, SIDECAR,
    )
    return counts["done"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", help="Only process archive key, e.g. M01, S01, N001-N010; repeatable")
    parser.add_argument("--limit", type=int, help="Process at most this many missing video IDs (resume-safe)")
    parser.add_argument("--model", default="small", help="faster-whisper model size (default: small)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compute-type", default="int8", help="CTranslate2 compute type (default: int8)")
    parser.add_argument("--language", default="auto", help="Whisper language code or 'auto' (default: auto)")
    parser.add_argument("--temp-dir", help="Temporary directory for one extracted video")
    parser.add_argument("--dry-run", action="store_true", help="Inspect archive entries and missing-video transfer sizes; write nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.dry_run:
        # Dry-run uses only archive central directories and never imports ASR.
        todo = _batch2_video_ids() - _existing_transcript_ids()
        if args.split:
            selected = {key for key in args.split if key in ARCHIVES}
            todo = {vid for vid in todo if _archive_key_for_video(vid) in selected}
        if args.limit is not None:
            todo = set(sorted(todo)[:args.limit])
        total_compressed = 0
        unsupported_ranges: list[str] = []
        for archive_key, archive_name in ARCHIVES.items():
            wanted = {vid for vid in todo if _archive_key_for_video(vid) == archive_key}
            if not wanted:
                continue
            try:
                remote = HttpRangeReader(f"{BASE_URL}/{archive_name}")
            except (RuntimeError, requests.RequestException) as exc:
                unsupported_ranges.append(archive_name)
                print(f"{archive_name}: SKIP (HTTP Range unavailable; full ZIP will not be downloaded)")
                logging.debug("Range failure detail: %s", exc)
                continue
            try:
                with zipfile.ZipFile(remote) as archive:
                    entries = [
                        info for info in archive.infolist()
                        if not info.is_dir()
                        and PurePosixPath(info.filename).suffix.lower() in VIDEO_SUFFIXES
                        and PurePosixPath(info.filename).stem in wanted
                    ]
                    compressed = sum(info.compress_size for info in entries)
                    total_compressed += compressed
                print(f"{archive_name}: {len(entries)} videos, {compressed/1024**3:.2f} GiB compressed", flush=True)
            finally:
                remote.close()
        print(f"Missing transcript IDs: {len(todo)}; compressed video bytes to read: {total_compressed/1024**3:.2f} GiB", flush=True)
        if unsupported_ranges:
            print("Archives to revisit later (no Range support): " + ", ".join(unsupported_ranges), flush=True)
        return 0

    try:
        _transcribe(args)
    except KeyboardInterrupt:
        logging.warning("Interrupted; completed video transcripts are already saved. Re-run to resume.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
