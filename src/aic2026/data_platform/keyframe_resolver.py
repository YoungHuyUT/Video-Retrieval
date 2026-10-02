"""Resolve local keyframes, including frames stored only inside BTC ZIPs.

Batch 2 contains tens of gigabytes of keyframe ZIPs. This module materializes
only requested images into a small cache instead of extracting every archive.
"""

from __future__ import annotations

import re
import shutil
from functools import lru_cache
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile


@lru_cache(maxsize=16)
def _open_zip(path: str) -> ZipFile:
    return ZipFile(path)


def _archive_name(video_id: str) -> str | None:
    upper = video_id.upper()
    if upper.startswith("S01-"):
        return "Keyframes_S01.zip"
    elif match := re.match(r"M(\d{2})_", upper):
        return f"Keyframes_M{match.group(1)}.zip"
    elif match := re.match(r"N(\d{3})[-_]", upper):
        number = int(match.group(1))
        first = ((number - 1) // 10) * 10 + 1
        return f"Keyframes_N{first:03d}-N{first + 9:03d}.zip"
    else:
        return None


def _archive_for(video_id: str, keyframes_root: Path) -> Path | None:
    name = _archive_name(video_id)
    if name is None:
        return None
    archive = keyframes_root / name
    return archive if archive.is_file() else None


def _zip_member(archive: ZipFile, video_id: str, filename: str) -> str | None:
    expected = f"keyframes/{video_id}/{filename}"
    try:
        archive.getinfo(expected)
        return expected
    except KeyError:
        # Be tolerant of ZIPs that capitalize the top-level folder or IDs.
        suffix = f"/{video_id}/{filename}".casefold()
        return next(
            (name for name in archive.namelist() if name.casefold().endswith(suffix)),
            None,
        )


def resolve_keyframe_path(keyframe_path: str | Path | None) -> Path | None:
    """Return an existing image path, lazily extracting one ZIP member if needed."""
    if not keyframe_path:
        return None

    normalized = str(keyframe_path).replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_file():
        return candidate

    project_root = Path.cwd()
    probes = (
        project_root / candidate,
        project_root / "data" / candidate,
        project_root / "data" / "raw" / candidate,
        project_root / "data" / "raw" / "Keyframes" / candidate,
        project_root / "data" / "raw" / "Keyframes" / candidate.parent.name / candidate.name,
    )
    for probe in probes:
        if probe.is_file():
            return probe

    video_id = candidate.parent.name
    filename = candidate.name
    keyframes_root = project_root / "data" / "raw" / "Keyframes"
    archive_name = _archive_name(video_id)
    if archive_name is not None:
        extracted_group = keyframes_root / Path(archive_name).stem
        for probe in (
            extracted_group / "keyframes" / video_id / filename,
            extracted_group / video_id / filename,
        ):
            if probe.is_file():
                return probe
    archive_path = _archive_for(video_id, keyframes_root)
    if archive_path is None:
        return None

    try:
        archive = _open_zip(str(archive_path))
        member = _zip_member(archive, video_id, filename)
        if member is None:
            return None
        cache_path = (
            project_root / "data" / "processed" / "keyframe_cache"
            / archive_path.stem / video_id / filename
        )
        if cache_path.is_file() and cache_path.stat().st_size > 0:
            return cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with archive.open(member) as source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        temporary.replace(cache_path)
        return cache_path
    except (OSError, BadZipFile, KeyError):
        return None

