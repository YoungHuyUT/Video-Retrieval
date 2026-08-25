from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile


@dataclass(frozen=True)
class OrganizerAssets:
    """Những tài nguyên BTC AIC 2026 công bố; không suy đoán từ video khi đã có asset."""
    videos: Path | None
    keyframes: Path | None
    objects: Path | None
    clip_features: tuple[Path, ...]
    metadata: Path | None
    map_keyframes: Path | None = None
    support_archives: tuple[Path, ...] = ()

    def missing_required(self) -> list[str]:
        required = {"Videos": self.videos, "Keyframes": self.keyframes, "CLIP features": self.clip_features, "Metadata": self.metadata}
        return [name for name, asset in required.items() if not asset]


def _folder(root: Path, expected: str) -> Path | None:
    for item in root.iterdir() if root.exists() else []:
        if item.is_dir() and item.name.casefold().replace(" ", "") == expected.casefold().replace(" ", ""):
            return item
    return None


def _folder_with_fallback(root: Path, expected: str, fallback_roots: tuple[Path, ...] = ()) -> Path | None:
    for candidate in (root, *fallback_roots):
        if candidate is None:
            continue
        found = _folder(candidate, expected)
        if found is not None:
            return found
    return None


def inspect_official_assets(raw_dir: Path) -> OrganizerAssets:
    raw_dir = Path(raw_dir)
    processed_root = None
    extracted_root = None
    if raw_dir.name.casefold() not in ("processed", "extracted"):
        processed_root = (raw_dir.parent / "processed") if raw_dir.parent.exists() else None
        extracted_root = (raw_dir.parent / "extracted") if raw_dir.parent.exists() else None
    fallbacks = tuple(r for r in (processed_root, extracted_root) if r is not None)
    clip_root = (
        _folder_with_fallback(raw_dir, "CLIP features", fallbacks)
        or _folder_with_fallback(raw_dir, "CLIP_features", fallbacks)
        or _folder_with_fallback(raw_dir, "clip_features", fallbacks)
        or _folder_with_fallback(raw_dir, "clip-features-32", fallbacks)
        or _folder_with_fallback(raw_dir, "clip-features", fallbacks)
    )
    if clip_root is not None and clip_root.exists():
        features = tuple(sorted(clip_root.glob("*.npy")))
    else:
        features = tuple(sorted(raw_dir.glob("*.npy")))
    zip_roots = (raw_dir, *fallbacks, raw_dir.parent if raw_dir.parent.exists() else None)
    archives_list = []
    for zr in zip_roots:
        if zr and zr.exists():
            archives_list.extend(path for path in zr.glob("*.zip") if path.name.casefold().startswith(("clip-features", "map-keyframes", "media-info", "objects")))
    archives = tuple(sorted(set(archives_list)))
    # map-keyframes CSV: ưu tiên thư mục được giải nén support (data/raw/map-keyframes.../map-keyframes)
    # hoặc data/downloads, fallback tìm bất kỳ thư mục con nào mang tên "map-keyframes".
    map_root = (
        _folder_with_fallback(raw_dir.parent, "map-keyframes-aic25-b1", (raw_dir, *fallbacks))
        or _folder_with_fallback(raw_dir, "map-keyframes-aic25-b1", fallbacks)
        or _folder_with_fallback(raw_dir, "map-keyframes", fallbacks)
        or _folder_with_fallback(raw_dir.parent, "map-keyframes", fallbacks)
    )

    return OrganizerAssets(
        videos=_folder_with_fallback(raw_dir, "Videos", fallbacks),
        keyframes=_folder_with_fallback(raw_dir, "Keyframes", fallbacks),
        objects=_folder_with_fallback(raw_dir, "Objects", fallbacks),
        clip_features=features,
        metadata=(
            _folder_with_fallback(raw_dir, "Metadata", fallbacks)
            or _folder_with_fallback(raw_dir, "media-info", fallbacks)
            or _folder_with_fallback(raw_dir, "media_info", fallbacks)
            or _folder_with_fallback(raw_dir, "media-info-aic25-b1", fallbacks)
        ),
        map_keyframes=map_root,
        support_archives=archives,
    )


def extract_support_archives(archives_dir: Path, destination: Path) -> list[Path]:
    """Giải nén an toàn các ZIP hỗ trợ BTC, chặn path traversal trong archive."""
    archives_dir, destination = Path(archives_dir), Path(destination)
    accepted = ("clip-features", "map-keyframes", "media-info", "objects")
    archives = sorted(path for path in archives_dir.glob("*.zip") if path.stem.casefold().startswith(accepted))
    if not archives:
        raise FileNotFoundError(f"Không tìm thấy ZIP hỗ trợ AIC trong {archives_dir}")
    extracted: list[Path] = []
    destination.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        folder = destination / archive.stem
        folder.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive) as zip_file:
            for member in zip_file.infolist():
                target = (folder / member.filename).resolve()
                if not target.is_relative_to(folder.resolve()):
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            zip_file.extractall(folder)
        extracted.append(folder)
    return extracted
