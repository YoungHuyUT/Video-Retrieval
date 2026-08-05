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
    if raw_dir.name.casefold() != "processed":
        processed_root = (raw_dir.parent / "processed") if raw_dir.parent.exists() else None
    clip_root = _folder_with_fallback(raw_dir, "CLIP features", (processed_root,)) or _folder_with_fallback(raw_dir, "CLIP_features", (processed_root,))
    features = tuple(sorted((clip_root or raw_dir).rglob("*.npy")))
    archives = tuple(sorted(path for path in raw_dir.rglob("*.zip") if path.name.casefold().startswith(("clip-features", "map-keyframes", "media-info", "objects"))))
    return OrganizerAssets(
        videos=_folder_with_fallback(raw_dir, "Videos", (processed_root,)),
        keyframes=_folder_with_fallback(raw_dir, "Keyframes", (processed_root,)),
        objects=_folder_with_fallback(raw_dir, "Objects", (processed_root,)),
        clip_features=features,
        metadata=_folder_with_fallback(raw_dir, "Metadata", (processed_root,)),
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
