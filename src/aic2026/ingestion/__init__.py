from __future__ import annotations

from pathlib import Path

from .asr_manifest import enrich_manifest_with_asr
from .manifest import build_manifest, load_manifest

__all__ = ["build_manifest", "enrich_manifest_with_asr", "load_manifest", "resolve_feature_sources"]

SIGLIP_MANIFEST = Path("data/processed/siglip_manifest.jsonl")
SIGLIP_FEATURES = Path("data/processed/siglip_features.npy")

OFFICIAL_MANIFEST = Path("data/processed/official_manifest.jsonl")
OFFICIAL_FEATURES = Path("data/processed/official_features.npy")

DERIVED_MANIFEST = Path("data/processed/derived_manifest.jsonl")
DERIVED_FEATURES = Path("data/processed/derived_features.npy")

ALT_DERIVED_MANIFEST = Path(r"D:\aichallenge\data\processed\derived_manifest.jsonl")
ALT_DERIVED_FEATURES = Path(r"D:\aichallenge\data\processed\derived_features.npy")


def resolve_feature_sources(
    manifest_path: str | Path | None = None,
    features_path: str | Path | None = None,
    *,
    prefer_official: bool = False,
) -> tuple[Path, Path]:
    if manifest_path is not None and features_path is not None:
        m = Path(manifest_path)
        f = Path(features_path)
        if not m.exists() and ALT_DERIVED_MANIFEST.exists() and m.name == ALT_DERIVED_MANIFEST.name:
            m = ALT_DERIVED_MANIFEST
        if not f.exists() and ALT_DERIVED_FEATURES.exists() and f.name == ALT_DERIVED_FEATURES.name:
            f = ALT_DERIVED_FEATURES
        return m, f

    if SIGLIP_MANIFEST.exists() and SIGLIP_FEATURES.exists():
        return SIGLIP_MANIFEST, SIGLIP_FEATURES

    if prefer_official and OFFICIAL_MANIFEST.exists() and OFFICIAL_FEATURES.exists():
        return OFFICIAL_MANIFEST, OFFICIAL_FEATURES

    if DERIVED_MANIFEST.exists() and DERIVED_FEATURES.exists():
        return DERIVED_MANIFEST, DERIVED_FEATURES

    if ALT_DERIVED_MANIFEST.exists() and ALT_DERIVED_FEATURES.exists():
        return ALT_DERIVED_MANIFEST, ALT_DERIVED_FEATURES

    return SIGLIP_MANIFEST, SIGLIP_FEATURES
