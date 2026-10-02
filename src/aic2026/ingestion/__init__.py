from __future__ import annotations

from pathlib import Path

from .manifest import build_manifest, load_manifest

__all__ = ["build_manifest", "load_manifest", "resolve_feature_sources"]

# OpenCLIP ViT-B/32 (512-dim) — thay thế SigLIP2 vì text encoder broken.
OPENCLIP_DIR = Path("data/processed/openclip")
OPENCLIP_MANIFEST = OPENCLIP_DIR / "manifest_openclip.jsonl"
OPENCLIP_FEATURES = OPENCLIP_DIR / "features_openclip.npy"

# SigLIP2 (768-dim) — canonical merged batch 1 + batch 2 retrieval source.
SIGLIP2_DIR = Path("data/processed/siglip2")
SIGLIP2_MANIFEST = SIGLIP2_DIR / "manifest_siglip2.jsonl"
SIGLIP2_FEATURES = SIGLIP2_DIR / "features_siglip2.npy"

# Derived CLIP (512-dim) — legacy fallback cuối cùng.
DERIVED_MANIFEST = Path("data/processed/derived_manifest.jsonl")
DERIVED_FEATURES = Path("data/processed/derived_features.npy")


def resolve_feature_sources(
    manifest_path: str | Path | None = None,
    features_path: str | Path | None = None,
    *,
    prefer_official: bool = True,
) -> tuple[Path, Path]:
    """Chọn cặp (manifest, features) để nạp retrieval.

    Ưu tiên: SigLIP2 đã gộp (768-dim) > OpenCLIP (512-dim) > Derived.

    Lưu ý: dù dùng vector nào, manifest vẫn phải trỏ ``keyframe_path``
    đến ảnh keyframe vật lý (cho QA/VLM xem ảnh trả lời) — hàm này chỉ chọn
    nguồn feature/vector, không ảnh hưởng đến việc có ảnh hay không.
    """
    if manifest_path is not None and features_path is not None:
        return Path(manifest_path), Path(features_path)

    # ƯU TIÊN 1: canonical merged SigLIP2 pair used by the API.
    if SIGLIP2_MANIFEST.exists() and SIGLIP2_FEATURES.exists():
        return SIGLIP2_MANIFEST, SIGLIP2_FEATURES

    # ƯU TIÊN 2: OpenCLIP fallback, only if SigLIP2 is unavailable.
    if OPENCLIP_MANIFEST.exists() and OPENCLIP_FEATURES.exists():
        return OPENCLIP_MANIFEST, OPENCLIP_FEATURES

    # Fallback cuối: derived (legacy).
    if DERIVED_MANIFEST.exists() and DERIVED_FEATURES.exists():
        return DERIVED_MANIFEST, DERIVED_FEATURES

    # Nếu không có gì cả, trả OpenCLIP default để caller báo lỗi rõ.
    return OPENCLIP_MANIFEST, OPENCLIP_FEATURES
