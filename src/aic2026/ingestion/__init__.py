from __future__ import annotations

from pathlib import Path

from .manifest import build_manifest, load_manifest

__all__ = ["build_manifest", "load_manifest", "resolve_feature_sources"]

# Đường dẫn mặc định do `prepare-official` sinh ra từ CLIP features chính thức của
# BTC (vector có sẵn, không encode lại keyframe). Ưu tiên dùng cặp này khi tồn tại.
OFFICIAL_MANIFEST = Path("data/processed/official_manifest.jsonl")
OFFICIAL_FEATURES = Path("data/processed/official_features.npy")
# Cặp fallback do encode keyframe (OpenCLIP) sinh ra.
DERIVED_MANIFEST = Path("data/processed/derived_manifest.jsonl")
DERIVED_FEATURES = Path("data/processed/derived_features.npy")


def resolve_feature_sources(
    manifest_path: str | Path | None = None,
    features_path: str | Path | None = None,
    *,
    prefer_official: bool = True,
) -> tuple[Path, Path]:
    """Chọn cặp (manifest, features) để nạp retrieval.

    Mặc định ưu tiên CLIP features **chính thức của BTC** (``official_*.npy``/
    ``official_*.jsonl`` từ lệnh ``prepare-official``) nếu tồn tại — tức là dùng
    vector có sẵn của BTC thay vì encode lại keyframe. Chỉ khi cặp official không
    đủ (thiếu 1 trong 2 file) mới fallback sang cặp ``derived_*`` (do encode
    keyframe). Truyền ``manifest_path``/``features_path`` tường minh sẽ ghi đè cả
    hai (không auto-detect), tiện khi muốn ép dùng 1 nguồn cụ thể.

    Lưu ý: dù dùng vector CLIP sẵn có, manifest vẫn phải trỏ ``keyframe_path``
    đến ảnh keyframe vật lý (cho QA/VLM xem ảnh trả lời) — hàm này chỉ chọn
    nguồn feature/vector, không ảnh hưởng đến việc có ảnh hay không.
    """
    if manifest_path is not None and features_path is not None:
        return Path(manifest_path), Path(features_path)

    if prefer_official and OFFICIAL_MANIFEST.exists() and OFFICIAL_FEATURES.exists():
        return OFFICIAL_MANIFEST, OFFICIAL_FEATURES

    return DERIVED_MANIFEST, DERIVED_FEATURES
