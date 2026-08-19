from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from aic2026.data_platform import inspect_official_assets
from aic2026.models import FrameRecord
from .manifest import (
    VideoMetadata,
    _frame_id,
    _load_metadata_keywords,
    _load_object_labels,
    _metadata,
    load_manifest,
)

logger = logging.getLogger(__name__)


def _load_keyframe_frame_ids(map_root: Path | None, video_id: str) -> list[int] | None:
    """Đọc CSV map-keyframes → list frame_idx (frame thật của video) theo thứ tự ordinal.

    CSV có cột: n, pts_time, fps, frame_idx. `frame_idx` = frame index gốc trong video,
    thứ tự hàng = ordinal keyframe (1-based). Đây là nguồn đáng tin cậy hơn metadata
    JSON (metadata BTC không có frame_indices). Trả về None nếu file CSV không có.
    """
    if map_root is None:
        return None
    csv_path = map_root / "map-keyframes" / f"{video_id}.csv"
    if not csv_path.exists():
        return None
    try:
        with csv_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [int(r["frame_idx"]) for r in reader if r.get("frame_idx")]
    except (OSError, ValueError, KeyError):
        return None


# Ngưỡng score tối thiểu để giữ lại object entity (detection_scores).
# Object JSON chứa đến 100 boxes/frame với score rất thấp (0.006...); giữ
# entity có score thấp sẽ làm BM25 nhiễu. 0.3 là mốc giữ được entity rõ rệt.
_OBJECT_SCORE_THRESHOLD = 0.3
# Ngưỡng "có vật thể rõ": nếu KHÔNG có entity nào đạt ngưỡng này, coi frame ảnh
# mờ / không có vật thể đáng tin → trả [] (rỗng) thay vì giữ entity yếu.
_OBJECT_PRESENT_THRESHOLD = 0.4


def _load_object_labels(object_path: Path | None) -> list[str]:
    """Đọc object JSON (dict chứa mảng song song) → entity list có score >= threshold.

    File object BTC là **dict** dạng::
        {"detection_class_entities": ["Tower", "Skyscraper", ...],
         "detection_scores": ["0.79...", "0.68...", ...],
         "detection_boxes": [...], "detection_class_labels": [...]}

    Code cũ giả định là list-of-dict `{label:...}` → vì lặp `for x in dict`
    chỉ lặp key (str) → `isinstance(x, dict)` luôn False → `object_labels`
    LUÔN rỗng → BM25 corpus rỗng → chỉ chạy CLIP vector. Sửa parser để lấy
    `detection_class_entities` có score >= ngưỡng.
    """
    if not object_path or not object_path.exists():
        return []
    try:
        data = json.loads(object_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    entities = data.get("detection_class_entities") or []
    scored = data.get("detection_scores") or []
    if not entities:
        return []
    # Ép kiểu scores; nếu thiếu thì coi tất cả = 1.0 (giữ lại).
    scores: list[float] = []
    for s in scored:
        try:
            scores.append(float(s))
        except (TypeError, ValueError):
            scores.append(1.0)
    # Dò đủ độ dài: nếu scores ngắn hơn entities, pad = keep lại.
    keep: list[str] = []
    has_strong = False
    for i, ent in enumerate(entities):
        sc = scores[i] if i < len(scores) else 1.0
        if sc >= _OBJECT_SCORE_THRESHOLD:
            keep.append(str(ent))
        if sc >= _OBJECT_PRESENT_THRESHOLD:
            has_strong = True
    if not has_strong:
        return []
    return sorted(set(keep))


def _expected_frame_indices(metadata_dir: Path | None, video_id: str) -> list[int] | None:
    """Return the ground-truth frame-index list for a video, if metadata has it.

    BTC metadata JSON may carry frame_indices / keyframe_indices / frames so that
    keyframe ordinals map to original video frame indices.
    """
    meta, _ = _metadata(metadata_dir, video_id)
    for key in ("frame_indices", "keyframe_indices", "frames"):
        values = meta.get(key)
        if isinstance(values, list):
            result: list[int] = []
            for v in values:
                if isinstance(v, int):
                    result.append(v)
                elif isinstance(v, dict):
                    inner = v.get("frame_id", v.get("index"))
                    if isinstance(inner, int):
                        result.append(inner)
            return result
    return None


def probe_official_features(
    features_path: Path,
    manifest_path: Path | None = None,
    metadata_dir: Path | None = None,
    raw_dir: Path | None = None,
) -> dict:
    """Inspect the official BTC CLIP `.npy` and how its rows relate to the manifest.

    Because the competition only documents that vectors are "in ascending keyframe
    order", this probe reports dimensionality, count, per-video frame counts from
    the manifest, and the first/last frame index of each video so the row order can
    be reverse-engineered before `build_official_index` is called.
    """
    features_path = Path(features_path)
    vectors = np.load(features_path, mmap_mode="r")
    result: dict = {
        "features_path": str(features_path),
        "dim": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
    }

    per_video: dict[str, dict] = {}
    manifest_path = Path(manifest_path) if manifest_path else None
    if manifest_path is not None and manifest_path.exists():
        records = load_manifest(manifest_path)
        result["manifest_count"] = len(records)
        for record in records:
            slot = per_video.setdefault(record.video_id, {"count": 0, "first_frame": None, "last_frame": None})
            slot["count"] += 1
            slot["first_frame"] = record.frame_id if slot["first_frame"] is None else min(slot["first_frame"], record.frame_id)
            slot["last_frame"] = max(slot["last_frame"] or 0, record.frame_id)
        result["per_video"] = per_video
        result["video_ids_in_order"] = list(per_video)
    elif raw_dir is not None:
        # Fallback: derive per-video counts from the keyframe folders directly.
        assets = inspect_official_assets(Path(raw_dir))
        if assets.keyframes is not None:
            for image in sorted([*assets.keyframes.rglob("*.jpg"), *assets.keyframes.rglob("*.png")]):
                video_id = image.parent.name
                slot = per_video.setdefault(video_id, {"count": 0, "first_frame": None, "last_frame": None})
                slot["count"] += 1
            result["per_video"] = per_video
            result["video_ids_in_order"] = list(per_video)

    if metadata_dir is not None:
        metadata_dir = Path(metadata_dir)
        sample = {}
        for video_id in (result.get("video_ids_in_order") or [])[:3]:
            indices = _expected_frame_indices(metadata_dir, video_id)
            if indices:
                sample[video_id] = {"n_expected": len(indices), "first": indices[0], "last": indices[-1]}
        result["metadata_frame_indices_sample"] = sample

    if raw_dir is not None:
        assets = inspect_official_assets(Path(raw_dir))
        result["detected"] = {
            "Videos": str(assets.videos) if assets.videos else None,
            "Keyframes": str(assets.keyframes) if assets.keyframes else None,
            "Objects": str(assets.objects) if assets.objects else None,
            "Metadata": str(assets.metadata) if assets.metadata else None,
            "CLIP_features": [str(p) for p in assets.clip_features],
        }
        if assets.clip_features:
            result["clip_feature_file_count"] = len(assets.clip_features)
            result["clip_feature_total_bytes"] = sum(p.stat().st_size for p in assets.clip_features)

    return result


def build_official_index(
    raw_dir: Path,
    features_path: Path,
    output_manifest: Path,
    output_features: Path,
    keyframes_dir: Path | None = None,
    objects_dir: Path | None = None,
    metadata_dir: Path | None = None,
) -> int:
    """Build a manifest + aligned `.npy` from the official BTC CLIP features.

    ``features_path`` may be either a single ``*.npy`` file (the legacy single-
    file layout) or a **directory** containing multiple ``*.npy`` files — one
    per video (the multi-file layout, e.g. ``data/raw/CLIP features``).  When a
    directory is given its ``.npy`` children are discovered, loaded in sorted
    (video-id) order, and concatenated along axis 0 into one matrix.

    **CLIP features are the authoritative source** (each row = one frame of one
    video).  Keyframe *images* are optional: if a video has downloaded keyframe
    images, the matching image path is attached to each record for gallery/VLM;
    if not (e.g. only 29/873 videos have images locally), ``keyframe_path`` is
    left empty and retrieval still runs on the CLIP vectors.  We therefore do
    **not** require the image count to equal the feature count — a mismatch only
    triggers a warning, never a crash.
    """
    raw_dir, features_path = Path(raw_dir), Path(features_path)
    output_manifest, output_features = Path(output_manifest), Path(output_features)
    assets = inspect_official_assets(raw_dir)

    keyframes_root = keyframes_dir and Path(keyframes_dir) or assets.keyframes
    if keyframes_root is None or not keyframes_root.exists():
        raise FileNotFoundError(f"Không tìm thấy Keyframes dưới {raw_dir}")
    metadata_root = metadata_dir and Path(metadata_dir) or assets.metadata
    objects_root = objects_dir and Path(objects_dir) or assets.objects

    # Load features: accept either a single .npy file or a directory of .npy files.
    feature_paths: list[Path]
    features_path = Path(features_path)
    if features_path.is_dir():
        discovered = tuple(sorted(p for p in features_path.rglob("*.npy")))
        if not discovered:
            raise FileNotFoundError(f"Không tìm thấy file .npy nào trong thư mục {features_path}")
        feature_paths = list(discovered)
        logger.info("Loaded %d CLIP feature files from directory %s", len(feature_paths), features_path)
    else:
        feature_paths = [features_path]

    # Load and concatenate all feature files. Each .npy is expected to contain
    # the CLIP vectors for one video's keyframes in ascending keyframe order.
    arrays: list[np.ndarray] = []
    for path in tqdm(feature_paths, desc="Load CLIP .npy", unit="video", smoothing=0.05):
        arr = np.load(path, mmap_mode="r")
        arrays.append(np.asarray(arr, dtype=np.float32))
        logger.debug("Loaded %s: shape %s", path.name, arr.shape)
    vectors = np.concatenate(arrays, axis=0)

    # CLIP features are the authoritative frame source: each .npy file is one
    # video, each row is one keyframe in ascending order. We iterate video by
    # video (in the sorted .npy order) and emit a record per row. Keyframe
    # *images* are matched opportunistically — if `Keyframes/<video>/<ord>.jpg`
    # exists we attach it, otherwise keyframe_path stays empty (retrieval still
    # works on the CLIP vectors). No crash when images are fewer than vectors.
    if keyframes_root is not None and keyframes_root.exists():
        image_paths = sorted([*keyframes_root.rglob("*.jpg"), *keyframes_root.rglob("*.png")])
    else:
        image_paths = []
    n_images = len(image_paths)
    if n_images != int(vectors.shape[0]):
        logger.warning(
            "Số keyframe ảnh (%d) khác tổng số feature CLIP BTC (%d). "
            "Đây là BÌNH THƯỜNG nếu chưa tải đủ ảnh keyframe (vd chỉ 29/873 video có ảnh): "
            "frame thiếu ảnh sẽ có keyframe_path rỗng, retrieval vẫn chạy bình thường trên CLIP vector.",
            n_images,
            int(vectors.shape[0]),
        )

    # Pre-index existing keyframe images by (video_id, ordinal) for O(1) lookup.
    image_by_slot: dict[tuple[str, int], Path] = {}
    for image in image_paths:
        image_by_slot.setdefault((image.parent.name, _frame_id(image)), image)

    # Pre-index object filenames per video (one listdir per video instead of one
    # stat per frame) to keep the manifest build fast over 177k+ frames.
    object_files_by_video: dict[str, set[str]] = {}
    if objects_root is not None and objects_root.exists():
        for video_dir in objects_root.iterdir():
            if video_dir.is_dir():
                object_files_by_video[video_dir.name] = {p.stem for p in video_dir.glob("*.json")}

    # Pre-index map-keyframes CSV root (ánh xạ keyframe ordinal → frame_idx gốc).
    map_root = assets.map_keyframes if hasattr(assets, "map_keyframes") else None

    # Pre-load and cache video metadata once per video (like manifest.py)
    video_meta_cache: dict[str, VideoMetadata] = {}
    for fpath in feature_paths:
        video_id = fpath.stem
        if video_id not in video_meta_cache:
            meta, metadata_path = _metadata(metadata_root, video_id)
            title = meta.get("title")
            description = meta.get("description")
            keywords = _load_metadata_keywords(meta)
            obj_files = object_files_by_video.get(video_id, set())
            # Extract frame_indices for frame_id mapping
            frame_indices = None
            for key in ("frame_indices", "keyframe_indices", "frames"):
                values = meta.get(key)
                if isinstance(values, list):
                    frame_indices = []
                    for v in values:
                        if isinstance(v, int):
                            frame_indices.append(v)
                        elif isinstance(v, dict):
                            inner = v.get("frame_id", v.get("index"))
                            if isinstance(inner, int):
                                frame_indices.append(inner)
                    break
            video_meta_cache[video_id] = VideoMetadata(
                video_id=video_id,
                title=title,
                description=description,
                metadata_keywords=keywords,
                metadata_path=str(metadata_path) if metadata_path else None,
                object_files=obj_files,
                frame_indices=frame_indices,
            )

    # GHI TĂNG DẦN (streaming) vào manifest: mỗi video xử lý xong là ghi ngay
    # từng dòng JSONL và flush. Mục đích: nếu chạy 2 tiếng mà bị ngắt (lỗi điện,
    # Ctrl+C, crash) thì file vẫn chứa dữ liệu hợp lệ của những video ĐÃ đọc —
    # không mất trắng, và user có thể mở file để theo dõi tiến độ bất cứ lúc nào
    # ("ghi dô file trước đã không break mất").
    total_videos = len(feature_paths)
    written_records = 0
    empty_object_videos = 0  # số video mà TẤT CẢ frame đều rỗng object (ảnh mờ)
    by_video_count: dict[str, int] = {}
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as mf:
        video_bar = tqdm(feature_paths, desc="Build manifest", unit="video", smoothing=0.05)
        for vi, fpath in enumerate(video_bar):
            video_id = fpath.stem  # e.g. "L21_V001"
            arr = arrays[feature_paths.index(fpath)]  # reuse already-loaded matrix
            n_frames = arr.shape[0]
            # Ưu tiên CSV map-keyframes (frame_idx chuẩn), fallback metadata.json.
            csv_frames = _load_keyframe_frame_ids(map_root, video_id)
            expected = csv_frames  # VideoMetadata already has frame_indices from metadata
            vmeta = video_meta_cache[video_id]
            # object_files: pre-indexed tên file (stem) để lookup nhanh.
            obj_files = object_files_by_video.get(video_id)
            video_empty = 0  # frame rỗng object trong video này
            for ordinal_index in range(n_frames):
                ordinal = ordinal_index + 1  # 1-based keyframe ordinal
                # Use cached frame_indices (from metadata) first, then CSV map, then ordinal
                if vmeta.frame_indices and ordinal_index < len(vmeta.frame_indices):
                    frame_id = vmeta.frame_indices[ordinal_index]
                elif expected and ordinal_index < len(expected):
                    frame_id = expected[ordinal_index]
                else:
                    frame_id = ordinal
                image = image_by_slot.get((video_id, ordinal))
                keyframe_path = str(image) if image is not None else ""
                single = f"{ordinal:03d}"
                object_path = None
                labels: list[str] = []
                if obj_files and single in obj_files:
                    object_path = objects_root / video_id / f"{single}.json"
                    labels = _load_object_labels(object_path)
                elif objects_root is not None:
                    # Fallback: thử đường dẫn .json trực tiếp (filename khớp ordinal).
                    # Sửa bug UnboundLocalError 'single': tính single trước if/else.
                    candidate = objects_root / video_id / f"{single}.json"
                    if candidate.exists():
                        object_path = candidate
                        labels = _load_object_labels(object_path)
                if not labels:
                    video_empty += 1
                # Ghi NGAY frame này ra file (streaming) thay vì gom RAM rồi ghi cuối.
                rec = FrameRecord(
                    vector_id=written_records,
                    clip_feature_index=written_records,
                    video_id=video_id,
                    frame_id=frame_id,
                    keyframe_path=keyframe_path,
                    # object_labels only — video-level text moved to video_metadata.jsonl
                    # (see _emit_video_metadata below) to avoid per-frame duplication.
                    object_labels=[label for label in labels if label],
                    object_path=object_path,
                    metadata_path=vmeta.metadata_path,
                )
                mf.write(rec.model_dump_json() + "\n")
                written_records += 1
            by_video_count[video_id] = n_frames
            if video_empty == n_frames:
                empty_object_videos += 1
            mf.flush()  # đẩy xuống đĩa ngay — an toàn nếu bị ngắt
            video_bar.set_postfix(frames=written_records)
            # Log tiến độ LIÊN TỤC (ra file + console) mỗi 25 video.
            if (vi + 1) % 25 == 0 or (vi + 1) == total_videos:
                logger.info(
                    "Tiến độ: %d/%d video | %d frame đã ghi | video ảnh-mờ (rỗng object): %d",
                    vi + 1, total_videos, written_records, empty_object_videos,
                )

    for video_id in sorted(by_video_count):
        expected = _expected_frame_indices(metadata_root, video_id)
        if expected and len(expected) != by_video_count[video_id]:
            logger.warning(
                "video %s: metadata lists %d frame indices but %d feature rows found; "
                "frame_id mapping may be wrong",
                video_id,
                len(expected),
                by_video_count[video_id],
            )

    # The official .npy is documented as one row per keyframe in ascending
    # keyframe order. Both the manifest rows and the .npy rows are produced in
    # that natural (sorted video, ascending ordinal) order, so row i of the
    # matrix corresponds to manifest record i. The streaming loop above already
    # wrote every record to disk incrementally; here we only verify the count
    # matches — reordering blindly without knowing the actual .npy video order
    # would risk silently misaligning every frame.
    if written_records != int(vectors.shape[0]):
        raise ValueError(
            f"Manifest records ({written_records}) do not match feature rows "
            f"({int(vectors.shape[0])})."
        )
    aligned = np.asarray(vectors, dtype=np.float32).copy()

    output_features.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_features, aligned)

    # Emit per-video metadata JSONL once per video (NOT per frame). Video-level
    # text (title/description/keywords) used to be duplicated inside every
    # FrameRecord row, which inflated the manifest ~10x and broke BM25 IDF.
    # Downstream code (RetrievalPipeline.filter_videos_by_metadata, BM25Index)
    # now reads this compact file instead.
    from aic2026.retrieval.video_metadata import VideoMetadataStore
    from aic2026.retrieval.video_metadata import VideoMetaRecord

    vm_records: list[VideoMetaRecord] = []
    for vmeta in video_meta_cache.values():
        # Normalize path relative to raw_dir so the file is portable.
        rel_meta_path = vmeta.metadata_path
        if rel_meta_path:
            try:
                rel_meta_path = str(Path(rel_meta_path).relative_to(raw_dir))
            except ValueError:
                pass  # keep absolute path
        vm_records.append(
            VideoMetaRecord(
                video_id=vmeta.video_id,
                title=vmeta.title,
                description=vmeta.description,
                metadata_keywords=list(vmeta.metadata_keywords),
                metadata_path=rel_meta_path,
            )
        )
    vm_store = VideoMetadataStore(vm_records)
    # Write next to the manifest as video_metadata.jsonl.
    vm_path = output_manifest.parent / "video_metadata.jsonl"
    vm_store.save(vm_path)
    logger.info("Wrote %d video-metadata records to %s", len(vm_records), vm_path)

    return written_records


def build_from_clip(
    raw_dir: Path,
    features_path: Path,
    output_manifest: Path,
    output_features: Path,
    keyframes_dir: Path | None = None,
    objects_dir: Path | None = None,
    metadata_dir: Path | None = None,
) -> int:
    """Build a manifest + aligned `.npy` from the official BTC CLIP features ONLY.

    Unlike :func:`build_official_index`, this does NOT require the Keyframes folder
    to exist or to match the feature count. Keyframes are only used for display and
    VLM Q&A, so they are optional here:

    - ``video_id`` is derived from each ``.npy`` file name (``L21_V001.npy`` -> ``L21_V001``).
    - ``frame_id`` is the 1-based ordinal within the video (or mapped via
      ``metadata.frame_indices`` when present).
    - ``keyframe_path`` uses the real image when a matching Keyframe exists, otherwise
      an empty string (so the gallery / VLM simply have no image for that video).

    ``features_path`` may be a single ``.npy`` file or a **directory** of per-video
    ``.npy`` files (the multi-file layout, e.g. ``data/raw/CLIP features``). The
    matrix rows are kept in the natural (sorted video, ascending keyframe) order,
    which matches the manifest records, so retrieval aligns without reordering.
    """
    raw_dir, features_path = Path(raw_dir), Path(features_path)
    output_manifest, output_features = Path(output_manifest), Path(output_features)
    assets = inspect_official_assets(raw_dir)

    keyframes_root = keyframes_dir and Path(keyframes_dir) or assets.keyframes
    metadata_root = metadata_dir and Path(metadata_dir) or assets.metadata
    objects_root = objects_dir and Path(objects_dir) or assets.objects

    # Load features: accept either a single .npy file or a directory of .npy files.
    feature_paths: list[Path]
    features_path = Path(features_path)
    if features_path.is_dir():
        discovered = tuple(sorted(p for p in features_path.rglob("*.npy")))
        if not discovered:
            raise FileNotFoundError(f"Không tìm thấy file .npy nào trong thư mục {features_path}")
        feature_paths = list(discovered)
        logger.info("Loaded %d CLIP feature files from directory %s", len(feature_paths), features_path)
    else:
        feature_paths = [features_path]

    # Pre-load per-video keyframe paths (optional) for display / VLM.
    keyframe_by_video: dict[str, list[Path]] = {}
    if keyframes_root is not None and keyframes_root.exists():
        for image in sorted([*keyframes_root.rglob("*.jpg"), *keyframes_root.rglob("*.png")]):
            keyframe_by_video.setdefault(image.parent.name, []).append(image)

    arrays: list[np.ndarray] = []
    for fi, path in enumerate(feature_paths):
        arr = np.load(path)  # direct read (faster than mmap for many small files on Windows)
        arrays.append(np.asarray(arr, dtype=np.float32))
        if fi % 100 == 0:
            print(f"[build_from_clip] loaded {fi}/{len(feature_paths)}: {path.name} {arr.shape}", flush=True)
    vectors = np.concatenate(arrays, axis=0)

    # Pre-index object filenames per video (one listdir per video instead of one
    # stat per frame) to keep the manifest build fast over 177k+ frames.
    object_files_by_video: dict[str, set[str]] = {}
    if objects_root is not None and objects_root.exists():
        for video_dir in objects_root.iterdir():
            if video_dir.is_dir():
                object_files_by_video[video_dir.name] = {
                    p.stem for p in video_dir.glob("*.json")
                }

    # Pre-load and cache video metadata once per video.
    video_meta_cache: dict[str, VideoMetadata] = {}
    for fi, path in enumerate(feature_paths):
        video_id = path.stem
        if video_id not in video_meta_cache:
            meta, metadata_path = _metadata(metadata_root, video_id)
            title = meta.get("title")
            description = meta.get("description")
            keywords = _load_metadata_keywords(meta)
            obj_files = object_files_by_video.get(video_id, set())
            # Extract frame_indices for frame_id mapping
            frame_indices = None
            for key in ("frame_indices", "keyframe_indices", "frames"):
                values = meta.get(key)
                if isinstance(values, list):
                    frame_indices = []
                    for v in values:
                        if isinstance(v, int):
                            frame_indices.append(v)
                        elif isinstance(v, dict):
                            inner = v.get("frame_id", v.get("index"))
                            if isinstance(inner, int):
                                frame_indices.append(inner)
                    break
            video_meta_cache[video_id] = VideoMetadata(
                video_id=video_id,
                title=title,
                description=description,
                metadata_keywords=keywords,
                metadata_path=str(metadata_path) if metadata_path else None,
                object_files=obj_files,
                frame_indices=frame_indices,
            )

    # Build one manifest record per feature row, in the same (sorted file, ascending
    # keyframe) order the matrix rows are in.
    rows: list[tuple[int, str, int, str, list[str], str | None]] = []
    vector_id = 0
    for fi, path in enumerate(feature_paths):
        video_id = path.stem  # e.g. L21_V001.npy -> L21_V001
        n = int(arrays[fi].shape[0])
        vmeta = video_meta_cache[video_id]
        kf_images = keyframe_by_video.get(video_id, [])
        obj_files = object_files_by_video.get(video_id)
        if fi % 100 == 0:
            logger.info("build_from_clip: %d/%d videos (%s, %d frames)", fi, len(feature_paths), video_id, n)
        for ordinal_index in range(n):
            # Use cached frame_indices from metadata, fallback to ordinal
            frame_id = vmeta.frame_indices[ordinal_index] if vmeta.frame_indices and ordinal_index < len(vmeta.frame_indices) else ordinal_index + 1
            keyframe_path = str(kf_images[ordinal_index]) if ordinal_index < len(kf_images) else ""
            labels: list[str] = []
            object_path = None
            if obj_files:
                single = f"{ordinal_index + 1:0{len(str(max(n, 1)))}d}"
                if single in obj_files:
                    single_path = objects_root / video_id / f"{single}.json"
                    try:
                        objects = json.loads(single_path.read_text(encoding="utf-8"))
                        labels = sorted({str(x.get("label", x.get("name", ""))) for x in objects if isinstance(x, dict)})
                        object_path = str(single_path) if single_path.exists() else None
                    except (OSError, json.JSONDecodeError):
                        pass
            rows.append(
                (
                    vector_id,
                    video_id,
                    frame_id,
                    keyframe_path,
                    labels,
                    object_path,
                )
            )
            vector_id += 1

    # Cross-check against available keyframes when present (informational only).
    if keyframes_root is not None and keyframes_root.exists():
        for video_id, kf_images in keyframe_by_video.items():
            npy_rows = next(
                (int(arrays[fi].shape[0]) for fi, p in enumerate(feature_paths) if p.stem == video_id),
                0,
            )
            if npy_rows and len(kf_images) != npy_rows:
                logger.warning(
                    "video %s: %d keyframe images but %d CLIP rows; gallery will be partial",
                    video_id,
                    len(kf_images),
                    npy_rows,
                )

    if len(rows) != int(vectors.shape[0]):
        raise ValueError(
            f"Manifest records ({len(rows)}) do not match feature rows ({int(vectors.shape[0])})."
        )
    aligned = np.asarray(vectors, dtype=np.float32).copy()

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_features.parent.mkdir(parents=True, exist_ok=True)
    records = [
        FrameRecord(
            vector_id=vid,
            clip_feature_index=vid,
            video_id=video_id,
            frame_id=frame_id,
            keyframe_path=keyframe_path,
            object_labels=labels,
            metadata_keywords=video_meta_cache[video_id].metadata_keywords,
            title=video_meta_cache[video_id].title,
            description=video_meta_cache[video_id].description,
            metadata_path=video_meta_cache[video_id].metadata_path,
            object_path=object_path,
            video_path=None,
        )
        for vid, video_id, frame_id, keyframe_path, labels, object_path in rows
    ]
    output_manifest.write_text("\n".join(record.model_dump_json() for record in records) + "\n", encoding="utf-8")
    np.save(output_features, aligned)
    return len(records)
