from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from aic2026.evaluation import evaluate_query
from aic2026.ingestion import build_manifest, resolve_feature_sources
from aic2026.models import Candidate, GroundTruth, Query

logger = logging.getLogger(__name__)

app = typer.Typer(help="AIC 2026 retrieval toolkit")

@app.command()
def prepare(raw_dir: Path = typer.Option(Path("data/raw")), output: Path = typer.Option(Path("data/processed/manifest.jsonl"))) -> None:
    """Discover raw assets and create the frame manifest."""
    typer.echo(f"Wrote {build_manifest(raw_dir, output)} frame records to {output}")

@app.command("inspect-data")
def inspect_data(raw_dir: Path = typer.Option(Path("data/raw"))) -> None:
    """Kiểm tra 5 nhóm dữ liệu BTC: Video, Keyframes, Objects, CLIP features, Metadata."""
    from aic2026.data_platform import inspect_official_assets
    assets = inspect_official_assets(raw_dir)
    typer.echo(json.dumps({
        "Videos": str(assets.videos) if assets.videos else None,
        "Keyframes": str(assets.keyframes) if assets.keyframes else None,
        "Objects": str(assets.objects) if assets.objects else None,
        "CLIP_features": [str(item) for item in assets.clip_features],
        "Metadata": str(assets.metadata) if assets.metadata else None,
        "support_archives": [str(item) for item in assets.support_archives],
        "missing_required": assets.missing_required(),
    }, ensure_ascii=False, indent=2))

@app.command("unpack-support-assets")
def unpack_support_assets(
    archives_dir: Path = typer.Option(Path("data/downloads"), help="Nơi chứa ZIP BTC tải về"),
    destination: Path = typer.Option(Path("data/raw/support")),
) -> None:
    """Giải nén clip-features/map-keyframes/media-info/objects ZIP do BTC cấp."""
    from aic2026.data_platform import extract_support_archives
    extracted = extract_support_archives(archives_dir, destination)
    typer.echo(json.dumps({"extracted": [str(path) for path in extracted]}, ensure_ascii=False, indent=2))

@app.command("extract-keyframes")
def extract_keyframes(
    video: Path = typer.Option(..., exists=True, help="Video chính thức do BTC cung cấp"),
    keyframes_dir: Path = typer.Option(Path("data/processed/keyframes"), help="Nơi ghi keyframe đã lọc vào processed"),
    features_dir: Path = typer.Option(Path("data/processed/clip_features"), help="Thư mục output để ghi .npz/.npy embedding tự tạo (không phải thư mục CLIP features BTC đầu vào)"),
    cosine_threshold: float = typer.Option(0.985, min=0.0, max=1.0),
    batch_size: int = typer.Option(32, min=1),
    clip_pretrained: str = typer.Option("openai", help="Checkpoint OpenCLIP: openai (chuẩn BTC) hoặc laion2b_s32b_b79k"),
) -> None:
    """Trích toàn bộ frame, CLIP-embed, rồi loại near-duplicate bằng cosine similarity."""
    from aic2026.data_platform import OpenCLIPFrameEncoder, extract_deduplicated_keyframes
    image_encoder = OpenCLIPFrameEncoder(model_name="ViT-B-32", pretrained=clip_pretrained)
    report = extract_deduplicated_keyframes(video, keyframes_dir, features_dir, image_encoder, cosine_threshold, batch_size)
    typer.echo(json.dumps({"video_id": report.video_id, "decoded_frames": report.decoded_frames, "kept_frames": report.kept_frames, "keyframes": str(report.output_dir), "features": str(report.feature_path)}, ensure_ascii=False, indent=2))

@app.command("build-derived-index-input")
def build_derived_index_input(
    keyframes_dir: Path = typer.Option(Path("data/processed/keyframes"), help="Thư mục keyframe đã sinh ra ở processed"),
    features_dir: Path = typer.Option(Path("data/processed/clip_features"), help="Thư mục embedding đã sinh ra ở processed"),
    manifest: Path = typer.Option(Path("data/processed/derived_manifest.jsonl"), help="Manifest output ở processed"),
    features: Path = typer.Option(Path("data/processed/derived_features.npy"), help="Feature index output ở processed"),
) -> None:
    """Gộp embedding frame đã giữ thành manifest và `.npy` dùng trực tiếp cho FAISS/retrieval."""
    from aic2026.data_platform import build_derived_artifacts
    count = build_derived_artifacts(keyframes_dir, features_dir, manifest, features)
    typer.echo(f"Wrote {count} retained frame records to {manifest} and {features}")

@app.command("embed-keyframes")
def embed_keyframes(
    keyframes_dir: Path = typer.Option(Path("data/raw/Keyframes"), help="Thư mục keyframe BTC đầu vào, nên đặt ở data/raw/Keyframes"),
    features_dir: Path = typer.Option(Path("data/processed/clip_features"), help="Thư mục output để ghi .npz/.npy embedding tự tạo (không phải thư mục CLIP features BTC đầu vào)"),
    manifest: Path = typer.Option(Path("data/processed/derived_manifest.jsonl"), help="Manifest output ở processed"),
    features: Path = typer.Option(Path("data/processed/derived_features.npy"), help="Feature index output ở processed"),
    clip_pretrained: str = typer.Option("openai", help="Checkpoint OpenCLIP: openai (chuẩn BTC) hoặc laion2b_s32b_b79k"),
    batch_size: int = typer.Option(4, min=1, help="Số ảnh mỗi batch để giảm mức dùng RAM"),
    video_id: str | None = typer.Option(None, help="Chỉ encode 1 thư mục keyframe, ví dụ L21_V001"),
    with_chroma: bool = typer.Option(False, help="Sau khi build .npy/.manifest, build luôn Chroma collection tại --chroma-dir"),
    chroma_dir: Path = typer.Option(Path("data/indexes/chroma"), help="Thư mục lưu Chroma index (dùng khi --with-chroma)"),
    with_ocr: bool = typer.Option(False, help="Sau khi build manifest, chạy OCR enrich object_labels ngay (cần paddleocr)"),
    ocr_lang: str = typer.Option("vi", help="Ngôn ngữ OCR khi --with-ocr"),
) -> None:
    """Encode existing JPG keyframes on disk into .npz archives and build the retrieval index inputs."""
    from aic2026.data_platform import (
        OpenCLIPFrameEncoder,
        build_derived_artifacts,
        embed_existing_keyframes,
    )
    image_encoder = OpenCLIPFrameEncoder(model_name="ViT-B-32", pretrained=clip_pretrained)

    archives = embed_existing_keyframes(keyframes_dir, features_dir, image_encoder, batch_size=batch_size, video_id=video_id)
    count = build_derived_artifacts(keyframes_dir, features_dir, manifest, features, video_ids=[video_id] if video_id else None)
    typer.echo(f"Created {len(archives)} feature archives under {features_dir}; wrote {count} records to {manifest} and {features}")

    if with_ocr:
        from aic2026.ingestion import load_manifest
        from aic2026.models import FrameRecord
        from aic2026.qa.ocr import OCRTextExtractor

        extractor = OCRTextExtractor(lang=ocr_lang)
        extractor._ensure_loaded()
        if extractor._ocr is None:
            typer.echo("paddleocr chưa được cài — bỏ qua OCR (cài: uv sync --extra models, cộng paddlepaddle).")
        else:
            records = load_manifest(manifest)
            enriched: list[FrameRecord] = []
            for index, record in enumerate(records):
                candidates = [
                    Path(record.keyframe_path),
                    keyframes_dir / record.video_id / Path(record.keyframe_path).name,
                    keyframes_dir / record.keyframe_path,
                ]
                chosen = next((c for c in candidates if c.exists()), None)
                texts = extractor.extract(chosen) if chosen else []
                merged = list(dict.fromkeys([*(record.object_labels or []), *texts]))
                enriched.append(record.model_copy(update={"object_labels": merged}))
                if (index + 1) % 50 == 0:
                    typer.echo(f"OCR {index + 1}/{len(records)}")
            manifest.write_text("\n".join(r.model_dump_json() for r in enriched) + "\n", encoding="utf-8")
            typer.echo(f"OCR-enriched {len(enriched)} records IN-PLACE to {manifest}")

    if with_chroma:
        from aic2026.ingestion import load_manifest
        from aic2026.retrieval.factory import build_vector_index
        build_vector_index(features, load_manifest(manifest), backend="chroma", chroma_dir=chroma_dir)
        typer.echo(f"Chroma collection ready at {chroma_dir}")

@app.command("probe-official-features")
def probe_official_features(
    features: Path = typer.Option(..., help="File CLIP features chính thức BTC (.npy)"),
    manifest: Path | None = typer.Option(None, help="Manifest hiện có để so count/thứ tự (tùy chọn)"),
    metadata_dir: Path | None = typer.Option(None, help="Thư mục Metadata BTC"),
    raw_dir: Path = typer.Option(Path("data/raw"), help="Gốc dữ liệu để phát hiện 5 nhóm asset"),
) -> None:
    """Probe file CLIP .npy chính thức: dim/count, frame_indices từng video, thứ tự video."""
    from aic2026.ingestion.official_index import probe_official_features as _probe
    report = _probe(features, manifest, metadata_dir, raw_dir)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))

@app.command("prepare-official")
def prepare_official(
    features: Path = typer.Option(..., help="File CLIP features chính thức BTC (.npy)"),
    raw_dir: Path = typer.Option(Path("data/raw"), help="Gốc dữ liệu chứa Keyframes/Objects/Metadata"),
    output_manifest: Path = typer.Option(Path("data/processed/official_manifest.jsonl")),
    output_features: Path = typer.Option(Path("data/processed/official_features.npy")),
) -> None:
    """Build manifest + aligned .npy từ CLIP features BTC chính thức (thứ tự khớp keyframe)."""
    from aic2026.data_platform import inspect_official_assets
    from aic2026.ingestion.official_index import build_official_index
    # Fail fast if the map-keyframes CSV (nguồn frame_idx gốc của video) is
    # missing. Without it, `build_official_index` silently falls back to keyframe
    # ordinal as `frame_id`, and TRAKE would then emit out-of-range frames versus
    # the BTC ground-truth `ranges`. BTC Metadata for this corpus does not carry
    # `frame_indices` (verified on L21_V001), so the CSV is the only source of
    # correct coordinates. This guard prevents building a subtly-wrong manifest.
    assets = inspect_official_assets(raw_dir)
    if assets.map_keyframes is None:
        raise typer.BadParameter(
            "Không tìm thấy thư mục map-keyframes (CSV frame_idx gốc). "
            "TRAKE cần tọa độ frame gốc; thiếu CSV sẽ sinh manifest sai tọa độ. "
            "Hãy giải nén ZIP hỗ trợ BTC vào data/raw (vd map-keyframes-aic25-b1)."
        )
    count = build_official_index(raw_dir, features, output_manifest, output_features)
    typer.echo(f"Wrote {count} records to {output_manifest} and {output_features}")

@app.command("build-from-clip")
def build_from_clip_cmd(
    features: Path = typer.Option(..., help="File hoặc thư mục CLIP features BTC (.npy)"),
    raw_dir: Path = typer.Option(Path("data/raw"), help="Gốc dữ liệu chứa Objects/Metadata (Keyframes tùy chọn)"),
    output_manifest: Path = typer.Option(Path("data/processed/official_manifest.jsonl")),
    output_features: Path = typer.Option(Path("data/processed/official_features.npy")),
    keyframes_dir: Path | None = typer.Option(None, help="Thư mục Keyframes (tùy chọn, chỉ để hiển thị/VLM). Thiếu vẫn chạy được."),
    objects_dir: Path | None = typer.Option(None, help="Thư mục Objects (tùy chọn)"),
    metadata_dir: Path | None = typer.Option(None, help="Thư mục Metadata (tùy chọn)"),
) -> None:
    """Build manifest + .npy CHỈ từ CLIP features BTC (không bắt buộc có Keyframes ảnh).

    Dùng khi đã có CLIP features của toàn bộ video nhưng Keyframes mới tải 1 phần.
    Keyframes nếu có sẽ dùng làm keyframe_path (gallery/VLM), thiếu thì để trống.
    """
    from aic2026.ingestion.official_index import build_from_clip
    count = build_from_clip(
        raw_dir, features, output_manifest, output_features,
        keyframes_dir=keyframes_dir, objects_dir=objects_dir, metadata_dir=metadata_dir,
    )
    typer.echo(f"Wrote {count} records to {output_manifest} and {output_features}")

@app.command("check-coverage")
def check_coverage(
    raw_dir: Path = typer.Option(Path("data/raw"), help="Gốc dữ liệu"),
) -> None:
    """So sánh CLIP features vs Keyframes: video nào có cả 2, số lượng có khớp không.

    In ra: tổng video CLIP, tổng video Keyframes, danh sách video chỉ có CLIP (thiếu ảnh),
    và với mỗi video có cả 2, số keyframe ảnh vs số row CLIP.
    """
    import sys

    import numpy as np

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from aic2026.data_platform import inspect_official_assets
    assets = inspect_official_assets(raw_dir)

    clip_videos: dict[str, int] = {}
    for f in assets.clip_features:
        n = int(np.load(f, mmap_mode="r").shape[0])
        clip_videos[f.stem] = clip_videos.get(f.stem, 0) + n
    kf_videos: dict[str, int] = {}
    if assets.keyframes is not None and assets.keyframes.exists():
        for image in sorted([*assets.keyframes.rglob("*.jpg"), *assets.keyframes.rglob("*.png")]):
            vid = image.parent.name
            kf_videos[vid] = kf_videos.get(vid, 0) + 1

    clip_set = set(clip_videos)
    kf_set = set(kf_videos)
    both = sorted(clip_set & kf_set)
    clip_only = sorted(clip_set - kf_set)
    kf_only = sorted(kf_set - clip_set)

    typer.echo(f"CLIP features videos : {len(clip_set)} (total rows {sum(clip_videos.values())})")
    typer.echo(f"Keyframes videos      : {len(kf_set)} (total images {sum(kf_videos.values())})")
    typer.echo(f"Video có CẢ 2         : {len(both)}")
    typer.echo(f"Video CHỈ có CLIP (thiếu ảnh hiển thị/VLM): {len(clip_only)}")
    if clip_only:
        typer.echo("  -> " + ", ".join(clip_only[:20]) + (" ..." if len(clip_only) > 20 else ""))
    if kf_only:
        typer.echo(f"Video CHỈ có Keyframes (không có CLIP, sẽ bị bỏ qua): {len(kf_only)}")
        typer.echo("  -> " + ", ".join(kf_only[:20]) + (" ..." if len(kf_only) > 20 else ""))

    mismatched = [(v, kf_videos[v], clip_videos[v]) for v in both if kf_videos[v] != clip_videos[v]]
    if mismatched:
        typer.echo(f"\nCẢNH BÁO: {len(mismatched)} video có số keyframe ảnh != số row CLIP:")
        for v, k, c in mismatched[:30]:
            typer.echo(f"  {v}: {k} ảnh vs {c} rows CLIP")
    else:
        typer.echo("\nTất cả video có cả 2 đều KHỚP số lượng (ảnh == rows CLIP).")

@app.command("check-features")
def check_features(features: Path, manifest: Path = Path("data/processed/derived_manifest.jsonl")) -> None:
    """Fail early unless provided CLIP features and the manifest have the same order/count."""
    import numpy as np

    from aic2026.ingestion import load_manifest
    # O(1) shape check via mmap — no need to load/normalize the full matrix.
    rows = int(np.load(features, mmap_mode="r").shape[0])
    count = len(load_manifest(manifest))
    if rows != count:
        raise typer.BadParameter(f"features has {rows} vectors; manifest has {count} records")
    typer.echo(f"OK: {count} feature vectors match the manifest")

@app.command("build-chroma-index")
def build_chroma_index(
    features: Path = typer.Option(Path("data/processed/derived_features.npy"), help="Feature .npy input"),
    manifest: Path = typer.Option(Path("data/processed/derived_manifest.jsonl"), help="Manifest input"),
    chroma_dir: Path = typer.Option(Path("data/indexes/chroma"), help="Thư mục lưu Chroma collection"),
    collection_name: str = typer.Option("aic2026_frames"),
    space: str = typer.Option("cosine", help="Khoảng cách Chroma: cosine | l2 | ip"),
) -> None:
    """Build/populate the on-disk ChromaDB collection from .npy + manifest."""
    from aic2026.ingestion import load_manifest
    from aic2026.retrieval.factory import build_vector_index

    manifest_records = load_manifest(manifest)
    store = build_vector_index(
        features,
        manifest_records,
        backend="chroma",
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        space=space,
    )
    typer.echo(
        f"Chroma collection '{collection_name}' ready: {len(store)} vectors @ {chroma_dir}"
    )


def _resolve_keyframe_path(record: object, keyframes_root: Path) -> Path | None:
    """Find the keyframe JPG for a record, trying absolute + two relative layouts."""
    candidates = [
        Path(record.keyframe_path),
        keyframes_root / record.video_id / Path(record.keyframe_path).name,
        keyframes_root / record.keyframe_path,
    ]
    return next((c for c in candidates if c.exists()), None)


def _run_ocr(
    records: list,
    extractor: object,
    keyframes_root: Path,
    batch_size: int,
    progress_label: str = "OCR keyframes",
) -> tuple[list, int, int]:
    """OCR a list of FrameRecords, merging recognized text into object_labels.

    Returns (enriched_records, found_images, recognized_lines). The progress bar
    is intentionally plain (no ETA) because the first batches are dominated by
    Paddle/MKLDNN warmup, which would otherwise show a wildly wrong ETA like
    "1504d". ETA appears and stabilizes after warmup.
    """
    from aic2026.models import FrameRecord

    enriched: list[FrameRecord] = []
    total = len(records)
    found_images = 0
    recognized_lines = 0
    typer.echo(f"OCR bắt đầu: {total:,} manifest records | lang={extractor.lang} | model={extractor.model_size}")
    with typer.progressbar(length=total, label=progress_label, show_percent=True, show_pos=True) as progress:
        for start in range(0, total, batch_size):
            record_batch = records[start : start + batch_size]
            resolved: list[Path | None] = []
            image_paths: list[Path] = []
            for record in record_batch:
                chosen = _resolve_keyframe_path(record, keyframes_root)
                resolved.append(chosen)
                if chosen is not None:
                    image_paths.append(chosen)

            # One Paddle inference pass for all present images in the batch;
            # records missing a JPG keep their labels and just advance progress.
            batch_texts = extractor.extract_many(image_paths, batch_size=batch_size)
            text_iter = iter(batch_texts)
            for record, chosen in zip(record_batch, resolved):
                texts = next(text_iter) if chosen is not None else []
                if chosen is not None:
                    found_images += 1
                recognized_lines += len(texts)
                merged = list(dict.fromkeys([*(record.object_labels or []), *texts]))
                enriched.append(record.model_copy(update={"object_labels": merged}))
                progress.update(1)
    return enriched, found_images, recognized_lines


@app.command("ocr-manifest")
def ocr_manifest(
    manifest: Path = typer.Option(Path("data/processed/derived_manifest.jsonl"), help="Manifest JSONL input"),
    output: Path | None = typer.Option(None, help="Manifest JSONL output. Mặc định ghi đè (in-place) vào --manifest để agent-query tự dùng."),
    keyframes_root: Path = typer.Option(Path("data/raw/Keyframes"), help="Gốc chứa thư mục keyframe để resolve đường dẫn tương đối"),
    batch_size: int = typer.Option(16, min=1),
    lang: str = typer.Option("vi", help="Ngôn ngữ OCR: vi | en"),
    model_size: str = typer.Option("medium", help="PP-OCRv6 checkpoint: medium (chuẩn, chính xác) | mobile (nhanh 3–5x, hơi kém chính xác)"),
    shard_count: int = typer.Option(1, min=1, help="Chia manifest thành N shard để chạy song song nhiều process (CPU). Dùng cùng --shard-id."),
    shard_id: int = typer.Option(0, min=0, help="Shard thứ i (0-based) khi chạy song song. Bình thường = 0."),
    video_prefix: str = typer.Option("", help="CHỈ OCR các video có tiền tố video_id này (vd: L25 — khóa học onl có bảng biểu). Để trống = OCR hết. Các video KHÔNG khớp vẫn được ghi nguyên vào output (không OCR, không drop)."),
) -> None:
    """OCR toàn bộ keyframe trong manifest, ghi text nhận dạng vào object_labels (build 1 lần).

    Mặc định ghi đè vào chính file --manifest (in-place) để bước agent-query /
    build-chroma-index / BM25 index tự động hưởng lợi từ text OCR. Dùng --output
    để ghi file riêng (không đụng manifest gốc).

    Song song hóa (CPU): chia manifest thành N phần bằng --shard-count N, rồi chạy
    N process đồng thời (mỗi process 1 --shard-id 0..N-1), mỗi process ghi file
    ``<output>.<shard_id>``. Sau khi N process xong, gộp bằng ``ocr-merge-shards``.
    Ví dụ trên máy 8 nhân: --shard-count 8, chạy 8 terminal/lệnh song song.
    """
    if shard_count > 1 and output is None:
        typer.echo("--shard-count > 1 yêu cầu --output (để ghi từng shard riêng, không ghi đè manifest).")
        raise typer.Exit(code=1)
    if shard_id >= shard_count:
        typer.echo(f"--shard-id {shard_id} phải nhỏ hơn --shard-count {shard_count}.")
        raise typer.Exit(code=1)

    from aic2026.ingestion import load_manifest
    from aic2026.qa.ocr import OCRTextExtractor

    records = load_manifest(manifest)
    if shard_count > 1:
        # Chia đều, shard cuối hứng phần dư (nếu có).
        chunk = (len(records) + shard_count - 1) // shard_count
        records = records[shard_id * chunk : (shard_id + 1) * chunk]
        progress_label = f"OCR shard {shard_id + 1}/{shard_count}"
    else:
        progress_label = "OCR keyframes"

    if not records:
        typer.echo(f"Shard {shard_id} rỗng — không có record nào.")
        raise typer.Exit(code=0)

    # --video-prefix: CHỈ OCR các video khớp tiền tố (vd L25 — khóa học onl có
    # bảng biểu). Các video KHÔNG khớp vẫn được giữ nguyên trong output (không
    # OCR, không drop) để manifest đầy đủ cho retrieval các task khác (TRAKE→L26).
    ocr_prefixes = [p.strip() for p in video_prefix.split(",") if p.strip()]
    if ocr_prefixes:
        ocr_records = [
            r for r in records
            if any(r.video_id.startswith(p) for p in ocr_prefixes)
        ]
        passthrough = [
            r for r in records
            if not any(r.video_id.startswith(p) for p in ocr_prefixes)
        ]
        typer.echo(
            f"--video-prefix {ocr_prefixes}: OCR {len(ocr_records):,} / {len(records):,} "
            f"records khớp; {len(passthrough):,} records còn lại giữ nguyên (không OCR)."
        )
    else:
        ocr_records = records
        passthrough = []

    extractor = OCRTextExtractor(lang=lang, model_size=model_size)
    extractor._ensure_loaded()
    if extractor._ocr is None:
        typer.echo(
            "PaddleOCR không khởi động được — xem warning phía trên. "
            "Nếu có libpaddle.pyd/DLL load failed, sửa PaddlePaddle/Visual C++ runtime trước."
        )
        raise typer.Exit(code=1)

    ocr_enriched, found_images, recognized_lines = _run_ocr(
        ocr_records, extractor, keyframes_root, batch_size, progress_label=progress_label
    )
    # Gộp toàn bộ: GIỮ NGUYÊN thứ tự manifest gốc. vector_id của mỗi record phải
    # khớp với dòng thứ tự tương ứng trong file feature .npy (được build theo
    # đúng thứ tự manifest), nên ta không được xáo trộn thứ tự. Record khớp
    # --video-prefix dùng bản đã OCR; record còn lại giữ nguyên (không OCR).
    if passthrough:
        enriched_by_id = {r.vector_id: r for r in ocr_enriched}
        enriched = [
            enriched_by_id.get(record.vector_id, record)
            for record in records
        ]
    else:
        enriched = ocr_enriched

    target = output if output is not None else manifest
    if shard_count > 1:
        # Mỗi shard ghi file riêng để không đè lên nhau; gộp sau bằng ocr-merge-shards.
        target = output.with_suffix(f".shard{shard_id}{output.suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(r.model_dump_json() for r in enriched) + ("\n" if enriched else ""), encoding="utf-8")
    if shard_count > 1:
        typer.echo(
            f"OCR shard {shard_id} xong: wrote {len(enriched):,} records to {target} "
            f"| images={found_images:,} | text-lines={recognized_lines:,}"
        )
    elif output is None:
        typer.echo(
            f"OCR hoàn tất: wrote {len(enriched):,} records IN-PLACE to {target} "
            f"| images={found_images:,} | text-lines={recognized_lines:,}"
        )
    else:
        typer.echo(
            f"OCR hoàn tất: wrote {len(enriched):,} records to {target} "
            f"| images={found_images:,} | text-lines={recognized_lines:,}"
        )


@app.command("ocr-merge-shards")
def ocr_merge_shards(
    output: Path = typer.Option(..., help="File manifest gộp đầu ra (ví dụ: official_manifest_ocr_vi.jsonl)"),
    shard_glob: str = typer.Option("data/processed/official_manifest_ocr_vi.jsonl.shard*.jsonl", help="Glob các file shard đã sinh bởi ocr-manifest --shard-count"),
) -> None:
    """Gộp các file shard do ``ocr-manifest --shard-count N`` sinh ra thành 1 manifest.

    Chạy SAU khi tất cả N process song song đã xong. Thứ tự record được giữ theo
    tên shard (shard0 trước, shard1 sau, ...), khớp với thứ tự manifest gốc.
    """
    from pathlib import Path as _Path

    shards = sorted(_Path(p) for p in __import__("glob").glob(shard_glob))
    if not shards:
        typer.echo(f"Không tìm thấy shard nào khớp glob: {shard_glob}")
        raise typer.Exit(code=1)
    total = 0
    lines: list[str] = []
    for shard in shards:
        text = shard.read_text(encoding="utf-8")
        kept = [ln for ln in text.splitlines() if ln.strip()]
        lines.extend(kept)
        total += len(kept)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(f"Gộp {len(shards)} shard → {total:,} records vào {output}")


@app.command("agent-query")
def agent_query(
    query: Path = typer.Option(..., help="JSON file matching the Query schema"),
    features: Path | None = typer.Option(None, help="BTC CLIP .npy feature file. Mặc định auto: ưu tiên official_features.npy (CLIP sẵn BTC) nếu có, fallback derived_features.npy (encode keyframe)."),
    manifest: Path | None = typer.Option(None, help="Manifest JSONL. Mặc định auto: ưu tiên official_manifest.jsonl nếu có, fallback derived_manifest.jsonl."),
    llm_model: str = typer.Option("qwen3.5:4b"),
    ollama_url: str = typer.Option("http://127.0.0.1:11434"),
    llm_timeout: int = typer.Option(600, help="Timeout (giây) cho mỗi lần gọi LLM local (CPU chậm cần lớn)"),
    backend: str = typer.Option("faiss", help="Index backend: faiss (mặc định) | numpy | chroma"),
    chroma_dir: Path = typer.Option(Path("data/indexes/chroma"), help="Thư mục Chroma (chỉ dùng khi backend=chroma)"),
    metadata_filter: str = typer.Option("", help="Từ khóa metadata cách nhau dấu phẩy để lọc video TRƯỚC retrieval (vd: cửa hàng,siêu thị). Để trống = không lọc."),
    vlm_backend: str = typer.Option("ollama", help="Backend VLM cho Q&A: ollama | transformers | none"),
    vlm_model: str = typer.Option("qwen2.5vl:3b", help="Model VLM cho Q&A (tên model trên Ollama; để trống để tắt)"),
    vlm_device: str | None = typer.Option(None, help="Device VLM khi backend=transformers (mặc định auto)"),
    vlm_dtype: str = typer.Option("bfloat16", help="Precision VLM khi backend=transformers: bfloat16 | float16 | float32"),
    vlm_timeout: int = typer.Option(120, help="Timeout (giây) mỗi lần gọi VLM qua Ollama (backend=ollama)"),
    coarse_top_k: int = typer.Option(200, help="TRAKE: giới hạn số video đưa vào DP alignment (coarse filter). 0 = xét hết."),
    trake_preferred_prefixes: str = typer.Option("", help="Ưu tiên (soft bias, KHÔNG loại trừ) TRAKE vào tiền tố video_id, cách nhau dấu phẩy (vd: L26). KIS/Q&A luôn xét TOÀN BỘ video. Để trống = mặc định ưu tiên L26. Truyền '.' để tắt ưu tiên."),
    translate: bool = typer.Option(False, help="Dịch VI→EN trước khi retrieval (CLIP là tiếng Anh). Mặc định tắt: dùng query nguyên bản."),
    output: Path | None = typer.Option(None),
) -> None:
    """Run the bounded local-LLM agent and save auditable candidates/trace."""
    from aic2026.agent import OllamaLLM, RetrievalAgent
    from aic2026.agent.tools import RetrievalTools
    from aic2026.embeddings import OpenCLIPTextEmbedder
    from aic2026.ingestion import load_manifest
    from aic2026.retrieval import RetrievalPipeline
    from aic2026.retrieval.factory import load_index_for_query

    parsed_query = Query.model_validate_json(query.read_text(encoding="utf-8"))
    resolved_manifest, resolved_features = resolve_feature_sources(manifest, features)
    if manifest is None or features is None:
        typer.echo(
            f"Auto-selected feature sources: manifest={resolved_manifest}, "
            f"features={resolved_features}"
        )
    manifest_records = load_manifest(resolved_manifest)

    index = load_index_for_query(resolved_features, manifest_records, backend=backend, chroma_dir=chroma_dir)
    pipeline = RetrievalPipeline(
        index,
        manifest_records,
        frames_per_video=20,
    )
    text_encoder = OpenCLIPTextEmbedder()
    encode_text = text_encoder.encode
    tools = RetrievalTools(pipeline, encode_text, encode_images=text_encoder.encode_images)
    # coarse_top_k áp dụng cho cả KIS (video-level rerank) và TRAKE (DP coarse
    # filter). TRAKE dùng agent.coarse_top_k; KIS dùng tools.coarse_top_k.
    tools.coarse_top_k = coarse_top_k
    # Objects/OCR are frame-specific; do not duplicate video metadata in BM25.
    from aic2026.retrieval import BM25Index
    tools.bm25_index = BM25Index(manifest_records)
    # Metadata pre-filter: chỉ retrieve trong các video có chứa từ khóa.
    tools.video_filter_terms = [t.strip() for t in metadata_filter.split(",") if t.strip()] or None
    # TRAKE soft preference: "." disables the default L26 bias; empty uses the
    # agent's built-in default (L26); otherwise split on commas into the
    # explicit preferred-prefix list. KIS/Q&A are intentionally never scoped —
    # they search the full corpus.
    def _parse_prefixes(raw: str) -> list[str] | None:
        if raw == ".":
            return None
        return [p.strip() for p in raw.split(",") if p.strip()] or None
    if parsed_query.type == "qa" and vlm_backend != "none" and vlm_model:
        if vlm_backend == "ollama":
            from aic2026.qa.vlm_ollama import OllamaVisionModel
            tools.visual_answerer = OllamaVisionModel(
                model_name=vlm_model,
                base_url=ollama_url,
                timeout_seconds=vlm_timeout,
            ).answer_question
        elif vlm_backend == "transformers":
            from aic2026.qa.vlm import QwenVLM
            tools.visual_answerer = QwenVLM(
                model_name=vlm_model,
                device=vlm_device,
                torch_dtype=vlm_dtype,
            ).answer_question
        else:
            raise typer.BadParameter(
                f"Unknown --vlm-backend '{vlm_backend}'. "
                "Expected one of: ollama, transformers, none."
            )
    agent = RetrievalAgent(
        tools,
        llm=OllamaLLM(model=llm_model, base_url=ollama_url, timeout_seconds=llm_timeout),
        trake_preferred_prefixes=_parse_prefixes(trake_preferred_prefixes),
    )
    agent.coarse_top_k = coarse_top_k
    agent.translate = translate
    result = agent.run(parsed_query)
    if parsed_query.type == "qa" and not any(c.answer for c in result.candidates):
        typer.echo(
            "⚠️ Cảnh báo: task Q&A nhưng không candidate nào có answer "
            "(VLM không gọi được hoặc không sinh nội dung). Submission sẽ không hợp lệ "
            "— kiểm tra: ollama đang serve model vision chưa (`ollama pull qwen2.5vl:3b`), "
            "--vlm-model, hoặc kết nối."
        )
    serialized = result.model_dump_json(indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        typer.echo(f"Wrote agent result to {output}")
    else:
        typer.echo(serialized)

@app.command()
def evaluate(query: Path, candidates: Path, ground_truth: Path) -> None:
    """Evaluate one query using AIC R@k scoring."""
    q = Query.model_validate_json(query.read_text(encoding="utf-8"))
    gt = GroundTruth.model_validate_json(ground_truth.read_text(encoding="utf-8"))
    items = [Candidate.model_validate(row) for row in json.loads(candidates.read_text(encoding="utf-8"))]
    typer.echo(json.dumps(evaluate_query(q, items, gt), ensure_ascii=False, indent=2))

@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the internal FastAPI service."""
    import uvicorn
    uvicorn.run("aic2026.app.api:app", host=host, port=port, reload=False)

if __name__ == "__main__": app()
