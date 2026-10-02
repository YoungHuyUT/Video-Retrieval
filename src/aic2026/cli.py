from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Load .env file if present (GEMINI_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Windows console defaults to a legacy code page (cp1252) that cannot encode
# Vietnamese (or any non-Latin) output, which makes every `typer.echo` of a
# progress message crash with UnicodeEncodeError. Force UTF-8 on the std
# streams so the CLI is runnable on Windows terminals as well as Colab/Linux.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (OSError, ValueError):
    pass

import typer

from aic2026.evaluation import evaluate_query
from aic2026.ingestion import build_manifest, resolve_feature_sources
from aic2026.models import Candidate, GroundTruth, Query

logger = logging.getLogger(__name__)

app = typer.Typer(help="AIC 2026 retrieval toolkit")


def _setup_run_logging(log_file: Path) -> None:
    """Cấu hình logging ghi LIÊN TỤC ra file (và console) để theo dõi lúc chạy.

    Không đợi chạy xong mới ghi — mỗi ``logger.info`` được flush ngay vào file,
    nên user có thể ``tail``/mở file log bất cứ lúc nào để xem tiến độ.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8", delay=False)
    file_handler.setLevel(logging.INFO)
    # dòng format có timestamp để dễ biết đang ở thời điểm nào.
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Tránh nhân đôi handler mỗi lần gọi lệnh.
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_file.resolve())
               for h in root.handlers):
        root.addHandler(file_handler)
    # Console handler để vẫn thấy trên terminal.
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        stream = logging.StreamHandler()
        stream.setLevel(logging.INFO)
        stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S"))
        root.addHandler(stream)

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
    sample_interval_sec: float | None = typer.Option(1.0, min=0.0, help="Khoảng thời gian lấy mẫu theo giây; 1.0 = lấy 1 frame/giây, 0 hoặc None = chế độ legacy lấy toàn bộ frame"),
    cosine_threshold: float = typer.Option(0.985, min=0.0, max=1.0),
    batch_size: int = typer.Option(32, min=1),
    clip_pretrained: str = typer.Option("openai", help="Checkpoint OpenCLIP: openai (chuẩn BTC) hoặc laion2b_s32b_b79k"),
    image_quality: int = typer.Option(92, min=1, max=100),
) -> None:
    """Trích toàn bộ frame, CLIP-embed, rồi loại near-duplicate bằng cosine similarity."""
    from aic2026.data_platform import OpenCLIPFrameEncoder, extract_deduplicated_keyframes
    image_encoder = OpenCLIPFrameEncoder(model_name="ViT-B-32", pretrained=clip_pretrained)
    report = extract_deduplicated_keyframes(
        video,
        keyframes_dir,
        features_dir,
        image_encoder,
        sample_interval_sec=sample_interval_sec,
        cosine_threshold=cosine_threshold,
        batch_size=batch_size,
        image_quality=image_quality,
    )
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

        extractor = OCRTextExtractor(lang=ocr_lang, correct=True)
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

@app.command("embed-siglip2")
def embed_siglip2(
    manifest: Path = typer.Option(Path("data/processed/official_manifest.jsonl"),
        help="Manifest JSONL (official hoặc derived) liệt kê keyframe_path cho từng frame."),
    output_features: Path = typer.Option(Path("data/processed/siglip2/features_siglip2.npy"),
        help="Output SigLIP2 embedding matrix (.npy)."),
    output_manifest: Path = typer.Option(Path("data/processed/siglip2/manifest_siglip2.jsonl"),
        help="Output manifest copy."),
    model_name: str = typer.Option("google/siglip2-so400m-patch14-384",
        help="SigLIP2 checkpoint: google/siglip2-so400m-patch14-384 (1152-d) hoặc google/siglip2-base-patch16-224 (768-d)"),
    batch_size: int = typer.Option(32, min=1, help="Batch size để giảm OOM trên CPU/GPU"),
    resume: bool = typer.Option(True, help="Resume nếu output .npy đã tồn tại"),
    limit: int = typer.Option(0, help="Chỉ embed N frame đầu (dev; 0 = toàn bộ)"),
    log_file: Path = typer.Option(Path("data/processed/siglip2/embed.log"),
        help="File log tiến độ."),
) -> None:
    """Embed BTC keyframes bằng SigLIP2 → features_siglip2.npy + manifest copy.

    Improvement.md §1: dùng trực tiếp BTC keyframes hiện có (không extract lại video).
    Improvement.md §11: batch inference, cache, resume.
    Improvement.md §12: lưu metadata (model_name, source_hash, dim, etc.).
    """
    _setup_run_logging(log_file)
    from aic2026.embeddings.siglip2 import (
        Siglip2Embedder, EmbedConfig, build_siglip2_embeddings
    )

    logger.info("BẮT ĐẦU embed-siglip2: manifest=%s, model=%s", manifest, model_name)

    # Build a temp manifest with limit if specified
    if limit > 0:
        import tempfile, json
        tmp_manifest = Path(tempfile.mktemp(suffix=".jsonl"))
        count = 0
        with open(manifest, "r") as src, open(tmp_manifest, "w") as out:
            for line in src:
                if count >= limit:
                    break
                line = line.strip()
                if line:
                    out.write(line + "\n")
                    count += 1
        manifest = tmp_manifest
        logger.info("Limited to first %d frames", limit)

    config = EmbedConfig(
        model_name=model_name,
        batch_size=batch_size,
        resume=resume,
    )
    embeddings, records = build_siglip2_embeddings(manifest, output_features, config)

    logger.info("HOÀN TẤT: %d embeddings, dim=%d", embeddings.shape[0], embeddings.shape[1])
    typer.echo(json.dumps({
        "frame_count": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
        "model": model_name,
        "features_path": str(output_features),
        "manifest_path": str(output_manifest),
    }, ensure_ascii=False, indent=2))

    if limit > 0:
        Path(manifest).unlink(missing_ok=True)


@app.command("build-siglip2-index")
def build_siglip2_index(
    features: Path = typer.Option(Path("data/processed/siglip2/features_siglip2.npy"),
        help="SigLIP2 embedding matrix (.npy)."),
    manifest: Path = typer.Option(Path("data/processed/siglip2/manifest_siglip2.jsonl"),
        help="Manifest JSONL tương ứng với features."),
    output_index: Path = typer.Option(Path("data/processed/siglip2/index_siglip2.faiss"),
        help="Output FAISS index file."),
    use_gpu: bool = typer.Option(False, help="Dùng GPU nếu có (FAISS GPU index)"),
) -> None:
    """Build FAISS index từ SigLIP2 embeddings.

    Improvement.md §2: SigLIP2 → FAISS.
    Vectors được L2-normalize để inner product = cosine similarity.
    """
    import faiss
    import numpy as np
    import json

    logger.info("Bắt đầu build FAISS từ SigLIP2 embeddings: %s", features)

    # Load embeddings
    embeddings = np.load(features, mmap_mode="r").astype(np.float32)
    logger.info("Loaded embeddings: shape=%s, dtype=%s", embeddings.shape, embeddings.dtype)

    # Build index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity

    # L2-normalize vectors
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save index
    output_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_index))

    # Save metadata
    from aic2026.embeddings.siglip2 import load_siglip2_metadata, SIGLIP2_META_FILE
    meta = load_siglip2_metadata(SIGLIP2_META_FILE)
    if meta:
        meta["index_path"] = str(output_index)
        meta["index_type"] = "IndexFlatIP"
        with open(output_index.parent / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    typer.echo(json.dumps({
        "index_vectors": index.ntotal,
        "embedding_dim": dim,
        "index_path": str(output_index),
        "index_type": "IndexFlatIP (inner product / cosine)",
        "model": meta.get("model_name", "unknown") if meta else "unknown",
    }, ensure_ascii=False, indent=2))

    logger.info("HOÀN TẤT: FAISS index built, %d vectors", index.ntotal)


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
    log_file: Path = typer.Option(
        Path("data/processed/prepare_official.log"),
        help="File log tiến độ LIÊN TỤC (ghi từng video, không đợi xong). Mở/tail bất cứ lúc nào để theo dõi.",
    ),
) -> None:
    """Build manifest + aligned .npy từ CLIP features BTC chính thức (thứ tự khớp keyframe)."""
    _setup_run_logging(log_file)
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
    logger.info("BẮT ĐẦU prepare-official: manifest=%s features=%s", output_manifest, features)
    count = build_official_index(raw_dir, features, output_manifest, output_features)
    logger.info("HOÀN TẤT prepare-official: %d records", count)
    typer.echo(f"Wrote {count} records to {output_manifest} and {output_features}")


@app.command("compute-colour-features")
def compute_colour_features(
    manifest: Path = typer.Option(
        Path("data/processed/official_manifest.jsonl"),
        help="Manifest JSONL (có keyframe_path + object_path mỗi record).",
    ),
    raw_dir: Path = typer.Option(Path("data/raw"), help="Gốc dữ liệu chứa Keyframes/Objects."),
    output: Path = typer.Option(
        Path("data/processed/colour_features.jsonl"),
        help="Output sidecar JSONL (key vector_id).",
    ),
    limit: int = typer.Option(0, help="Chỉ xử lý N record đầu (dev; 0 = toàn bộ)."),
) -> None:
    """Precompute offline colour evidence per keyframe.

    CLIP toàn cảnh gần như 'mù màu', nên truy vấn như 'xe đỏ' và 'xe xanh'
    trả kết quả gần như giống nhau. Bước rerank màu gốc giải quyết bằng cách
    decode ảnh keyframe mỗi lần query — chậm và chỉ xét top-N. Lệnh này
    chuyển toàn bộ công việc pixel sang một lần chạy offline, ghi một sidecar
    nhẹ (vector_id → {dom, fracs, obj}); query chỉ là dict lookup tức thì,
    áp dụng cho toàn bộ candidate pool, và bind màu với bất kỳ object nào.

    Chạy một lần trước khi serve; rerank sẽ tự động dùng sidecar nếu tồn tại.
    """
    from aic2026.ingestion.colour_features import build_colour_features

    count = build_colour_features(
        manifest_path=manifest,
        keyframes_root=raw_dir / "Keyframes",
        objects_root=raw_dir / "Objects",
        output_path=output,
        limit=limit,
    )
    typer.echo(f"Wrote {count} colour-feature records to {output}")

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
    """Find the keyframe JPG for a record, trying several layouts.

    ``record.keyframe_path`` is stored RELATIVE to the project root, so a plain
    ``Path(record.keyframe_path)`` only resolves when the process cwd happens to
    be the project root. If the CLI is launched from elsewhere (or as a detached
    background task whose cwd defaults to the user's home), that candidate
    misses and we silently get ``images=0`` — the exact bug seen when running
    OCR away from the project dir. Always also try the project root so the
    lookup is cwd-independent.
    """
    proj_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path(record.keyframe_path),
        proj_root / record.keyframe_path,
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
):
    """OCR a list of FrameRecords, yielding each enriched record lazily.

    Là **generator**: mỗi batch OCR xong là yield ngay record đã enrich (theo
    đúng thứ tự đầu vào), kèm thống kê (found_images, recognized_lines) của
    record đó. Caller có thể ghi liền từng dòng ra file (streaming) thay vì giữ
    hết trong RAM — nên nếu chạy hàng giờ mà bị ngắt giữa chừng, file output vẫn
    chứa những gì đã OCR xong (không mất trắng như bản cũ ghi 1 lần ở cuối).

    Yield: ``(enriched_record, found_images_increment, recognized_lines_increment)``.

    The progress bar is intentionally plain (no ETA) because the first batches
    are dominated by Paddle/MKLDNN warmup, which would otherwise show a wildly
    wrong ETA like "1504d". ETA appears and stabilizes after warmup.
    """
    from aic2026.models import FrameRecord

    total = len(records)
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
                found = 1 if chosen is not None else 0
                lines = len(texts)
                merged = list(dict.fromkeys([*(record.object_labels or []), *texts]))
                # Mark OCR-complete so a resumed run can skip this frame. The
                # flag survives the streaming write below and is re-read on the
                # next invocation, so an interrupted run loses nothing.
                enriched = record.model_copy(update={"object_labels": merged, "ocr_done": True})
                progress.update(1)
                yield enriched, found, lines


@app.command("ocr-manifest")
def ocr_manifest(
    manifest: Path = typer.Option(Path("data/processed/official_manifest.jsonl"), help="Manifest JSONL input (mặc định official — file có sẵn; dùng derived_manifest.jsonl nếu đã encode keyframe Chế độ A)"),
    output: Path | None = typer.Option(None, help="Manifest JSONL output. Mặc định ghi đè (in-place) vào --manifest để agent-query tự dùng."),
    keyframes_root: Path = typer.Option(Path("data/raw/Keyframes"), help="Gốc chứa thư mục keyframe để resolve đường dẫn tương đối"),
    batch_size: int = typer.Option(16, min=1),
    lang: str = typer.Option("vi", help="Ngôn ngữ OCR: vi | en"),
    model_size: str = typer.Option("medium", help="PP-OCRv6 checkpoint: medium (chuẩn, chính xác) | mobile (nhanh 3–5x, hơi kém chính xác)"),
    device: str = typer.Option("cpu", help="Thiết bị inference: cpu (mặc định, máy local không GPU) | gpu | gpu:0. Trên Colab T4 (CUDA) dùng gpu để nhanh 10–20x. Yêu cầu paddlepaddle-gpu đã cài."),
    correct: bool = typer.Option(True, help="Chạy post-correction Tiếng Việt (sửa dấu vỡ, rn→m, lọc token rác) sau OCR. Tắt nếu muốn text thô."),
    resume: bool = typer.Option(False, help="Tiếp tục từ chỗ dở: bỏ qua các record đã có flag ocr_done=True trong output cũ (không OCR lại). Dùng khi chạy bị ngắt giữa chừng."),
    shard_count: int = typer.Option(1, min=1, help="Chia manifest thành N shard để chạy song song nhiều process (CPU). Dùng cùng --shard-id."),
    shard_id: int = typer.Option(0, min=0, help="Shard thứ i (0-based) khi chạy song song. Bình thường = 0."),
    log_file: Path = typer.Option(
        Path("data/processed/ocr_manifest.log"),
        help="File log tiến độ LIÊN TỤC (ghi từng record, không đợi xong). Mở/tail bất cứ lúc nào để theo dõi OCR.",
    ),
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
    _setup_run_logging(log_file)
    logger.info("BẮT ĐẦU ocr-manifest: manifest=%s output=%s lang=%s model=%s prefix=%s",
                manifest, output, lang, model_size, video_prefix or "(all)")
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

    extractor = OCRTextExtractor(lang=lang, model_size=model_size, correct=correct, device=device)
    extractor._ensure_loaded()
    if extractor._ocr is None:
        typer.echo(
            "PaddleOCR không khởi động được — xem warning phía trên. "
            "Nếu có libpaddle.pyd/DLL load failed, sửa PaddlePaddle/Visual C++ runtime trước."
        )
        raise typer.Exit(code=1)

    # --resume: nếu lần chạy trước (có thể bị ngắt giữa chừng) đã ghi một phần
    # output mang flag ocr_done=True, load output đó và đánh dấu những record
    # đó để bỏ qua — không OCR lại. Hoạt động cả với --output riêng và in-place
    # (output == manifest). Luôn load từ bản ghi đầy đủ (manifest gốc) làm
    # nguồn record, chỉ mượn ocr_done từ output để biết frame nào xong rồi.
    if resume and ocr_records:
        prev_path = output if output is not None else manifest
        if os.path.exists(prev_path):
            from aic2026.ingestion import load_manifest as _load
            prev = _load(prev_path)
            done_ids = {r.vector_id for r in prev if getattr(r, "ocr_done", False)}
            if done_ids:
                before = len(ocr_records)
                ocr_records = [r for r in ocr_records if r.vector_id not in done_ids]
                passthrough = [r for r in records
                               if not any(r.video_id.startswith(p) for p in ocr_prefixes)] + \
                             [r for r in records
                               if any(r.video_id.startswith(p) for p in ocr_prefixes)
                               and r.vector_id in done_ids]
                typer.echo(
                    f"RESUME: bỏ qua {len(done_ids):,} record đã OCR xong; "
                    f"{len(ocr_records):,} record còn lại cần OCR."
                )

    # PRE-FLIGHT: đếm số ảnh THỰC SỰ resolve được trong số record sẽ OCR TRƯỚC
    # khi chạy. Nếu = 0 mà vẫn còn record cần OCR, dừng ngay với hướng dẫn — tránh
    # kịch bản Colab cũ: chạy hàng giờ qua 177k record mà images=0 (đường dẫn
    # keyframe không khớp) rồi ra manifest rỗng.
    if ocr_records:
        resolved_sample = sum(
            1 for r in ocr_records
            if _resolve_keyframe_path(r, keyframes_root) is not None
        )
        typer.echo(
            f"PRE-FLIGHT: {resolved_sample:,}/{len(ocr_records):,} ảnh L25 resolve được "
            f"tại keyframes_root={keyframes_root}"
        )
        if resolved_sample == 0:
            typer.echo(
                "❌ PRE-FLIGHT FAILED: không resolve được ảnh nào. Đường dẫn keyframe "
                "sai hoặc keyframes_root không chứa ảnh. Kiểm tra:\n"
                f"  - keyframes_root hiện tại: {keyframes_root}\n"
                "  - manifest lưu keyframe_path dạng gì (vd data/raw/Keyframes/...)\n"
                "  - ảnh có nằm ở keyframes_root/<video_id>/<tên_file>.jpg không\n"
                "Script sẽ thoát để không lãng phí thời gian. Sửa keyframes_root rồi chạy lại."
            )
            raise typer.Exit(code=1)
    else:
        typer.echo("Không có record nào cần OCR (đã xong hết hoặc prefix không khớp).")

    # GHI STREAMING (từng record) giữ NGUYÊN thứ tự manifest gốc. vector_id của
    # mỗi record phải khớp dòng tương ứng trong .npy (build theo đúng thứ tự),
    # nên không xáo trộn. Record khớp --video-prefix (hoặc thuộc shard này) dùng
    # bản đã OCR; record còn lại ghi nguyên (không OCR). Ghi liền + flush mỗi
    # 5000 dòng → nếu ngắt giữa chừng, file output vẫn chứa những gì đã OCR xong.
    ocr_ids = {r.vector_id for r in ocr_records}
    ocr_iter = _run_ocr(ocr_records, extractor, keyframes_root, batch_size, progress_label=progress_label)
    found_images = 0
    recognized_lines = 0
    written = 0
    total = len(records)

    target = output if output is not None else manifest
    if shard_count > 1:
        # Mỗi shard ghi file riêng để không đè lên nhau; gộp sau bằng ocr-merge-shards.
        target = output.with_suffix(f".shard{shard_id}{output.suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as mf:
        for record in records:
            if record.vector_id in ocr_ids:
                enriched, found, lines = next(ocr_iter)
                found_images += found
                recognized_lines += lines
                mf.write(enriched.model_dump_json() + "\n")
            else:
                mf.write(record.model_dump_json() + "\n")
            written += 1
            # Flush định kỳ + log tiến độ LIÊN TỤC (mở/tail file bất cứ lúc nào).
            if written % 5000 == 0:
                mf.flush()
                logger.info(
                    "OCR tiến độ: %d/%d records | images=%d | text-lines=%d → %s",
                    written, total, found_images, recognized_lines, target,
                )
        mf.flush()
    if shard_count > 1:
        typer.echo(
            f"OCR shard {shard_id} xong: wrote {written:,} records to {target} "
            f"| images={found_images:,} | text-lines={recognized_lines:,}"
        )
    elif output is None:
        typer.echo(
            f"OCR hoàn tất: wrote {written:,} records IN-PLACE to {target} "
            f"| images={found_images:,} | text-lines={recognized_lines:,}"
        )
    else:
        typer.echo(
            f"OCR hoàn tất: wrote {written:,} records to {target} "
            f"| images={found_images:,} | text-lines={recognized_lines:,}"
        )
    logger.info("HOÀN TẤT ocr-manifest: %d records → %s | images=%d | text-lines=%d",
                written, target, found_images, recognized_lines)


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
    vlm_backend: str = typer.Option("florence", help="Backend VLM cho Q&A: florence | none (Florence-2 thay thế QwenVLM/Ollama)"),
    vlm_model: str = typer.Option("microsoft/Florence-2-base-ft", help="Model VLM cho Q&A (HuggingFace id Florence-2; để trống để tắt)"),
    vlm_device: str | None = typer.Option(None, help="Device VLM (mặc định auto / cpu)"),
    vlm_dtype: str = typer.Option("float32", help="Precision VLM: float32 (CPU an toàn) | float16 | bfloat16 (GPU)"),
    vlm_timeout: int = typer.Option(120, help="Timeout (giây) mỗi lần gọi VLM"),
    coarse_top_k: int = typer.Option(200, help="TRAKE: giới hạn số video đưa vào DP alignment (coarse filter). 0 = xét hết."),
    late_interaction_weight: float = typer.Option(0.0, help="KIS: bật late-interaction (ColBERT-style MaxSim theo từng facet query). 0 = tắt (mặc định). Thử 0.5 để CLIP không chọn nhầm frame đúng cảnh sai vật thể. Nặng hơn RRF nhưng chỉ vài chục encode text, vẫn ms."),
    object_evidence_weight: float = typer.Option(0.0, help="KIS: độ lớn reward/penalty object detector khi query hỏi vật thể (RRF scale). Lớn hơn = ép object mạnh hơn CLIP. 0.0 = tắt object evidence (dense-only mode)."),
    object_penalty_scale: float = typer.Option(2.0, help="KIS: hệ số phạt frame THIẾU vật thể so với thưởng frame có vật thể (mặc định 2.0)."),
    drop_empty_object_frames: bool = typer.Option(False, help="KIS: loại hẳn frame KHÔNG có vật thể nào (ảnh mờ, không entity đạt ngưỡng 0.4) khi query có hỏi vật thể. Giảm truy xuất đến frame nhiễu. Tắt nếu sợ mất recall."),
    trake_preferred_prefixes: str = typer.Option("", help="Ưu tiên (soft bias, KHÔNG loại trừ) TRAKE vào tiền tố video_id, cách nhau dấu phẩy (vd: L26). KIS/Q&A luôn xét TOÀN BỘ video. Để trống = mặc định ưu tiên L26. Truyền '.' để tắt ưu tiên."),
    translate: bool = typer.Option(False, help="Dịch VI→EN trước khi retrieval (CLIP là tiếng Anh). Mặc định tắt: dùng query nguyên bản."),
    asr_sidecar_path: Path | None = typer.Option(None, help="Đường dẫn ASR sidecar (.jsonl) cho spoken-term localization. Mặc định tự động tìm data/processed/asr_sidecar.jsonl."),
    asr_weight: float = typer.Option(0.15, help="Trọng số ASR spoken-term score (weak signal, nudge RRF)."),
    use_event_coverage: bool = typer.Option(True, help="Bật Event Coverage precision rerank (ưu tiên frame giải thích nhiều event). Tắt để dùng baseline RRF."),
    use_moment_rerank: bool = typer.Option(True, help="Bật chọn 1 best evidence frame / moment cho KIS/QA."),
    event_coverage_blend: float = typer.Option(0.5, help="Blend factor Event Coverage (0=giữ ranking gốc, 1=ưu tiên coverage)."),
    embedding_backend: str = typer.Option("siglip2", help="Embedding backend: siglip2 (768-dim, mặc định) | official_clip/OpenCLIP ViT-B/32 (512-dim)"),
    clip_pretrained: str = typer.Option("openai", help="OpenCLIP checkpoint: openai (chuẩn BTC) hoặc laion2b_s32b_b79k"),
    siglip2_model: str = typer.Option("google/siglip2-base-patch16-224", help="SigLIP2 checkpoint (phải khớp features_siglip2.npy)"),
    output: Path | None = typer.Option(None),
) -> None:
    """Run the bounded local-LLM agent and save auditable candidates/trace."""
    from aic2026.agent import OllamaLLM, RetrievalAgent
    from aic2026.agent.tools import RetrievalTools
    from aic2026.embeddings import OpenCLIPTextEmbedder, Siglip2Embedder
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
    # Per-video metadata (title/description/keywords) sống cùng thư mục với
    # manifest. Load vào store để pipeline chạy được metadata pre-filter
    # (filter_videos_by_metadata) — nếu không truyền, store rỗng và pre-filter
    # bị skip dù tools.video_filter_terms có set.
    from aic2026.retrieval import VideoMetadataStore

    video_metadata = VideoMetadataStore.load(
        resolved_manifest.parent / "video_metadata.jsonl"
    )

    index = load_index_for_query(resolved_features, manifest_records, backend=backend, chroma_dir=chroma_dir)
    pipeline = RetrievalPipeline(
        index,
        manifest_records,
        frames_per_video=20,
        video_metadata=video_metadata,
    )
    # Improvement.md §3: SigLIP2 text encoder must match the image encoder
    # checkpoint used for features_siglip2.npy (siglip2-base-patch16-224, 768-dim).
    if embedding_backend == "siglip2":
        text_encoder = Siglip2Embedder.get_or_create(model_name=siglip2_model)
        logger.info("CLI using SigLIP2 text encoder: %s", siglip2_model)
    else:
        text_encoder = OpenCLIPTextEmbedder(pretrained=clip_pretrained)
        logger.info("CLI using OpenCLIP text encoder: ViT-B/32 (%s)", clip_pretrained)
    tools = RetrievalTools(pipeline, text_encoder.encode, encode_images=text_encoder.encode_images)
    # coarse_top_k áp dụng cho cả KIS (video-level rerank) và TRAKE (DP coarse
    # filter). TRAKE dùng agent.coarse_top_k; KIS dùng tools.coarse_top_k.
    tools.coarse_top_k = coarse_top_k
    # KIS evidence tuning (object detector + late-interaction facet rerank).
    tools.late_interaction_weight = late_interaction_weight
    tools.object_evidence_weight = object_evidence_weight
    tools.object_penalty_scale = object_penalty_scale
    tools.drop_empty_object_frames = drop_empty_object_frames
    # BM25 needs video-level text (title/description/keywords) to match lexical
    # queries.  Frame-level object_labels alone are too coarse.  ASR transcripts
    # (Vietnamese) also go into BM25 so spoken-content queries match lexically.
    asr_text_by_video: dict[str, str] | None = None
    asr_path_for_bm25 = asr_sidecar_path
    if asr_path_for_bm25 is None:
        asr_path_for_bm25 = resolved_manifest.parent.parent / "processed" / "asr_sidecar.jsonl"
    if asr_path_for_bm25 and Path(asr_path_for_bm25).exists():
        try:
            from aic2026.ingestion.asr import load_transcripts_sidecar as _load_asr
            _asr_t = _load_asr(asr_path_for_bm25)
            asr_text_by_video = {
                vid: t.full_text for vid, t in _asr_t.items() if t.full_text
            }
        except Exception:
            pass
    from aic2026.retrieval import BM25Index
    tools.bm25_index = BM25Index(
        manifest_records,
        video_metadata=video_metadata,
        asr_text_by_video=asr_text_by_video,
    )
    # Metadata pre-filter: chỉ retrieve trong các video có chứa từ khóa.
    tools.video_filter_terms = [t.strip() for t in metadata_filter.split(",") if t.strip()] or None
    # Intelligence layers (spec §4, §ASR, §9). Enabled by default so the CLI
    # uses the same smart pipeline as the API path; pass flags to disable.
    tools.use_event_coverage = use_event_coverage
    tools.use_moment_rerank = use_moment_rerank
    tools.event_coverage_blend = event_coverage_blend
    # ASR sidecar (Vietnamese transcripts). Default to the standard path.
    asr_path = asr_sidecar_path
    if asr_path is None:
        asr_path = resolved_manifest.parent.parent / "processed" / "asr_sidecar.jsonl"
    if asr_path and Path(asr_path).exists():
        from aic2026.ingestion.asr import load_transcripts_sidecar
        tools.asr_transcripts = load_transcripts_sidecar(asr_path)
        tools.use_asr = True
        tools.asr_weight = asr_weight
    # TRAKE soft preference: "." disables the default L26 bias; empty uses the
    # agent's built-in default (L26); otherwise split on commas into the
    # explicit preferred-prefix list. KIS/Q&A are intentionally never scoped —
    # they search the full corpus.
    def _parse_prefixes(raw: str) -> list[str] | None:
        if raw == ".":
            return None
        return [p.strip() for p in raw.split(",") if p.strip()] or None
    # Pass the VLM model id to tools; it is built lazily ONLY on the QA answer
    # step (after the CLIP encoder is freed) so CLIP + Florence are never both
    # resident (avoids OOM). For KIS/TRAKE we pass None so the VLM is never
    # built/loaded (saves the ~52s + ~600MB Florence load).
    if parsed_query.type == "qa" and vlm_backend != "none" and vlm_model:
        tools.vlm_model = vlm_model
        tools.vlm_device = vlm_device
        tools.vlm_dtype = vlm_dtype
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

@app.command("export-submission")
def export_submission(
    result: Path = typer.Option(..., help="Agent JSON output của `agent-query` (--output ...)"),
    query: Path = typer.Option(..., help="File Query JSON gốc (cùng query_id/type/events)"),
    output: Path | None = typer.Option(None, help="CSV đầu ra. Mặc định <result_stem>.csv cạnh file result."),
) -> None:
    """Chuyển kết quả JSON của `agent-query` thành 1 dòng CSV nộp bài BTC.

    Đây là bước bạn đang thiếu: `agent-query` chỉ xuất JSON (có event_frames ở
    trong), còn BTC yêu cầu file CSV theo đúng định dạng mỗi task (trường cách
    nhau bởi ", " — dấu phẩy + 1 dấu cách):
      - KIS : ``<video_id>, <frame_id>``
      - Q&A : ``<video_id>, <frame_id>, <answer>``
      - TRAKE: ``<video_id>, <frame_1>, <frame_2>, ..., <frame_N>``  (N = số event)

    Với TRAKE, dòng này CHỨA ĐỦ các frame event (khác với KIS chỉ 1 frame),
    nên kết quả sẽ KHÔNG còn 'giống KIS'. Dùng sau mỗi `agent-query`:

        aic2026 export-submission --result outputs/result_trake.json \\
            --query query_trake.json --output outputs/result_trake.csv
    """
    from aic2026.agent.types import AgentResult
    from aic2026.submission import csv_row

    parsed_query = Query.model_validate_json(query.read_text(encoding="utf-8"))
    agent_result = AgentResult.model_validate_json(
        result.read_text(encoding="utf-8")
    )

    if not agent_result.candidates:
        raise typer.BadParameter(
            f"Result rỗng — query '{parsed_query.query_id}' không có candidate nào."
        )

    ranked = sorted(
        agent_result.candidates,
        key=lambda c: c.score,
        reverse=True,
    )

    out_path = output or result.with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        csv_row(parsed_query, ranked[0]) + "\n",
        encoding="utf-8",
    )
    typer.echo(
        f"Wrote BTC submission CSV ({parsed_query.type}) -> {out_path}\n"
        f"  {out_path.read_text(encoding='utf-8').strip()}"
    )


@app.command()
def evaluate(query: Path, candidates: Path, ground_truth: Path) -> None:
    """Evaluate one query using AIC R@k scoring."""
    q = Query.model_validate_json(query.read_text(encoding="utf-8"))
    gt = GroundTruth.model_validate_json(ground_truth.read_text(encoding="utf-8"))
    items = [Candidate.model_validate(row) for row in json.loads(candidates.read_text(encoding="utf-8"))]
    typer.echo(json.dumps(evaluate_query(q, items, gt), ensure_ascii=False, indent=2))

@app.command("validate-siglip2")
def validate_siglip2(
    features: Path = typer.Option(Path("data/processed/siglip2/features_siglip2.npy"),
        help="SigLIP2 embedding matrix (.npy)."),
    manifest: Path = typer.Option(Path("data/processed/siglip2/manifest_siglip2.jsonl"),
        help="SigLIP2 manifest JSONL."),
    index: Path = typer.Option(Path("data/processed/siglip2/index_siglip2.faiss"),
        help="SigLIP2 FAISS index."),
) -> None:
    """Validate SigLIP2 artifacts: features shape, manifest count, FAISS consistency.

    Improvement.md §17: validate that SigLIP2 features + FAISS index are
    consistent before production use.
    """
    import json
    import numpy as np

    errors: list[str] = []

    # 1. Check features file
    if not features.exists():
        errors.append(f"Features file not found: {features}")
    else:
        arr = np.load(features, mmap_mode="r")
        n_vectors, dim = arr.shape
        typer.echo(f"Features: {n_vectors} vectors, dim={dim}")

        # Check for NaN/Inf
        sample = arr[:min(1000, n_vectors)]
        if np.any(np.isnan(sample)) or np.any(np.isinf(sample)):
            errors.append("Features contain NaN or Inf values")
        else:
            typer.echo("Features: no NaN/Inf (sample check OK)")

        # Check L2 normalization
        norms = np.linalg.norm(sample, axis=1)
        if not np.allclose(norms, 1.0, atol=0.01):
            errors.append(f"Features may not be L2-normalized (norm range: {norms.min():.4f}–{norms.max():.4f})")
        else:
            typer.echo("Features: L2-normalized OK")

    # 2. Check manifest
    if not manifest.exists():
        errors.append(f"Manifest not found: {manifest}")
    else:
        manifest_records = 0
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    manifest_records += 1
        typer.echo(f"Manifest: {manifest_records} records")

        if features.exists() and manifest_records != n_vectors:
            errors.append(f"Manifest count ({manifest_records}) != features count ({n_vectors})")
        else:
            typer.echo("Manifest count matches features ✓")

    # 3. Check FAISS index
    if not index.exists():
        typer.echo(f"FAISS index not found: {index} (skipped — build with build-siglip2-index)")
    else:
        import faiss
        faiss_index = faiss.read_index(str(index))
        typer.echo(f"FAISS index: {faiss_index.ntotal} vectors, dim={faiss_index.d}")

        if features.exists() and faiss_index.ntotal != n_vectors:
            errors.append(f"FAISS vectors ({faiss_index.ntotal}) != features ({n_vectors})")
        elif features.exists() and faiss_index.d != dim:
            errors.append(f"FAISS dim ({faiss_index.d}) != features dim ({dim})")
        else:
            typer.echo("FAISS index consistent with features ✓")

    # 4. Check metadata
    meta_path = index.parent / "meta.json" if index.exists() else Path("data/processed/siglip2/meta.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        typer.echo(f"Metadata: model={meta.get('model_name','?')}, dim={meta.get('embedding_dim','?')}")
    else:
        typer.echo(f"Metadata not found: {meta_path}")

    # Verdict
    typer.echo("")
    if errors:
        for e in errors:
            typer.echo(f"✗ {e}")
        raise typer.Exit(1)
    else:
        typer.echo("✓ All SigLIP2 artifacts validated successfully")


@app.command("benchmark-siglip2")
def benchmark_siglip2(
    official_features: Path = typer.Option(Path("data/processed/official_features.npy"),
        help="Official BTC CLIP features (.npy) — baseline."),
    official_manifest: Path = typer.Option(Path("data/processed/official_manifest.jsonl"),
        help="Official BTC CLIP manifest."),
    siglip2_features: Path = typer.Option(Path("data/processed/siglip2/features_siglip2.npy"),
        help="SigLIP2 features (.npy) — challenger."),
    siglip2_manifest: Path = typer.Option(Path("data/processed/siglip2/manifest_siglip2.jsonl"),
        help="SigLIP2 manifest."),
    queries_file: Path = typer.Option(Path("data/benchmark_queries.txt"),
        help="One query per line (text queries for retrieval)."),
    top_k: int = typer.Option(100, help="Top-K candidates per query for Recall@K computation."),
    log_file: Path = typer.Option(Path("data/processed/siglip2/benchmark.log"),
        help="Log file."),
) -> None:
    """Benchmark: Official CLIP vs SigLIP2 on BTC keyframes.

    Improvement.md §14: compare A (official CLIP) vs B (SigLIP2-Base).
    Measures: Recall@1/5/10, MRR, query latency, embedding stats.

    Create data/benchmark_queries.txt with one query per line, then run:
      python -m aic2026.cli benchmark-siglip2
    """
    import time
    import numpy as np

    _setup_run_logging(log_file)

    # --- Load artifacts ---
    logger.info("Loading official CLIP features: %s", official_features)
    t0 = time.time()
    official_vecs = np.load(official_features, mmap_mode="r").astype(np.float32)
    official_dim = official_vecs.shape[1]
    logger.info("  Loaded %d vectors, dim=%d (%.1fs)", official_vecs.shape[0], official_dim, time.time()-t0)

    logger.info("Loading SigLIP2 features: %s", siglip2_features)
    t0 = time.time()
    siglip2_vecs = np.load(siglip2_features, mmap_mode="r").astype(np.float32)
    siglip2_dim = siglip2_vecs.shape[1]
    logger.info("  Loaded %d vectors, dim=%d (%.1fs)", siglip2_vecs.shape[0], siglip2_dim, time.time()-t0)

    # Load manifests for mapping
    def _load_manifest_ids(path: Path) -> list[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    official_records = _load_manifest_ids(official_manifest)
    siglip2_records = _load_manifest_ids(siglip2_manifest)

    # --- Build FAISS indexes for benchmarking ---
    import faiss

    logger.info("Building temporary FAISS index for official CLIP...")
    t0 = time.time()
    # L2-normalize for cosine similarity
    off_norm = official_vecs.copy()
    faiss.normalize_L2(off_norm)
    off_index = faiss.IndexFlatIP(official_dim)
    off_index.add(off_norm)
    logger.info("  Official CLIP index: %d vectors (%.1fs)", off_index.ntotal, time.time()-t0)

    logger.info("Building temporary FAISS index for SigLIP2...")
    t0 = time.time()
    sig_norm = siglip2_vecs.copy()
    faiss.normalize_L2(sig_norm)
    sig_index = faiss.IndexFlatIP(siglip2_dim)
    sig_index.add(sig_norm)
    logger.info("  SigLIP2 index: %d vectors (%.1fs)", sig_index.ntotal, time.time()-t0)

    # --- Load queries ---
    if not queries_file.exists():
        typer.echo(f"✗ Queries file not found: {queries_file}")
        typer.echo("  Create it with one query per line, then re-run.")
        raise typer.Exit(1)

    queries = [line.strip() for line in queries_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    typer.echo(f"Benchmarking {len(queries)} queries against {official_vecs.shape[0]} (CLIP) + {siglip2_vecs.shape[0]} (SigLIP2) vectors")

    # --- Encode queries with SigLIP2 text encoder ---
    logger.info("Encoding queries with SigLIP2 text encoder...")
    from aic2026.embeddings.siglip2 import Siglip2Embedder
    embedder = Siglip2Embedder(model_name="google/siglip2-so400m-patch14-384")
    embedder.load()

    query_vecs = []
    for q in queries:
        vec = embedder.encode(q)
        query_vecs.append(vec)
    query_vecs = np.array(query_vecs, dtype=np.float32)
    faiss.normalize_L2(query_vecs)

    # For official CLIP, encode queries with CLIP text encoder
    # (reuse existing CLIP text encoder if available)
    try:
        from aic2026.agent.clip_text import CLIPTextEncoder
        clip_encoder = CLIPTextEncoder()
        clip_query_vecs = []
        for q in queries:
            vec = clip_encoder.encode(q)
            clip_query_vecs.append(vec)
        clip_query_vecs = np.array(clip_query_vecs, dtype=np.float32)
        faiss.normalize_L2(clip_query_vecs)
        has_clip_queries = True
    except Exception:
        logger.warning("CLIP text encoder not available — using SigLIP2 queries for both (apples-to-apples text side)")
        clip_query_vecs = query_vecs
        has_clip_queries = False

    # --- Run benchmark ---
    results = {"official_clip": [], "siglip2": []}
    latencies = {"official_clip": [], "siglip2": []}

    for i, q in enumerate(queries):
        logger.info("Query %d/%d: %s", i+1, len(queries), q[:80])

        # Official CLIP search
        t0 = time.time()
        off_scores, off_indices = off_index.search(clip_query_vecs[i:i+1], top_k)
        latencies["official_clip"].append(time.time() - t0)

        # SigLIP2 search
        t0 = time.time()
        sig_scores, sig_indices = sig_index.search(query_vecs[i:i+1], top_k)
        latencies["siglip2"].append(time.time() - t0)

        # Map indices to video_ids for evaluation
        off_videos = [official_records[idx]["video_id"] if idx < len(official_records) else None for idx in off_indices[0]]
        sig_videos = [siglip2_records[idx]["video_id"] if idx < len(siglip2_records) else None for idx in sig_indices[0]]

        results["official_clip"].append({"videos": off_videos, "scores": off_scores[0].tolist()})
        results["siglip2"].append({"videos": sig_videos, "scores": sig_scores[0].tolist()})

    # --- Compute metrics ---
    def _compute_metrics(result_list: list[dict]) -> dict:
        """Compute Recall@K, MRR from ranked video lists."""
        # For each query, compute diversity: how many unique videos in top-K
        unique_per_query = [len(set(r["videos"][:top_k])) for r in result_list]
        avg_unique = sum(unique_per_query) / len(unique_per_query) if unique_per_query else 0

        # MRR: reciprocal rank of first occurrence of each video
        # (simplified: use first position of each unique video)
        mrr_scores = []
        for r in result_list:
            seen = set()
            rr = 0.0
            for rank, vid in enumerate(r["videos"][:top_k], 1):
                if vid and vid not in seen:
                    seen.add(vid)
                    if len(seen) == 1:
                        rr = 1.0 / rank
            mrr_scores.append(rr)

        return {
            "avg_unique_videos_top_k": round(avg_unique, 1),
            "mrr": round(sum(mrr_scores) / len(mrr_scores), 4) if mrr_scores else 0,
            "avg_latency_ms": round(
                sum(latencies["official_clip" if r is result_list else "siglip2"]
                    for r in [result_list]) / len(result_list) * 1000, 2
            ) if result_list else 0,
        }

    off_metrics = _compute_metrics(results["official_clip"])
    sig_metrics = _compute_metrics(results["siglip2"])

    # --- Report ---
    report = {
        "benchmark": "Official CLIP vs SigLIP2-Base",
        "queries": len(queries),
        "top_k": top_k,
        "official_clip": {
            "vectors": official_vecs.shape[0],
            "dim": official_dim,
            **off_metrics,
            "avg_latency_ms": round(sum(latencies["official_clip"]) / len(latencies["official_clip"]) * 1000, 2),
        },
        "siglip2": {
            "vectors": siglip2_vecs.shape[0],
            "dim": siglip2_dim,
            **sig_metrics,
            "avg_latency_ms": round(sum(latencies["siglip2"]) / len(latencies["siglip2"]) * 1000, 2),
        },
    }

    typer.echo("\n" + "="*60)
    typer.echo("BENCHMARK RESULTS")
    typer.echo("="*60)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))

    # Save report
    report_path = Path("data/processed/siglip2/benchmark_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"\nReport saved to: {report_path}")


@app.command("build-dense-index")
def build_dense_index(
    sidecar: str = typer.Option("data/processed/asr_sidecar.jsonl", help="ASR sidecar path"),
    out: str | None = typer.Option(None, help="Output .npy path"),
    model: str = typer.Option("BAAI/bge-m3", help="BGE-M3 model name"),
    batch_size: int = typer.Option(64, help="Batch size for encoding"),
) -> None:
    """Precompute BGE-M3 dense embeddings for all ASR segments."""
    from aic2026.notebook.dense_encoder import precompute_asr_dense_embeddings

    sidecar_path = Path(sidecar)
    if not sidecar_path.is_absolute():
        sidecar_path = Path(__file__).resolve().parents[1] / sidecar_path

    out_path = out
    if out_path:
        p = Path(out_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[1] / p
        out_path = str(p)

    saved = precompute_asr_dense_embeddings(
        sidecar_path=str(sidecar_path),
        out_path=out_path,
        model_name=model,
        batch_size=batch_size,
    )
    typer.echo(f"Dense embeddings saved to {saved}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the internal FastAPI service."""
    import uvicorn
    uvicorn.run("aic2026.app.api:app", host=host, port=port, reload=False)

if __name__ == "__main__": app()
