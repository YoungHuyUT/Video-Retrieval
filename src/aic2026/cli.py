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
    from aic2026.ingestion.official_index import build_official_index
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


@app.command("ocr-manifest")
def ocr_manifest(
    manifest: Path = typer.Option(Path("data/processed/derived_manifest.jsonl"), help="Manifest JSONL input"),
    output: Path | None = typer.Option(None, help="Manifest JSONL output. Mặc định ghi đè (in-place) vào --manifest để agent-query tự dùng."),
    keyframes_root: Path = typer.Option(Path("data/raw/Keyframes"), help="Gốc chứa thư mục keyframe để resolve đường dẫn tương đối"),
    batch_size: int = typer.Option(16, min=1),
    lang: str = typer.Option("vi", help="Ngôn ngữ OCR: vi | en"),
) -> None:
    """OCR toàn bộ keyframe trong manifest, ghi text nhận dạng vào object_labels (build 1 lần).

    Mặc định ghi đè vào chính file --manifest (in-place) để bước agent-query /
    build-chroma-index / BM25 index tự động hưởng lợi từ text OCR. Dùng --output
    để ghi file riêng (không đụng manifest gốc).
    """
    from aic2026.ingestion import load_manifest
    from aic2026.models import FrameRecord
    from aic2026.qa.ocr import OCRTextExtractor

    records = load_manifest(manifest)
    extractor = OCRTextExtractor(lang=lang)
    extractor._ensure_loaded()
    if extractor._ocr is None:
        typer.echo(
            "PaddleOCR không khởi động được — xem warning phía trên. "
            "Nếu có libpaddle.pyd/DLL load failed, sửa PaddlePaddle/Visual C++ runtime trước."
        )
        raise typer.Exit(code=1)

    enriched: list[FrameRecord] = []
    total = len(records)
    found_images = 0
    recognized_lines = 0
    typer.echo(
        f"OCR bắt đầu: {total:,} manifest records | lang={lang}"
    )
    # Click/Typer render một progress bar thật trong cmd/PowerShell, thay vì
    # in hàng trăm dòng log. Các counter chi tiết được in ở summary cuối.
    with typer.progressbar(
        length=total,
        label="OCR keyframes",
        show_percent=True,
        show_pos=True,
    ) as progress:
        for start in range(0, total, batch_size):
            record_batch = records[start : start + batch_size]
            resolved: list[Path | None] = []
            image_paths: list[Path] = []
            for record in record_batch:
                # keyframe_path có thể là đường dẫn đầy đủ hoặc tương đối.
                candidates = [
                    Path(record.keyframe_path),
                    keyframes_root / record.video_id / Path(record.keyframe_path).name,
                    keyframes_root / record.keyframe_path,
                ]
                chosen = next((candidate for candidate in candidates if candidate.exists()), None)
                resolved.append(chosen)
                if chosen is not None:
                    image_paths.append(chosen)

            # Một lần Paddle inference cho cả batch ảnh có tồn tại; những record
            # thiếu JPG vẫn giữ nguyên object_labels và chỉ cập nhật progress.
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

    target = output if output is not None else manifest
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(r.model_dump_json() for r in enriched) + ("\n" if enriched else ""), encoding="utf-8")
    if output is None:
        typer.echo(
            f"OCR hoàn tất: wrote {len(enriched):,} records IN-PLACE to {target} "
            f"| images={found_images:,} | text-lines={recognized_lines:,}"
        )
    else:
        typer.echo(
            f"OCR hoàn tất: wrote {len(enriched):,} records to {target} "
            f"| images={found_images:,} | text-lines={recognized_lines:,}"
        )


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
