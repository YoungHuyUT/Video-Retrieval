from __future__ import annotations

import json
from pathlib import Path
import typer

from aic2026.evaluation import evaluate_query
from aic2026.ingestion import build_manifest
from aic2026.models import Candidate, GroundTruth, Query
from aic2026.retrieval import VectorIndex

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
    encoder: str = typer.Option("siglip2", help="siglip2 hoặc clip"),
    model_id: str = typer.Option("google/siglip2-base-patch16-224"),
    clip_pretrained: str = typer.Option("openai", help="Checkpoint OpenCLIP khi encoder=clip"),
) -> None:
    """Trích toàn bộ frame, CLIP-embed, rồi loại near-duplicate bằng cosine similarity."""
    from aic2026.data_platform import OpenCLIPFrameEncoder, extract_deduplicated_keyframes
    from aic2026.embeddings import SigLIPEncoder
    if encoder == "siglip2":
        image_encoder = SigLIPEncoder(model_id)
    elif encoder == "clip":
        image_encoder = OpenCLIPFrameEncoder(pretrained=clip_pretrained)
    else:
        raise typer.BadParameter("encoder chỉ nhận siglip2 hoặc clip")
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
    encoder: str = typer.Option("siglip2", help="siglip2 hoặc clip"),
    model_id: str = typer.Option("google/siglip2-base-patch16-224"),
    clip_pretrained: str = typer.Option("openai", help="Checkpoint OpenCLIP khi encoder=clip"),
    batch_size: int = typer.Option(4, min=1, help="Số ảnh mỗi batch để giảm mức dùng RAM"),
    video_id: str | None = typer.Option(None, help="Chỉ encode 1 thư mục keyframe, ví dụ L21_V001"),
) -> None:
    """Encode existing JPG keyframes on disk into .npz archives and build the retrieval index inputs."""
    from aic2026.data_platform import build_derived_artifacts, embed_existing_keyframes
    from aic2026.embeddings import SigLIPEncoder

    if encoder == "siglip2":
        image_encoder = SigLIPEncoder(model_id)
    elif encoder == "clip":
        from aic2026.data_platform import OpenCLIPFrameEncoder
        image_encoder = OpenCLIPFrameEncoder(pretrained=clip_pretrained)
    else:
        raise typer.BadParameter("encoder chỉ nhận siglip2 hoặc clip")

    archives = embed_existing_keyframes(keyframes_dir, features_dir, image_encoder, batch_size=batch_size, video_id=video_id)
    count = build_derived_artifacts(keyframes_dir, features_dir, manifest, features, video_ids=[video_id] if video_id else None)
    typer.echo(f"Created {len(archives)} feature archives under {features_dir}; wrote {count} records to {manifest} and {features}")

@app.command("check-features")
def check_features(features: Path, manifest: Path = Path("data/processed/manifest.jsonl")) -> None:
    """Fail early unless provided CLIP features and the manifest have the same order/count."""
    from aic2026.ingestion import load_manifest
    index = VectorIndex.from_npy(features)
    count = len(load_manifest(manifest))
    if len(index.vectors) != count:
        raise typer.BadParameter(f"features has {len(index.vectors)} vectors; manifest has {count} records")
    typer.echo(f"OK: {count} feature vectors match the manifest")

@app.command("agent-query")
def agent_query(
    query: Path = typer.Option(..., help="JSON file matching the Query schema"),
    features: Path = typer.Option(..., help="BTC CLIP .npy feature file"),
    manifest: Path = typer.Option(Path("data/processed/manifest.jsonl")),
    llm_model: str = typer.Option("qwen3:8b"),
    ollama_url: str = typer.Option("http://127.0.0.1:11434"),
    encoder: str = typer.Option("clip", help="clip cho feature BTC, siglip2 cho index tự tạo"),
    encoder_model: str = typer.Option("google/siglip2-base-patch16-224"),
    output: Path | None = typer.Option(None),
) -> None:
    """Run the bounded local-LLM agent and save auditable candidates/trace."""
    from aic2026.agent import OllamaLLM, RetrievalAgent
    from aic2026.agent.tools import RetrievalTools
    from aic2026.embeddings import OpenCLIPTextEmbedder, SigLIPEncoder
    from aic2026.ingestion import load_manifest
    from aic2026.retrieval import RetrievalPipeline

    parsed_query = Query.model_validate_json(query.read_text(encoding="utf-8"))
    pipeline = RetrievalPipeline(VectorIndex.from_npy(features), load_manifest(manifest))
    if encoder == "clip":
        text_encoder = OpenCLIPTextEmbedder()
        encode_text = text_encoder.encode
    elif encoder == "siglip2":
        text_encoder = SigLIPEncoder(encoder_model)
        encode_text = text_encoder.encode_text
    else:
        raise typer.BadParameter("encoder chỉ nhận clip hoặc siglip2")
    agent = RetrievalAgent(OllamaLLM(model=llm_model, base_url=ollama_url), RetrievalTools(pipeline, encode_text))
    result = agent.run(parsed_query)
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
