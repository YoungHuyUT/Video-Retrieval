from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from aic2026.agent.types import AgentResult
from aic2026.models import Query

app = FastAPI(title="AIC 2026 Agent API", version="0.2.0")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    """Trang chào để mở localhost:8000 không trả 404."""
    return """<!doctype html><html lang='vi'><head><meta charset='utf-8'><title>AIC 2026 Agent API</title>
    <style>body{font-family:system-ui;max-width:760px;margin:4rem auto;line-height:1.6}a{color:#2563eb}</style></head>
    <body><h1>AIC 2026 Agent API đang chạy</h1><p>Backend ở cổng 8000. Giao diện thi Streamlit chạy riêng ở cổng 8501.</p>
    <ul><li><a href='/docs'>API docs (/docs)</a></li><li><a href='/health'>Health check (/health)</a></li><li><a href='http://127.0.0.1:8501'>Giao diện Streamlit (cần chạy lệnh riêng)</a></li></ul>
    <pre>aic2026 serve
streamlit run src/aic2026/app/ui.py</pre></body></html>"""


class RuntimeConfig(BaseModel):
    """Cấu hình nội bộ; UI gửi để backend nạp đúng index/encoder."""
    manifest_path: str = "data/processed/derived_manifest.jsonl"
    features_path: str = "data/processed/derived_features.npy"
    clip_pretrained: str = "openai"
    llm_model: str = "qwen3.5:4b"
    ollama_url: str = "http://127.0.0.1:11434"
    backend: str = "auto"
    chroma_dir: str = "data/indexes/chroma"
    metadata_filter: str = ""
    vlm_backend: str = "ollama"
    vlm_model: str = "qwen2.5vl:3b"
    vlm_device: str | None = None
    vlm_dtype: str = "bfloat16"
    vlm_timeout: int = 120
    coarse_top_k: int = 200


class TaskRequest(BaseModel):
    query_id: str = "live-query"
    text: str = Field(min_length=1)
    question: str | None = None
    events: list[str] = Field(default_factory=list)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


@lru_cache(maxsize=4)
def load_orchestrator(
    manifest_path: str,
    features_path: str,
    clip_pretrained: str,
    llm_model: str,
    ollama_url: str,
    backend: str = "auto",
    chroma_dir: str = "data/indexes/chroma",
    metadata_filter: str = "",
    vlm_backend: str = "ollama",
    vlm_model: str = "qwen2.5vl:3b",
    vlm_device: str | None = None,
    vlm_dtype: str = "bfloat16",
    vlm_timeout: int = 120,
    coarse_top_k: int = 200,
):
    """Shared layer: index + encoders load once per runtime configuration."""
    from aic2026.agent import OllamaLLM, RetrievalAgent
    from aic2026.agent.tools import RetrievalTools
    from aic2026.embeddings import OpenCLIPTextEmbedder
    from aic2026.ingestion import load_manifest
    from aic2026.retrieval import RetrievalPipeline
    from aic2026.retrieval.factory import load_index_for_query

    manifest_file, features_file = Path(manifest_path), Path(features_path)
    missing = [str(path) for path in (manifest_file, features_file) if not path.exists()]
    if missing:
        hint = (
            "Chạy: python -m aic2026.cli embed-keyframes --video-id L21_V001 "
            "(hoặc bỏ --video-id để encode toàn bộ keyframes)."
        )
        raise FileNotFoundError(f"Thiếu index retrieval: {', '.join(missing)}. {hint}")
    manifest_records = load_manifest(manifest_file)
    index = load_index_for_query(
        features_file,
        manifest_records,
        backend=backend,
        chroma_dir=chroma_dir,
    )
    pipeline = RetrievalPipeline(index, manifest_records, frames_per_video=20)
    text_encoder = OpenCLIPTextEmbedder(pretrained=clip_pretrained)
    tools = RetrievalTools(pipeline, text_encoder.encode)
    tools.coarse_top_k = coarse_top_k
    tools.video_filter_terms = [t.strip() for t in metadata_filter.split(",") if t.strip()] or None
    # Nối BM25 lexical vào retrieval (text từ Objects/Metadata). tools.retrieve tự
    # động bỏ qua khi manifest không có text (is_empty), nên an toàn cả khi chưa nạp data.
    from aic2026.retrieval import BM25Index
    tools.bm25_index = BM25Index(manifest_records)
    if vlm_backend != "none" and vlm_model:
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
    agent = RetrievalAgent(
        tools,
        llm=OllamaLLM(
            model=llm_model,
            base_url=ollama_url,
            timeout_seconds=600,
            num_predict=1500,
            keep_alive="0",
        ),
    )
    agent.coarse_top_k = coarse_top_k
    return agent


def run_task(task_type: str, request: TaskRequest) -> AgentResult:
    if task_type == "qa" and not request.question:
        raise HTTPException(422, "Q&A cần trường question")
    if task_type == "trake" and not request.events:
        raise HTTPException(422, "TRAKE cần events theo đúng thứ tự")
    try:
        config = request.runtime
        orchestrator = load_orchestrator(
            config.manifest_path,
            config.features_path,
            config.clip_pretrained,
            config.llm_model,
            config.ollama_url,
            backend=config.backend,
            chroma_dir=config.chroma_dir,
            vlm_backend=config.vlm_backend,
            vlm_model=config.vlm_model,
            vlm_device=config.vlm_device,
            vlm_dtype=config.vlm_dtype,
            vlm_timeout=config.vlm_timeout,
        )
        return orchestrator.run(Query(query_id=request.query_id, type=task_type, text=request.text, question=request.question, events=request.events))
    except HTTPException:
        raise
    except FileNotFoundError as error:
        raise HTTPException(503, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        raise HTTPException(500, str(error)) from error


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "flows": "kis,qa,trake"}


@app.get("/health/ready")
def ready(
    manifest_path: str = "data/processed/derived_manifest.jsonl",
    features_path: str = "data/processed/derived_features.npy",
    chroma_dir: str = "data/indexes/chroma",
) -> dict[str, object]:
    """Kiểm tra nhanh trước khi UI gọi agent (tránh chờ load model rồi mới báo thiếu file)."""
    from aic2026.retrieval import ChromaVectorStore
    manifest_ok = Path(manifest_path).exists()
    features_ok = Path(features_path).exists()
    chroma_ok = ChromaVectorStore.available() and (Path(chroma_dir) / "chroma.sqlite3").exists()
    return {
        "ready": manifest_ok and features_ok,
        "manifest_path": manifest_path,
        "manifest_exists": manifest_ok,
        "features_path": features_path,
        "features_exists": features_ok,
        "chroma_dir": chroma_dir,
        "chroma_ok": chroma_ok,
    }


@app.post("/tasks/kis/run", response_model=AgentResult)
def run_kis(request: TaskRequest) -> AgentResult:
    """KIS flow: plan -> retrieval -> judge -> KIS format candidates."""
    return run_task("kis", request)


@app.post("/tasks/qa/run", response_model=AgentResult)
def run_qa(request: TaskRequest) -> AgentResult:
    """Q&A flow: plan -> retrieval -> judge -> local VLM/manual answer stage."""
    return run_task("qa", request)


@app.post("/tasks/trake/run", response_model=AgentResult)
def run_trake(request: TaskRequest) -> AgentResult:
    """TRAKE flow: plan -> per-event retrieval -> judge -> deterministic temporal alignment."""
    return run_task("trake", request)
