from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from aic2026.agent.types import AgentResult
from aic2026.ingestion import resolve_feature_sources
from aic2026.models import Query

app = FastAPI(title="AIC 2026 Agent API", version="0.2.0")
logger = logging.getLogger(__name__)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    """Trang chào để mở localhost:8000 không trả 404."""
    return """<!doctype html><html lang='vi'><head><meta charset='utf-8'><title>AIC 2026 Agent API</title>
    <style>body{font-family:system-ui;max-width:760px;margin:4rem auto;line-height:1.6}a{color:#2563eb}</style></head>
    <body><h1>AIC 2026 Agent API đang chạy</h1><p>Backend ở cổng 8000. Giao diện thi Streamlit chạy riêng ở cổng 8501.</p>
    <ul><li><a href='/docs'>API docs (/docs)</a></li><li><a href='/health'>Health check (/health)</a></li><li><a href='http://127.0.0.1:8501'>Giao diện Streamlit (cần chạy lệnh riêng)</a></li></ul>
    <pre>aic2026 serve
streamlit run src/aic2026/app/ui.py</pre></body></html>"""


# RAM Cache lưu trữ ảnh để giảm thiểu I/O đĩa
_IMAGE_CACHE: dict[str, bytes] = {}


@app.get("/images", include_in_schema=True)
def get_image(path: str) -> Response:
    """Trả về file ảnh trực tiếp. Cache lại dạng byte trong RAM để tăng tốc tối đa."""
    safe_path = Path(path).resolve()
    if not safe_path.exists():
        raise HTTPException(404, "Không tìm thấy file ảnh.")

    path_str = str(safe_path)
    if path_str in _IMAGE_CACHE:
        return Response(content=_IMAGE_CACHE[path_str], media_type="image/jpeg")

    try:
        with open(safe_path, "rb") as handle:
            img_bytes = handle.read()
    except Exception as exc:
        raise HTTPException(500, f"Không thể đọc file ảnh: {exc}")

    # Giới hạn kích thước cache khoảng 300 ảnh để tránh OOM
    if len(_IMAGE_CACHE) > 300:
        _IMAGE_CACHE.clear()

    _IMAGE_CACHE[path_str] = img_bytes
    return Response(content=img_bytes, media_type="image/jpeg")



class RuntimeConfig(BaseModel):
    """Cấu hình nội bộ; UI gửi để backend nạp đúng index/encoder."""
    manifest_path: str | None = None
    features_path: str | None = None
    clip_model: str = "ViT-SO400M-14-SigLIP-384"
    clip_pretrained: str = "webli"
    llm_model: str = "qwen3.5:4b"
    ollama_url: str = "http://127.0.0.1:11434"
    # The official CLIP matrix is static; FAISS is faster than starting a
    # Chroma collection and is the default execution path.
    backend: str = "faiss"
    chroma_dir: str = "data/indexes/chroma"
    late_interaction_weight: float = 0.0
    metadata_filter: str = ""
    translate_query: bool = False
    vlm_backend: str = "florence"
    vlm_model: str = "microsoft/Florence-2-base-ft"
    vlm_device: str | None = None
    vlm_dtype: str = "float32"
    vlm_timeout: int = 120
    coarse_top_k: int = 200
    # Số video gửi VLM tối đa (top-N video score cao nhất). 0 = tất cả.
    # Giới hạn này giúp giảm thời gian VLM khi có nhiều candidate video.
    vlm_top_videos: int = 20
    # Số thread song song gọi VLM (mỗi video 1 thread). Giảm xuống 3 để tránh
    # nghẽn Ollama local — Ollama xử lý tuần tự nên thread quá nhiều chỉ tăng queue.
    vlm_max_workers: int = 3


class TaskRequest(BaseModel):
    query_id: str = "live-query"
    text: str = Field(min_length=1)
    question: str | None = None
    events: list[str] = Field(default_factory=list)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


class AnswerRequest(BaseModel):
    """Phase-2 QA: gửi lại candidates đã retrieve, chỉ chạy VLM để sinh đáp án."""
    question: str = Field(min_length=1)
    candidates: list[dict] = Field(default_factory=list)  # Candidate.model_dump() list
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


@lru_cache(maxsize=4)
def load_orchestrator(
    manifest_path: str,
    features_path: str,
    clip_pretrained: str,
    llm_model: str,
    ollama_url: str,
    backend: str = "faiss",
    chroma_dir: str = "data/indexes/chroma",
    metadata_filter: str = "",
    translate_query: bool = False,
    vlm_backend: str = "florence",
    late_interaction_weight: float = 0.0,
    vlm_model: str = "microsoft/Florence-2-base-ft",
    vlm_device: str | None = None,
    vlm_dtype: str = "float32",
    vlm_timeout: int = 120,
    coarse_top_k: int = 200,
    query_type: str | None = None,
    clip_model: str = "ViT-SO400M-14-SigLIP-384",
):
    """Shared layer: index + encoders load once per runtime configuration."""
    from aic2026.agent import OllamaLLM, RetrievalAgent
    from aic2026.agent.tools import RetrievalTools
    from aic2026.embeddings import OpenCLIPTextEmbedder
    from aic2026.ingestion import load_manifest
    from aic2026.retrieval import RetrievalPipeline
    from aic2026.retrieval.factory import load_index_for_query

    manifest_file, features_file = resolve_feature_sources(manifest_path, features_path)
    missing = [str(path) for path in (manifest_file, features_file) if not path.exists()]
    if missing:
        hint = (
            "Thiếu index retrieval. Ưu tiên official_features.npy (CLIP sẵn BTC, "
            "sinh bởi `aic2026 prepare-official`); fallback derived_features.npy "
            "(sinh bởi `aic2026 embed-keyframes`). Chạy một trong hai lệnh trên."
        )
        raise FileNotFoundError(f"Thiếu index retrieval: {', '.join(missing)}. {hint}")
    manifest_records = load_manifest(manifest_file)
    # Per-video metadata (title/description/keywords) sống cùng thư mục với
    # manifest. Load vào store để pipeline chạy được metadata pre-filter
    # (filter_videos_by_metadata) — nếu không truyền, store rỗng và pre-filter
    # bị skip dù tools.video_filter_terms có set.
    from aic2026.retrieval import VideoMetadataStore

    video_metadata = VideoMetadataStore.load(
        manifest_file.parent / "video_metadata.jsonl"
    )
    index = load_index_for_query(
        features_file,
        manifest_records,
        backend=backend,
        chroma_dir=chroma_dir,
    )
    pipeline = RetrievalPipeline(
        index, manifest_records, frames_per_video=20, video_metadata=video_metadata
    )
    text_encoder = OpenCLIPTextEmbedder(model_name=clip_model, pretrained=clip_pretrained)
    tools = RetrievalTools(
        pipeline, text_encoder.encode, encode_images=text_encoder.encode_images
    )
    tools.coarse_top_k = coarse_top_k
    tools.video_filter_terms = [t.strip() for t in metadata_filter.split(",") if t.strip()] or None
    tools.late_interaction_weight = late_interaction_weight
    # Objects are frame-specific and useful for lexical retrieval.  Video
    # metadata is deliberately excluded: it is near-duplicate across frames and
    # adds memory/IDF noise without improving frame selection.
    from aic2026.retrieval import BM25Index
    tools.bm25_index = BM25Index(manifest_records)
    # Pass the VLM model id to tools; it is built lazily ONLY on the QA answer
    # step (after the CLIP encoder is freed) so CLIP + Florence are never both
    # resident (avoids OOM). For KIS/TRAKE we force "none" so the VLM is never
    # built/loaded (saves the ~52s + ~600MB Florence load).
    effective_vlm_backend = vlm_backend
    if query_type and query_type != "qa":
        effective_vlm_backend = "none"
    if effective_vlm_backend != "none" and vlm_model:
        tools.vlm_model = vlm_model
        tools.vlm_device = vlm_device
        tools.vlm_dtype = vlm_dtype
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
    agent.translate = translate_query
    return agent


def run_task(task_type: str, request: TaskRequest) -> AgentResult:
    if task_type == "qa" and not request.question:
        request.question = request.text
    if task_type == "trake" and not request.events:
        import re
        parts = [p.strip() for p in re.split(r'[\n;.]+', request.text) if p.strip()]
        request.events = parts or [request.text]
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
            metadata_filter=config.metadata_filter,
            translate_query=config.translate_query,
            vlm_backend=config.vlm_backend,
            vlm_model=config.vlm_model,
            vlm_device=config.vlm_device,
            vlm_dtype=config.vlm_dtype,
            vlm_timeout=config.vlm_timeout,
            late_interaction_weight=config.late_interaction_weight,
            query_type=task_type,
            clip_model=config.clip_model,
        )
        return orchestrator.run(Query(query_id=request.query_id, type=task_type, text=request.text, question=request.question, events=request.events))
    except HTTPException:
        raise
    except FileNotFoundError as error:
        raise HTTPException(503, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        # Log the full traceback so the exact failing line is visible in the
        # server log instead of just the client-facing message.
        logger.exception("run_task failed for task=%s: %s", task_type, error)
        raise HTTPException(500, f"{type(error).__name__}: {error}") from error


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "flows": "kis,qa,trake"}


@app.get("/health/ready")
def ready(
    manifest_path: str | None = None,
    features_path: str | None = None,
    chroma_dir: str = "data/indexes/chroma",
) -> dict[str, object]:
    """Kiểm tra nhanh trước khi UI gọi agent (tránh chờ load model rồi mới báo thiếu file)."""
    from aic2026.ingestion import resolve_feature_sources
    from aic2026.retrieval import ChromaVectorStore

    resolved_manifest, resolved_features = resolve_feature_sources(manifest_path, features_path)
    manifest_ok = resolved_manifest.exists()
    features_ok = resolved_features.exists()
    chroma_ok = ChromaVectorStore.available() and (Path(chroma_dir) / "chroma.sqlite3").exists()
    return {
        "ready": manifest_ok and features_ok,
        "manifest_path": str(resolved_manifest),
        "manifest_exists": manifest_ok,
        "features_path": str(resolved_features),
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


@app.post("/tasks/qa/candidates", response_model=AgentResult)
def run_qa_candidates(request: TaskRequest) -> AgentResult:
    """Phase-1 QA: chỉ retrieval (CLIP + BM25), không VLM. Trả candidates ngay.

    UI gọi endpoint này trước: hiển thị gallery ảnh ngay (giống KIS), sau đó mới
    gọi /tasks/qa/answers để điền đáp án từ VLM mà không block UI.
    """
    if not request.question:
        request.question = request.text
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
            metadata_filter=config.metadata_filter,
            translate_query=config.translate_query,
            # VLM không cần cho phase-1: truyền vlm_backend="none" để bỏ qua
            vlm_backend="none",
            vlm_model=config.vlm_model,
            vlm_device=config.vlm_device,
            vlm_dtype=config.vlm_dtype,
            vlm_timeout=config.vlm_timeout,
            late_interaction_weight=config.late_interaction_weight,
            query_type="qa",
            clip_model=config.clip_model,
        )
        # Phase-1 QA: retrieval only, no VLM. We pass vlm_backend="none" above so
        # tools.vlm_model is unset and ensure_vlm() is a no-op in _finalize — no
        # need to null the callables (that previously corrupted the cached agent).
        query = Query(
            query_id=request.query_id,
            type="qa",
            text=request.text,
            question=request.question,
            events=request.events,
        )
        return orchestrator.run(query)
    except HTTPException:
        raise
    except FileNotFoundError as error:
        raise HTTPException(503, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        raise HTTPException(500, str(error)) from error


@app.post("/tasks/qa/answers")
def run_qa_answers(request: AnswerRequest) -> dict[str, str]:
    """Phase-2 QA: nhận candidates đã có, chạy VLM song song. Trả {vector_id: answer}.

    Chỉ xử lý top-N video (config.vlm_top_videos) theo score giảm dần để kiểm soát
    thời gian VLM. Số thread song song: config.vlm_max_workers.
    """
    from aic2026.models import Candidate

    config = request.runtime

    if not config.vlm_backend or config.vlm_backend == "none":
        return {}

    # Parse candidates từ dict
    try:
        candidates = [Candidate.model_validate(c) for c in request.candidates]
    except Exception as exc:
        raise HTTPException(422, f"candidates không hợp lệ: {exc}") from exc

    if not candidates:
        return {}

    # Giới hạn top-N video đưa vào VLM
    if config.vlm_top_videos and config.vlm_top_videos > 0:
        seen_videos: set[str] = set()
        top_candidates: list[Candidate] = []
        for c in sorted(candidates, key=lambda x: x.score, reverse=True):
            if c.video_id not in seen_videos:
                seen_videos.add(c.video_id)
                if len(seen_videos) > config.vlm_top_videos:
                    break
            top_candidates.append(c)
        candidates = top_candidates

    try:
        from aic2026.qa.florence import FlorenceVLM

        florence = FlorenceVLM(
            model_name=config.vlm_model,
            device=config.vlm_device,
            torch_dtype=config.vlm_dtype,
        )
        answers = florence.answer_question(request.question, candidates)
    except Exception as error:
        raise HTTPException(500, str(error)) from error

    # Trả về string key (JSON không hỗ trợ int key)
    return {str(k): v for k, v in answers.items()}


@app.post("/tasks/trake/run", response_model=AgentResult)
def run_trake(request: TaskRequest) -> AgentResult:
    """TRAKE flow: plan -> per-event retrieval -> judge -> deterministic temporal alignment."""
    return run_task("trake", request)
