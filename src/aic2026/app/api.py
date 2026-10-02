from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

# Load .env file if present (GEMINI_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Literal

from aic2026.agent.types import AgentResult
from aic2026.data_platform.keyframe_resolver import resolve_keyframe_path
from aic2026.ingestion import SIGLIP2_FEATURES, SIGLIP2_MANIFEST, resolve_feature_sources
from aic2026.models import Query
from aic2026.notebook import NotebookAgent, NotebookRequest, NotebookResult

# Configure stdout logging so query processing steps are immediately visible in terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("aic2026.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load heavy assets (SigLIP2 + FAISS + manifest + BM25) at startup.

    The _HeavyAssets singleton is built once during lifespan so that every
    subsequent KIS/QA/TRAKE request takes <2s instead of 60+s.
    """
    import time as _time
    t0 = _time.time()
    try:
        _get_heavy()  # Build the singleton NOW at startup
        logger.info("Heavy assets pre-loaded in %.1fs", _time.time() - t0)
    except Exception as exc:
        logger.exception("Heavy asset preload failed; refusing to start an unusable API: %s", exc)
        raise
    yield


app = FastAPI(title="AIC 2026 Agent API", version="0.2.0", lifespan=lifespan)

# CORS: Streamlit UI (port 8501) calls API (port 8000) cross-origin,
# so browser sends preflight OPTIONS. Without this middleware, FastAPI
# returns 405 Method Not Allowed for OPTIONS and UI cannot call API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8501",
        "http://localhost:8501",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    """Trang cho de mo localhost:8000 khong tra 404."""
    return """<!doctype html><html lang='vi'><head><meta charset='utf-8'><title>AIC 2026 Agent API</title>
    <style>body{font-family:system-ui;max-width:760px;margin:4rem auto;line-height:1.6}a{color:#2563eb}</style></head>
    <body><h1>AIC 2026 Agent API dang chay</h1><p>Backend o cong 8000. Giao dien thi Streamlit chay rieng o cong 8501.</p>
    <ul><li><a href='/docs'>API docs (/docs)</a></li><li><a href='/health'>Health check (/health)</a></li><li><a href='http://127.0.0.1:8501'>Giao dien Streamlit (can chay lenh rieng)</a></li></ul>
    <pre>aic2026 serve
streamlit run src/aic2026/app/ui.py</pre></body></html>"""


# RAM Cache luu tru anh de giam thieu I/O dia
_IMAGE_CACHE: dict[str, bytes] = {}


@app.get("/images", include_in_schema=True)
def get_image(path: str) -> Response:
    """Tra ve file anh truc tiep. Cache lai dang byte trong RAM de tang toc toi da."""
    safe_path = resolve_keyframe_path(path)
    if safe_path is None:
        raise HTTPException(404, "Khong tim thay file anh.")

    safe_path = safe_path.resolve()

    path_str = str(safe_path)
    if path_str in _IMAGE_CACHE:
        return Response(content=_IMAGE_CACHE[path_str], media_type="image/jpeg")

    try:
        with open(safe_path, "rb") as handle:
            img_bytes = handle.read()
    except Exception as exc:
        raise HTTPException(500, f"Khong the doc file anh: {exc}")

    # Gioi han kich thuoc cache khoang 300 anh de tranh OOM
    if len(_IMAGE_CACHE) > 300:
        _IMAGE_CACHE.clear()

    _IMAGE_CACHE[path_str] = img_bytes
    return Response(content=img_bytes, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Heavy Assets Singleton — loaded ONCE at startup, shared across all requests.
# This eliminates the 60s+ per-request reload that was caused by @lru_cache
# misses (any UI toggle changed a param -> cache miss -> full reload).
# ---------------------------------------------------------------------------

class _HeavyAssets:
    """Cached heavy objects (FAISS index, manifest, BM25, pipeline, encoder)."""

    def __init__(self):
        import time as _t
        t0 = _t.time()

        from aic2026.embeddings import Siglip2Embedder
        from aic2026.ingestion import load_manifest
        from aic2026.retrieval import RetrievalPipeline, VideoMetadataStore
        from aic2026.retrieval.factory import load_index_for_query

        # Use the canonical merged batch 1 + batch 2 pair shared with CLI ingestion.
        sig_features = SIGLIP2_FEATURES
        sig_manifest = SIGLIP2_MANIFEST

        # Load the text tower before the manifest, vector mmap and BM25
        # structures consume the remaining Windows commit budget. Keep it in
        # FP32 to match the model space used by the stored FP16/FP32 image exports.
        t_enc = _t.time()
        self.text_encoder = Siglip2Embedder.get_or_create(
            model_name="google/siglip2-base-patch16-224",
            # The persisted image embeddings are FP32 exports produced from
            # the unquantized SigLIP2 model (Batch 2 used FP16 inference before
            # FP32 normalization). Dynamic INT8 changes the text embedding
            # space and can rank unrelated concepts above the query.
            quantize=False,
            text_only=True,
        )
        logger.info("SigLIP2 text-only encoder ready (%.1fs)", _t.time() - t_enc)

        # Load manifest
        self.manifest_records = load_manifest(sig_manifest)
        logger.info("Manifest loaded: %d records (%.1fs)", len(self.manifest_records), _t.time() - t0)

        # Load video metadata
        self.video_metadata = VideoMetadataStore.load(
            Path("data/processed/video_metadata.jsonl")
        )

        # SigLIP2 exports L2-normalized vectors. Keep the merged matrix
        # memory-mapped and skip a full-corpus norm pass at startup.
        t_faiss = _t.time()
        self.index = load_index_for_query(
            sig_features,
            self.manifest_records,
            backend="numpy",
            features_normalized=True,
            ann_index_path=Path("data/processed/siglip2/index_ivfpq.faiss"),
        )
        logger.info("Memory-mapped vector index loaded (%.1fs)", _t.time() - t_faiss)

        # Build retrieval pipeline
        self.pipeline = RetrievalPipeline(
            self.index, self.manifest_records, frames_per_video=40,
            video_metadata=self.video_metadata,
        )

        # NOTE: ASR text is intentionally EXCLUDED from the shared BM25 index.
        # ASR (Vietnamese transcripts) should NOT affect KIS/QA/TRAKE scoring —
        # only NOTEBOOK uses ASR, and it builds its own BM25 sidecar separately.
        self.asr_text_by_video = None

        # Build BM25 index (WITHOUT ASR text — KIS/QA/TRAKE only)
        t_bm25 = _t.time()
        self.bm25_index = self._build_bm25()
        logger.info("BM25 index built (%.1fs)", _t.time() - t_bm25)

        logger.info("All heavy assets ready (%.1fs total)", _t.time() - t0)

    def _build_bm25(self):
        from aic2026.retrieval import BM25Index
        return BM25Index(
            self.manifest_records,
            video_metadata=self.video_metadata,
            asr_text_by_video=self.asr_text_by_video,
        )


_heavy: _HeavyAssets | None = None


def _get_heavy() -> _HeavyAssets:
    global _heavy
    if _heavy is None:
        _heavy = _HeavyAssets()
    return _heavy


def _attach_candidate_fps(result: AgentResult) -> AgentResult:
    """Attach FPS and playback seconds using cached maps, without video I/O."""
    from aic2026.frame_time import frame_timestamp_seconds, video_fps_info

    for candidate in result.candidates:
        candidate.fps, candidate.fps_source = video_fps_info(candidate.video_id)
        if candidate.timestamp is None:
            candidate.timestamp = frame_timestamp_seconds(
                candidate.video_id, candidate.frame_id,
                frame_unit=candidate.frame_unit, fps_hint=candidate.fps,
            )
    return result


# ---------------------------------------------------------------------------
# load_orchestrator — lightweight per-request setup (no I/O, <10ms)
# ---------------------------------------------------------------------------

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
    vlm_backend: str = "none",
    late_interaction_weight: float = 0.0,
    vlm_model: str = "microsoft/Florence-2-base-ft",
    vlm_device: str | None = None,
    vlm_dtype: str = "float32",
    vlm_timeout: int = 120,
    coarse_top_k: int = 200,
    query_type: str = "",
    asr_sidecar_path: str | None = None,
    asr_dense_path: str | None = None,
    embedding_backend: str = "siglip2",
    siglip2_model: str = "google/siglip2-base-patch16-224",
    use_event_coverage: bool = False,
    use_moment_rerank: bool = False,
    use_blip2_rerank: bool = False,
    blip2_rerank_top_k: int = 70,
    blip2_rerank_weight: float = 0.3,
    asr_weight: float = 0.15,
    event_coverage_blend: float = 0.5,
    use_llm_query_analyzer: bool = False,
    llm_query_model: str = "qwen2.5:1.5b",
    llm_provider: str = "auto",
    use_cascade_rerank: bool = True,
    cascade_stage_a_top_n: int = 100,
    video_dedupe_window: float = 1.5,
    use_llm_verifier: bool = False,
    llm_verifier_max_candidates: int = 20,
    llm_verifier_timeout: float = 10.0,
    use_ltr_ranker: bool = False,
    use_fast_kis: bool = False,
    retrieval_profile: str = "speed",
):
    """Build RetrievalAgent from pre-loaded heavy assets. Config-only, no I/O."""
    from aic2026.agent import OllamaLLM, RetrievalAgent
    from aic2026.agent.tools import RetrievalTools

    heavy = _get_heavy()

    # Build tools from pre-loaded assets
    tools = RetrievalTools(
        heavy.pipeline, heavy.text_encoder.encode, encode_images=None
    )

    # Set config flags (fast, no I/O)
    tools.coarse_top_k = coarse_top_k
    tools.video_filter_terms = [t.strip() for t in metadata_filter.split(",") if t.strip()] or None
    tools.late_interaction_weight = late_interaction_weight
    tools.bm25_index = heavy.bm25_index
    tools.use_fast_kis = use_fast_kis
    tools.strict_object_match = False
    tools.use_event_coverage = use_event_coverage
    tools.use_moment_rerank = use_moment_rerank
    tools.use_blip2_rerank = use_blip2_rerank
    tools.blip2_rerank_top_k = blip2_rerank_top_k
    tools.blip2_rerank_weight = blip2_rerank_weight
    tools.event_coverage_blend = event_coverage_blend
    tools.use_llm_query_analyzer = use_llm_query_analyzer
    tools.llm_model = llm_query_model
    tools.llm_base_url = ollama_url
    tools.llm_provider = llm_provider
    tools.use_cascade_rerank = use_cascade_rerank
    tools.cascade_stage_a_top_n = cascade_stage_a_top_n
    tools.video_dedupe_window = video_dedupe_window
    tools.use_llm_verifier = use_llm_verifier
    tools.llm_verifier_max_candidates = llm_verifier_max_candidates
    tools.llm_verifier_timeout = llm_verifier_timeout
    tools.use_ltr_ranker = use_ltr_ranker

    # Apply retrieval_profile defaults (speed vs precision)
    # Speed profile: optimized for Recall@10-20 + low latency
    # Precision profile: optimized for Recall@1 (full rerank stack)
    if retrieval_profile == "speed":
        # Keep KIS latency bounded; QA has a separate explicit Florence answer stage.
        tools.use_blip2_rerank = False
        tools.use_llm_verifier = False
        # Event coverage parses events cheaply and does no extra work for a
        # single-scene query. For ordered multi-event KIS it adds a few cached
        # text vectors to prefer frames/videos that cover more of the request.
        tools.use_event_coverage = True
        tools.use_moment_rerank = True
        # Late-interaction (ColBERT-style MaxSim): catches specific facet matches
        # that pooled CLIP misses. Light weight since it only encodes ~10 facets.
        tools.late_interaction_weight = 0.15
        # LLM query analyzer decomposes query into events, entities, expansion variants
        tools.use_llm_query_analyzer = True
        tools.llm_model = llm_query_model
        tools.llm_base_url = ollama_url
        tools.llm_provider = llm_provider
        tools.use_fast_kis = True
        tools.coarse_top_k = 100
        tools.cascade_stage_a_top_n = 50
    elif retrieval_profile == "precision":
        tools.use_blip2_rerank = True
        tools.blip2_rerank_top_k = 70
        tools.use_llm_verifier = True
        tools.use_event_coverage = True
        tools.use_moment_rerank = True
        tools.late_interaction_weight = 0.3
        tools.use_llm_query_analyzer = True
        tools.llm_model = llm_query_model
        tools.llm_base_url = ollama_url
        tools.llm_provider = llm_provider
        tools.use_fast_kis = False
        tools.coarse_top_k = 200
        tools.cascade_stage_a_top_n = 100

    # VLM config (only for QA path)
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
            use_event_coverage=config.use_event_coverage,
            use_moment_rerank=config.use_moment_rerank,
            use_blip2_rerank=config.use_blip2_rerank,
            blip2_rerank_top_k=config.blip2_rerank_top_k,
            blip2_rerank_weight=config.blip2_rerank_weight,
            asr_weight=config.asr_weight,
            event_coverage_blend=config.event_coverage_blend,
            embedding_backend=config.embedding_backend,
            siglip2_model=config.siglip2_model,
            use_llm_query_analyzer=config.use_llm_query_analyzer,
            llm_query_model=config.llm_query_model,
            llm_provider=config.llm_provider,
            use_cascade_rerank=config.use_cascade_rerank,
            cascade_stage_a_top_n=config.cascade_stage_a_top_n,
            video_dedupe_window=config.video_dedupe_window,
            use_llm_verifier=config.use_llm_verifier,
            llm_verifier_max_candidates=config.llm_verifier_max_candidates,
            llm_verifier_timeout=config.llm_verifier_timeout,
            use_ltr_ranker=config.use_ltr_ranker,
            use_fast_kis=config.use_fast_kis,
            retrieval_profile=config.retrieval_profile,
        )
        result = orchestrator.run(
            Query(
                query_id=request.query_id,
                type=task_type,
                text=request.text,
                question=request.question,
                events=request.events,
                asr_query=request.asr_query,
            )
        )
        return _attach_candidate_fps(result)
    except HTTPException:
        raise
    except FileNotFoundError as error:
        raise HTTPException(503, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        logger.exception("run_task failed for task=%s: %s", task_type, error)
        raise HTTPException(500, f"{type(error).__name__}: {error}") from error


class RuntimeConfig(BaseModel):
    """Cau hinh noi bo; UI gui de backend nap dung index/encoder."""
    manifest_path: str | None = None
    features_path: str | None = None
    clip_pretrained: str = "openai"
    llm_model: str = "qwen3.5:4b"
    ollama_url: str = "http://127.0.0.1:11434"
    backend: str = "faiss"
    chroma_dir: str = "data/indexes/chroma"
    late_interaction_weight: float = 0.3
    metadata_filter: str = ""
    translate_query: bool = False
    vlm_backend: str = "none"
    vlm_model: str = "microsoft/Florence-2-base-ft"
    vlm_device: str | None = None
    vlm_dtype: str = "float32"
    vlm_timeout: int = 120
    coarse_top_k: int = 200
    vlm_top_videos: int = 0
    vlm_max_workers: int = 0
    asr_sidecar_path: str | None = None
    asr_weight: float = 0.15
    use_event_coverage: bool = False
    event_coverage_blend: float = 0.5
    use_moment_rerank: bool = False
    use_blip2_rerank: bool = False
    blip2_rerank_top_k: int = 30
    blip2_rerank_weight: float = 0.3
    use_llm_query_analyzer: bool = False
    llm_query_model: str = "qwen2.5:1.5b"
    llm_provider: str = "gemini"
    asr_dense_path: str | None = None
    embedding_backend: str = "siglip2"
    siglip2_model: str = "google/siglip2-base-patch16-224"
    use_cascade_rerank: bool = True
    cascade_stage_a_top_n: int = 100
    video_dedupe_window: float = 1.5
    use_llm_verifier: bool = False
    llm_verifier_max_candidates: int = 20
    llm_verifier_timeout: float = 10.0
    use_ltr_ranker: bool = False
    use_fast_kis: bool = True  # Fast KIS enabled by default for max speed
    retrieval_profile: Literal["speed", "precision"] = "speed"


class TaskRequest(BaseModel):
    query_id: str = "live-query"
    text: str = Field(min_length=1)
    question: str | None = None
    events: list[str] = Field(default_factory=list)
    asr_query: str | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


class AnswerRequest(BaseModel):
    """Phase-2 QA: gui lai candidates da retrieve, chi chay VLM de sinh dap an."""
    question: str = Field(min_length=1)
    candidates: list[dict] = Field(default_factory=list)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


@app.get("/video-time/{video_id}")
def video_time(video_id: str, frame_id: int = 0) -> dict[str, float | None]:
    """Return the timestamp for a given frame_id. Uses BTC CSV map + cv2 fps."""
    from aic2026.frame_time import _keyframe_csv_path, _video_fps
    import csv as _csv

    csv_path = _keyframe_csv_path(video_id)
    if csv_path is not None:
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                for row in _csv.DictReader(f):
                    try:
                        if int(row["frame_idx"]) == frame_id:
                            return {"timestamp": float(row["pts_time"]), "frame_id": frame_id}
                    except (KeyError, ValueError):
                        continue
        except Exception:
            pass
    fps = _video_fps(video_id) or 30.0
    return {"timestamp": frame_id / fps, "frame_id": frame_id}


@app.get("/frame-at-time/{video_id}")
def frame_at_time(video_id: str, timestamp: float = 0.0) -> dict[str, int | float]:
    """Return the frame_id closest to timestamp. Uses BTC CSV map + cv2 fps."""
    from aic2026.frame_time import resolve_frame_at_timestamp

    info = resolve_frame_at_timestamp(video_id, timestamp)
    return {"frame_id": info.frame_number, "timestamp": info.actual_frame_timestamp}


@app.get("/video-player/{video_id}", response_class=HTMLResponse, include_in_schema=False)
def video_player(video_id: str, frame_id: int = 0, timestamp: float | None = None) -> str:
    """Serve a player for a local video or an on-demand remote ZIP member."""
    import glob
    from aic2026.video_archive import archive_key_for_video

    patterns = [
        f"data/raw/Videos/**/{video_id}.mp4",
        f"data/raw/Videos_L01_a/video/{video_id}.mp4",
        f"data/raw/Videos/{video_id}.mp4",
    ]
    video_path = None
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            video_path = matches[0]
            break
    # Remote BTC batches are stored inside ZIPs. The page can open immediately;
    # its video request below extracts only the requested member into cache.
    if not video_path and archive_key_for_video(video_id) is None:
        raise HTTPException(404, f"Video {video_id} not found")

    video_url = f"/video-file/{video_id}"
    from aic2026.frame_time import _video_fps
    _fps = _video_fps(video_id) or 30.0
    seek_js = ""
    if timestamp is not None and timestamp > 0:
        seek_js = f"document.getElementById('vid').currentTime={timestamp:.3f};"
    elif frame_id > 0:
        seek_js = f"document.getElementById('vid').currentTime={(frame_id / _fps):.3f};"

    ts_label = f", t={timestamp:.1f}s" if timestamp else ""
    return f"""<!doctype html><html><head>
<meta charset="utf-8"><title>{video_id}</title>
<style>
body{{margin:0;background:#111;display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;color:#eee;font-family:system-ui}}
video{{max-width:95vw;max-height:85vh}}
h3{{margin:0 0 8px}}
#frame-info{{position:fixed;bottom:20px;right:20px;background:rgba(0,0,0,0.85);border:1px solid #555;border-radius:8px;padding:12px 18px;font-size:15px;display:none;z-index:999;min-width:220px}}
#frame-info .label{{color:#aaa;font-size:12px}}
#frame-info .value{{color:#4fc3f7;font-size:18px;font-weight:bold}}
#frame-info .hint{{color:#888;font-size:11px;margin-top:6px}}
</style></head><body>
<h3>{video_id} &mdash; frame {frame_id}{ts_label}</h3>
<video id="vid" controls autoplay src="{video_url}"></video>
<div id="frame-info">
  <div class="label">FRAME AT CURRENT TIME</div>
  <div class="value" id="frame-val">&mdash;</div>
  <div class="hint">Press Ctrl+G to query &middot; Esc to close</div>
</div>
<script>
{seek_js}
(function() {{
  const vid = document.getElementById('vid');
  const info = document.getElementById('frame-info');
  const frameVal = document.getElementById('frame-val');
  let hideTimer = null;

  document.addEventListener('keydown', function(e) {{
    if ((e.ctrlKey || e.metaKey) && (e.key === 'g' || e.key === 'G')) {{
      e.preventDefault();
      var t = vid.currentTime;
      fetch('/frame-at-time/{video_id}?timestamp=' + t.toFixed(3))
        .then(function(r) {{ return r.json(); }})
        .then(function(d) {{
          var fid = d.frame_id;
          var ts = d.timestamp;
          frameVal.textContent = 'frame ' + fid + '  (t=' + ts.toFixed(2) + 's)';
          info.style.display = 'block';
          clearTimeout(hideTimer);
          hideTimer = setTimeout(function() {{ info.style.display = 'none'; }}, 8000);
        }})
        .catch(function() {{
          frameVal.textContent = 'frame ' + Math.round(t * {_fps}) + '  (t=' + t.toFixed(2) + 's, approx)';
          info.style.display = 'block';
          clearTimeout(hideTimer);
          hideTimer = setTimeout(function() {{ info.style.display = 'none'; }}, 8000);
        }});
    }}
    if (e.key === 'Escape') {{
      info.style.display = 'none';
    }}
  }});
}})();
</script>
</body></html>"""


@app.get("/video-file/{video_id}", include_in_schema=False)
def video_file(video_id: str):
    """Stream a local video, or cache one member from a BTC remote ZIP first."""
    from pathlib import Path as _P
    import glob
    import mimetypes
    from fastapi.responses import FileResponse

    patterns = [
        f"data/raw/Videos/**/{video_id}.mp4",
        f"data/raw/Videos_L01_a/video/{video_id}.mp4",
        f"data/raw/Videos/{video_id}.mp4",
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            video_path = _P(matches[0])
            media_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
            return FileResponse(str(video_path), media_type=media_type)
    # Do not download a whole ZIP: fetch_video_to_cache reads its directory and
    # exactly one compressed video entry through HTTP Range.
    try:
        from aic2026.video_archive import fetch_video_to_cache

        cached = fetch_video_to_cache(video_id)
    except Exception as exc:  # noqa: BLE001 - report remote availability clearly
        logger.warning("Remote video extraction failed for %s: %s", video_id, exc)
        raise HTTPException(502, f"Could not load remote video {video_id}: {exc}") from exc
    if cached is not None:
        media_type = mimetypes.guess_type(str(cached))[0] or "video/mp4"
        return FileResponse(str(cached), media_type=media_type)
    raise HTTPException(404, f"Video {video_id} not found")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "flows": "kis,qa,trake,notebook"}


@app.get("/health/ready")
def ready(
    manifest_path: str | None = None,
    features_path: str | None = None,
    chroma_dir: str = "data/indexes/chroma",
) -> dict[str, object]:
    """Kiem tra nhanh truoc khi UI goi agent."""
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
    """Phase-1 QA: chi retrieval (CLIP + BM25), khong VLM. Tra candidates ngay."""
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
            vlm_backend="none",
            vlm_model=config.vlm_model,
            vlm_device=config.vlm_device,
            vlm_dtype=config.vlm_dtype,
            vlm_timeout=config.vlm_timeout,
            late_interaction_weight=config.late_interaction_weight,
            query_type="qa",
            asr_sidecar_path=config.asr_sidecar_path,
            use_event_coverage=config.use_event_coverage,
            use_moment_rerank=config.use_moment_rerank,
            use_blip2_rerank=config.use_blip2_rerank,
            blip2_rerank_top_k=config.blip2_rerank_top_k,
            blip2_rerank_weight=config.blip2_rerank_weight,
            asr_weight=config.asr_weight,
            event_coverage_blend=config.event_coverage_blend,
            embedding_backend=config.embedding_backend,
            siglip2_model=config.siglip2_model,
            use_llm_query_analyzer=config.use_llm_query_analyzer,
            llm_query_model=config.llm_query_model,
            llm_provider=config.llm_provider,
            use_cascade_rerank=config.use_cascade_rerank,
            cascade_stage_a_top_n=config.cascade_stage_a_top_n,
            video_dedupe_window=config.video_dedupe_window,
            use_llm_verifier=config.use_llm_verifier,
            llm_verifier_max_candidates=config.llm_verifier_max_candidates,
            llm_verifier_timeout=config.llm_verifier_timeout,
            use_ltr_ranker=config.use_ltr_ranker,
            use_fast_kis=config.use_fast_kis,
            retrieval_profile=config.retrieval_profile,
        )
        query = Query(
            query_id=request.query_id,
            type="qa",
            text=request.text,
            question=request.question,
            events=request.events,
            asr_query=request.asr_query,
        )
        return _attach_candidate_fps(orchestrator.run(query))
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
    """Phase-2 QA: nhan candidates da co, chay VLM song song. Tra {vector_id: answer}."""
    from aic2026.models import Candidate

    config = request.runtime

    if not config.vlm_backend or config.vlm_backend == "none":
        return {}

    try:
        candidates = [Candidate.model_validate(c) for c in request.candidates]
    except Exception as exc:
        raise HTTPException(422, f"candidates khong hop le: {exc}") from exc

    if not candidates:
        return {}

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

    return {str(k): v for k, v in answers.items()}


@app.post("/tasks/trake/run", response_model=AgentResult)
def run_trake(request: TaskRequest) -> AgentResult:
    """TRAKE flow: plan -> per-event retrieval -> judge -> deterministic temporal alignment."""
    return run_task("trake", request)


@app.get("/notebook/agent", response_model=None)
@lru_cache(maxsize=1)
def get_notebook_agent(
    ollama_url: str = "http://127.0.0.1:11434",
    llm_model: str = "qwen2.5:1.5b",
    asr_sidecar_path: str | None = "data/processed/asr_sidecar.jsonl",
) -> NotebookAgent:
    """Build (and cache) the NotebookAgent — separate from KIS/QA/TRAKE orchestrator."""
    sidecar = asr_sidecar_path
    if sidecar:
        from pathlib import Path as _P
        s = _P(sidecar)
        if not s.is_absolute():
            s = _P(__file__).resolve().parents[3] / s
        sidecar = str(s) if s.exists() else None

    if not sidecar:
        logger.warning(
            "NotebookAgent: ASR sidecar not found (asr_sidecar_path=%s) — ASR retrieval will be disabled",
            asr_sidecar_path,
        )
        sidecar = None

    agent = NotebookAgent(
        asr_sidecar_path=sidecar or "",
        ollama_url=ollama_url,
        llm_model=llm_model,
    )
    logger.info("NotebookAgent ready (asr_sidecar=%s)", sidecar)
    return agent


@app.post("/tasks/notebook/run", response_model=NotebookResult)
def run_notebook(request: NotebookRequest) -> NotebookResult:
    """NOTEBOOK / Video Search flow (spec section 11)."""
    try:
        agent = get_notebook_agent(
            ollama_url=request.ollama_url,
            llm_model=request.llm_model,
        )
        return agent.run(
            text=request.text,
            asr_query=request.asr_query,
            top_n=request.top_n,
            use_visual=request.use_visual,
        )
    except FileNotFoundError as error:
        raise HTTPException(503, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        logger.exception("run_notebook failed: %s", error)
        raise HTTPException(500, f"{type(error).__name__}: {error}") from error
