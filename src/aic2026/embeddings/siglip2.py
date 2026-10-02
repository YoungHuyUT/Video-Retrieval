"""SigLIP2 image + text embedder (Improvement.md Phase 1-3).

Production visual retrieval backend — replaces official CLIP ViT-B/32 index
with a SigLIP2 frame-side index on the SAME BTC keyframes.

WHY THIS EXISTS
---------------
Per Improvement.md §1-2:
  BTC Keyframes → SigLIP2 → FAISS → RRF → Adaptive Fusion → Final result

Official BTC CLIP features (official_features.npy) remain as BASELINE ONLY.

ARCHITECTURE (spec §2, §4)
---------------------------
- SigLIP2 image encoder: encode BTC keyframes → normalized embeddings
- SigLIP2 text encoder: encode query text → normalized embeddings
- Same model checkpoint for text & image (required: Improvement.md §4 "text
  encoder & index PHẢI cùng checkpoint")

Model variants (per Improvement.md §2):
- google/siglip2-so400m-patch14-384 — SigLIP2-SO400M (default, 1152-dim, strongest)
- google/siglip2-base-patch16-224 — SigLIP2-Base (lighter, 768-dim, CPU-friendly)
- google/siglip2-large-patch16-384 — SigLIP2-L (if GPU available)

Embedding dims:
- SigLIP2-Base: 768 (vision_config.hidden_size)
- SigLIP2-Large: 1024

INTERFACE
----------
Implements the same surface as OpenCLIPTextEmbedder:
  encode(text), encode_images(list[PIL.Image]), load(), unload()

Plus frame-side utilities for batch embedding keyframes with caching/resume
(Improvement.md §11-12).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Model configuration
# ----------------------------------------------------------------------------

# SigLIP2-Base (768-dim) — matches the pre-built features_siglip2.npy (768-dim).
# Text encoder MUST use the same checkpoint as the image features (Improvement.md §4).
DEFAULT_SIGLIP2_MODEL = "google/siglip2-base-patch16-224"
SIGLIP2_LARGE = "google/siglip2-so400m-patch14-384"  # for future use with so400m features
SIGLIP2_BASE = DEFAULT_SIGLIP2_MODEL

# Fallback embedding dim (will be auto-detected from model config on load).
# SigLIP2-Base (google/siglip2-base-patch16-224) produces 768-dim vectors —
# this MUST match the pre-built features_siglip2.npy and the indexed matrix.
SIGLIP2_EMBEDDING_DIM = 768

# Output directory structure (Improvement.md §3)
SIGLIP2_OUTPUT_DIR = Path("data/processed/siglip2")
SIGLIP2_FEATURES_FILE = SIGLIP2_OUTPUT_DIR / "features_siglip2.npy"
SIGLIP2_MANIFEST_FILE = SIGLIP2_OUTPUT_DIR / "manifest_siglip2.jsonl"
SIGLIP2_INDEX_FILE = SIGLIP2_OUTPUT_DIR / "index_siglip2.faiss"
SIGLIP2_META_FILE = SIGLIP2_OUTPUT_DIR / "meta.json"


@dataclass
class Siglip2Metadata:
    """Metadata for reproducibility (Improvement.md §12)."""

    model_name: str
    model_version: str
    source_hash: str
    embedding_dim: int
    dtype: str
    batch_size: int
    created_at: str
    frame_count: int
    video_count: int = 0

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "source_hash": self.source_hash,
            "embedding_dim": self.embedding_dim,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "created_at": self.created_at,
            "frame_count": self.frame_count,
            "video_count": self.video_count,
        }


_STANCE_CACHE: dict[str, "Siglip2Embedder"] = {}


class Siglip2Embedder:
    """Image + text embedder backed by a local SigLIP2 checkpoint.

    Usage:
        embedder = Siglip2Embedder()
        text_vec = embedder.encode("a person riding a bicycle")
        image_vec = embedder.encode_images([img1, img2])

    Memory management:
        - call unload() before loading a VLM on low-RAM machines
        - load() is idempotent and safe to call after unload()

    Query optimization:
        - Use get_or_create() to reuse a singleton per model (avoids reload)
        - INT8 quantization on CPU for ~1.5x speedup
        - Warmup on first load to eliminate first-query latency
    """

    @classmethod
    def get_or_create(
        cls,
        model_name: str = DEFAULT_SIGLIP2_MODEL,
        device: str | None = None,
        quantize: bool = True,
        text_only: bool = False,
    ) -> "Siglip2Embedder":
        """Get a cached embedder or create one (singleton per model/config)."""
        key = f"{model_name}:{device or 'auto'}:{quantize}:text_only={text_only}"
        if key not in _STANCE_CACHE:
            _STANCE_CACHE[key] = cls(model_name, device, quantize, text_only=text_only)
        return _STANCE_CACHE[key]

    @classmethod
    def preload(
        cls,
        model_name: str = DEFAULT_SIGLIP2_MODEL,
        device: str | None = None,
        quantize: bool = True,
        text_only: bool = False,
    ) -> None:
        """Load model synchronously. Call at server startup to warm the cache.

        The first encode() call after this is instant.
        """
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        key = f"{model_name}:{device or 'auto'}:{quantize}:text_only={text_only}"
        if key in _STANCE_CACHE and _STANCE_CACHE[key].model is not None:
            return  # already loaded

        try:
            cls.get_or_create(model_name, device, quantize, text_only=text_only)
            logger.info("Siglip2Embedder: preload complete")
        except Exception as exc:
            logger.warning("Siglip2Embedder: preload failed: %s", exc)

    def __init__(
        self,
        model_name: str = DEFAULT_SIGLIP2_MODEL,
        device: str | None = None,
        quantize: bool = True,
        text_only: bool = False,
    ) -> None:
        """Initialize SigLIP2 embedder.

        Args:
            model_name: HuggingFace model ID (default: google/siglip2-so400m-patch14-384).
            device: "cuda", "cpu", or None (auto-detect).
            quantize: If True and on CPU, apply INT8 dynamic quantization for
                ~1.5x speedup (Improvement.md §11 performance optimization).
        """
        # Set offline mode before importing Transformers. Recent versions may
        # otherwise make a Hub capability request during import, delaying startup.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModel,
                AutoProcessor,
                AutoTokenizer,
                SiglipTextModel,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install model extras: pip install torch transformers"
            ) from exc

        # Suppress the bos/eos_token_id warnings from SigLIP2 config
        warnings.filterwarnings("ignore", message=".*bos_token_id.*")
        warnings.filterwarnings("ignore", message=".*eos_token_id.*")

        self.model_name = model_name
        self._text_only = text_only
        self._torch = torch
        self._AutoConfig = AutoConfig
        self._AutoModel = AutoModel
        self._SiglipTextModel = SiglipTextModel
        self._AutoProcessor = AutoProcessor
        self._AutoTokenizer = AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[object] = None
        self.processor: Optional[object] = None
        self._cache: dict[str, np.ndarray] = {}
        self._cache_max_size: int = 2048
        self._embedding_dim: int = SIGLIP2_EMBEDDING_DIM  # Will be updated on load
        self._quantize = quantize and self.device == "cpu"
        # Increase thread count for CPU inference (Improvement.md §11)
        if self.device == "cpu":
            self._torch.set_num_threads(min(8, self._torch.get_num_threads()))
        self._load_model()

    @property
    def embedding_dim(self) -> int:
        """Get the actual embedding dimension from the loaded model."""
        return self._embedding_dim

    def _load_model(self) -> None:
        """Load SigLIP2 model and processor from HuggingFace cache (offline).

        The API loads only the text tower; image embedding jobs still load the
        full checkpoint. This keeps query serving below the full-model peak.
        """
        import time as _time

        # Prevent HuggingFace from checking for updates online
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        t0 = _time.time()

        if self._text_only:
            # AutoProcessor selects GemmaTokenizerFast here and parses the
            # 33 MB tokenizer.json into a large in-memory structure. Under the
            # API's mapped corpus/model memory footprint that can exhaust the
            # Windows commit limit. The bundled SentencePiece model is 4 MB and
            # has identical token IDs; text-only serving does not need an image
            # processor or the fast-tokenizer JSON.
            self.processor = self._AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True,
                use_fast=False,
            )
        else:
            self.processor = self._AutoProcessor.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
        # Load the text tower in float32. Quantization is an optional runtime
        # optimization and must only be enabled when the stored image vectors
        # were generated with the same quantization path.
        import torch
        # The cached google/siglip2-base checkpoint has a root `siglip` config
        # and a nested `siglip_text_model` config (vocab_size=256000). Loading
        # Siglip2TextModel invents a different 32k-vocabulary config, while
        # AutoModel loads both towers. Extract the actual nested text config
        # and load only matching text weights for query serving.
        load_kwargs = {"local_files_only": True, "dtype": torch.float32}
        if self._text_only:
            root_config = self._AutoConfig.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
            text_config = getattr(root_config, "text_config", None)
            if (
                text_config is None
                or getattr(text_config, "model_type", None) != "siglip_text_model"
            ):
                raise ValueError(
                    f"{self.model_name} does not contain a SigLIP text tower config"
                )
            self.model = self._load_text_tower_mmap(text_config)
        else:
            self.model = self._AutoModel.from_pretrained(
                self.model_name,
                **load_kwargs,
            )

        # Optional INT8 dynamic quantization for CPU inference. The API keeps
        # this off because the merged Batch 2 vectors were exported from an
        # FP16 model and metadata does not establish an INT8-compatible space.
        if self._quantize:
            logger.info("Applying INT8 dynamic quantization for CPU acceleration...")
            self.model = self._torch.quantization.quantize_dynamic(
                self.model,
                {self._torch.nn.Linear},
                dtype=self._torch.qint8,
                inplace=True,
            )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Auto-detect embedding dimension from model config
        # SigLIP2 stores hidden_size in vision_config
        if hasattr(self.model.config, "vision_config"):
            self._embedding_dim = self.model.config.vision_config.hidden_size
        elif hasattr(self.model.config, "text_config"):
            self._embedding_dim = self.model.config.text_config.hidden_size
        else:
            self._embedding_dim = self.model.config.hidden_size

        # Warmup: run dummy inference to avoid first-query latency
        # (JIT compilation + memory allocation happens here, not on first real query)
        try:
            import torch as _torch
            with _torch.no_grad():
                dummy_text = self.processor(
                    text="warmup", return_tensors="pt",
                    padding="max_length", max_length=64, truncation=True,
                ).to(self.device)
                _ = self._text_features(dummy_text)
        except Exception:
            pass  # warmup failure is non-fatal

        elapsed = _time.time() - t0
        logger.info(
            f"SigLIP2 model loaded: {self.model_name} on {self.device}, "
            f"dim={self._embedding_dim}, warmup={elapsed:.1f}s"
        )

    def _load_text_tower_mmap(self, text_config):
        """Load only text tensors as file-backed arrays, without safe_open.

        On Windows, safetensors maps the entire 1.5 GB joint image+text
        checkpoint. That can fail with OS error 1455 even though the API only
        needs the 1.08 GiB text tower. Map each text tensor's byte range from
        the safetensors file, and assign those arrays to a meta-initialized
        model. This avoids both mapping vision weights and allocating a second
        in-memory copy of the text weights.
        """
        import gc
        import numpy as np

        from transformers.utils import cached_file

        checkpoint = cached_file(
            self.model_name,
            "model.safetensors",
            local_files_only=True,
        )
        if checkpoint is None:
            raise FileNotFoundError(
                f"No local model.safetensors found for {self.model_name}"
            )

        # Safetensors starts with an 8-byte little-endian header size, followed
        # by JSON and then the raw tensor bytes. Read just the small header.
        with open(checkpoint, "rb") as source:
            raw_length = source.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"Invalid safetensors header: {checkpoint}")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 0 or header_length > 64 * 1024 * 1024:
                raise ValueError(f"Invalid safetensors header size: {header_length}")
            header = json.loads(source.read(header_length))

        data_start = 8 + header_length
        source_tensors = {
            name: info
            for name, info in header.items()
            if name.startswith("text_model.")
        }
        if not source_tensors:
            raise ValueError(f"No text_model tensors found in {checkpoint}")

        # The AIC index and this checkpoint are float32. Refuse an implicit
        # conversion here: it would allocate a full additional copy and defeat
        # the low-memory loader.
        dtype_map = {"F32": np.dtype("<f4"), "F16": np.dtype("<f2")}
        state_dict = {}
        for source_name, info in source_tensors.items():
            if info.get("dtype") not in dtype_map:
                raise ValueError(
                    f"Unsupported tensor dtype {info.get('dtype')} in {source_name}; "
                    "the low-memory loader supports F32 and F16"
                )
            key = source_name.removeprefix("text_model.")
            start, end = info["data_offsets"]
            expected_items = int(np.prod(info["shape"], dtype=np.int64))
            tensor_dtype = dtype_map[info["dtype"]]
            if (end - start) != expected_items * tensor_dtype.itemsize:
                raise ValueError(f"Invalid byte range for safetensors tensor {source_name}")
            array = np.memmap(
                checkpoint,
                mode="r",
                dtype=tensor_dtype,
                offset=data_start + start,
                shape=tuple(info["shape"]),
                order="C",
            )
            # Weights are inference-only and must not reserve writable
            # copy-on-write pages against the already tight Windows commit
            # limit. Silence PyTorch's expected warning for read-only arrays.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="The given NumPy array is not writable.*"
                )
                state_dict[key] = self._torch.from_numpy(array)

        # Constructing the model on `meta` avoids allocating random parameter
        # buffers before they are replaced with file-backed checkpoint tensors.
        with self._torch.device("meta"):
            model = self._SiglipTextModel(text_config)
        incompat = model.load_state_dict(state_dict, strict=True, assign=True)
        if incompat.missing_keys or incompat.unexpected_keys:
            raise RuntimeError(
                "Text checkpoint does not exactly match SiglipTextModel: "
                f"missing={incompat.missing_keys}, unexpected={incompat.unexpected_keys}"
            )

        # `position_ids` is a non-persistent buffer, so it is intentionally
        # absent from the checkpoint and remains on `meta` after assign=True.
        # Recreate it from its registered shape before Module.to()/inference.
        for module in model.modules():
            for name, buffer in tuple(module._buffers.items()):
                if buffer is None or not buffer.is_meta:
                    continue
                if name != "position_ids":
                    raise RuntimeError(
                        f"Unexpected uninitialized meta buffer: {name} "
                        f"on {type(module).__name__}"
                    )
                module._buffers[name] = self._torch.arange(
                    buffer.numel(), dtype=buffer.dtype, device="cpu"
                ).reshape(buffer.shape)

        del state_dict
        gc.collect()
        logger.info(
            "Loaded %d SigLIP text tensors from file-backed mappings (%s)",
            len(source_tensors),
            checkpoint,
        )
        return model

    def load(self) -> None:
        """(Re)create model/processor if missing. Idempotent."""
        if self.model is not None and self.processor is not None:
            return
        self._load_model()

    def unload(self) -> None:
        """Drop weights to free RAM before a VLM loads on low-RAM machines."""
        self.model = None
        self.processor = None
        try:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:
            pass

    # -- encoding -----------------------------------------------------------

    def split_long_query(
        self, text: str, *, chunk_size: int = 60, overlap: int = 8
    ) -> list[str]:
        """Create focused SigLIP2 branches for sentence-level and long queries.

        Keep the full query as the global intent, add its meaningful sentences
        as facets, and add overlapping tokenizer windows only when SigLIP2's
        64-token input would otherwise truncate content.
        """
        if not text:
            return []
        if self.processor is None:
            self.load()
        try:
            from aic2026.query.expansion import _split_sentences

            sentences = _split_sentences(text)
        except Exception:
            sentences = []
        branches = [text]
        seen = {text.casefold()}
        for sentence in sentences[:5]:
            if sentence.casefold() not in seen:
                seen.add(sentence.casefold())
                branches.append(sentence)

        token_ids = self.processor.encode(text, add_special_tokens=False)
        if len(token_ids) <= chunk_size:
            return branches
        step = max(1, chunk_size - overlap)
        chunks: list[str] = []
        for start in range(0, len(token_ids), step):
            decoded = self.processor.decode(
                token_ids[start : start + chunk_size],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            if decoded and decoded.casefold() not in seen:
                seen.add(decoded.casefold())
                chunks.append(decoded)
            if start + chunk_size >= len(token_ids):
                break
        return [*branches, *chunks]

    def encode_many(self, texts: list[str]) -> list[np.ndarray]:
        """Batch-encode query variants with one SigLIP2 text-tower pass."""
        if not texts:
            return []
        if self.model is None or self.processor is None:
            self.load()

        unique_misses = list(
            dict.fromkeys(text for text in texts if text not in self._cache)
        )
        if unique_misses:
            inputs = self.processor(
                text=unique_misses,
                return_tensors="pt",
                padding="max_length",
                max_length=64,
                truncation=True,
            ).to(self.device)
            with getattr(self._torch, "inference_mode", self._torch.no_grad)():
                outputs = self._text_features(inputs)
                if hasattr(outputs, "pooler_output"):
                    text_embeds = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state"):
                    text_embeds = outputs.last_hidden_state[:, 0, :]
                else:
                    text_embeds = outputs
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                vectors = text_embeds.detach().cpu().float().numpy()
            for text, vector in zip(unique_misses, vectors):
                self._cache_text(text, vector)

        return [self._cache[text].copy() for text in texts]
    def encode(self, text: str) -> np.ndarray:
        """Encode query text into an L2-normalized SigLIP2 text vector.

        Returns: (embedding_dim,) float32 vector, L2-normalized.
        """
        cached = self._cache.get(text)
        if cached is not None:
            return cached.copy()

        if self.model is None or self.processor is None:
            self.load()

        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        ).to(self.device)

        with getattr(self._torch, "inference_mode", self._torch.no_grad)():
            outputs = self._text_features(inputs)
            # SigLIP2 output: BaseModelOutputWithPooling
            # text_embeds shape: (batch_size, hidden_size)
            if hasattr(outputs, "pooler_output"):
                text_embeds = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                text_embeds = outputs.last_hidden_state[:, 0, :]
            else:
                text_embeds = outputs

            # L2-normalize
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            vec = text_embeds[0].detach().cpu().float().numpy()

        self._cache_text(text, vec)
        return vec.copy()

    def _text_features(self, inputs):
        """Run the text tower, including when vision weights are omitted."""
        if self._text_only:
            return self.model(**inputs)
        return self.model.get_text_features(**inputs)

    def encode_images(
        self,
        images: list,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Encode RGB PIL images into L2-normalized SigLIP2 image vectors.

        Args:
            images: List of PIL.Image objects (RGB).
            batch_size: Batch size for inference (Improvement.md §11).

        Returns:
            (N, embedding_dim) float32 array, L2-normalized.
        """
        if self._text_only:
            raise RuntimeError("This SigLIP2 instance was loaded in text-only mode")
        if not images:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        if self.model is None or self.processor is None:
            self.load()

        outputs: list[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch = images[start : start + batch_size]
                inputs = self.processor(
                    images=batch,
                    return_tensors="pt",
                ).to(self.device)

                out = self.model.get_image_features(**inputs)
                if hasattr(out, "pooler_output"):
                    image_embeds = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    image_embeds = out.last_hidden_state[:, 0, :]
                else:
                    image_embeds = out

                # L2-normalize
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                outputs.append(image_embeds.detach().cpu().float().numpy())

        return np.vstack(outputs)

    def _cache_text(self, text: str, vec: np.ndarray) -> None:
        """Simple LRU cache for text embeddings."""
        if len(self._cache) >= self._cache_max_size:
            for k in list(self._cache.keys())[: self._cache_max_size // 2]:
                del self._cache[k]
        self._cache[text] = vec


# ============================================================================
# Frame-side embedding (Improvement.md §1-2: embed BTC keyframes)
# ============================================================================


@dataclass
class EmbedConfig:
    """Configuration for batch keyframe embedding."""

    model_name: str = DEFAULT_SIGLIP2_MODEL
    batch_size: int = 32
    output_dir: Path = field(default_factory=lambda: SIGLIP2_OUTPUT_DIR)
    resume: bool = True
    quantize: bool = True  # INT8 dynamic quantization for CPU (Improvement.md §11)
    fp16: bool = False  # BF16/FP16 if GPU supports (Improvement.md §11)
    max_workers: int = 4  # Parallel image loading


def compute_source_hash(manifest_path: Path) -> str:
    """Compute hash of manifest file for cache invalidation."""
    h = hashlib.sha256()
    with open(manifest_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def save_siglip2_metadata(metadata: Siglip2Metadata, path: Path) -> None:
    """Save embedding metadata for reproducibility (Improvement.md §12)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)


def load_siglip2_metadata(path: Path) -> Optional[dict]:
    """Load embedding metadata if exists."""
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def build_siglip2_embeddings(
    manifest_path: Path,
    features_output: Path,
    config: EmbedConfig,
    save_every: int = 1000,
) -> tuple[np.ndarray, list[dict]]:
    """Build SigLIP2 embeddings for all keyframes listed in manifest.

    Implements: Improvement.md §1 (use BTC keyframes directly), §11 (batch/cache/resume),
    §12 (reproducibility).

    Resume-safe: writes incrementally to a .tmp file and atomically renames to
    the final path. If the job is killed, re-running resumes from the last
    saved frame (no re-computation).

    Args:
        manifest_path: Path to official_manifest.jsonl (or siglip2 manifest).
        features_output: Output path for features_siglip2.npy.
        config: Embedding configuration.
        save_every: Save partial progress every N frames (resume safety).

    Returns:
        (embeddings_matrix, manifest_records)
    """
    from PIL import Image

    # Ensure output directory exists
    features_output.parent.mkdir(parents=True, exist_ok=True)

    # Load manifest records
    records = _load_manifest_records(manifest_path)
    total_frames = len(records)
    logger.info(f"Processing {total_frames} frames from manifest")

    # Resume setup: find the latest checkpoint.
    # tmp_path uses a name that ENDS in .npy so np.save does not append another
    # .npy (which was the old bug: "features_siglip2.npy.tmp" -> np.save wrote
    # "features_siglip2.npy.tmp.npy" and resume could never find it).
    tmp_path = features_output.with_name(features_output.stem + ".tmp.npy")  # features_siglip2.tmp.npy
    # Backward-compat: checkpoint left by the buggy old code (np.save appended .npy)
    legacy_tmp = features_output.with_suffix(".npy.tmp")
    legacy_tmp = legacy_tmp.parent / (legacy_tmp.name + ".npy")  # features_siglip2.npy.tmp.npy

    checkpoint = None
    if config.resume:
        for cand in (tmp_path, legacy_tmp, features_output):
            if cand.exists():
                checkpoint = cand
                break

    if checkpoint is features_output:
        # Final file exists. If complete, skip entirely.
        with open(features_output, "rb") as _f:
            _arr = np.load(_f, allow_pickle=False)
            if _arr.shape[0] >= total_frames:
                logger.info("Existing embeddings complete, skipping.")
                return _arr, records
            logger.info(f"Final file incomplete: {_arr.shape[0]}/{total_frames}; will resume from checkpoint.")
        checkpoint = None
        for cand in (tmp_path, legacy_tmp):
            if cand.exists():
                checkpoint = cand
                break

    # Determine model (Improvement.md §11: quantization + thread optimization)
    embedder = Siglip2Embedder(
        model_name=config.model_name,
        quantize=config.quantize,
    )
    embedding_dim = embedder.embedding_dim

    completed_frames = 0
    head = None
    if checkpoint is not None:
        # Load fully into RAM (NOT mmap) so the file handle is released and
        # Windows does not block the subsequent np.save() to the same path.
        # (mmap + np.save-to-same-path = Errno 22 on Windows.)
        with open(checkpoint, "rb") as _f:
            partial = np.load(_f, allow_pickle=False)
        completed_frames = min(partial.shape[0], total_frames)
        head = np.asarray(partial[:completed_frames], dtype=np.float32)
        del partial
        logger.info(f"Resuming from {checkpoint.name}: {completed_frames}/{total_frames} frames")

    # Pre-allocate full matrix (resume-safe: fill from completed_frames)
    embeddings = np.zeros((total_frames, embedding_dim), dtype=np.float32)
    if head is not None:
        embeddings[:completed_frames] = head

    # Collect remaining image paths with their indices
    frame_indices = list(range(completed_frames, total_frames))
    image_paths = [records[i]["keyframe_path"].replace("\\", "/") for i in frame_indices]

    # Batch inference with incremental save
    last_save = completed_frames
    with embedder._torch.no_grad():
        for start in range(0, len(image_paths), config.batch_size):
            end = min(start + config.batch_size, len(image_paths))
            batch_paths = image_paths[start:end]
            batch_indices = frame_indices[start:end]

            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(img)

            batch_embeds = embedder.encode_images(images, config.batch_size)
            for j, idx in enumerate(batch_indices):
                embeddings[idx] = batch_embeds[j]

            processed = end + completed_frames
            logger.info(
                f"Processed {processed}/{total_frames} "
                f"(frame {processed}/{total_frames})"
            )

            # Incremental save (resume-safe)
            if processed - last_save >= save_every or processed >= total_frames:
                np.save(tmp_path, embeddings[:processed])
                last_save = processed
                logger.info(f"  -> Saved checkpoint: {processed} frames")

    # Final save: atomic rename (tmp_path already ends in .npy)
    np.save(tmp_path, embeddings)
    tmp_path.replace(features_output)  # atomic on same filesystem
    logger.info(f"Saved {total_frames} SigLIP2 embeddings to {features_output}")

    # Save metadata
    metadata = Siglip2Metadata(
        model_name=config.model_name,
        model_version=embedder.model.config.model_type if embedder.model else "unknown",
        source_hash=compute_source_hash(manifest_path),
        embedding_dim=embedding_dim,
        dtype=str(embeddings.dtype),
        batch_size=config.batch_size,
        created_at=_timestamp_now(),
        frame_count=total_frames,
    )
    save_siglip2_metadata(metadata, features_output.parent / "meta.json")

    # Save manifest copy
    _save_manifest_copy(records, features_output.parent / "manifest_siglip2.jsonl")

    return embeddings, records


def _is_complete(embeddings: np.ndarray, manifest_path: Path) -> bool:
    """Check if existing embedding matrix matches manifest length."""
    records = _load_manifest_records(manifest_path)
    return embeddings.shape[0] == len(records)


def _load_manifest_records(manifest_path: Path) -> list[dict]:
    """Load JSONL manifest records."""
    records = []
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _save_manifest_copy(records: list[dict], output_path: Path) -> None:
    """Save a copy of the manifest alongside the embeddings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _timestamp_now() -> str:
    """Get current timestamp for metadata."""
    from datetime import datetime

    return datetime.utcnow().isoformat()


# ============================================================================
# FAISS index building (Improvement.md §2, §3)
# ============================================================================


def build_faiss_index(
    embeddings: np.ndarray,
    index_path: Path,
    use_gpu: bool = False,
) -> "object":
    """Build a FAISS index from SigLIP2 embeddings.

    Args:
        embeddings: (N, dim) L2-normalized embeddings.
        index_path: Output path for FAISS index.
        use_gpu: Whether to use GPU for index building.

    Returns:
        FAISS index object.
    """
    import faiss

    dim = embeddings.shape[1]
    metric = faiss.METRIC_INNER_PRODUCT  # Since vectors are L2-normalized, inner product = cosine

    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, dim)
    else:
        index = faiss.IndexFlatIP(dim)

    # Add vectors (ensure L2-normalized for cosine similarity)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info(
        f"FAISS index built: {index.ntotal} vectors, dim={dim}, saved to {index_path}"
    )

    return index


def search_faiss_index(
    index_path: Path,
    query_vector: np.ndarray,
    k: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Search FAISS index with a query vector.

    Args:
        index_path: Path to FAISS index file.
        query_vector: L2-normalized query vector (embedding_dim,).
        k: Number of results to retrieve.

    Returns:
        (scores, indices) - both (k,) arrays.
    """
    import faiss

    index = faiss.read_index(str(index_path))

    # Ensure query is L2-normalized
    query = query_vector.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query)

    scores, indices = index.search(query, k)
    return scores[0], indices[0]


# ============================================================================
# Backward compatibility
# ============================================================================

# Siglip2TextEmbedder is an alias for Siglip2Embedder for backward compatibility
# with code that imports the old class name
Siglip2TextEmbedder = Siglip2Embedder
