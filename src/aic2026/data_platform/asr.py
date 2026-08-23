from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import os

logger = logging.getLogger(__name__)


def _setup_windows_cuda_dlls() -> None:
    """Ensure NVIDIA cuBLAS / cuDNN dlls from pip are visible to CTranslate2 on Windows."""
    if os.name != "nt":
        return
    import sys
    for path in sys.path:
        nvidia_dir = Path(path) / "nvidia"
        if nvidia_dir.is_dir():
            for sub in ("cublas/bin", "cudnn/bin", "cuda_nvrtc/bin", "cublas/lib", "cudnn/lib"):
                p = (nvidia_dir / sub).resolve()
                if p.is_dir():
                    try:
                        os.add_dll_directory(str(p))
                    except Exception:
                        pass
                    if str(p) not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")


class ASRExtractor:
    """Automatic Speech Recognition (ASR) extractor using faster-whisper.

    Extracts timestamped speech transcript segments [start, end, text] directly
    from .mp4 video files (via PyAV audio decoding).

    Whisper models:
      - 'tiny' / 'base': fastest, very light (~75MB–140MB)
      - 'small': recommended balance for Vietnamese/English (~460MB)
      - 'medium': accurate (~1.5GB)
      - 'large-v3' / 'large-v3-turbo': highest accuracy (~3GB)
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "default",
        lang: str | None = "vi",
        beam_size: int = 5,
        vad_filter: bool = True,
        initial_prompt: str | None = "Đây là bản tin thời sự, phóng sự truyền hình, tài liệu, thể thao, văn hóa bằng tiếng Việt chuẩn chính tả.",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.lang = lang
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.initial_prompt = initial_prompt
        self._model: Any = None
        self._load_failed: bool = False

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_failed:
            return
        _setup_windows_cuda_dlls()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            logger.warning(
                "ASRExtractor: faster-whisper is not installed (%s). "
                "Install with: pip install faster-whisper",
                exc,
            )
            self._load_failed = True
            return

        resolved_device = self.device
        if resolved_device == "auto":
            try:
                import ctranslate2
                resolved_device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                try:
                    import torch
                    resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    resolved_device = "cpu"

        resolved_compute = self.compute_type
        if resolved_compute == "default":
            resolved_compute = "int8" if resolved_device == "cpu" else "float32"

        try:
            logger.info(
                "Loading faster-whisper model '%s' on %s (%s)...",
                self.model_size,
                resolved_device,
                resolved_compute,
            )
            self._model = WhisperModel(
                self.model_size,
                device=resolved_device,
                compute_type=resolved_compute,
            )
        except Exception as exc:  # noqa: BLE001
            # If float16 or CUDA load fails, retry with float32 or cpu
            if resolved_device == "cuda" and resolved_compute != "float32":
                try:
                    logger.info("Retrying faster-whisper with float32 on CUDA...")
                    self._model = WhisperModel(
                        self.model_size,
                        device="cuda",
                        compute_type="float32",
                    )
                except Exception:
                    logger.warning("Retrying faster-whisper on CPU fallback: %s", exc)
                    self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            else:
                logger.warning("ASRExtractor: failed to load WhisperModel: %s", exc)
                self._load_failed = True

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    def transcribe_video(self, video_path: str | Path) -> list[dict[str, Any]]:
        """Transcribe a single video file.

        Returns a list of segments:
        ``[{"start": 0.0, "end": 4.5, "text": "..."}, ...]``
        """
        self._ensure_loaded()
        if self._model is None:
            return []

        path = Path(video_path)
        if not path.exists():
            logger.warning("ASRExtractor: video path does not exist: %s", path)
            return []

        try:
            segments, info = self._model.transcribe(
                str(path),
                language=self.lang,
                beam_size=self.beam_size,
                initial_prompt=self.initial_prompt,
                vad_filter=self.vad_filter,
                word_timestamps=False,
            )
            results = []
            for seg in segments:
                text = seg.text.strip()
                if text:
                    results.append({
                        "start": round(float(seg.start), 2),
                        "end": round(float(seg.end), 2),
                        "text": text,
                    })
            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning("ASRExtractor: error transcribing %s: %s", path.name, exc)
            return []

    def transcribe_directory(
        self,
        videos_dir: Path | str,
        output_dir: Path | str,
        video_prefix: str = "",
        resume: bool = True,
        progress_callback: callable | None = None,
    ) -> list[Path]:
        """Transcribe all .mp4 videos in ``videos_dir``, saving per-video JSON in ``output_dir``."""
        videos_dir = Path(videos_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_paths = sorted(videos_dir.glob("*.mp4"))
        if not video_paths:
            video_paths = sorted(videos_dir.rglob("*.mp4"))

        prefixes = [p.strip() for p in video_prefix.split(",") if p.strip()]
        if prefixes:
            video_paths = [
                v for v in video_paths
                if any(v.stem.startswith(p) for p in prefixes)
            ]

        written: list[Path] = []
        total = len(video_paths)
        for idx, video_path in enumerate(video_paths):
            video_id = video_path.stem
            out_file = output_dir / f"{video_id}.json"

            if resume and out_file.exists() and out_file.stat().st_size > 0:
                logger.debug("Skipping already transcribed: %s", video_id)
                written.append(out_file)
                if progress_callback is not None:
                    progress_callback(idx + 1, total, video_id, True)
                continue

            segments = self.transcribe_video(video_path)
            out_file.write_text(
                json.dumps(segments, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            written.append(out_file)
            if progress_callback is not None:
                progress_callback(idx + 1, total, video_id, False)

        return written
