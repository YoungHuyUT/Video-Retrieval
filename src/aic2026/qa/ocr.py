from __future__ import annotations

import logging
import json
import os
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_paddle_stderr():
    """Hide Paddle's raw C++ stderr noise (e.g. ``ReduceMeanCheckIfOneDNNSupport``).

    That line is printed straight to the C-level stderr stream by Paddle's OneDNN
    backend, so env vars like ``GLOG_minloglevel`` cannot suppress it. We redirect
    the OS file descriptor 2 to a pipe, drain it in a background thread, and
    restore it on exit. Only active on the non-Windows (fileno) path; on Windows
    (no usable fileno) we silently skip so behaviour is unchanged.
    """
    raw_stderr = getattr(sys.stderr, "fileno", None)
    if raw_stderr is None or not hasattr(sys.stderr, "fileno"):
        yield
        return
    try:
        fd = sys.stderr.fileno()
    except (OSError, ValueError):
        yield
        return

    old_fd = os.dup(fd)
    pipe_r, pipe_w = os.pipe()
    os.dup2(pipe_w, fd)

    import threading

    stop = threading.Event()

    def _drain():
        try:
            while not stop.is_set():
                try:
                    chunk = os.read(pipe_r, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                # Discard — these are Paddle C++ diagnostic lines, not user output.
        finally:
            try:
                os.close(pipe_r)
            except OSError:
                pass

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(old_fd, fd)
        os.close(old_fd)
        os.close(pipe_w)
        stop.set()
        t.join(timeout=1.0)


class OCRTextExtractor:
    """PaddleOCR wrapper for reading Vietnamese text off keyframes.

    Optional dependency: if ``paddleocr`` is missing, every ``extract`` call
    returns ``[]`` instead of crashing, so retrieval degrades gracefully.
    Used to enrich the BM25 index with on-screen text (titles, logos, banners)
    that CLIP cannot read.

    ``correct`` (default True) runs a lightweight Vietnamese post-correction
    pass (see :mod:`aic2026.qa.vi_correct`) to fix systematic OCR errors
    (garbled diacritics, confusable letter pairs, noise tokens) without
    swapping the model — a public VN fine-tune was tested on real keyframes
    and scored *worse* (it overfits document-scan fonts).
    """

    def __init__(
        self,
        lang: str = "vi",
        model_size: str = "medium",
        correct: bool = True,
        device: str = "cpu",
        **paddle_kwargs: object,
    ) -> None:
        self.lang = lang
        self.model_size = model_size
        self.correct = correct
        # Inference device for PaddleOCR 3.x: "cpu" (default), "gpu", or
        # "gpu:0". The local dev machine has no GPU, so the default keeps OCR
        # running on CPU; pass --device gpu on a Colab T4 / CUDA box for a
        # 10-20x speedup. We validate CUDA availability here so a misconfigured
        # GPU request fails loudly (with a clear fix) instead of dying deep in
        # Paddle's C++ init.
        self.device = device
        if device.startswith("gpu"):
            import paddle  # local import keeps CPU installs light

            if not paddle.is_compiled_with_cuda():
                raise RuntimeError(
                    "OCR device='gpu' nhưng PaddlePaddle hiện tại KHÔNG được build "
                    "với CUDA (paddle.is_compiled_with_cuda() == False). Cài đặt "
                    "paddlepaddle-gpu tương ứng (vd trên Colab: "
                    "`pip install paddlepaddle-gpu==3.2.0 -f "
                    "https://www.paddlepaddle.org.cn/whl/linux/cuda12.0/`) rồi chạy lại, "
                    "hoặc dùng --device cpu."
                )
        # Keyframes are ordinary video frames, not scanned documents. PaddleOCR
        # v3 enables three document-preprocessing models by default; they add
        # several downloads and substantial RAM use but do not help normal
        # scene-text retrieval. Keep OCR to detector + recognizer by default.
        self._paddle_kwargs = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": self.device,
            **paddle_kwargs,
        }
        # PP-OCRv6 ships both a `medium` (default, accurate) and a `mobile`
        # (3–5x faster, slightly less accurate) recognizer/detector. The
        # PaddleOCR 3.x constructor only accepts `det_model_dir` /
        # `rec_model_dir` (the bare `*_model_name` keys raise
        # "Unknown argument"), so pass the checkpoint names through the
        # `*_model_dir` slots — Paddle resolves them from its model zoo. Only
        # set when explicitly requested so the default path is unchanged.
        if model_size == "mobile":
            self._paddle_kwargs["det_model_dir"] = "PP-OCRv6_mobile_det"
            self._paddle_kwargs["rec_model_dir"] = "PP-OCRv6_mobile_rec"
        self._ocr = None
        self._load_failed: bool = False

    def _post_correct(self, texts: list[str]) -> list[str]:
        if not self.correct:
            return texts
        from aic2026.qa.vi_correct import correct_texts

        return correct_texts(texts)

    def _ensure_loaded(self) -> None:
        if self._ocr is not None or self._load_failed:
            return
        try:
            # PaddleX otherwise probes every model host before downloading. In
            # restricted/offline environments that probe is slow and can fail
            # before the actual OCR models are initialized.
            import os

            # Quiet Paddle's glog-level diagnostics. The raw C++ OneDNN stderr
            # noise (ReduceMeanCheckIfOneDNNSupport) is filtered separately at the
            # file-descriptor level during inference (_suppress_paddle_stderr).
            os.environ.setdefault("GLOG_minloglevel", "3")
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR
        except ImportError as exc:
            logger.warning(
                "OCRTextExtractor: PaddleOCR/PaddlePaddle cannot be imported: %s. "
                "If libpaddle.pyd is named, repair the PaddlePaddle Windows CPU "
                "install and Microsoft Visual C++ x64 runtime.",
                exc,
            )
            self._load_failed = True
            return
        try:
            self._ocr = PaddleOCR(lang=self.lang, **self._paddle_kwargs)
        except Exception as exc:  # noqa: BLE001 — OCR init must degrade, not crash
            logger.warning("OCRTextExtractor: failed to init PaddleOCR: %s", exc)
            self._load_failed = True

    def extract(self, frame_path: str | Path) -> list[str]:
        """Return the list of recognized text lines in a keyframe (or [])."""
        self._ensure_loaded()
        if self._ocr is None:
            return []
        try:
            # PaddleOCR 3.x returns OCRResult objects from ``predict`` with
            # ``rec_texts``. The former 2.x ``ocr`` output is a nested list of
            # [box, (text, confidence)]. Supporting both avoids silently
            # writing an unchanged manifest when PaddleOCR is upgraded.
            with _suppress_paddle_stderr():
                if hasattr(self._ocr, "predict"):
                    result = self._ocr.predict(str(frame_path))
                else:
                    result = self._ocr.ocr(str(frame_path), cls=True)
            if hasattr(self._ocr, "predict"):
                return self._post_correct(self._extract_v3(result))
        except Exception as exc:  # noqa: BLE001 — a bad frame must not abort a batch
            logger.debug("OCRTextExtractor: error on %s: %s", frame_path, exc)
            return []

        texts: list[str] = []
        if not result:
            return texts
        for page in result:
            if not page:
                continue
            for line in page:
                # PaddleOCR line: [box, (text, confidence)]
                try:
                    text = line[1][0]
                except (IndexError, TypeError):
                    continue
                if text and text.strip():
                    texts.append(text.strip())
        return self._post_correct(texts)

    @staticmethod
    def _extract_v3(results: object) -> list[str]:
        """Read ``rec_texts`` from PaddleOCR 3.x OCRResult objects."""

        texts: list[str] = []
        try:
            pages = iter(results)  # ``predict`` returns an iterator/generator.
        except TypeError:
            return texts

        for page in pages:
            payload: object = page
            if not isinstance(payload, Mapping):
                payload = getattr(page, "json", None)
                if callable(payload):
                    payload = payload()
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        payload = None
                if not isinstance(payload, Mapping):
                    payload = getattr(page, "res", None)
            if not isinstance(payload, Mapping):
                continue

            result = payload.get("res", payload)
            if not isinstance(result, Mapping):
                continue
            recognized = result.get("rec_texts", [])
            if isinstance(recognized, str):
                recognized = [recognized]
            if not isinstance(recognized, (list, tuple)):
                continue
            texts.extend(
                text.strip()
                for text in recognized
                if isinstance(text, str) and text.strip()
            )

        return list(dict.fromkeys(texts))

    def extract_many(self, frame_paths: list[str | Path], batch_size: int = 32) -> list[list[str]]:
        """Extract text for many frames, using PaddleOCR v3 batch prediction.

        ``PaddleOCR.predict`` accepts a list of inputs and returns one result per
        image.  If an older PaddleOCR build rejects list input, fall back to the
        single-image path so OCR remains functional.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self._ensure_loaded()
        if self._ocr is None:
            return [[] for _ in frame_paths]

        outputs: list[list[str]] = []
        for start in range(0, len(frame_paths), batch_size):
            chunk = frame_paths[start : start + batch_size]
            if hasattr(self._ocr, "predict"):
                try:
                    # Suppress Paddle's raw C++ stderr (ReduceMeanCheckIfOneDNNSupport,
                    # oneDNN noise) during the actual inference call.
                    with _suppress_paddle_stderr():
                        pages = list(self._ocr.predict([str(path) for path in chunk]))
                    if len(pages) != len(chunk):
                        raise RuntimeError(
                            f"PaddleOCR returned {len(pages)} results for {len(chunk)} inputs"
                        )
                    outputs.extend(
                        self._post_correct(self._extract_v3([page])) for page in pages
                    )
                    continue
                except Exception as exc:  # noqa: BLE001 - compatibility fallback
                    logger.warning(
                        "PaddleOCR batch prediction failed; falling back to one image at a time: %s",
                        exc,
                    )
            outputs.extend(self.extract(path) for path in chunk)
        return outputs
