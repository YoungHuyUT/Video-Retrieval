from __future__ import annotations

import logging
import json
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)


class OCRTextExtractor:
    """PaddleOCR wrapper for reading Vietnamese text off keyframes.

    Optional dependency: if ``paddleocr`` is missing, every ``extract`` call
    returns ``[]`` instead of crashing, so retrieval degrades gracefully.
    Used to enrich the BM25 index with on-screen text (titles, logos, banners)
    that CLIP cannot read.
    """

    def __init__(self, lang: str = "vi", **paddle_kwargs: object) -> None:
        self.lang = lang
        # Keyframes are ordinary video frames, not scanned documents. PaddleOCR
        # v3 enables three document-preprocessing models by default; they add
        # several downloads and substantial RAM use but do not help normal
        # scene-text retrieval. Keep OCR to detector + recognizer by default.
        self._paddle_kwargs = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            **paddle_kwargs,
        }
        self._ocr = None
        self._load_failed: bool = False

    def _ensure_loaded(self) -> None:
        if self._ocr is not None or self._load_failed:
            return
        try:
            # PaddleX otherwise probes every model host before downloading. In
            # restricted/offline environments that probe is slow and can fail
            # before the actual OCR models are initialized.
            import os
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
            if hasattr(self._ocr, "predict"):
                return self._extract_v3(self._ocr.predict(str(frame_path)))
            result = self._ocr.ocr(str(frame_path), cls=True)
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
        return texts

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
                    pages = list(self._ocr.predict([str(path) for path in chunk]))
                    if len(pages) != len(chunk):
                        raise RuntimeError(
                            f"PaddleOCR returned {len(pages)} results for {len(chunk)} inputs"
                        )
                    outputs.extend(self._extract_v3([page]) for page in pages)
                    continue
                except Exception as exc:  # noqa: BLE001 - compatibility fallback
                    logger.warning(
                        "PaddleOCR batch prediction failed; falling back to one image at a time: %s",
                        exc,
                    )
            outputs.extend(self.extract(path) for path in chunk)
        return outputs
