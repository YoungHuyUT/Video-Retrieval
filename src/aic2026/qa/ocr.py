from __future__ import annotations

import logging
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
        self._paddle_kwargs = paddle_kwargs
        self._ocr = None
        self._load_failed: bool = False

    def _ensure_loaded(self) -> None:
        if self._ocr is not None or self._load_failed:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            logger.warning(
                "OCRTextExtractor: paddleocr unavailable. "
                "Install with: uv sync --extra models  (plus paddlepaddle)"
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

    def extract_many(self, frame_paths: list[str | Path], batch_size: int = 32) -> list[list[str]]:
        """Extract text for many frames, batching to bound memory."""
        outputs: list[list[str]] = []
        for start in range(0, len(frame_paths), batch_size):
            chunk = frame_paths[start : start + batch_size]
            for path in chunk:
                outputs.append(self.extract(path))
        return outputs
