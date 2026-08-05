from __future__ import annotations

from collections.abc import Iterable
import numpy as np

from .text import HashingTextEmbedder


class SigLIPEncoder:
    """Joint image-text encoder for a self-generated SigLIP/SigLIP2 feature index."""
    def __init__(self, model_id: str = "google/siglip2-base-patch16-224", device: str | None = None):
        self._fallback_embedder: HashingTextEmbedder | None = None
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            self._fallback_embedder = HashingTextEmbedder(768)
            self.torch = None
            self.device = device or "cpu"
            self.processor = None
            self.model = None
            return

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        except Exception:
            self._fallback_embedder = HashingTextEmbedder(768)
            self.processor = None
            self.model = None

    def _extract_embedding(self, output):
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state
        if isinstance(output, tuple):
            return output[0]
        return output

    def _normalize(self, values):
        return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _fallback_encode(self, text: str) -> np.ndarray:
        if self._fallback_embedder is None:
            self._fallback_embedder = HashingTextEmbedder(768)
        return self._fallback_embedder.encode(text)

    def encode_text(self, text: str) -> np.ndarray:
        if self.model is None or self.processor is None or self.torch is None:
            return self._fallback_encode(text)
        batch = self.processor(text=[text], padding=True, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model.get_text_features(**batch)
            values = self._normalize(self._extract_embedding(output))
        return values[0].detach().cpu().float().numpy()

    def encode_images(self, images: Iterable[object]) -> np.ndarray:
        images = list(images)
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        if self.model is None or self.processor is None or self.torch is None:
            return np.array([self._fallback_encode(str(image)) for image in images], dtype=np.float32)
        batch = self.processor(images=images, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model.get_image_features(**batch)
            values = self._normalize(self._extract_embedding(output))
        return values.detach().cpu().float().numpy()

    def encode_images_in_chunks(self, images: Iterable[object], batch_size: int = 4) -> list[np.ndarray]:
        items = list(images)
        if not items:
            return []
        outputs: list[np.ndarray] = []
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            outputs.append(self.encode_images(chunk))
        return outputs
