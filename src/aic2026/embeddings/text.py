from __future__ import annotations

import numpy as np


class OpenCLIPTextEmbedder:
    """Text encoder matching CLIP ViT-B/32 features when the checkpoint is the same."""
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str | None = None):
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError("Install model extras: uv sync --extra models") from exc
        self.model_name = model_name
        self.pretrained = pretrained
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self._cache: dict[str, np.ndarray] = {}
        self._cache_max_size: int = 2048

    def encode(self, text: str) -> np.ndarray:
        cached = self._cache.get(text)
        if cached is not None:
            return cached.copy()

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is unloaded. Call load() first.")

        with self.torch.no_grad():
            features = self.model.encode_text(self.tokenizer([text]).to(self.device))
            features = features / features.norm(dim=-1, keepdim=True)
            vec = features[0].detach().cpu().float().numpy()

        if len(self._cache) >= self._cache_max_size:
            # Simple eviction: clear half when full
            for k in list(self._cache.keys())[: self._cache_max_size // 2]:
                del self._cache[k]
        self._cache[text] = vec
        return vec.copy()

    def encode_images(self, images: list[object], batch_size: int = 32) -> np.ndarray:
        """Encode RGB PIL images with the same CLIP checkpoint as ``encode``."""
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        outputs: list[np.ndarray] = []
        with self.torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch = self.torch.stack(
                    [self.preprocess(image) for image in images[start:start + batch_size]]
                ).to(self.device)
                features = self.model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
                outputs.append(features.detach().cpu().float().numpy())
        return np.vstack(outputs)

    def unload(self) -> None:
        """Drop the model/tokenizer to free RAM (used before loading a VLM on
        low-memory machines)."""
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        try:
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def load(self) -> None:
        """(Re)create the model/tokenizer if missing. Safe to call repeatedly;
        a no-op when already loaded. Enables the encoder to be unloaded (to free
        RAM for a VLM) and reloaded for a subsequent retrieve on a cached agent."""
        if self.model is not None and self.tokenizer is not None:
            return
        import open_clip
        import torch

        self.torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()
