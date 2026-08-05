from __future__ import annotations

import hashlib
import numpy as np


class HashingTextEmbedder:
    """Offline deterministic fallback; replace with OpenCLIP multilingual encoder in model config."""
    def __init__(self, dimension: int): self.dimension = dimension
    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in text.lower().split():
            index = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimension
            vector[index] += 1
        return vector


class OpenCLIPTextEmbedder:
    """Text encoder matching CLIP ViT-B/32 features when the checkpoint is the same."""
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str | None = None):
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError("Install model extras: uv sync --extra models") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        with self.torch.no_grad():
            features = self.model.encode_text(self.tokenizer([text]).to(self.device))
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].detach().cpu().float().numpy()
