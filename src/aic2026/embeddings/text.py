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
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        with self.torch.no_grad():
            features = self.model.encode_text(self.tokenizer([text]).to(self.device))
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].detach().cpu().float().numpy()

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
