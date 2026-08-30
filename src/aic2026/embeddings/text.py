from __future__ import annotations

import numpy as np


_DEFAULT_PROMPT_TEMPLATES = (
    "{}",
    "a photo of {}",
    "a video frame showing {}",
    "a scene of {}",
    "a clear view of {}",
)


class MultilingualSemanticTextEmbedder:
    """Lightweight multilingual semantic encoder for Vietnamese text queries.

    This is intentionally optional: the project keeps the main CLIP branch for
    visual retrieval, while this helper offers a stronger semantic text signal
    for Vietnamese queries when a sentence-transformers model is available.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - import-time dependency guard
            raise RuntimeError(
                "Install model extras: uv sync --extra models"
            ) from exc

        self.model_name = model_name
        self.device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, text: str) -> np.ndarray:
        clean_text = (text or "").strip()
        if not clean_text:
            return np.zeros(self.model.get_sentence_embedding_dimension(), dtype=np.float32)
        vec = self.model.encode(clean_text, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    def unload(self) -> None:
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name, device=self.device)


class OpenCLIPTextEmbedder:
    """Text encoder matching CLIP features with multi-template prompt ensembling."""
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str | None = None,
        use_ensemble: bool = True,
    ):
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError("Install model extras: uv sync --extra models") from exc
        self.model_name = model_name
        self.pretrained = pretrained
        self.torch = torch
        self.use_ensemble = use_ensemble
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def encode(
        self,
        text: str,
        ensemble: bool | None = None,
        negative_prompt: str | None = None,
        neg_weight: float = 0.15,
    ) -> np.ndarray:
        """Encode text query into normalized CLIP feature vector with prompt ensembling and negative suppression."""
        clean_text = (text or "").strip()
        if not clean_text:
            clean_text = "a photo"
        do_ensemble = self.use_ensemble if ensemble is None else ensemble
        with self.torch.no_grad():
            if do_ensemble:
                prompts = [tmpl.format(clean_text) for tmpl in _DEFAULT_PROMPT_TEMPLATES]
                tokens = self.tokenizer(prompts).to(self.device)
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                mean_feat = features.mean(dim=0, keepdim=True)
                pos_vec = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
            else:
                tokens = self.tokenizer([clean_text]).to(self.device)
                features = self.model.encode_text(tokens)
                pos_vec = features / features.norm(dim=-1, keepdim=True)

            if negative_prompt and negative_prompt.strip():
                neg_tokens = self.tokenizer([negative_prompt.strip()]).to(self.device)
                neg_features = self.model.encode_text(neg_tokens)
                neg_vec = neg_features / neg_features.norm(dim=-1, keepdim=True)
                combined = pos_vec - neg_weight * neg_vec
                combined = combined / combined.norm(dim=-1, keepdim=True)
                return combined[0].detach().cpu().float().numpy()

            return pos_vec[0].detach().cpu().float().numpy()

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
