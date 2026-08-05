import numpy as np
import sys
import types

from PIL import Image
import torch


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeProcessor:
    @staticmethod
    def from_pretrained(_model_id):
        return FakeProcessor()

    def __call__(self, *args, **kwargs):
        return FakeBatch({"input_ids": torch.tensor([[1]])})


class FakeOutput:
    def __init__(self, tensor):
        self.pooler_output = tensor


class FakeModel:
    @staticmethod
    def from_pretrained(_model_id):
        return FakeModel()

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def get_text_features(self, **kwargs):
        return FakeOutput(torch.tensor([[3.0, 4.0]], dtype=torch.float32))

    def get_image_features(self, **kwargs):
        return FakeOutput(torch.tensor([[1.0, 2.0]], dtype=torch.float32))


def test_encode_text_supports_wrapped_transformer_output(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoProcessor=FakeProcessor, AutoModel=FakeModel))

    from aic2026.embeddings.siglip import SigLIPEncoder

    encoder = SigLIPEncoder(model_id="dummy")
    vector = encoder.encode_text("hello")

    assert vector.shape == (2,)
    np.testing.assert_allclose(vector, np.array([0.6, 0.8], dtype=np.float32))


def test_encode_images_in_chunks_splits_large_inputs(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoProcessor=FakeProcessor, AutoModel=FakeModel))

    from aic2026.embeddings.siglip import SigLIPEncoder

    encoder = SigLIPEncoder(model_id="dummy")
    vectors = encoder.encode_images_in_chunks([object(), object(), object()], batch_size=2)

    assert len(vectors) == 2
    assert vectors[0].shape == (1, 2)
    assert vectors[1].shape == (1, 2)


def test_fallback_encoder_works_when_model_load_fails(monkeypatch):
    class FailingProcessor:
        @staticmethod
        def from_pretrained(_model_id):
            raise RuntimeError("boom")

    class FailingModel:
        @staticmethod
        def from_pretrained(_model_id):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoProcessor=FailingProcessor, AutoModel=FailingModel))

    from aic2026.embeddings.siglip import SigLIPEncoder

    encoder = SigLIPEncoder(model_id="dummy")
    vector = encoder.encode_text("hello")

    assert vector.shape == (768,)
    assert np.isfinite(vector).all()


def test_embed_existing_keyframes_supports_single_video_and_chunking(tmp_path):
    from aic2026.data_platform.video_frames import embed_existing_keyframes

    class FakeEncoder:
        def __init__(self):
            self.calls = []

        def encode_images(self, images):
            self.calls.append(len(images))
            return np.array([[float(len(images)), 0.0] for _ in images], dtype=np.float32)

    keyframes_root = tmp_path / "keyframes"
    features_root = tmp_path / "features"
    video_dir = keyframes_root / "L21_V001"
    video_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(video_dir / "000000001.jpg")
    Image.fromarray(np.ones((4, 4, 3), dtype=np.uint8)).save(video_dir / "000000002.jpg")
    (keyframes_root / "L21_V002").mkdir(parents=True)
    Image.fromarray(np.full((4, 4, 3), 2, dtype=np.uint8)).save(keyframes_root / "L21_V002" / "000000001.jpg")

    encoder = FakeEncoder()
    archives = embed_existing_keyframes(keyframes_root, features_root, encoder, batch_size=2, video_id="L21_V001")

    assert len(archives) == 1
    assert encoder.calls == [2]
    archive = np.load(features_root / "L21_V001.npz")
    assert np.array_equal(archive["frame_ids"], np.array([1, 2], dtype=np.int64))
    assert archive["features"].shape == (2, 2)
