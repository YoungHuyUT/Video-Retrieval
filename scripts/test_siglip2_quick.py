"""Quick test of SigLIP2 embedding with quantization."""
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(__file__).parent.parent / "src")

import json
import time
import numpy as np
from pathlib import Path
from PIL import Image

from aic2026.embeddings.siglip2 import Siglip2Embedder, EmbedConfig, build_siglip2_embeddings

embedder = Siglip2Embedder(quantize=True)
print(f"Dim: {embedder.embedding_dim}")

records = []
with open("data/processed/official_manifest.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i >= 50:
            break
        records.append(json.loads(line))

images = []
for r in records:
    path = Path(r["keyframe_path"].replace("\\", "/"))
    img = Image.open(path).convert("RGB")
    images.append(img)

start = time.time()
embeddings = embedder.encode_images(images, batch_size=32)
elapsed = time.time() - start
print(f"50 images: {elapsed:.2f}s = {50/elapsed:.1f} img/s")
print(f"Shape: {embeddings.shape}")
print(f"Norms OK: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=0.01)}")

embedder.unload()
print("PASS!")
