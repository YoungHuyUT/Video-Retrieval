"""Full SigLIP2 embedding for entire BTC keyframe corpus.

Chạy trong background để embed toàn bộ 177k BTC keyframes bằng SigLIP2-Base
(INT8 quantized) → data/processed/siglip2/features_siglip2.npy

Usage:
    python scripts/embed_siglip2_full.py [--limit N] [--batch-size B]

Improvement.md Phase 1: BTC Keyframes → SigLIP2 embedding
Improvement.md §11: batch inference, cache, resume, quantization
Improvement.md §12: metadata saved for reproducibility
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Setup logging
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("data/processed/siglip2/embed_full.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SigLIP2 full keyframe embedding")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N frames (0 = all)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume if output exists")
    parser.add_argument("--quantize", action="store_true", default=True, help="Use INT8 quantization")
    args = parser.parse_args()

    from aic2026.embeddings.siglip2 import (
        Siglip2Embedder, EmbedConfig, build_siglip2_embeddings,
        SIGLIP2_OUTPUT_DIR, SIGLIP2_FEATURES_FILE, SIGLIP2_MANIFEST_FILE
    )

    logger.info("=" * 60)
    logger.info("SigLIP2 Full Keyframe Embedding")
    logger.info("=" * 60)
    logger.info(f"Model: google/siglip2-so400m-patch14-384 (float16)")
    logger.info(f"Quantization: {args.quantize}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Limit: {args.limit if args.limit > 0 else 'ALL'}")

    # Setup output directory
    SIGLIP2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create manifest (full or limited)
    manifest_path = Path("data/processed/official_manifest.jsonl")
    if args.limit > 0:
        temp_manifest = Path("data/processed/siglip2/_temp_manifest.jsonl")
        with open(manifest_path, "r") as src:
            with open(temp_manifest, "w") as out:
                for i, line in enumerate(src):
                    if i >= args.limit:
                        break
                    out.write(line)
        manifest_path = temp_manifest
        logger.info(f"Using limited manifest: {temp_manifest} ({args.limit} records)")

    # Count total frames
    with open(manifest_path, "r") as f:
        total_frames = sum(1 for line in f if line.strip())
    logger.info(f"Total frames to process: {total_frames}")

    # Create config — use so400m (1152-dim) with float16 to fit in 8GB RAM
    config = EmbedConfig(
        model_name="google/siglip2-so400m-patch14-384",
        batch_size=args.batch_size,
        resume=args.resume and args.limit == 0,
        quantize=False,  # INT8 quantization not compatible with float16
    )

    # Output paths
    output_features = SIGLIP2_OUTPUT_DIR / "features_siglip2.npy"
    if args.limit > 0:
        output_features = SIGLIP2_OUTPUT_DIR / f"features_test_{args.limit}.npy"

    # Estimate time
    estimated_time = total_frames * 0.18 / 3600  # ~0.18s/frame with quantization
    logger.info(f"Estimated time: ~{estimated_time:.1f}h")

    # Run embedding
    start = time.time()
    embeddings, records = build_siglip2_embeddings(
        manifest_path,
        output_features,
        config,
        save_every=1000,  # Save checkpoint every 1000 frames (resume-safe)
    )
    elapsed = time.time() - start

    # Summary
    logger.info("=" * 60)
    logger.info("EMBEDDING COMPLETE")
    logger.info(f"  Frames: {embeddings.shape[0]}")
    logger.info(f"  Dim: {embeddings.shape[1]}")
    logger.info(f"  Time: {elapsed/3600:.2f}h ({embeddings.shape[0]/elapsed:.1f} fps)")
    logger.info(f"  Output: {output_features}")

    # Verify norms
    norms = np.linalg.norm(embeddings, axis=1)
    logger.info(f"  Norm range: [{norms.min():.4f}, {norms.max():.4f}]")

    # Save manifest copy
    with open(SIGLIP2_MANIFEST_FILE, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info(f"  Manifest: {SIGLIP2_MANIFEST_FILE}")

    # Cleanup temp manifest
    if args.limit > 0 and manifest_path != Path("data/processed/official_manifest.jsonl"):
        manifest_path.unlink(missing_ok=True)

    logger.info("=" * 60)
    print(f"\nSUCCESS: {embeddings.shape[0]} frames embedded in {elapsed/3600:.2f}h")


if __name__ == "__main__":
    main()
