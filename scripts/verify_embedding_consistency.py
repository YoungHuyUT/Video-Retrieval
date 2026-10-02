#!/usr/bin/env python3
"""Verify embedding dimension consistency between index and text encoder.

Checks that the CLIP text encoder and image index have matching dimensions.
Fails fast if they don't match (prevents silent recall degradation).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def verify_embedding_consistency(
    features_path: str | Path,
    text_encoder_model: str = "google/siglip2-base-patch16-224",
) -> bool:
    """Verify that the feature index and text encoder have matching dimensions.

    Args:
        features_path: Path to the .npy feature file
        text_encoder_model: Text encoder model name to check

    Returns:
        True if dimensions match, False otherwise
    """
    features_path = Path(features_path)

    # Load feature index to get dimension
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        return False

    features = np.load(features_path, mmap_mode="r")
    index_dim = features.shape[1]
    logger.info(f"Feature index dimension: {index_dim} (shape: {features.shape})")

    # Load text encoder to get its output dimension
    try:
        if "siglip2" in text_encoder_model.lower():
            from aic2026.embeddings import Siglip2Embedder
            encoder = Siglip2Embedder.get_or_create(model_name=text_encoder_model)
        elif "openclip" in text_encoder_model.lower() or "vit-b" in text_encoder_model.lower() or "vit-l" in text_encoder_model.lower():
            from aic2026.embeddings import OpenCLIPTextEmbedder
            encoder = OpenCLIPTextEmbedder(model_name=text_encoder_model)
        else:
            logger.warning(f"Unknown encoder type for {text_encoder_model}, assuming 512")
            return False

        encoder.load()
        test_vec = encoder.encode("test query")
        encoder_dim = test_vec.shape[0]
        logger.info(f"Text encoder dimension: {encoder_dim}")

    except Exception as exc:
        logger.error(f"Failed to load text encoder {text_encoder_model}: {exc}")
        return False

    # Check match
    if index_dim != encoder_dim:
        logger.error(
            f"DIMENSION MISMATCH: index={index_dim}, encoder={encoder_dim}. "
            f"This will cause silent recall failure! "
            f"Fix: rebuild index with matching encoder or switch encoder to match index."
        )
        return False

    logger.info(f"✓ Dimensions match: {index_dim} == {encoder_dim}")
    return True


def check_manifest_consistency(
    manifest_path: str | Path,
    features_path: str | Path,
) -> bool:
    """Verify that manifest record count matches feature count.

    Args:
        manifest_path: Path to manifest JSONL
        features_path: Path to .npy feature file

    Returns:
        True if counts match, False otherwise
    """
    manifest_path = Path(manifest_path)
    features_path = Path(features_path)

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return False

    if not features_path.exists():
        logger.error(f"Features not found: {features_path}")
        return False

    # Count manifest records
    import json
    manifest_count = 0
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                manifest_count += 1

    # Count feature vectors
    features = np.load(features_path, mmap_mode="r")
    feature_count = features.shape[0]

    logger.info(f"Manifest records: {manifest_count}")
    logger.info(f"Feature vectors: {feature_count}")

    if manifest_count != feature_count:
        logger.error(
            f"COUNT MISMATCH: manifest={manifest_count}, features={feature_count}. "
            f"This indicates index/manifest drift. Rebuild with `aic2026 prepare`."
        )
        return False

    logger.info(f"✓ Counts match: {manifest_count} == {feature_count}")
    return True


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Verify embedding consistency")
    parser.add_argument(
        "--features",
        required=True,
        help="Path to .npy feature file",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest JSONL",
    )
    parser.add_argument(
        "--encoder",
        default="google/siglip2-base-patch16-224",
        help="Text encoder model name",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ok = True
    ok &= verify_embedding_consistency(args.features, args.encoder)
    ok &= check_manifest_consistency(args.manifest, args.features)

    if ok:
        print("\n✓ All consistency checks passed!")
        return 0
    else:
        print("\n✗ Consistency check FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())