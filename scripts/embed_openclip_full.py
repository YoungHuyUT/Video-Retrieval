"""Full OpenCLIP ViT-B/32 embedding for entire BTC keyframe corpus.

Replaces SigLIP2 embeddings which are broken for cosine-based retrieval.

Memory-efficient: saves embeddings in chunks without pre-allocating full matrix.
Resumes by checking which chunks exist.

Usage:
    python scripts/embed_openclip_full.py [--limit N] [--batch-size B]

Output:
    data/processed/openclip/features_openclip.npy   (177k × 512 float32)
    data/processed/openclip/manifest_openclip.jsonl
    data/processed/openclip/meta.json
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OPENCLIP_OUTPUT_DIR = Path("data/processed/openclip")
OPENCLIP_FEATURES_FILE = OPENCLIP_OUTPUT_DIR / "features_openclip.npy"
OPENCLIP_MANIFEST_FILE = OPENCLIP_OUTPUT_DIR / "manifest_openclip.jsonl"
CHUNKS_DIR = OPENCLIP_OUTPUT_DIR / "chunks"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(OPENCLIP_OUTPUT_DIR / "embed_full.log" if OPENCLIP_OUTPUT_DIR.exists() else "embed_openclip.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def compute_source_hash(manifest_path: Path) -> str:
    h = hashlib.sha256()
    with open(manifest_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_image(args) -> tuple[int, Image.Image | None]:
    """Load a single image, return (index, image_or_None)."""
    idx, path = args
    try:
        img = Image.open(path).convert("RGB")
        return idx, img
    except Exception as e:
        logger.warning("Failed to load frame %d: %s — using zeros", idx, path)
        return idx, None


CHUNK_SIZE = 5000  # Save every 5000 frames


def main():
    parser = argparse.ArgumentParser(description="OpenCLIP ViT-B/32 full keyframe embedding (memory-efficient)")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N frames (0 = all)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--workers", type=int, default=4, help="Parallel image loaders")
    args = parser.parse_args()

    from aic2026.embeddings.text import OpenCLIPTextEmbedder

    logger.info("=" * 60)
    logger.info("OpenCLIP ViT-B/32 Full Keyframe Embedding (memory-efficient)")
    logger.info("=" * 60)

    # Setup output directory
    OPENCLIP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Re-setup file handler with correct path
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and "embed_openclip.log" in h.baseFilename:
            h.close()
    file_handler = logging.FileHandler(OPENCLIP_OUTPUT_DIR / "embed_full.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(file_handler)

    # Load manifest
    manifest_path = Path("data/processed/official_manifest.jsonl")
    if args.limit > 0:
        temp_manifest = OPENCLIP_OUTPUT_DIR / "_temp_manifest.jsonl"
        with open(manifest_path, "r") as src, open(temp_manifest, "w") as out:
            for i, line in enumerate(src):
                if i >= args.limit:
                    break
                out.write(line)
        manifest_path = temp_manifest
        logger.info("Using limited manifest: %s (%d records)", temp_manifest, args.limit)

    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    total_frames = len(records)
    logger.info("Total frames to process: %d", total_frames)

    # Initialize OpenCLIP encoder
    logger.info("Loading OpenCLIP ViT-B-32 (openai pretrained)...")
    encoder = OpenCLIPTextEmbedder(model_name="ViT-B-32", pretrained="openai")
    logger.info("Device: %s | Embedding dim: 512", encoder.device)

    # Check for existing chunks (resume support)
    existing_chunks = sorted(CHUNKS_DIR.glob("chunk_*.npy"))
    start_frame = len(existing_chunks) * CHUNK_SIZE
    if start_frame > 0:
        logger.info("Resuming from %d chunks (%d frames already embedded)", len(existing_chunks), start_frame)

    # Estimate time
    frames_remaining = total_frames - start_frame
    estimated_time = frames_remaining * 0.05 / 3600
    logger.info("Estimated time for remaining: ~%.1fh", estimated_time)

    # Process frames in batches, save in chunks
    t_start = time.time()
    processed = start_frame
    chunk_embeddings = []
    failed_count = 0

    try:
        for batch_start in range(start_frame, total_frames, args.batch_size):
            batch_end = min(batch_start + args.batch_size, total_frames)
            batch_records = records[batch_start:batch_end]

            # Load images in parallel
            image_args = []
            for i, rec in enumerate(batch_records):
                frame_path = rec.get("keyframe_path", "")
                image_args.append((batch_start + i, frame_path))

            images = [None] * len(batch_records)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(load_image, a): a for a in image_args}
                for fut in as_completed(futures):
                    idx, img = fut.result()
                    local_idx = idx - batch_start
                    images[local_idx] = img

            # Encode batch
            valid_images = []
            valid_indices = []
            for i, img in enumerate(images):
                if img is not None:
                    valid_images.append(img)
                    valid_indices.append(batch_start + i)

            if valid_images:
                try:
                    features = encoder.encode_images(valid_images, batch_size=args.batch_size)
                    for feat, idx in zip(features, valid_indices):
                        chunk_embeddings.append((idx, feat))
                except Exception as e:
                    logger.error("Batch encoding failed at frame %d: %s", batch_start, e)
                    failed_count += len(valid_images)

            processed = batch_end

            # Save chunk when we have enough embeddings
            if len(chunk_embeddings) >= CHUNK_SIZE:
                chunk_idx = len(existing_chunks) + (processed // CHUNK_SIZE)
                chunk_path = CHUNKS_DIR / f"chunk_{chunk_idx:04d}.npy"
                chunk_arr = np.array([e[1] for e in chunk_embeddings[:CHUNK_SIZE]], dtype=np.float32)
                np.save(str(chunk_path), chunk_arr)
                chunk_embeddings = chunk_embeddings[CHUNK_SIZE:]
                logger.info("Saved chunk %d (%d frames)", chunk_idx, CHUNK_SIZE)

            # Log progress
            elapsed = time.time() - t_start
            fps = (processed - start_frame) / max(elapsed, 0.01)
            eta = (total_frames - processed) / max(fps, 0.001) if processed < total_frames else 0
            if processed % 100 == 0 or processed == total_frames:
                logger.info(
                    "Progress: %d/%d (%.1f%%) | %.1f fps | ETA: %.1fm",
                    processed, total_frames, 100 * processed / total_frames,
                    fps, eta / 60,
                )

    except KeyboardInterrupt:
        logger.info("Interrupted at frame %d — saving partial chunk...", processed)
        if chunk_embeddings:
            chunk_idx = len(existing_chunks) + (processed // CHUNK_SIZE)
            chunk_path = CHUNKS_DIR / f"chunk_{chunk_idx:04d}.npy"
            chunk_arr = np.array([e[1] for e in chunk_embeddings], dtype=np.float32)
            np.save(str(chunk_path), chunk_arr)
        logger.info("Saved partial chunk. Re-run to resume.")
        return

    # Save any remaining embeddings as final chunk
    if chunk_embeddings:
        chunk_idx = len(existing_chunks) + (processed // CHUNK_SIZE)
        chunk_path = CHUNKS_DIR / f"chunk_{chunk_idx:04d}.npy"
        chunk_arr = np.array([e[1] for e in chunk_embeddings], dtype=np.float32)
        np.save(str(chunk_path), chunk_arr)
        logger.info("Saved final chunk %d (%d frames)", chunk_idx, len(chunk_embeddings))

    elapsed_total = time.time() - t_start
    logger.info("=" * 60)
    logger.info("EMBEDDING COMPLETE")
    logger.info("  Frames: %d (failed: %d)", processed, failed_count)
    logger.info("  Time: %.2fh (%.1f fps)", elapsed_total / 3600, (processed - start_frame) / max(elapsed_total, 0.01))

    # Merge all chunks into final file
    logger.info("Merging chunks into final file...")
    all_chunks = sorted(CHUNKS_DIR.glob("chunk_*.npy"))
    if all_chunks:
        embeddings = np.concatenate([np.load(str(c)) for c in all_chunks], axis=0)
        # Trim to actual frame count
        embeddings = embeddings[:total_frames]
        np.save(str(OPENCLIP_FEATURES_FILE), embeddings)
        logger.info("  Output: %s (%d × %d)", OPENCLIP_FEATURES_FILE, embeddings.shape[0], embeddings.shape[1])

        # Verify norms
        norms = np.linalg.norm(embeddings, axis=1)
        zero_count = np.sum(norms < 0.01)
        logger.info("  Norm range: [%.4f, %.4f] (zero-norm: %d)", norms.min(), norms.max(), zero_count)

    # Save manifest
    with open(OPENCLIP_MANIFEST_FILE, "w", encoding="utf-8") as f:
        for rec in records[:processed]:
            f.write(json.dumps(rec) + "\n")
    logger.info("  Manifest: %s (%d records)", OPENCLIP_MANIFEST_FILE, processed)

    # Save metadata
    meta = {
        "model_name": "ViT-B-32",
        "model_version": "openai",
        "source_hash": compute_source_hash(Path("data/processed/official_manifest.jsonl")),
        "embedding_dim": 512,
        "dtype": "float32",
        "batch_size": args.batch_size,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frame_count": int(processed),
        "source": "official_manifest.jsonl",
        "replaces": "features_siglip2.npy (broken text encoder)",
        "zero_norm_frames": int(zero_count) if 'zero_count' in dir() else -1,
    }
    meta_path = OPENCLIP_OUTPUT_DIR / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("  Metadata: %s", meta_path)

    # Cleanup chunks
    logger.info("Cleaning up chunks...")
    shutil.rmtree(CHUNKS_DIR, ignore_errors=True)

    # Cleanup temp manifest
    if args.limit > 0 and manifest_path != Path("data/processed/official_manifest.jsonl"):
        manifest_path.unlink(missing_ok=True)

    logger.info("=" * 60)
    print(f"\nSUCCESS: {processed} frames embedded in {elapsed_total/3600:.2f}h")


if __name__ == "__main__":
    main()
