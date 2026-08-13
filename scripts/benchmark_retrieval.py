"""Benchmark retrieval backends (FAISS/numpy vs ChromaDB).

Prints a README-worthy markdown table:
    backend | p50 ms | p95 ms | recall@5 | recall@20 | recall@50

Ground truth for Chroma recall is the FAISS/numpy cosine ranking (exact).
Run from the project root:
    python scripts/benchmark_retrieval.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from aic2026.embeddings import OpenCLIPTextEmbedder
from aic2026.ingestion import load_manifest
from aic2026.retrieval import ChromaVectorStore, VectorIndex

DEFAULT_FEATURES = Path("data/processed/derived_features.npy")
DEFAULT_MANIFEST = Path("data/processed/derived_manifest.jsonl")
CHROMA_DIR = Path("data/indexes/chroma")
COLLECTION = "aic2026_frames"


def _load_vectors(features: Path, manifest_path: Path) -> tuple[np.ndarray, list]:
    records = load_manifest(manifest_path)
    return np.load(features), records


def _build_query_set(manifest: list, limit: int = 20) -> list[str]:
    """Sample queries from manifest titles/descriptions, fall back to prompts."""
    prompts = [
        "Một người đang nói chuyện trước khán giả",
        "Diễn giả mặc áo đỏ phát biểu tại họp báo ngoài trời",
        "Người đi bộ trong cửa hàng",
        "Lễ trao giải thưởng âm nhạc",
        "Vận động viên thực hiện cú nhảy cao",
    ]
    collected = list(prompts)
    for record in manifest:
        if record.title and len(collected) < limit:
            collected.append(record.title)
        if record.description and len(collected) < limit:
            collected.append(record.description)
    return collected[:limit]


def _recall_at_k(gt: set[int], got: set[int], k: int) -> float:
    if not gt:
        return 0.0
    return len(gt & got) / min(len(gt), k)


def _bench_backend(backend, vectors, queries, k: int, runs: int, chroma_dir: Path) -> dict:
    manifest = queries["manifest"]
    if backend == "chroma":
        store = ChromaVectorStore.from_manifest(manifest, persist_dir=chroma_dir, collection_name=COLLECTION)
    else:
        store = VectorIndex(vectors)

    # Exact ground-truth index: build once outside the loop instead of inside
    # every query × run (previously it re-created FAISS + re-normalized each time).
    exact_index = VectorIndex(vectors)

    latencies: list[float] = []
    recalls = {k_sel: [] for k_sel in (5, 20, 50)}

    for _ in range(runs):
        for q in queries["texts"]:
            emb = queries["encode"](q)
            start = time.perf_counter()
            ids, _ = store.search(emb, k)
            latencies.append((time.perf_counter() - start) * 1000.0)

            # Ground truth: exact cosine on the FAISS/numpy index.
            exact_ids, _ = exact_index.search(emb, k)
            gt_set = set(exact_ids[:50].tolist())
            got = set(ids.tolist())
            for k_sel, bucket in recalls.items():
                bucket.append(_recall_at_k(gt_set, got, k_sel))

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    return {
        "backend": backend,
        "p50_ms": p50,
        "p95_ms": p95,
        "recall5": float(np.mean(recalls[5])),
        "recall20": float(np.mean(recalls[20])),
        "recall50": float(np.mean(recalls[50])),
    }


def main(
    features: Path = DEFAULT_FEATURES,
    manifest_path: Path = DEFAULT_MANIFEST,
    limit: int = 20,
    k: int = 100,
    runs: int = 5,
) -> None:
    vectors, manifest = _load_vectors(features, manifest_path)
    texts = _build_query_set(manifest, limit)

    try:
        encoder = OpenCLIPTextEmbedder()
        encode = encoder.encode
    except RuntimeError:
        # Fallback for machines without OpenCLIP: use a stable bag-of-words so
        # the script still runs and reports relative latency/recall.
        def encode(text: str) -> np.ndarray:
            vec = np.zeros(vectors.shape[1], dtype=np.float32)
            for token in text.lower().split():
                vec[hash(token) % vectors.shape[1]] += 1
            return vec

    queries = {"texts": texts, "manifest": manifest, "encode": encode}

    print(f"Vectors: {vectors.shape} | queries: {len(texts)} | k={k} | runs={runs}\n")
    header = "| backend | p50 ms | p95 ms | recall@5 | recall@20 | recall@50 |"
    print(header)
    print("|---|---|---|---|---|---|")

    results = []
    for backend in ("faiss", "chroma"):
        if backend == "chroma" and not ChromaVectorStore.available():
            print("| chroma | — (chưa cài chromadb) | | | | |")
            continue
        try:
            result = _bench_backend(backend, vectors, queries, k, runs, CHROMA_DIR)
        except Exception as exc:  # noqa: BLE001
            print(f"| {backend} | error: {exc} | | | | |")
            continue
        results.append(result)
        print(
            f"| {result['backend']} | {result['p50_ms']:.1f} | {result['p95_ms']:.1f} "
            f"| {result['recall5']:.3f} | {result['recall20']:.3f} | {result['recall50']:.3f} |"
        )

    out = Path("outputs/benchmark_retrieval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
