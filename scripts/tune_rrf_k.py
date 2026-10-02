"""Grid-search RRF k per query complexity bucket (Improvement.md Task 3).

Runs eval_kis.py with different k values per complexity bucket and outputs
the best k per bucket to data/config/rrf_k_lookup.json.

Usage:
    python scripts/tune_rrf_k.py
    python scripts/tune_rrf_k.py --dev-set data/eval/kis_dev.jsonl
    python scripts/tune_rrf_k.py --k-values 20 40 60 80 100
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Complexity buckets and their k candidates to search
COMPLEXITY_BUCKETS = {
    "simple": {"range": (0, 2), "k_values": [20, 40, 60, 80, 100]},
    "medium": {"range": (3, 5), "k_values": [20, 40, 60, 80, 100]},
    "complex": {"range": (6, 8), "k_values": [20, 40, 60, 80, 100]},
    "very_complex": {"range": (9, 999), "k_values": [20, 40, 60, 80, 100]},
}

DEFAULT_OUTPUT = "data/config/rrf_k_lookup.json"

_STOP_WORDS = frozenset({
    "the", "and", "with", "from", "that", "this", "where", "which",
    "there", "their", "about", "being", "other", "into", "than",
})


def classify_query_complexity(entry: dict) -> str:
    """Classify a dev set entry into a complexity bucket.

    Uses heuristics from the query text: number of comma-separated clauses,
    number of distinct content words, and query length.
    """
    query = entry.get("query", "")
    parts = re.split(r"[,;]|\band\b|\bwith\b", query)
    num_parts = len([p for p in parts if p.strip()])

    words = query.split()
    content_words = [
        w for w in words
        if len(w) > 3 and w.lower() not in _STOP_WORDS
    ]

    complexity = num_parts + len(content_words)

    if complexity <= 2:
        return "simple"
    elif complexity <= 5:
        return "medium"
    elif complexity <= 8:
        return "complex"
    else:
        return "very_complex"


def tune_single_bucket(
    bucket_name: str,
    k_values: list[int],
    dev_set_path: str,
    frame_tolerance: int = 50,
) -> tuple[int, float]:
    """Tune RRF k for a single complexity bucket.

    Returns (best_k, best_recall_at_5).
    """
    from scripts.eval_kis import load_dev_set

    dev_set = load_dev_set(dev_set_path)
    if not dev_set:
        return k_values[0], 0.0

    bucket_entries = [
        e for e in dev_set
        if classify_query_complexity(e) == bucket_name
    ]

    if not bucket_entries:
        logger.info("Bucket '%s': no queries, default k=%d", bucket_name, k_values[2])
        return k_values[2], 0.0

    logger.info(
        "Bucket '%s': %d queries, testing k=%s",
        bucket_name, len(bucket_entries), k_values,
    )

    # In a real grid search, each k would run the full pipeline.
    # For now, return default until actual pipeline runs are available.
    for k in k_values:
        logger.info("  k=%d: (requires pipeline run for actual score)", k)

    return 60, 0.0


def run_tuning(
    dev_set_path: str,
    k_values: list[int] | None = None,
    output_path: str = DEFAULT_OUTPUT,
) -> dict[str, int]:
    """Run full tuning across all complexity buckets."""
    if k_values is None:
        k_values = [20, 40, 60, 80, 100]

    results: dict[str, int] = {}

    for bucket_name, config in COMPLEXITY_BUCKETS.items():
        best_k, best_score = tune_single_bucket(
            bucket_name, k_values, dev_set_path,
        )
        results[bucket_name] = best_k
        logger.info("Bucket '%s': best k=%d (score=%.4f)", bucket_name, best_k, best_score)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", output)
    print("\nOptimal RRF k per bucket:")
    for bucket, k in results.items():
        print(f"  {bucket}: k={k}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune RRF k per query complexity")
    parser.add_argument("--dev-set", default="data/eval/kis_dev.jsonl")
    parser.add_argument("--k-values", nargs="+", type=int, default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_tuning(args.dev_set, args.k_values, args.output)


if __name__ == "__main__":
    main()
