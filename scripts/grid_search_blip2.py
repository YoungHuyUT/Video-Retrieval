#!/usr/bin/env python3
"""Grid search for BLIP-2 reranker hyperparameters.

Searches over:
- blip2_rerank_top_k: {20, 30, 40, 50, 70}
- use_llm_verifier: {True, False}
- use_event_coverage: {True, False}
- late_interaction_weight: {0.0, 0.1, 0.3}

Outputs: Recall@10, Recall@20, p50, p95 latency for each combo.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GridSearchResult:
    """Result for a single grid point."""
    config: dict
    recall_at_10: float
    recall_at_20: float
    p50_latency_ms: float
    p95_latency_ms: float
    elapsed_seconds: float


def run_single_config(
    config_dict: dict,
    dev_set_path: str | Path,
    frame_tolerance: int = 50,
    limit: int = 100,
) -> GridSearchResult:
    """Run evaluation for a single config."""
    from aic2026.app.api import RuntimeConfig
    from aic2026.scripts.eval_kis import run_eval

    config = RuntimeConfig(**config_dict)
    start = time.time()
    result = run_eval(config, dev_set_path, frame_tolerance, limit)
    elapsed = time.time() - start

    return GridSearchResult(
        config=config_dict,
        recall_at_10=result.avg_recall_at_10,
        recall_at_20=result.avg_recall_at_20,
        p50_latency_ms=result.p50_latency_ms,
        p95_latency_ms=result.p95_latency_ms,
        elapsed_seconds=elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="BLIP-2 Hyperparameter Grid Search")
    parser.add_argument(
        "--dev-set",
        default="data/eval/kis_dev.jsonl",
        help="Path to JSONL dev set",
    )
    parser.add_argument(
        "--frame-tolerance",
        type=int,
        default=50,
        help="Max frame offset for a match",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max candidates to retrieve per query",
    )
    parser.add_argument(
        "--output",
        default="results/grid_search_blip2.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--profile",
        choices=["speed", "precision"],
        default="speed",
        help="Base retrieval profile",
    )
    parser.add_argument(
        "--param",
        choices=["all", "blip2_top_k", "llm_verifier", "event_coverage", "late_interaction"],
        default="all",
        help="Which parameter group to sweep",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Base config from profile
    from aic2026.app.api import RuntimeConfig
    base_config = RuntimeConfig()
    base_config.retrieval_profile = args.profile

    # Build param grid
    param_grid = {}

    if args.param in ("all", "blip2_top_k"):
        param_grid["blip2_rerank_top_k"] = [20, 30, 40, 50, 70]

    if args.param in ("all", "llm_verifier"):
        param_grid["use_llm_verifier"] = [False, True]

    if args.param in ("all", "event_coverage"):
        param_grid["use_event_coverage"] = [False, True]

    if args.param in ("all", "late_interaction"):
        param_grid["late_interaction_weight"] = [0.0, 0.1, 0.3]

    # If no params selected, just run base config
    if not param_grid:
        logger.info("No params to sweep, running base config...")
        result = run_single_config(base_config.__dict__, args.dev_set, args.frame_tolerance, args.limit)
        results = [result]
    else:
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        logger.info(f"Running grid search over {len(combinations)} combinations...")
        logger.info(f"Parameters: {keys}")

        results = []
        for i, combo in enumerate(combinations, 1):
            config_dict = base_config.__dict__.copy()
            for k, v in zip(keys, combo):
                config_dict[k] = v

            logger.info(f"[{i}/{len(combinations)}] Testing: {dict(zip(keys, combo))}")
            try:
                result = run_single_config(config_dict, args.dev_set, args.frame_tolerance, args.limit)
                results.append(result)
                logger.info(
                    f"  -> R@10={result.recall_at_10:.4f}, R@20={result.recall_at_20:.4f}, "
                    f"p50={result.p50_latency_ms:.1f}ms, p95={result.p95_latency_ms:.1f}ms"
                )
            except Exception as exc:
                logger.error(f"  -> FAILED: {exc}")

    # Sort by Recall@20 descending, then p95 ascending
    results.sort(key=lambda r: (-r.recall_at_20, r.p95_latency_ms))

    # Print summary table
    print("\n" + "=" * 100)
    header = f"{'Rank':>4} {'blip2_k':>7} {'llm_vfy':>7} {'ev_cov':>7} {'late_w':>7} {'R@10':>8} {'R@20':>8} {'p50(ms)':>10} {'p95(ms)':>10} {'Time':>8}"
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(results, 1):
        cfg = r.config
        print(
            f"{rank:>4} "
            f"{cfg.get('blip2_rerank_top_k', 'N/A'):>7} "
            f"{str(cfg.get('use_llm_verifier', False))[0]:>7} "
            f"{str(cfg.get('use_event_coverage', False))[0]:>7} "
            f"{cfg.get('late_interaction_weight', 0.0):>7.1f} "
            f"{r.recall_at_10:>8.4f} {r.recall_at_20:>8.4f} "
            f"{r.p50_latency_ms:>10.1f} {r.p95_latency_ms:>10.1f} "
            f"{r.elapsed_seconds:>7.1f}s"
        )
    print("=" * 100)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for r in results:
        data.append({
            "config": r.config,
            "recall_at_10": r.recall_at_10,
            "recall_at_20": r.recall_at_20,
            "p50_latency_ms": r.p50_latency_ms,
            "p95_latency_ms": r.p95_latency_ms,
            "elapsed_seconds": r.elapsed_seconds,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print best config
    if results:
        best = results[0]
        print(f"\nBest config (by R@20 then p95):")
        print(f"  blip2_rerank_top_k: {best.config.get('blip2_rerank_top_k')}")
        print(f"  use_llm_verifier: {best.config.get('use_llm_verifier')}")
        print(f"  use_event_coverage: {best.config.get('use_event_coverage')}")
        print(f"  late_interaction_weight: {best.config.get('late_interaction_weight')}")
        print(f"  Recall@10: {best.recall_at_10:.4f}")
        print(f"  Recall@20: {best.recall_at_20:.4f}")
        print(f"  p50 latency: {best.p50_latency_ms:.1f}ms")
        print(f"  p95 latency: {best.p95_latency_ms:.1f}ms")


if __name__ == "__main__":
    main()