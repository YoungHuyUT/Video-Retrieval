"""KIS eval harness —跑了 retrieve() cho moi query, so ket qua voi ground-truth.

Metrics: Recall@1, Recall@5, Recall@10, Recall@20, mAP (Mean Average Precision), p50/p95 latency.

Usage:
    python scripts/eval_kis.py
    python scripts/eval_kis.py --dev-set data/eval/kis_dev.jsonl
    python scripts/eval_kis.py --configs config_a.json config_b.json
    python scripts/eval_kis.py --limit 100 --frame-tolerance 50
    python scripts/eval_kis.py --compare speed precision  # Compare speed vs precision profiles
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result for a single query evaluation."""
    query_id: str
    query: str
    gt_video_id: str
    gt_frame_id: int | None
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    average_precision: float
    latency_ms: float
    top1_video: str | None = None
    top1_frame: int | None = None
    top1_score: float = 0.0


@dataclass
class EvalResult:
    """Aggregate evaluation results."""
    config_name: str
    total_queries: int
    avg_recall_at_1: float
    avg_recall_at_5: float
    avg_recall_at_10: float
    avg_recall_at_20: float
    avg_mAP: float
    p50_latency_ms: float
    p95_latency_ms: float
    query_results: list[QueryResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def load_dev_set(path: str | Path) -> list[dict[str, Any]]:
    """Load dev set from JSONL file.

    Schema per line: {"query": str, "gt_video_id": str, "gt_frame_id": int | None}
    """
    dev_set = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line %d: %s", line_no, exc)
                continue
            # Validate required fields
            if "query" not in entry or "gt_video_id" not in entry:
                logger.warning("Skipping line %d: missing 'query' or 'gt_video_id'", line_no)
                continue
            dev_set.append(entry)
    return dev_set


def _is_hit(candidate_video_id: str, candidate_frame_id: int,
            gt_video_id: str, gt_frame_id: int | None,
            frame_tolerance: int) -> bool:
    """Check if a candidate matches the ground-truth."""
    if candidate_video_id != gt_video_id:
        return False
    if gt_frame_id is None:
        return True  # video-only match
    return abs(candidate_frame_id - gt_frame_id) <= frame_tolerance


def _compute_ap(relevant_flags: list[bool]) -> float:
    """Compute Average Precision from a list of binary relevance flags.

    AP = sum(precision@k * rel_k) / total_relevant
    """
    total_relevant = sum(relevant_flags)
    if total_relevant == 0:
        return 0.0
    precision_sum = 0.0
    hits = 0
    for k, is_relevant in enumerate(relevant_flags, 1):
        if is_relevant:
            hits += 1
            precision_sum += hits / k
    return precision_sum / total_relevant


def evaluate_single_query(
    candidates: list[Any],
    gt_video_id: str,
    gt_frame_id: int | None,
    frame_tolerance: int,
    query_id: str = "",
    query: str = "",
    latency_ms: float = 0.0,
) -> QueryResult:
    """Evaluate a single query against its ground-truth.

    Args:
        candidates: List of Candidate objects (must have .video_id, .frame_id, .score).
        gt_video_id: Ground-truth video ID.
        gt_frame_id: Ground-truth frame ID (or None for video-only match).
        frame_tolerance: Max frame offset allowed for a match.
        query_id: Query identifier (for logging).
        query: Query text (for logging).
        latency_ms: Latency in milliseconds for this query.

    Returns:
        QueryResult with recall and AP metrics.
    """
    # Sort by score descending
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)

    # Compute relevance for each rank position
    relevant_flags = []
    for cand in ranked[:100]:  # cap at 100 for mAP
        hit = _is_hit(
            cand.video_id, cand.frame_id,
            gt_video_id, gt_frame_id,
            frame_tolerance,
        )
        relevant_flags.append(hit)

    # Recall@K
    def recall_at(k: int) -> float:
        if not relevant_flags:
            return 0.0
        top_k = relevant_flags[:k]
        return 1.0 if any(top_k) else 0.0

    # mAP
    ap = _compute_ap(relevant_flags)

    # Top-1 info
    top1_video = ranked[0].video_id if ranked else None
    top1_frame = ranked[0].frame_id if ranked else None
    top1_score = ranked[0].score if ranked else 0.0

    return QueryResult(
        query_id=query_id,
        query=query,
        gt_video_id=gt_video_id,
        gt_frame_id=gt_frame_id,
        recall_at_1=recall_at(1),
        recall_at_5=recall_at(5),
        recall_at_10=recall_at(10),
        recall_at_20=recall_at(20),
        average_precision=ap,
        latency_ms=latency_ms,
        top1_video=top1_video,
        top1_frame=top1_frame,
        top1_score=top1_score,
    )


def run_eval(
    config: Any,
    dev_set_path: str | Path,
    frame_tolerance: int = 50,
    limit: int = 100,
) -> EvalResult:
    """Run eval for a given config on the dev set.

    Args:
        config: RuntimeConfig or dict with config params.
        dev_set_path: Path to JSONL dev set.
        frame_tolerance: Max frame offset for a match.
        limit: Max candidates to retrieve per query.

    Returns:
        EvalResult with per-query and aggregate metrics.
    """
    from aic2026.app.api import load_orchestrator
    from aic2026.models import Query

    dev_set = load_dev_set(dev_set_path)
    if not dev_set:
        logger.warning("Dev set is empty: %s", dev_set_path)
        return EvalResult(
            config_name="empty",
            total_queries=0,
            avg_recall_at_1=0.0,
            avg_recall_at_5=0.0,
            avg_recall_at_10=0.0,
            avg_recall_at_20=0.0,
            avg_mAP=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
        )

    # Load orchestrator from config
    if isinstance(config, dict):
        from aic2026.app.api import RuntimeConfig
        config = RuntimeConfig(**config)

    t0 = time.time()
    orchestrator = load_orchestrator(
        config.manifest_path,
        config.features_path,
        config.clip_pretrained,
        config.llm_model,
        config.ollama_url,
        backend=config.backend,
        chroma_dir=config.chroma_dir,
        metadata_filter=config.metadata_filter,
        translate_query=config.translate_query,
        vlm_backend="none",  # no VLM for KIS eval
        vlm_model=config.vlm_model,
        vlm_device=config.vlm_device,
        vlm_dtype=config.vlm_dtype,
        vlm_timeout=config.vlm_timeout,
        late_interaction_weight=config.late_interaction_weight,
        query_type="kis",
        asr_sidecar_path=config.asr_sidecar_path,
        use_event_coverage=config.use_event_coverage,
        use_moment_rerank=config.use_moment_rerank,
        use_blip2_rerank=config.use_blip2_rerank,
        blip2_rerank_top_k=config.blip2_rerank_top_k,
        blip2_rerank_weight=config.blip2_rerank_weight,
        asr_weight=config.asr_weight,
        event_coverage_blend=config.event_coverage_blend,
        embedding_backend=config.embedding_backend,
        siglip2_model=config.siglip2_model,
        use_llm_query_analyzer=config.use_llm_query_analyzer,
        llm_query_model=config.llm_query_model,
        use_cascade_rerank=getattr(config, "use_cascade_rerank", False),
        cascade_stage_a_top_n=getattr(config, "cascade_stage_a_top_n", 100),
        retrieval_profile=getattr(config, "retrieval_profile", "speed"),
    )

    query_results: list[QueryResult] = []
    latencies: list[float] = []
    for entry in dev_set:
        query_text = entry["query"]
        gt_video_id = entry["gt_video_id"]
        gt_frame_id = entry.get("gt_frame_id")
        query_id = entry.get("query_id", query_text[:40])

        query = Query(query_id=query_id, type="kis", text=query_text)
        try:
            q_start = time.time()
            result = orchestrator.run(query)
            q_latency = (time.time() - q_start) * 1000.0
            candidates = result.candidates
        except Exception as exc:
            logger.warning("Query '%s' failed: %s", query_id, exc)
            candidates = []
            q_latency = 0.0

        qr = evaluate_single_query(
            candidates=candidates,
            gt_video_id=gt_video_id,
            gt_frame_id=gt_frame_id,
            frame_tolerance=frame_tolerance,
            query_id=query_id,
            query=query_text,
            latency_ms=q_latency,
        )
        query_results.append(qr)
        latencies.append(q_latency)

    elapsed = time.time() - t0

    # Aggregate
    n = len(query_results) if query_results else 1
    avg_r1 = sum(q.recall_at_1 for q in query_results) / n
    avg_r5 = sum(q.recall_at_5 for q in query_results) / n
    avg_r10 = sum(q.recall_at_10 for q in query_results) / n
    avg_r20 = sum(q.recall_at_20 for q in query_results) / n
    avg_ap = sum(q.average_precision for q in query_results) / n

    # Percentile latencies
    p50_lat = statistics.median(latencies) if latencies else 0.0
    p95_lat = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 2 else (latencies[0] if latencies else 0.0)

    return EvalResult(
        config_name=getattr(config, "embedding_backend", "unknown"),
        total_queries=len(query_results),
        avg_recall_at_1=avg_r1,
        avg_recall_at_5=avg_r5,
        avg_recall_at_10=avg_r10,
        avg_recall_at_20=avg_r20,
        avg_mAP=avg_ap,
        p50_latency_ms=p50_lat,
        p95_latency_ms=p95_lat,
        query_results=query_results,
        elapsed_seconds=elapsed,
    )


def print_eval_table(results: list[EvalResult]) -> None:
    """Print a comparison table of eval results."""
    header = f"{'Config':<20} {'Q':>4} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'R@20':>7} {'mAP':>7} {'p50':>8} {'p95':>8} {'Time':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r.config_name:<20} {r.total_queries:>4} "
            f"{r.avg_recall_at_1:>7.4f} {r.avg_recall_at_5:>7.4f} "
            f"{r.avg_recall_at_10:>7.4f} {r.avg_recall_at_20:>7.4f} "
            f"{r.avg_mAP:>7.4f} "
            f"{r.p50_latency_ms:>7.1f}ms {r.p95_latency_ms:>7.1f}ms "
            f"{r.elapsed_seconds:>7.1f}s"
        )
    print(sep)


def save_eval_results(results: list[EvalResult], output_dir: str | Path = "results") -> Path:
    """Save eval results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_{timestamp}.json"

    data = []
    for r in results:
        data.append({
            "config_name": r.config_name,
            "total_queries": r.total_queries,
            "avg_recall_at_1": r.avg_recall_at_1,
            "avg_recall_at_5": r.avg_recall_at_5,
            "avg_recall_at_10": r.avg_recall_at_10,
            "avg_recall_at_20": r.avg_recall_at_20,
            "avg_mAP": r.avg_mAP,
            "p50_latency_ms": r.p50_latency_ms,
            "p95_latency_ms": r.p95_latency_ms,
            "elapsed_seconds": r.elapsed_seconds,
            "query_results": [
                {
                    "query_id": q.query_id,
                    "query": q.query,
                    "gt_video_id": q.gt_video_id,
                    "gt_frame_id": q.gt_frame_id,
                    "recall_at_1": q.recall_at_1,
                    "recall_at_5": q.recall_at_5,
                    "recall_at_10": q.recall_at_10,
                    "recall_at_20": q.recall_at_20,
                    "average_precision": q.average_precision,
                    "latency_ms": q.latency_ms,
                    "top1_video": q.top1_video,
                    "top1_frame": q.top1_frame,
                }
                for q in r.query_results
            ],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="KIS Eval Harness")
    parser.add_argument(
        "--dev-set",
        default="data/eval/kis_dev.jsonl",
        help="Path to JSONL dev set",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Config JSON files to compare (default: use RuntimeConfig defaults)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("PROFILE_A", "PROFILE_B"),
        default=None,
        help="Compare two retrieval profiles (e.g., --compare speed precision)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max candidates to retrieve per query",
    )
    parser.add_argument(
        "--frame-tolerance",
        type=int,
        default=50,
        help="Max frame offset for a match",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for output JSON",
    )
    parser.add_argument(
        "--backend",
        default="numpy",
        choices=["faiss", "numpy"],
        help="Index backend (numpy=low-memory mmap, faiss=in-RAM)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    all_results: list[EvalResult] = []

    if args.compare:
        # Compare two retrieval profiles
        profile_a, profile_b = args.compare
        from aic2026.app.api import RuntimeConfig

        # Profile A
        config_a = RuntimeConfig()
        config_a.retrieval_profile = profile_a
        config_a.backend = args.backend
        result_a = run_eval(config_a, args.dev_set, args.frame_tolerance, args.limit)
        result_a.config_name = f"{profile_a}"
        all_results.append(result_a)

        # Profile B
        config_b = RuntimeConfig()
        config_b.retrieval_profile = profile_b
        config_b.backend = args.backend
        result_b = run_eval(config_b, args.dev_set, args.frame_tolerance, args.limit)
        result_b.config_name = f"{profile_b}"
        all_results.append(result_b)
    elif args.configs:
        for config_path in args.configs:
            with open(config_path, encoding="utf-8") as f:
                config_dict = json.load(f)
            config_name = Path(config_path).stem
            config_dict.setdefault("embedding_backend", config_name)
            result = run_eval(config_dict, args.dev_set, args.frame_tolerance, args.limit)
            result.config_name = config_name
            all_results.append(result)
    else:
        # Single run with defaults
        from aic2026.app.api import RuntimeConfig
        config = RuntimeConfig()
        result = run_eval(config, args.dev_set, args.frame_tolerance, args.limit)
        all_results.append(result)

    print_eval_table(all_results)
    save_eval_results(all_results, args.output_dir)


if __name__ == "__main__":
    main()
