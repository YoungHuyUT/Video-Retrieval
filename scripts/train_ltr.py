"""Train LTR ranker from dev set (Improvement.md Task 7)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LTR ranker")
    parser.add_argument("--dev-set", default="data/eval/kis_dev.jsonl")
    parser.add_argument("--output", default="models/ltr_ranker.joblib")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from scripts.eval_kis import load_dev_set
    from aic2026.app.api import RuntimeConfig, load_orchestrator
    from aic2026.models import Query
    from aic2026.reranking.ltr_ranker import FEATURE_NAMES

    dev_set = load_dev_set(args.dev_set)
    if not dev_set:
        logger.error("Dev set empty: %s", args.dev_set)
        return

    config = RuntimeConfig()
    orchestrator = load_orchestrator(
        config.manifest_path, config.features_path, config.clip_pretrained,
        config.llm_model, config.ollama_url,
        backend=config.backend, embedding_backend=config.embedding_backend,
        siglip2_model=config.siglip2_model, vlm_backend="none",
        use_blip2_rerank=config.use_blip2_rerank,
        blip2_rerank_top_k=config.blip2_rerank_top_k,
        blip2_rerank_weight=config.blip2_rerank_weight,
        use_cascade_rerank=config.use_cascade_rerank,
        cascade_stage_a_top_n=config.cascade_stage_a_top_n,
    )

    all_features, all_labels = [], []
    for entry in dev_set:
        query = Query(query_id=entry.get("query_id", ""), type="kis", text=entry["query"])
        try:
            result = orchestrator.run(query)
            candidates = result.candidates
        except Exception as exc:
            logger.warning("Query '%s' failed: %s", entry["query"][:40], exc)
            continue
        gt_video, gt_frame = entry["gt_video_id"], entry.get("gt_frame_id")
        for cand in candidates[:50]:
            label = 1 if (cand.video_id == gt_video and (gt_frame is None or abs(cand.frame_id - gt_frame) <= 50)) else 0
            all_features.append([getattr(cand, a, 0.0) for a in ["_object_score", "_colour_score", "_blip2_score", "_late_interaction_score", "_event_coverage_score", "_asr_score"]])
            all_labels.append(label)

    if not all_features:
        logger.error("No training data")
        return

    X, y = np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.int32)
    logger.info("Data: %d samples, %d pos", len(y), sum(y))

    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    model.fit(X, y)

    for name, imp in zip(FEATURE_NAMES, model.feature_importances_):
        logger.info("  %s: %.4f", name, imp)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, output)
    logger.info("Saved to %s", output)


if __name__ == "__main__":
    main()
