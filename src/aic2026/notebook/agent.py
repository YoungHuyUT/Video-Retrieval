"""NOTEBOOK agent — Query → Plan → Retrieve → Evidence → Rank (spec §11).

This is the main orchestrator for the NOTEBOOK/VIDEO SEARCH mode.
It is a SEPARATE module from the KIS/QA/TRAKE RetrievalAgent — it does NOT
route through load_orchestrator / RetrievalAgent per spec §17.

Pipeline:
    raw query
    → NotebookPlan (deterministic rule-based keyword plan)
    → ASR BM25 retrieval (per-segment evidence)
    → Evidence aggregation (event coverage + temporal + modality)
    → Scoring & ranking (Top-N video IDs)
    → NotebookResult

Default: no model load and no LLM call.  NOTEBOOK is a lexical ASR search.

Spec §17: NOTEBOOK is a separate module that reuses existing tools but does NOT
go through load_orchestrator/RetrievalAgent.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from aic2026.notebook.types import NotebookResult, NotebookPlan, NotebookCandidate
from aic2026.notebook.notebook_planner import plan_notebook
from aic2026.notebook.asr_retriever import ASRRetriever, fuse_bm25_dense
from aic2026.notebook.dense_retriever import DenseRetriever
from aic2026.notebook.evidence import aggregate_evidence
from aic2026.notebook.scorer import score_candidates
from aic2026.notebook.ranker import rank_videos, maybe_second_pass

logger = logging.getLogger(__name__)


class NotebookAgent:
    """Orchestrates the NOTEBOOK/VIDEO SEARCH retrieval pipeline.

    Separate from RetrievalAgent (spec §17). Uses a local LLM planner
    (qwen2.5:1.5b, no-think) to decompose queries, then ASR BM25 +
    temporal/event scoring to rank videos.

    Attributes:
        asr_sidecar_path: Path to asr_sidecar.jsonl (Vietnamese transcripts)
        ollama_url / llm_model: retained only for API compatibility; NOTEBOOK
            lexical mode never invokes them.
    """

    def __init__(
        self,
        asr_sidecar_path: str,
        ollama_url: str = "http://127.0.0.1:11434",
        llm_model: str = "qwen2.5:1.5b",
    ) -> None:
        self.asr_sidecar_path = asr_sidecar_path
        self.ollama_url = ollama_url
        self.llm_model = llm_model
        self._asr_retriever: ASRRetriever | None = None
        # A NotebookAgent is cached by FastAPI.  Track the loaded sidecar so a
        # newly merged ASR batch becomes visible without a server restart.
        self._asr_sidecar_signature: tuple[int, int] | None = None
        self._dense_retriever: DenseRetriever | None = None

    def _get_retriever(self) -> ASRRetriever:
        """Lazily load the ASR retriever (builds BM25 on first use, ~190s)."""
        if self._asr_retriever is None:
            from pathlib import Path as _P
            # Resolve sidecar path: try self.asr_sidecar_path first, then
            # fallback to project root data/processed/asr_sidecar.jsonl.
            # NOTE: Path("") resolves to "." on Windows and .exists() returns True,
            # so we must check for empty/blank string explicitly BEFORE Path().
            sidecar_path = (self.asr_sidecar_path or "").strip()
            sidecar = _P(sidecar_path) if sidecar_path else _P()

            # Always try project-root fallback first (most reliable)
            proj_root = _P(__file__).resolve().parents[3]
            candidate = proj_root / "data" / "processed" / "asr_sidecar.jsonl"
            if candidate.is_file():
                sidecar = candidate
            elif sidecar_path and sidecar.is_file():
                pass  # use the user-provided path
            else:
                logger.warning(
                    "NotebookAgent: ASR sidecar not found (path=%s, candidate=%s) "
                    "— ASR retrieval will be disabled",
                    self.asr_sidecar_path, candidate,
                )
                self._asr_retriever = ASRRetriever.empty()
                return self._asr_retriever

            self.asr_sidecar_path = str(sidecar)
            logger.info("NotebookAgent: loading ASR sidecar from %s", sidecar)
            self._asr_retriever = ASRRetriever.from_sidecar(str(sidecar))
            stat = sidecar.stat()
            self._asr_sidecar_signature = (stat.st_mtime_ns, stat.st_size)
        else:
            sidecar = Path(self.asr_sidecar_path)
            if sidecar.is_file():
                stat = sidecar.stat()
                signature = (stat.st_mtime_ns, stat.st_size)
                if signature != self._asr_sidecar_signature:
                    logger.info("NotebookAgent: ASR sidecar changed; rebuilding BM25 index")
                    self._asr_retriever = ASRRetriever.from_sidecar(str(sidecar))
                    self._asr_sidecar_signature = signature
                    # Dense embeddings are ordered by ASR segments, so any old
                    # instance is invalid after a sidecar merge.
                    self._dense_retriever = None
        return self._asr_retriever

    def _get_dense_retriever(self) -> DenseRetriever:
        """Lazily load the Dense retriever (loads precomputed embeddings)."""
        if self._dense_retriever is None:
            self._dense_retriever = DenseRetriever(self.asr_sidecar_path)
        return self._dense_retriever

    def run(
        self,
        text: str,
        asr_query: str | None = None,
        top_n: int = 5,
        use_visual: bool = False,
    ) -> NotebookResult:
        """Run the full NOTEBOOK pipeline on a query.

        Pipeline:
        1. PLAN: LLM decomposes query → NotebookPlan (1 LLM call)
        2. RETRIEVE: ASR BM25 search for each asr_query
        3. EVIDENCE: Group segments by video, compute coverage/temporal
        4. SCORE: Normalise + score each video
        5. RANK: Sort by score, apply Top-N, check confidence gap

        Args:
            text: Raw query text
            asr_query: Optional Vietnamese query for ASR matching
            top_n: Number of top videos to return
            use_visual: Whether to attempt visual retrieval (off by default)

        Returns:
            NotebookResult with candidates and timing
        """
        total_start = time.time()
        llm_latency = 0.0
        second_pass_triggered = False
        error: str | None = None

        # --- Step 1: deterministic keyword plan ---
        # NOTEBOOK is intentionally lexical-only. Passing use_llm=False is
        # essential: the planner otherwise defaults to Ollama/Qwen and makes a
        # simple keyword query wait for model loading.
        # SPEED OPT: Reduce LLM timeout from 120s to 15s, skip second LLM pass entirely
        plan_start = time.time()
        try:
            plan = plan_notebook(
                text,
                asr_query=asr_query,
                use_llm=False,
                model=self.llm_model,
                base_url=self.ollama_url,
                timeout_seconds=15.0,  # Reduced from default 120s
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("NotebookAgent: keyword planner failed")
            error = f"Keyword planner error: {exc}"
            plan = self._fallback_plan(text, asr_query)
        llm_latency = time.time() - plan_start

        logger.info(
            "NotebookAgent: lexical plan done — %d events, %d asr_queries, %d visual_clues, "
            "ASR weight=%.1f",
            len(plan.events),
            len(plan.asr_queries),
            len(plan.visual_clues),
            plan.modality_weights.get("asr", 0.0),
        )

        # --- Step 2: RETRIEVE ASR evidence (BM25 + optional dense fuse + PRF) ---
        retriever = self._get_retriever()
        # SPEED OPT: Disable PRF by default (saves ~1-2s), reduce top_k_per_query from 20 to 10
        use_prf = False  # plan.modality_weights.get("asr", 1.0) >= 0.5
        bm25_matches = retriever.retrieve(plan, top_k_per_query=10, use_prf=use_prf)

        # NOTEBOOK is reserved for ASR keyword evidence only. Visual retrieval
        # stays OFF by default because the user explicitly wants NOTEBOOK to be
        # keyword search over ASR transcripts rather than a visual frame route.
        if use_visual:
            dense_retriever = self._get_dense_retriever()
            if dense_retriever.is_available:
                dense_matches = dense_retriever.retrieve(plan, top_k_per_query=20)
                matches = fuse_bm25_dense(bm25_matches, dense_matches, dynamic_k=True)
                logger.info(
                    "NotebookAgent: fused %d BM25 + %d dense → %d matches (dynamic RRF)",
                    len(bm25_matches), len(dense_matches), len(matches),
                )
            else:
                matches = bm25_matches
        else:
            matches = bm25_matches
            logger.info(
                "NotebookAgent: visual retrieval disabled — returning ASR-only BM25 evidence",
            )

        # --- Step 3: EVIDENCE aggregation ---
        evidence_by_video = aggregate_evidence(matches, plan)

        # --- Step 4: SCORE & RANK ---
        rank_start = time.time()
        candidates = rank_videos(evidence_by_video, plan, top_n=top_n)
        rank_latency = time.time() - rank_start

        # --- Step 5: Confidence gap check (optional second pass) ---
        # Second LLM pass is disabled by default for speed (saves ~2-5s).
        # Enable via UI or pass use_llm=True when a second opinion is needed.
        # SPEED OPT: Second pass disabled by default - no LLM call
        candidates, second_pass_triggered = maybe_second_pass(
            candidates, plan, use_llm=False,
            llm_model=self.llm_model, llm_base_url=self.ollama_url,
        )

        total_latency = time.time() - total_start
        logger.info(
            "NotebookAgent: done in %.2fs (plan=%.2fs, rank=%.2fs, top %d candidates)",
            total_latency, llm_latency, rank_latency, len(candidates),
        )

        # Attach video paths to candidates for the UI
        for c in candidates:
            c.video_path = self._resolve_video_path(c.video_id)

        return NotebookResult(
            query_id=f"notebook-{int(total_start * 1000) % 100000}",
            plan=plan,
            candidates=candidates,
            second_pass=second_pass_triggered,
            llm_latency=round(llm_latency, 3),
            total_latency=round(total_latency, 3),
            error=error,
        )

    def _fallback_plan(self, text: str, asr_query: str | None) -> NotebookPlan:
        """Build a minimal plan when the LLM planner completely fails."""
        from aic2026.notebook.notebook_planner import _rule_based_notebook_plan
        return _rule_based_notebook_plan(text, asr_query)

    def _resolve_video_path(self, video_id: str) -> str | None:
        """Resolve the source video file path for the UI's [Open Video] button."""
        import os
        from aic2026.models import project_path

        candidates = [
            project_path(f"data/raw/Videos/**/{video_id}.mp4"),
            project_path(f"data/raw/Videos_L01_a/video/{video_id}.mp4"),
            project_path(f"data/raw/Videos/{video_id}.mp4"),
        ]
        # Use glob to find the video
        import glob
        for pattern in candidates:
            matches = glob.glob(str(pattern), recursive=True)
            if matches:
                return matches[0]
        return None

    # --- Caching helpers for API layer ---

    def run_cached(
        self,
        text: str,
        asr_query: str | None = None,
        top_n: int = 5,
        use_visual: bool = False,
    ) -> NotebookResult:
        """Run with result caching (spec §18: cache raw_query → results).

        Same as run() but caches the NotebookResult so repeated identical
        queries skip LLM + retrieval entirely.
        """
        cache_key = (text, asr_query, top_n)
        cache = _RESULT_CACHE.get(cache_key)
        if cache is not None:
            logger.debug("NotebookAgent: result cache hit for %s...", text[:60])
            return cache

        result = self.run(text, asr_query, top_n, use_visual)
        _RESULT_CACHE[cache_key] = result
        return result


# Per-process result cache (spec §18).
_RESULT_CACHE: dict[tuple[str, str | None, int], NotebookResult] = {}


__all__ = ["NotebookAgent"]
