"""NOTEBOOK / VIDEO SEARCH pipeline (Improvement.md section 53-126).

A 4th independent retrieval mode alongside KIS/QA/TRAKE:

    QUERY (long/complex natural language)
    -> UNDERSTAND (local LLM planner, qwen2.5:1.5b no-think)
    -> RETRIEVE EVIDENCE (parallel ASR BM25 + visual SigLIP2)
    -> GROUP EVIDENCE BY VIDEO
    -> REASON (event coverage + temporal consistency)
    -> RANK VIDEOS
    -> RETURN TOP N VIDEO IDs

Design principles (spec section 11, 12, 20, 21):
- Deterministic tools do retrieval; LLM only does query understanding/reasoning.
- Default: EXACTLY 1 LLM CALL (planner only).
- Never let LLM read the full corpus or scan 115k ASR segments.
- Grounded on ASR/evidence, no hallucinated video IDs.
- Easy to debug, easy to toggle (separate module, does NOT touch KIS/QA/TRAKE).

Reuses (spec section 17):
- ``aic2026.agent.local_llm.OllamaLLM`` -- local LLM client
- ``aic2026.query`` -- planner utilities, parse_query
- ``aic2026.ingestion.asr`` -- ASR sidecar load
- ``aic2026.retrieval.bm25_index.BM25Index`` -- ASR BM25
- ``aic2026.embeddings.siglip2`` -- SigLIP2 encoder (optional visual)
- ``aic2026.temporal.alignment`` -- temporal decay / lambdas
"""

from aic2026.notebook.types import (
    NotebookPlan,
    NotebookEvent,
    NotebookEvidence,
    NotebookCandidate,
    NotebookResult,
    NotebookRequest,
    EventCoverage,
    TemporalEvidence,
    ModalitySignal,
)
from aic2026.notebook.agent import NotebookAgent
from aic2026.notebook.notebook_planner import (
    plan_notebook,
    DEFAULT_NOTEBOOK_MODEL,
)
from aic2026.notebook.dense_encoder import DenseTextEncoder
from aic2026.notebook.dense_retriever import DenseRetriever

__all__ = [
    "NotebookPlan",
    "NotebookEvent",
    "NotebookEvidence",
    "NotebookCandidate",
    "NotebookResult",
    "NotebookRequest",
    "EventCoverage",
    "TemporalEvidence",
    "ModalitySignal",
    "NotebookAgent",
    "plan_notebook",
    "DEFAULT_NOTEBOOK_MODEL",
    "DenseTextEncoder",
    "DenseRetriever",
]
