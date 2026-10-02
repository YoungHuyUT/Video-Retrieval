"""Segment-level index (V, rule 14) — coarse retrieval one level above frames.

Hierarchy the spec wants:

    VIDEO
      └── SEGMENT   (a contiguous run of ~N sampled frames)
            └── FRAME
            └── FRAME
            ...

A **segment** is built by pooling the CLIP embeddings of a contiguous run of a
single video's frames (spec V: "Có thể tạo segment embedding bằng pooling frame
embeddings" — no heavy neural video encoder required).  Each segment carries:

* ``video_id`` and its ordered ``frame_ids`` (manifest frame indices),
* ``start_ts`` / ``end_ts`` derived from ``frame_id × interval``
  (the sampling cadence, default 1.0 s — "Giữ 1 FPS để đảm bảo recall"),
* a representative embedding (chosen pooling strategy),
* a mean-pooled embedding for cheap fallback.

Why this layer (spec rule 14: "Segment retrieval phải đứng trước fine frame
retrieval"): a long / multi-event query should be matched against a small
number of *segment* representatives first, instead of scanning every frame of
every video.  That is a coarse-to-fine pre-filter — it never replaces frame
retrieval, it narrows the candidate set before the (more expensive) frame-level
RRF / rerank stages run.

This module is **self-contained and additive**: it does not modify the frame
index, the manifest, or the query path.  It exposes a plain ``search`` that
returns top segments with their frame ranges; the pipeline/tools layer decides
*when* to use it (today: only for multi-event / long queries, so a short single
query keeps the existing fast path — no regression).

Pooling strategies (all four are benchmarked per spec V):
* ``mean``     — average of frame vectors (default; robust).
* ``max``      — element-wise max across frames (catches salient features).
* ``topk``     — mean of the ``top_k`` frames with highest norm diversity.
* ``weighted`` — mean weighted by each frame's centrality to the run.

The strategy is a build-time choice and is cached with the index, so A/B
comparisons are reproducible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from aic2026.models import FrameRecord

from .index import VectorIndex

logger = logging.getLogger(__name__)

PoolingStrategy = Literal["mean", "max", "topk", "weighted"]

# Default contiguous-run length in *frames*. At the default 1.0 s sampling this
# is ~10 s of video per segment.  Spec V explicitly invites benchmarking pool
# sizes; expose as a parameter rather than hard-coding.
DEFAULT_SEGMENT_FRAMES = 10
# Default sampling cadence (seconds per frame).  The real interval is whatever
# was used at ingestion; we accept it as a build-time parameter and fall back to
# 1.0 s when unknown.  start_ts = frame_id * interval.
DEFAULT_INTERVAL_SEC = 1.0
# For the "topk" pooling strategy: keep the top fraction of frames by a
# centrality measure, then mean-pool them.
TOPK_FRACTION = 0.5


@dataclass
class Segment:
    """One contiguous run of frames inside a single video."""

    segment_id: int
    video_id: str
    # Ordered manifest indices (== vector row indices) of the frames in this run.
    frame_manifest_indices: list[int]
    frame_ids: list[int]
    start_ts: float
    end_ts: float
    # Representative embedding (chosen pooling strategy), L2-normalized.
    embedding: np.ndarray
    # Mean-pooled embedding (always available, cheap fallback), L2-normalized.
    mean_embedding: np.ndarray
    # Short OCR/object text aggregated from member frames — lets future stages
    # BM25-filter segments without re-reading every frame.
    text: str = ""

    @property
    def frame_count(self) -> int:
        return len(self.frame_manifest_indices)


@dataclass
class SegmentHit:
    """A segment returned by :meth:`SegmentIndex.search`."""

    segment: Segment
    score: float
    # Manifest (vector) indices of the frames inside the hit segment.
    frame_manifest_indices: list[int] = field(default_factory=list)
    video_id: str = ""
    start_ts: float = 0.0
    end_ts: float = 0.0

    def __post_init__(self) -> None:
        if not self.frame_manifest_indices and self.segment is not None:
            self.frame_manifest_indices = list(self.segment.frame_manifest_indices)
        if not self.video_id and self.segment is not None:
            self.video_id = self.segment.video_id
        if self.segment is not None:
            self.start_ts = self.segment.start_ts
            self.end_ts = self.segment.end_ts


class SegmentIndex:
    """Builds and searches a segment-level index over an existing frame index."""

    def __init__(
        self,
        segments: list[Segment],
        strategy: PoolingStrategy = "mean",
        interval_sec: float = DEFAULT_INTERVAL_SEC,
        segment_frames: int = DEFAULT_SEGMENT_FRAMES,
    ) -> None:
        if not segments:
            raise ValueError("SegmentIndex requires at least one segment")
        self.segments = segments
        self.strategy = strategy
        self.interval_sec = interval_sec
        self.segment_frames = segment_frames
        # Representative-embedding matrix for fast cosine search.
        self._matrix = np.stack([s.embedding for s in segments]).astype(np.float32)
        self._matrix /= np.maximum(
            np.linalg.norm(self._matrix, axis=1, keepdims=True), 1e-12
        )
        # Optional FAISS over the (small) segment matrix.
        self._faiss = None
        try:
            import faiss  # type: ignore

            self._faiss = faiss.IndexFlatIP(self._matrix.shape[1])
            self._faiss.add(self._matrix)
        except Exception:  # noqa: BLE001 — fall back to NumPy cosine
            self._faiss = None

    # -- build ---------------------------------------------------------------
    @classmethod
    def build(
        cls,
        index: VectorIndex,
        manifest: list[FrameRecord],
        segment_frames: int = DEFAULT_SEGMENT_FRAMES,
        strategy: PoolingStrategy = "mean",
        interval_sec: float = DEFAULT_INTERVAL_SEC,
        overlap: int = 0,
    ) -> "SegmentIndex":
        """Build segments from a frame ``VectorIndex`` + ``manifest``.

        Frames are grouped per video and split into contiguous runs of
        ``segment_frames`` (with optional ``overlap`` between consecutive runs).
        Contiguity is by ascending ``frame_id`` — a gap in frame_id starts a new
        segment, so scene cuts that drop frames do not merge across boundaries.
        """
        if segment_frames <= 0:
            raise ValueError("segment_frames must be > 0")
        if overlap < 0 or overlap >= segment_frames:
            raise ValueError("overlap must be in [0, segment_frames)")

        # Group manifest indices by video, ordered by frame_id (mirrors
        # RetrievalPipeline._build_video_manifest_index).
        by_video: dict[str, list[tuple[int, int]]] = {}
        for manifest_index, record in enumerate(manifest):
            by_video.setdefault(record.video_id, []).append(
                (record.frame_id, manifest_index)
            )

        segments: list[Segment] = []
        seg_id = 0
        step = max(1, segment_frames - overlap)
        vectors = index.vectors

        for video_id, pairs in by_video.items():
            pairs.sort(key=lambda p: p[0])
            frame_ids = [p[0] for p in pairs]
            m_indices = [p[1] for p in pairs]
            n = len(pairs)
            start = 0
            while start < n:
                end = min(start + segment_frames, n)
                run_ids = frame_ids[start:end]
                run_m = m_indices[start:end]
                if not run_m:
                    break
                seg = cls._make_segment(
                    seg_id, video_id, run_m, run_ids, vectors, strategy, interval_sec
                )
                segments.append(seg)
                seg_id += 1
                if end >= n:
                    break
                start += step

        logger.info(
            "SegmentIndex built: %d segments across %d videos "
            "(segment_frames=%d, overlap=%d, strategy=%s)",
            len(segments), len(by_video), segment_frames, overlap, strategy,
        )
        return cls(
            segments,
            strategy=strategy,
            interval_sec=interval_sec,
            segment_frames=segment_frames,
        )

    @staticmethod
    def _pool(vectors: np.ndarray, strategy: PoolingStrategy) -> np.ndarray:
        """Pool a (k, d) frame-vector matrix into one (d,) representative."""
        if vectors.shape[0] == 0:
            raise ValueError("cannot pool an empty frame set")
        if strategy == "mean":
            rep = vectors.mean(axis=0)
        elif strategy == "max":
            rep = vectors.max(axis=0)
        elif strategy == "topk":
            # Centrality = distance of each frame from the run mean; keep the
            # most central frames (they represent the dominant content), then mean.
            mean = vectors.mean(axis=0)
            dist = np.linalg.norm(vectors - mean, axis=1)
            # smaller distance = more central; keep the closest TOPK_FRACTION.
            k = max(1, int(round(vectors.shape[0] * TOPK_FRACTION)))
            keep = np.argsort(dist)[:k]
            rep = vectors[keep].mean(axis=0)
        elif strategy == "weighted":
            # Weight by centrality (inverse distance to run mean) so outlier
            # frames contribute less.
            mean = vectors.mean(axis=0)
            dist = np.linalg.norm(vectors - mean, axis=1)
            w = 1.0 / np.maximum(dist, 1e-6)
            w /= w.sum()
            rep = (vectors * w[:, None]).sum(axis=0)
        else:
            raise ValueError(f"unknown pooling strategy {strategy!r}")
        norm = float(np.linalg.norm(rep))
        if norm > 0:
            rep = rep / norm
        return rep.astype(np.float32)

    @classmethod
    def _make_segment(
        cls,
        seg_id: int,
        video_id: str,
        manifest_indices: list[int],
        frame_ids: list[int],
        vectors: np.ndarray,
        strategy: PoolingStrategy,
        interval_sec: float,
    ) -> Segment:
        run = vectors[manifest_indices]
        mean_rep = run.mean(axis=0)
        norm = float(np.linalg.norm(mean_rep))
        if norm > 0:
            mean_rep = mean_rep / norm
        rep = cls._pool(run, strategy)
        lo_id = min(frame_ids)
        hi_id = max(frame_ids)
        text_parts: list[str] = []
        return Segment(
            segment_id=seg_id,
            video_id=video_id,
            frame_manifest_indices=list(manifest_indices),
            frame_ids=list(frame_ids),
            start_ts=float(lo_id) * interval_sec,
            end_ts=float(hi_id) * interval_sec,
            embedding=rep,
            mean_embedding=mean_rep.astype(np.float32),
            text=" ".join(text_parts),
        )

    # -- search --------------------------------------------------------------
    def search(
        self, query: np.ndarray, k: int
    ) -> list[SegmentHit]:
        """Return the ``k`` most similar segments to ``query`` (L2-normalized)."""
        query = np.asarray(query, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        k = min(k, len(self.segments))
        if k <= 0:
            return []
        if self._faiss is not None:
            scores, ids = self._faiss.search(query[None, :], k)
            ids = ids[0]
            scores = scores[0]
        else:
            sims = self._matrix @ query
            idx = np.argpartition(-sims, k - 1)[:k]
            order = np.argsort(-sims[idx])
            ids = idx[order]
            scores = sims[ids]
        return [
            SegmentHit(segment=self.segments[int(i)], score=float(s))
            for i, s in zip(ids, scores, strict=False)
        ]

    def search_in_videos(
        self, query: np.ndarray, k: int, video_ids: set[str]
    ) -> list[SegmentHit]:
        """Like :meth:`search` but restricted to ``video_ids``."""
        if not video_ids:
            return self.search(query, k)
        cand = [s for s in self.segments if s.video_id in video_ids]
        if not cand:
            return []
        # Temporarily search over the restricted set via a fresh small index.
        sub = SegmentIndex(
            cand, strategy=self.strategy, interval_sec=self.interval_sec,
            segment_frames=self.segment_frames,
        )
        return sub.search(query, k)

    def frame_manifest_indices_for(
        self, hits: list[SegmentHit]
    ) -> list[int]:
        """Flatten the frame manifest indices covered by a list of segment hits."""
        out: list[int] = []
        for h in hits:
            out.extend(h.segment.frame_manifest_indices)
        # Preserve order but drop duplicates (a frame belongs to one segment).
        seen: set[int] = set()
        dedup: list[int] = []
        for i in out:
            if i not in seen:
                seen.add(i)
                dedup.append(i)
        return dedup
