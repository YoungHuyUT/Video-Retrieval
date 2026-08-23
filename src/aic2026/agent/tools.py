from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from aic2026.models import Candidate, FrameRecord

logger = logging.getLogger(__name__)
from aic2026.reranking import (
    late_interaction_rerank,
    contrastive_clip_colour_rerank,
    rerank_with_colour_evidence,
    object_evidence_adjustment,
    rerank_with_metadata,
    rerank_with_object_evidence,
)
from aic2026.retrieval import RetrievalPipeline
from aic2026.temporal import align_events, refine_trake_candidates

if TYPE_CHECKING:
    from aic2026.retrieval.bm25_index import BM25Index


_TASKS_REQUIRING_DENSE_VIDEO_EVIDENCE = frozenset(
    {
        "qa",
        "trake",
    }
)

@dataclass
class RetrievalTools:
    """The agent's controlled interface to competition evidence."""

    pipeline: RetrievalPipeline
    encode_text: Callable[[str], np.ndarray]
    visual_answerer: (
        Callable[[str, list[Candidate]], dict[int, str]] | None
    ) = None
    # Optional handle to the VLM instance so we can free its weights after QA
    # answering (low-RAM machines cannot keep CLIP + Florence resident at once).
    visual_answerer_obj: Any | None = None
    # VLM construction params. The VLM is NOT created at agent build time; it is
    # lazily built (see ``ensure_vlm``) only on the QA answer step, AFTER the CLIP
    # encoder has been freed (``close_text_encoder``). This keeps only ONE heavy
    # model resident at a time (avoids OOM on low-RAM machines) and means KIS/TRAKE
    # never pay the Florence load cost. ``visual_answerer_obj`` stays None for
    # KIS/TRAKE (vlm_model None) so ``ensure_vlm`` is a no-op there.
    vlm_model: str | None = None
    vlm_device: str | None = None
    vlm_dtype: str = "float32"
    encode_images: Callable[[list[object]], np.ndarray] | None = None
    bm25_index: BM25Index | None = None
    # Trọng số metadata bonus (Direction B): nhỏ, cộng trực tiếp lên RRF gốc
    # (KHÔNG normalize). RRF base ~0.01-0.03; weight 0.01 → bonus tối đa +0.01,
    # đủ nudging mà không đảo ngược thứ tự RRF.
    rerank_weight: float = 0.01
    # Asymmetry of the object-evidence signal (KIS).  A frame shown by the
    # detector to LACK every object the query asks for is penalized
    # `object_evidence_weight * object_penalty_scale`, while a full match is only
    # rewarded `object_evidence_weight`.  This counters CLIP ranking a frame high
    # for the right scene but the wrong object.  Default 2.0.
    object_penalty_scale: float = 2.0
    # Base magnitude of the object-evidence reward/penalty (KIS).  On the RRF
    # scale (~0.0018–0.016) a value of 0.02–0.05 is a strong but bounded nudge.
    object_evidence_weight: float = 0.03
    # Hard-drop frames that have NO object labels at all (ingest marked them as
    # blurry / no-clear-object, i.e. no entity reached the 0.4 present-threshold)
    # WHEN the query actually asks for an object.  This is the "blurry frame"
    # filter: such frames never enter the final KIS ranking.  Off by default so
    # scene-only queries and weak-detector corpora are unaffected; turn on for
    # KIS via --drop-empty-object-frames.  Falls back to keeping the pool if
    # dropping would empty it.
    drop_empty_object_frames: bool = False
    # Trọng số late-interaction (ColBERT-style MaxSim): encode_text ~200 lần trong
    # retrieve(), RẤT nặng cho thi tốc độ. Mặc định 0.0 (TẮT) — chỉ bật nếu cần
    # tăng recall query dài nhiều từ (rare cho BTC). Có thể bật qua runtime config.
    late_interaction_weight: float = 0.0
    # HSV colour evidence reads only the top image candidates and never calls a
    # VLM.  It is deliberately bounded so CLIP remains the primary signal.
    colour_rerank_weight: float = 0.08
    contrastive_colour_rerank_weight: float = 0.06
    colour_sidecar_path: str | None = None
    # Restrict retrieval to a dataset split by video_id prefix (e.g. ``["L25"]``
    # for QA on online-course videos, ``["L26"]`` for TRAKE). Optional; when set
    # it is intersected with `video_filter_terms` (metadata) so both filters
    # apply. ``None``/empty = no prefix restriction. The agent sets this per task
    # (QA→L25, TRAKE→L26) before each retrieve call.
    video_prefixes: list[str] | None = None
    video_filter_terms: list[str] | None = None
    # Số video tối đa giữ lại sau bước video-level coarse filter (KIS). Dataset
    # lớn (> số video này) thì chỉ top-K video theo video_score mới được giữ, còn
    # lại loại bỏ trước khi vào MMR — tránh video nhiễu (1 frame outlier) lọt top.
    # 0 / <= 0 hoặc >= số video = xét hết (backward-compatible, dataset nhỏ).
    coarse_top_k: int = 200
    # Coarse keyframes select the video; only its short event windows are then
    # decoded from the source video for finer TRAKE timestamps.
    #
    # Default is OFF for the official BTC corpus. The base TRAKE path already
    # emits source-video frame coordinates: the manifest's `frame_id` is mapped
    # from the official map-keyframes CSV (frame_idx gốc của video) inside
    # `prepare-official`, so `event_frames` already land inside the ground-truth
    # `ranges`. Enabling dense refinement re-encodes raw frames with a *different*
    # CLIP embedding (open_clip ViT-B/32 vs the official BTC CLIP vectors used for
    # retrieval) and re-runs DP; because the metric is `start <= frame <= end`
    # (range, not exact), picking a nearby frame in the same scene that falls
    # outside the range silently zeroes that event. It also adds the cost of
    # decoding `.mp4` for up to `trake_refine_top_videos` videos. Keep off unless
    # a dev set shows it helps for a specific reason.
    trake_dense_refine: bool = False
    trake_refine_top_videos: int = 20
    trake_refine_sample_fps: float = 3.0
    trake_refine_window_seconds: float = 2.0
    trake_video_root: Path = Path("data/raw/Videos")

    @property
    def has_lexical_objects(self) -> bool:
        """Whether a frame-level objects/OCR BM25 index is available."""
        return self.bm25_index is not None and not self.bm25_index.is_empty

    def retrieve(
        self,
        query: str | list[str],
        limit: int,
        task_type: str | None = None,
    ) -> list[Candidate]:
        """Retrieve task-appropriate candidates.

        KIS receives diversified results. QA and TRAKE receive raw candidates
        so that multiple relevant frames from the same video are preserved.
        Every non-TRAKE path is then reranked by lexical keyword overlap with
        object/metadata text (a small deterministic score bonus), so frames
        whose labels/titles mention the query terms float above pure vector
        matches.

        ``query`` may be a single string or a list of query variants
        (multi-query expansion). When multiple variants are given, each is
        encoded and the resulting ranked lists are fused with Reciprocal Rank
        Fusion before reranking, improving recall.
        """

        if limit <= 0:
            return []

        queries = [query] if isinstance(query, str) else list(query)
        embeddings = [self.encode_text(q) for q in queries]

        requires_dense_evidence = (
            task_type in _TASKS_REQUIRING_DENSE_VIDEO_EVIDENCE
        )

        # Metadata pre-filter: narrow the candidate video set before the
        # (relatively expensive) embedding/BM25 retrieval so frames from
        # off-topic videos never enter the pool.  Combined with the optional
        # `video_prefixes` split restriction (e.g. QA→L25): intersection means
        # a video must satisfy BOTH filters to stay in the pool.
        allowed_video_ids = self.pipeline.filter_terms_to_video_ids(
            self.video_filter_terms or []
        )
        if self.video_prefixes:
            prefix_ids = self.pipeline.prefixes_to_video_ids(
                self.video_prefixes, allowed_video_ids
            )
            if prefix_ids is not None:
                allowed_video_ids = prefix_ids
        if allowed_video_ids is not None:
            logger.info(
                "video pool after filters: %d/%d videos (%s)%s",
                len(allowed_video_ids),
                len({record.video_id for record in self.pipeline.manifest}),
                ", ".join(sorted(allowed_video_ids))[:200],
                f" prefixes={self.video_prefixes}" if self.video_prefixes else "",
            )

        if self.bm25_index is not None and self.bm25_index.is_empty:
            logger.warning(
                "BM25 skipped: manifest has no Objects/Metadata text. "
                "Run `aic2026 prepare` to load them for lexical retrieval."
            )

        # Bound the retrieval work to the requested candidate pool.  A global
        # per-video scan is disproportionately expensive for a static CLIP index.
        pool = limit

        if self.bm25_index is not None and not self.bm25_index.is_empty:
            logger.debug("BM25 lexical index active; fusing with vector retrieval.")
            if requires_dense_evidence:
                candidates = self.pipeline.hybrid_retrieve_raw(
                    text_query=queries[0],
                    text_embedding=embeddings[0],
                    bm25_index=self.bm25_index,
                    top_frames=pool,
                    video_ids=allowed_video_ids,
                )
            else:
                candidates = self.pipeline.hybrid_retrieve_raw(
                    text_query=queries[0],
                    text_embedding=embeddings[0],
                    bm25_index=self.bm25_index,
                    top_frames=pool,
                    video_ids=allowed_video_ids,
                )
        elif requires_dense_evidence:
            candidates = self.pipeline.retrieve_raw(
                text_embedding=embeddings[0],
                top_frames=pool,
                video_ids=allowed_video_ids,
            )
        elif len(embeddings) > 1:
            # Multi-query expansion: RRF-fuse per-variant vector rankings.
            ranked_lists = []
            for emb in embeddings:
                ids, _ = self.pipeline.search_with_filter(
                    emb, pool, allowed_video_ids
                )
                ranked_lists.append(np.asarray(ids, dtype=np.int64))
            fused = self.pipeline._rrf_fuse(ranked_lists)
            candidates = self.pipeline._candidates_from_scores(fused, limit=limit)
        else:
            candidates = self.pipeline.retrieve_raw(
                text_embedding=embeddings[0],
                top_frames=pool,
                video_ids=allowed_video_ids,
            )

        # Safety net: drop any candidate whose video slipped past the DB filter
        # (e.g. matched only via BM25 lexical text, not the vector tier).
        if allowed_video_ids is not None:
            candidates = self.pipeline._mask_video_ids(candidates, allowed_video_ids)

        # Lexical metadata bonus: nudge frames whose object/title/description
        # text literally mentions the query terms (deterministic, not learned).
        candidates = rerank_with_metadata(
            query=queries[0],
            candidates=candidates,
            records=self._record_lookup(),
            weight=self.rerank_weight,
        )
        # Object detector evidence is a stronger, signed signal than generic
        # lexical overlap: an exact/synonym match is promoted; a known object
        # mismatch is softly penalized.  No advanced UI field is required.
        candidates = rerank_with_object_evidence(
            query=queries[0],
            candidates=candidates,
            records=self._record_lookup(),
            weight=self.object_evidence_weight,
            penalty_scale=self.object_penalty_scale,
            drop_empty_object_frames=self.drop_empty_object_frames,
        )

        # Contrastive CLIP Colour Reranking (Option 1):
        # Crops object boxes/full keyframe image and compares contrastive prompt variants
        # (e.g. "red car" vs "blue car", "yellow car", etc.) using CLIP.
        if self.encode_text is not None and self.encode_images is not None:
            candidates = contrastive_clip_colour_rerank(
                query=queries[0],
                candidates=candidates,
                records=self._record_lookup(),
                encode_text=self.encode_text,
                encode_images=self.encode_images,
                weight=self.contrastive_colour_rerank_weight,
            )

        # KIS is ranked per frame.  Do not aggregate/cap by video here: a
        # candidate rises or falls only on its own retrieval evidence.

        # Late-interaction (ColBERT-style MaxSim): bắt khớp cục bộ theo từng facet
        # của query — một frame khớp BẤT KỲ facet nào (vd "water bottle") đều được
        # nâng, điều vector CLIP pooled đơn lẻ không làm được. Chỉ chạy khi index
        # thực sự chứa ĐỦ toàn bộ corpus: Chroma nếu chỉ build 1 phần (thiếu mấy
        # chục nghìn vector) thì vector_id sẽ lệch → skip an toàn, fallback về
        # RRF + metadata bonus, không crash.
        if self.encode_text is not None and candidates:
            index = getattr(self.pipeline, "index", None)
            index_has_all = index is not None
            manifest_size = len(self.pipeline.manifest) if self.pipeline.manifest is not None else 0
            if hasattr(index, "collection_count"):  # ChromaVectorStore
                try:
                    if index.collection_count != manifest_size:
                        index_has_all = False
                        logger.warning(
                            "late-interaction skipped: Chroma index has %d vectors "
                            "but manifest has %d — index is partial, vector_id would "
                            "mismatch. Rebuild with `aic2026 build-chroma-index`.",
                            index.collection_count,
                            manifest_size,
                        )
                except Exception as exc:  # noqa: BLE001
                    index_has_all = False
                    logger.warning("late-interaction skipped (index count error): %s", exc)
            elif getattr(index, "vectors", None) is None:
                index_has_all = False

            if index_has_all:
                # Late-interaction indexes ``index.vectors[c.vector_id]`` — the
                # manifest row. Refined TRAKE candidates can carry ``vector_id=None``
                # (dense_refinement drops it), which would IndexError here. Skip
                # those: rerank only the candidates that still map to a row, and
                # merge the untouched ones back in afterwards.
                indexed = [c for c in candidates if c.vector_id is not None]
                if indexed:
                    try:
                        frame_vectors = np.stack(
                            [index.vectors[c.vector_id] for c in indexed]
                        )
                        reranked = late_interaction_rerank(
                            query=queries[0],
                            candidates=indexed,
                            encode_text=self.encode_text,
                            frame_vectors=frame_vectors,
                            top_n=200,
                            weight=self.late_interaction_weight,
                        )
                        # Re-merge: keep candidates without a vector_id untouched.
                        reranked_ids = {id(c) for c in reranked}
                        candidates = reranked + [
                            c for c in candidates if id(c) not in reranked_ids
                        ]
                        # late_interaction chỉ sửa score tại chỗ, cần sort lại.
                        candidates = sorted(
                            candidates, key=lambda c: c.score, reverse=True
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("late-interaction rerank skipped: %s", exc)

        # KIS and QA are BOTH video-level: a single high-scoring outlier frame
        # must not lift the wrong video. Aggregate each video's strongest frames
        # into one video_score (log-sum-exp), keep the top videos, and re-emit
        # frames ordered by that video-aware score — so a video with *consistently*
        # good frames rises above a lone-outlier video, and we still surface other
        # strong videos instead of spamming frames from one. QA inherits this so
        # that (a) its candidate ordering matches KIS for the same query, and
        # (b) the VLM only answers frames from the strongest videos instead of a
        # raw pool of hundreds. TRAKE keeps the raw pool (multiple frames per
        # video are meaningful there and it has its own alignment stage).
        if task_type == "kis":
            top_videos = self.coarse_top_k if self.coarse_top_k and self.coarse_top_k > 0 else None
            candidates = self.pipeline.video_level_rerank(
                candidates,
                top_videos=top_videos,
            )

        return candidates

    def _record_lookup(self) -> dict[int, FrameRecord]:
        """Map ``vector_id`` → manifest record, for reranking."""

        manifest = getattr(self.pipeline, "manifest", None)
        if not manifest:
            return {}
        return {
            record.vector_id: record
            for record in manifest
        }

    def retrieve_trake(
        self,
        events: list[str],
        limit: int,
        prefilter_frames_per_event: int = 500,
        penalty_weight: float = 0.005,
        coarse_top_k: int = 200,
        # Soft *preference* (not a hard restriction) for certain video_id
        # prefixes (e.g. ["L26"]). Preferred videos get a small bounded score
        # nudge so they rank above equally-similar non-preferred videos, but a
        # non-preferred video is never dropped — BTC event queries are generic
        # and match many splits, so a hard restrict would zero recall.
        preferred_prefixes: list[str] | None = None,
    ) -> list[Candidate]:
        """Run deterministic event-wise TRAKE retrieval and alignment.

        ``coarse_top_k`` giới hạn số video đưa vào DP alignment (xem
        ``RetrievalPipeline.retrieve_trake``): chỉ top-K video theo coarse
        video-level similarity mới được xét, tránh DP chạy trên vài nghìn video.
        Object labels do not change this cap; it remains the guardrail that
        keeps dynamic-programming alignment bounded on the full corpus.
        """

        cleaned_events = [
            event.strip()
            for event in events
            if event.strip()
        ]

        if not cleaned_events:
            raise ValueError(
                "TRAKE requires at least one non-empty event"
            )

        if limit <= 0:
            return []

        event_embeddings = np.stack(
            [
                np.asarray(
                    self.encode_text(event),
                    dtype=np.float32,
                ).reshape(-1)
                for event in cleaned_events
            ],
            axis=0,
        )

        # Metadata pre-filter (TRAKE): restrict candidate videos before the
        # per-event vector scan inside retrieve_trake (pushed down to the DB).
        # Combined with the optional `video_prefixes` split restriction so a
        # video must satisfy BOTH to stay in the pool.
        allowed_video_ids = self.pipeline.filter_terms_to_video_ids(
            self.video_filter_terms or []
        )
        if self.video_prefixes:
            prefix_ids = self.pipeline.prefixes_to_video_ids(
                self.video_prefixes, allowed_video_ids
            )
            if prefix_ids is not None:
                allowed_video_ids = prefix_ids

        # Unit-test/lightweight pipelines may not expose manifest records.  The
        # production pipeline always does; in that case apply evidence inside
        # the event×frame matrix (vectorized) before temporal DP selects an
        # alignment. Building the adjustment as a per-video NumPy matrix — instead
        # of a per-(event,frame) Python closure — cuts TRAKE latency from ~145s to
        # a few seconds on the full 177k-frame corpus for a single query.
        object_adjustment = None
        object_adjustment_matrices = None
        manifest = getattr(self.pipeline, "manifest", None)
        video_to_manifest = getattr(
            self.pipeline, "_video_to_manifest_indices", None
        )
        if (
            manifest
            and video_to_manifest is not None
            and not self.trake_dense_refine
        ):
            from aic2026.reranking.lexical import (
                build_object_adjustment_matrices,
            )

            all_videos = (
                set(self.pipeline._video_embeddings.keys())
                if hasattr(self.pipeline, "_video_embeddings")
                else set(video_to_manifest.keys())
            )
            object_adjustment_matrices = (
                build_object_adjustment_matrices(
                    events=cleaned_events,
                    candidate_videos=all_videos,
                    video_to_manifest_indices=video_to_manifest,
                    manifest=manifest,
                    weight=0.08,
                )
            )

        candidates = self.pipeline.retrieve_trake(
            event_embeddings=event_embeddings,
            top_videos=limit,
            prefilter_frames_per_event=(
                prefilter_frames_per_event
            ),
            penalty_weight=penalty_weight,
            video_ids=allowed_video_ids,
            coarse_top_k=coarse_top_k,
            object_adjustment=object_adjustment,
            object_adjustment_matrices=object_adjustment_matrices,
            preferred_prefixes=preferred_prefixes,
        )

        # Safety net for any candidate the DB filter could not exclude.
        if allowed_video_ids is not None:
            candidates = self.pipeline._mask_video_ids(candidates, allowed_video_ids)

        if self.trake_dense_refine:
            candidates = refine_trake_candidates(
                candidates,
                event_embeddings,
                self.encode_images,
                video_root=self.trake_video_root,
                top_videos=self.trake_refine_top_videos,
                sample_fps=self.trake_refine_sample_fps,
                window_seconds=self.trake_refine_window_seconds,
                penalty_weight=penalty_weight,
            )

        return candidates

    def candidates_for_video(
        self,
        candidates: list[Candidate],
        video_id: str,
    ) -> list[Candidate]:
        return [
            candidate
            for candidate in candidates
            if candidate.video_id == video_id
        ]

    def temporal_alignment(
        self,
        candidates: list[Candidate],
        event_count: int,
    ) -> list[int]:
        return align_events(candidates, event_count)

    def answer_question(
        self,
        question: str,
        candidates: list[Candidate],
    ) -> dict[int, str]:
        if self.visual_answerer is None:
            return {}
        # Reload the VLM if it was previously closed (cached agent reuse).
        obj = self.visual_answerer_obj
        if obj is not None and not getattr(obj, "available", False):
            try:
                obj.load()
            except Exception as exc:  # noqa: BLE001
                logger.warning("answer_question: VLM reload failed: %s", exc)
        return self.visual_answerer(
            question,
            candidates,
        )

    def reload_text_encoder(self) -> None:
        """Rebuild the CLIP encoder if it was unloaded (cached agent reuse)."""
        enc = getattr(self.encode_text, "__self__", None)
        if enc is not None and hasattr(enc, "load"):
            try:
                enc.load()
            except Exception as exc:  # noqa: BLE001
                logger.warning("reload_text_encoder: load failed: %s", exc)

    def close_text_encoder(self) -> None:
        """Free the CLIP text/image encoder weights to reclaim RAM before the VLM
        runs (low-RAM machines cannot keep CLIP + Florence resident at once).

        Only the underlying model weights are dropped — the ``encode_text`` /
        ``encode_images`` callables are preserved so a later retrieve on a *cached*
        agent can reload them (see ``reload_text_encoder``). We never null out the
        callables, which previously caused ``'NoneType' object is not callable'``.
        """
        enc = getattr(self.encode_text, "__self__", None)
        if enc is not None and hasattr(enc, "unload"):
            try:
                enc.unload()
            except Exception as exc:  # noqa: BLE001
                logger.warning("close_text_encoder: unload failed: %s", exc)

    def ensure_vlm(self) -> None:
        """Lazily build the VLM only on the QA answer step, AFTER the CLIP encoder
        has been freed (``close_text_encoder``). Avoids holding CLIP + Florence
        resident at once (OOM on low-RAM machines) and means KIS/TRAKE (vlm_model
        None) never pay the Florence load. No-op when already built or no model.
        """
        if self.visual_answerer is not None or not self.vlm_model:
            return
        try:
            from aic2026.qa.florence import FlorenceVLM
        except Exception as exc:  # noqa: BLE001
            logger.warning("ensure_vlm: cannot import FlorenceVLM: %s", exc)
            return
        florence = FlorenceVLM(
            model_name=self.vlm_model,
            device=self.vlm_device,
            torch_dtype=self.vlm_dtype,
        )
        florence.load()  # build processor + model now (CLIP already freed)
        if florence.available:  # only wire up if it actually loaded
            self.visual_answerer = florence.answer_question
            self.visual_answerer_obj = florence

    def close_vlm(self) -> None:
        """Free the VLM (Florence-2) weights after QA answering is complete.

        Only the underlying model is closed — the ``visual_answerer`` callable and
        ``visual_answerer_obj`` handle are preserved so a later request on a
        *cached* agent can reload the VLM (see ``answer_question``). We never null
        them out, which previously caused ``'NoneType' object is not callable'``.
        """
        obj = self.visual_answerer_obj
        if obj is not None and hasattr(obj, "close"):
            try:
                obj.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("close_vlm: close failed: %s", exc)

    @staticmethod
    def evidence(
        candidates: list[Candidate],
        maximum: int = 40,
    ) -> list[dict]:
        fields = {
            "vector_id",
            "video_id",
            "frame_id",
            "score",
            "keyframe_path",
        }

        return [
            candidate.model_dump(include=fields)
            for candidate in candidates[:maximum]
        ]

    @staticmethod
    def by_vector_id(
        candidates: list[Candidate],
    ) -> dict[int, Candidate]:
        return {
            candidate.vector_id: candidate
            for candidate in candidates
            if candidate.vector_id is not None
        }

    @staticmethod
    def group_by_video(
        candidates: list[Candidate],
    ) -> dict[str, list[Candidate]]:
        grouped: dict[str, list[Candidate]] = defaultdict(list)

        for candidate in candidates:
            grouped[candidate.video_id].append(candidate)

        return grouped
