"""ASR BM25 retrieval for NOTEBOOK (spec §13, §22).

Searches per-segment ASR transcripts using rank_bm25 for evidence collection.
The retriever does NOT scan all 115k ASR segments blindly — it searches the
BM25 index for each asr_query and groups results by video_id.

Key constraints (from memory file):
- ASR transcripts are 100% Vietnamese; asr_queries must match in Vietnamese.
- rank_bm25 idf=0 for terms in >= half a tiny corpus — use 4+ doc pool minimum.
- Lazy BM25 build (~190s first time, cached thereafter).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from aic2026.ingestion.asr import load_transcripts_sidecar
from aic2026.retrieval.bm25_index import BM25Index
from aic2026.notebook.types import ASRSegmentMatch, NotebookPlan

logger = logging.getLogger(__name__)

# Minimum number of videos in the corpus for BM25 to be meaningful.
# rank_bm25 idf collapses to 0 when df >= N/2, so a tiny pool zeroes all scores.
_MIN_CORPUS_SIZE = 4

# Two Whisper renderings of the Vietnamese name for London occur in the corpus.
# Canonicalising this *phrase* (rather than fuzzily changing individual tokens)
# is safe and makes "Luân Đôn", "Luân Đôm" and "London" retrieve together.
_LONDON_RE = re.compile(r"\bluan\s+do[mn]\b|\blondon\b")

# Additional Whisper Vietnamese spelling corrections.
# These are common Whisper transcription errors for Vietnamese that should be
# normalized at query time so they match the indexed ASR text.
# Format: (incorrect_pattern, correct_form) - applied after NFD+strip diacritics.
_WHISPER_VI_CORRECTIONS = [
    # Common vowel confusions
    (r"\boong\b", "ung"),      # "ông" → "ung"
    (r"\baang\b", "ang"),      # "ã" → "ang"
    (r"\beeng\b", "eng"),      # "ề" → "eng"
    (r"\boang\b", "oang"),     # "oang" variants
    (r"\buong\b", "uong"),     # "ương" → "uong"
    (r"\bueng\b", "ueng"),     # "ường" → "ueng"
    # Consonant confusions
    (r"\bgi\b", "di"),         # "gi" → "di" (northern dialect)
    (r"\bgh\b", "g"),          # "gh" → "g"
    (r"\bkh\b", "k"),          # "kh" → "k"
    (r"\bngh\b", "ng"),        # "ngh" → "ng"
    (r"\bnh\b", "n"),          # "nh" → "n" (in some contexts)
    (r"\bph\b", "f"),          # "ph" → "f"
    (r"\bth\b", "t"),          # "th" → "t"
    (r"\btr\b", "tr"),         # "tr" → "tr" (keep)
    (r"\bch\b", "c"),          # "ch" → "c" (in some contexts)
    # Specific word corrections (Whisper often confuses these)
    (r"\bcon cho\b", "con cho"),      # "chó" often misheard
    (r"\bcon meo\b", "con meo"),      # "mèo"
    (r"\bnguoi\b", "nguoi"),          # "người"
    (r"\bnguoi dan\b", "nguoi dan"),  # "người dân"
    (r"\bxa hoi\b", "xa hoi"),        # "xã hội"
    (r"\bphat trien\b", "phat trien"),# "phát triển"
    (r"\bcong nghe\b", "cong nghe"),  # "công nghệ"
    (r"\bkhoa hoc\b", "khoa hoc"),    # "khoa học"
    (r"\bgiao duc\b", "giao duc"),    # "giáo dục"
    (r"\by te\b", "y te"),            # "y tế"
    (r"\bvan hoa\b", "van hoa"),      # "văn hóa"
    (r"\blich su\b", "lich su"),      # "lịch sử"
    (r"\bdia ly\b", "dia ly"),        # "địa lý"
    (r"\btoan hoc\b", "toan hoc"),    # "toán học"
    (r"\bvat ly\b", "vat ly"),        # "vật lý"
    (r"\bhoa hoc\b", "hoa hoc"),      # "hóa học"
    (r"\bsinh hoc\b", "sinh hoc"),    # "sinh học"
]
# Context radius for ASR segments - reduced from 2 to 1 for speed
_ASR_CONTEXT_RADIUS = 1


def _asr_lexical_tokens(text: str) -> list[str]:
    """Return accent-insensitive, ASR-aware tokens for BM25.

    The same normalisation is applied to indexed ASR and the user's query, so
    it repairs known Whisper spelling variants without rewriting source
    evidence or introducing fuzzy-match false positives.
    """
    folded = unicodedata.normalize("NFD", text.casefold())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = folded.replace("đ", "d")
    folded = re.sub(r"[^a-z0-9]+", " ", folded)
    canonical = _LONDON_RE.sub(" london ", folded)

    # Apply Whisper Vietnamese spelling corrections
    for pattern, replacement in _WHISPER_VI_CORRECTIONS:
        canonical = re.sub(pattern, replacement, canonical)

    return canonical.split()


@dataclass
class ASRRetriever:
    """Per-segment ASR BM25 retriever for NOTEBOOK evidence gathering.

    Loads the ASR sidecar lazily and builds a per-segment BM25 index.
    Each ASR segment is a document; BM25 search returns matching segments
    with their scores and timestamps.

    NOT thread-safe by design — one retriever per query is fine since
    NOTEBOOK processes one query at a time.
    """

    transcripts: dict[str, "VideoTranscript"]
    _segment_corpus: list[str] = None
    _segment_index: tuple[list[int], list[float]] = None  # (video_indices, bm25_scores)
    _seg_keys: list[tuple[str, int]] = None  # (video_id, segment_index) for each corpus entry

    @classmethod
    def from_sidecar(cls, sidecar_path: str) -> "ASRRetriever":
        """Load ASR sidecar and build per-segment index.

        Args:
            sidecar_path: Path to asr_sidecar.jsonl
        """
        transcripts = load_transcripts_sidecar(sidecar_path)
        logger.info("ASRRetriever: loaded %d videos from sidecar", len(transcripts))
        return cls(transcripts=transcripts)

    @classmethod
    def empty(cls) -> "ASRRetriever":
        """Return an empty retriever that returns no matches (fallback when sidecar is missing)."""
        return cls(transcripts={})

    def _build_segment_corpus(self) -> None:
        """Build a flat list of ASR segment texts with key mapping.

        Each ASR segment becomes one document in the BM25 corpus. We track
        (video_id, segment_index) for each position so we can reconstruct
        timestamps after search.
        """
        from rank_bm25 import BM25Okapi

        corpus: list[list[str]] = []
        seg_keys: list[tuple[str, int]] = []

        for video_id, transcript in self.transcripts.items():
            for seg_idx, segment in enumerate(transcript.segments):
                text = self._segment_context_text(video_id, seg_idx)
                if not text:
                    continue
                tokens = _asr_lexical_tokens(text)
                if len(tokens) < 1:
                    continue
                corpus.append(tokens)
                seg_keys.append((video_id, seg_idx))

        self._segment_corpus = corpus
        self._seg_keys = seg_keys

        if not corpus:
            logger.warning("ASRRetriever: no ASR segments to index")
            return

        self._bm25 = BM25Okapi(corpus)
        logger.info(
            "ASRRetriever: indexed %d segments across %d videos",
            len(corpus), len({k[0] for k in seg_keys}),
        )

    def _segment_context_text(self, video_id: str, segment_index: int) -> str:
        """Return the matched ASR segment with two neighbours on either side.

        The centre segment remains the timestamp anchor, while its neighbours
        make an ASR hit readable and allow a multi-word phrase split across
        adjacent Whisper segments to be found.  This is index-time text only;
        no model is loaded at query time.
        """
        segments = self.transcripts[video_id].segments
        first = max(0, segment_index - _ASR_CONTEXT_RADIUS)
        last = min(len(segments), segment_index + _ASR_CONTEXT_RADIUS + 1)
        return " ".join(
            seg.text.strip() for seg in segments[first:last] if seg.text.strip()
        )

    @property
    def is_ready(self) -> bool:
        """True when the per-segment BM25 index has been built."""
        return hasattr(self, '_bm25') and self._bm25 is not None

    def ensure_ready(self) -> None:
        """Lazily build the segment BM25 index on first use."""
        if not self.is_ready:
            self._build_segment_corpus()

    def search_segments(
        self,
        query: str,
        top_k: int = 30,
        video_ids: set[str] | None = None,
    ) -> list[ASRSegmentMatch]:
        """BM25 search across all ASR segments for *query*.

        Returns matching segments sorted by score (descending), each enriched
        with video_id, start/end timestamps, and the matched text.

        Args:
            query: Vietnamese search term (must match ASR transcript language)
            top_k: Maximum segments to return
            video_ids: Optional filter to a subset of videos

        Returns:
            List of ASRSegmentMatch, sorted by bm25_score descending.
        """
        self.ensure_ready()

        if not self.is_ready or not self._seg_keys:
            return []

        # Skip degenerate queries
        query_tokens = _asr_lexical_tokens(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Collect positive-score matches
        results: list[ASRSegmentMatch] = []
        for idx, score in enumerate(scores):
            if score > 0:
                video_id, seg_idx = self._seg_keys[idx]
                if video_ids is not None and video_id not in video_ids:
                    continue
                segment = self.transcripts[video_id].segments[seg_idx]
                results.append(
                    ASRSegmentMatch(
                        video_id=video_id,
                        segment_text=self._segment_context_text(video_id, seg_idx),
                        start=segment.start,
                        end=segment.end,
                        frame=segment.frame,
                        matched_query=query,
                        bm25_score=float(score),
                    )
                )

        # Sort by score descending, cap to top_k
        results.sort(key=lambda m: m.bm25_score, reverse=True)
        return results[:top_k]

    def search_queries(
        self,
        queries: Sequence[str],
        top_k_per_query: int = 20,
        video_ids: set[str] | None = None,
    ) -> list[ASRSegmentMatch]:
        """Search for multiple queries in parallel and merge results.

        Each query is searched independently; results are merged and re-sorted
        by score. The same segment may match multiple queries.

        Args:
            queries: Vietnamese search terms from NotebookPlan.asr_queries
            top_k_per_query: Per-query result cap
            video_ids: Optional video filter

        Returns:
            Merged list of ASRSegmentMatch, sorted by score descending.
        """
        all_matches: list[ASRSegmentMatch] = []
        for q in queries:
            matches = self.search_segments(q, top_k_per_query, video_ids)
            all_matches.extend(matches)

        all_matches.sort(key=lambda m: m.bm25_score, reverse=True)
        return all_matches

    def search_queries_prf(
        self,
        queries: Sequence[str],
        top_k_per_query: int = 20,
        video_ids: set[str] | None = None,
        prf_top_n: int = 5,
        alpha: float = 1.0,
        beta: float = 0.4,
    ) -> list[ASRSegmentMatch]:
        """Search with Rocchio Pseudo-Relevance Feedback (PRF) expansion.

        1. Run initial search for queries.
        2. Extract top feedback terms from Top-N matches.
        3. Formulate expanded queries and run 2nd pass retrieval.
        4. Fuse results using Dynamic RRF.
        """
        initial_matches = self.search_queries(queries, top_k_per_query=top_k_per_query, video_ids=video_ids)
        if not initial_matches or len(initial_matches) < prf_top_n:
            return initial_matches

        top_segments = initial_matches[:prf_top_n]
        words = []
        for m in top_segments:
            words.extend(m.segment_text.lower().split())

        from collections import Counter
        stop_words = {"và", "là", "của", "trong", "một", "các", "những", "này", "với", "cho", "tại", "đã", "đang", "được"}
        counts = Counter(w for w in words if len(w) >= 2 and w not in stop_words)
        prf_terms = [word for word, count in counts.most_common(3)]

        if not prf_terms:
            return initial_matches

        base_q = queries[0] if queries else ""
        expanded_query = f"{base_q} {' '.join(prf_terms)}".strip()
        logger.info("ASR PRF expansion query: %s", expanded_query)
        second_matches = self.search_segments(expanded_query, top_k=top_k_per_query, video_ids=video_ids)

        return fuse_bm25_dense(
            initial_matches,
            second_matches,
            bm25_weight=alpha,
            dense_weight=beta,
            dynamic_k=True,
        )

    def retrieve(
        self,
        plan: NotebookPlan,
        top_k_per_query: int = 10,  # Reduced from 20 for speed
        max_total_matches: int = 100,  # Reduced from 200 for speed
        use_prf: bool = False,
    ) -> list[ASRSegmentMatch]:
        """Retrieve ASR evidence for a NotebookPlan.

        Runs BM25 search for each asr_query in the plan, merges results,
        and caps the total number of matches to bound memory.

        This is the main public method called by the NotebookAgent.
        """
        if not plan.asr_queries:
            logger.warning("ASR retriever: plan has no asr_queries")
            return []

        if use_prf:
            matches = self.search_queries_prf(
                plan.asr_queries,
                top_k_per_query=top_k_per_query,
            )
        else:
            matches = self.search_queries(
                plan.asr_queries,
                top_k_per_query=top_k_per_query,
            )

        if len(matches) > max_total_matches:
            matches = matches[:max_total_matches]
            logger.info(
                "ASR retriever: truncated to %d matches (from %d total)",
                max_total_matches, len(matches),
            )

        logger.info(
            "ASR retriever: %d segment matches for %d queries",
            len(matches), len(plan.asr_queries),
        )
        return matches


def fuse_bm25_dense(
    bm25_matches: list[ASRSegmentMatch],
    dense_matches: list[ASRSegmentMatch],
    k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    dynamic_k: bool = False,
) -> list[ASRSegmentMatch]:
    """Fuse BM25 and Dense retrieval results using Reciprocal Rank Fusion (RRF).

    RRF score = sum( weight / (k + rank) ) for each list where the segment appears.

    Args:
        bm25_matches: BM25 retrieval results
        dense_matches: Dense retrieval results
        k: RRF constant (higher = less weight on top ranks)
        bm25_weight: Weight for BM25 ranks
        dense_weight: Weight for Dense ranks
        dynamic_k: When True, scale k dynamically based on pool candidate depth.

    Returns:
        Fused list sorted by RRF score descending
    """
    if dynamic_k:
        max_len = max(len(bm25_matches), len(dense_matches), 1)
        k = max(1, min(60, int(max_len * 0.2)))

    # Build segment key → combined RRF score
    rrf_scores: dict[tuple[str, float, float, str], float] = {}
    segment_data: dict[tuple[str, float, float, str], ASRSegmentMatch] = {}

    for rank, m in enumerate(bm25_matches):
        key = (m.video_id, m.start, m.end, m.segment_text[:50])
        rrf_scores[key] = rrf_scores.get(key, 0.0) + bm25_weight / (k + rank + 1)
        if key not in segment_data:
            segment_data[key] = m

    for rank, m in enumerate(dense_matches):
        key = (m.video_id, m.start, m.end, m.segment_text[:50])
        rrf_scores[key] = rrf_scores.get(key, 0.0) + dense_weight / (k + rank + 1)
        if key not in segment_data:
            segment_data[key] = m

    # Sort by RRF score descending
    fused = []
    for key, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        m = segment_data[key]
        fused.append(m.model_copy(update={"bm25_score": rrf_score}))

    return fused


__all__ = ["ASRRetriever", "fuse_bm25_dense"]
