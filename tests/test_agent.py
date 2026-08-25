from __future__ import annotations

import numpy as np

from aic2026.agent import RetrievalAgent
from aic2026.agent.tools import RetrievalTools
from aic2026.agent.types import AgentPlan
from aic2026.models import Candidate, Query


def test_cached_agent_handles_consecutive_qa_requests() -> None:
    """Regression: a cached agent (lru_cache in api.py) must survive multiple
    QA requests. The first QA request unloads CLIP + Florence to free RAM; the
    second request must reload the encoder (reload_text_encoder in run()) instead
    of crashing with ``'NoneType' object is not callable'``.

    We use a fake encoder that raises once its underlying model is unloaded, and
    a fake VLM that records load/close, to exercise the exact free→reload path
    without loading multi-hundred-MB models.
    """
    unloaded = {"encoder": False, "encoder_reloaded": False, "vlm_loads": 0, "vlm_closes": 0}

    class FakeEncoder:
        model = object()

        def unload(self) -> None:
            unloaded["encoder"] = True
            self.model = None

        def load(self) -> None:
            if self.model is None:
                self.model = object()
                unloaded["encoder_reloaded"] = True

        def encode(self, text: str) -> np.ndarray:
            if self.model is None:
                raise RuntimeError("encoder unloaded — callable must be reloaded")
            return np.asarray([1.0], dtype=np.float32)

    class FakeVLM:
        available = False

        def load(self) -> None:
            self.available = True
            unloaded["vlm_loads"] += 1

        def close(self) -> None:
            self.available = False
            unloaded["vlm_closes"] += 1

        def answer_question(self, question, candidates):
            return {c.vector_id: "ok" for c in candidates if c.vector_id is not None}

    enc = FakeEncoder()
    vlm = FakeVLM()

    class FakePipeline:
        manifest = None

        def filter_terms_to_video_ids(self, terms, video_ids=None):
            return None

        def prefixes_to_video_ids(self, prefixes, video_ids=None):
            return None

        def filter_videos_by_metadata(self, terms, video_ids=None):
            return sorted(video_ids or [])

        def retrieve(self, text_embedding, top_frames, max_answers, video_ids=None):
            return [Candidate(video_id="L01_V001", frame_id=505, score=0.9, vector_id=7)]

        def retrieve_raw(self, text_embedding, top_frames, video_ids=None):
            return [Candidate(video_id="L01_V001", frame_id=505, score=0.9, vector_id=7)]

        def video_level_rerank(self, candidates, top_videos=None, frames_per_video=None, aggregation_top_k=3):
            return candidates

    tools = RetrievalTools(
        FakePipeline(),
        encode_text=enc.encode,
        vlm_model="fake-model",
    )
    # Inject the fake VLM so ensure_vlm/answer_question use it without importing
    # the real (heavy) Florence wrapper.
    tools.visual_answerer = vlm.answer_question
    tools.visual_answerer_obj = vlm

    agent = RetrievalAgent(tools)

    # Simulate the QA _finalize path manually: close encoder, ensure+answer, close vlm
    q1 = Query(query_id="q1", type="qa", text="red speaker", question="What color?")
    tools.close_text_encoder()
    tools.ensure_vlm()
    _ = tools.answer_question(q1.question, [Candidate(video_id="L01_V001", frame_id=505, score=0.9, vector_id=7)])
    tools.close_vlm()

    assert unloaded["encoder"] is True  # encoder was freed after request 1
    assert unloaded["vlm_closes"] == 1  # vlm closed after request 1

    # Second QA request on the SAME (cached) agent — this used to raise
    # 'NoneType' object is not callable because encode_text was nulled out.
    q2 = Query(query_id="q2", type="qa", text="blue screen", question="What is shown?")
    result = agent.run(q2)  # run() calls reload_text_encoder() first

    assert result is not None
    assert [c.vector_id for c in result.candidates] == [7]
    # The encoder must have been reloaded for request 2 (reload_text_encoder
    # in run()), proving the callable was NOT left as None. We check the flag
    # set by FakeEncoder.load() rather than enc.model (which is unloaded again
    # at the end of request 2's _finalize to free RAM).
    assert unloaded["encoder_reloaded"] is True
    assert unloaded["vlm_loads"] >= 1


class FakePipeline:
    """Minimal deterministic retrieval backend used by agent unit tests.

    Mirrors only the methods ``RetrievalTools`` actually calls so the agent's
    deterministic pipeline can run end-to-end without a real index.
    """

    manifest: list | None = None
    frames_per_video: int | None = None

    def filter_terms_to_video_ids(
        self,
        terms: list[str],
        video_ids: set[str] | None = None,
    ) -> set[str] | None:
        # No metadata filter → return None so the agent considers all videos.
        return None

    def prefixes_to_video_ids(
        self,
        prefixes: list[str] | None,
        video_ids: set[str] | None = None,
    ) -> set[str] | None:
        # No prefix restriction → return None so the agent considers all videos.
        return None

    def filter_videos_by_metadata(
        self,
        terms: list[str],
        video_ids: set[str] | None = None,
    ) -> list[str]:
        return sorted(video_ids or [])

    def retrieve(
        self,
        text_embedding: np.ndarray,
        top_frames: int,
        max_answers: int,
        video_ids: set[str] | None = None,
    ) -> list[Candidate]:
        return [
            Candidate(
                video_id="L01_V001",
                frame_id=505,
                score=0.90,
                vector_id=7,
            )
        ]

    def retrieve_raw(
        self,
        text_embedding: np.ndarray,
        top_frames: int,
        video_ids: set[str] | None = None,
    ) -> list[Candidate]:
        return [
            Candidate(
                video_id="L01_V001",
                frame_id=505,
                score=0.90,
                vector_id=7,
            )
        ]

    def retrieve_trake(
        self,
        event_embeddings: np.ndarray,
        top_videos: int,
        prefilter_frames_per_event: int,
        penalty_weight: float,
        video_ids: set[str] | None = None,
        coarse_top_k: int = 200,
        object_adjustment=None,
        object_adjustment_matrices=None,
        preferred_prefixes: list[str] | None = None,
    ) -> list[Candidate]:
        assert event_embeddings.shape[0] == 3
        assert top_videos > 0
        assert prefilter_frames_per_event > 0
        assert penalty_weight >= 0

        return [
            Candidate(
                video_id="L21_V001",
                frame_id=200,
                score=0.92,
                vector_id=2,
                event_frames=[
                    100,
                    200,
                    300,
                ],
            )
        ]

    @staticmethod
    def _mask_video_ids(
        candidates: list[Candidate],
        video_ids: set[str] | None,
    ) -> list[Candidate]:
        if not video_ids:
            return candidates
        allowed = set(video_ids)
        return [c for c in candidates if c.video_id in allowed]

    # KIS / multi-query expansion paths (keep candidates intact for the test).
    def _rrf_fuse(
        self,
        ranked_lists: list[np.ndarray],
        k: int = 60,
    ) -> dict[int, float]:
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, vid in enumerate(ranked.tolist()):
                scores[vid] = scores.get(vid, 0.0) + 1.0 / (rank + 1)
        return scores

    def _candidates_from_scores(
        self,
        scores_by_manifest_idx: dict[int, float],
        limit: int | None = None,
    ) -> list[Candidate]:
        items = sorted(
            scores_by_manifest_idx.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        out = [
            Candidate(video_id="L01_V001", frame_id=i, score=s, vector_id=i)
            for i, s in items
        ]
        return out[:limit] if limit else out

    def search_with_filter(
        self,
        text_embedding: np.ndarray,
        top_frames: int,
        video_ids: set[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        ids = np.asarray([7, 1, 2], dtype=np.int64)
        scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
        return ids, scores

    def video_level_rerank(
        self,
        candidates: list[Candidate],
        top_videos: int | None = None,
        frames_per_video: int | None = None,
        aggregation_top_k: int = 3,
    ) -> list[Candidate]:
        return candidates


class BuggyLLM:
    """Simulates an LLM that throws an unexpected programming error.

    The deterministic pipeline calls the LLM only for translation and is
    expected to *swallow* that failure (falling back to the original text) so
    retrieval still runs — an LLM hiccup must never abort the whole pipeline.
    """

    def structured(
        self,
        system: str,
        user: str,
        schema: type,
    ):
        raise RuntimeError("programming bug")


def make_tools() -> RetrievalTools:
    return RetrievalTools(
        pipeline=FakePipeline(),
        encode_text=lambda _: np.asarray(
            [1.0],
            dtype=np.float32,
        ),
    )


def test_agent_can_only_return_retrieved_evidence() -> None:
    """The deterministic agent returns only candidates produced by retrieval."""
    agent = RetrievalAgent(
        make_tools(),
        llm=BuggyLLM(),
    )

    result = agent.run(
        Query(
            query_id="q-kis",
            type="kis",
            text="red speaker",
        )
    )

    assert [
        candidate.vector_id
        for candidate in result.candidates
    ] == [7]


def test_explicit_trake_events_override_planner_events() -> None:
    """TRAKE uses the query's explicit events, not any planner-supplied ones."""
    query = Query(
        query_id="q-trake",
        type="trake",
        text="A person enters, sits, and starts speaking.",
        events=[
            "The person enters the room.",
            "The person sits down.",
            "The person starts speaking.",
        ],
    )

    # The BuggyLLM would raise if the planner were still consulted; the
    # deterministic pipeline must not call it for TRAKE/KIS at all.
    agent = RetrievalAgent(
        make_tools(),
        llm=BuggyLLM(),
    )

    result = agent.run(query)

    assert result.plan.events == query.events
    assert len(result.candidates) == 1

    candidate = result.candidates[0]

    assert candidate.video_id == "L21_V001"
    assert candidate.event_frames == [
        100,
        200,
        300,
    ]
    assert len(candidate.event_frames) == len(
        query.events
    )


def test_plan_fallback_preserves_explicit_trake_events() -> None:
    """When nothing else is available, the plan keeps the query's events."""
    query = Query(
        query_id="q-fallback",
        type="trake",
        text="A three-event sequence.",
        events=[
            "Event one.",
            "Event two.",
            "Event three.",
        ],
    )

    agent = RetrievalAgent(
        make_tools(),
        llm=BuggyLLM(),
    )

    # The deterministic plan is built without an LLM; its events are the
    # query's events verbatim.
    result = agent.run(query)

    assert isinstance(result.plan, AgentPlan)
    assert result.plan.events == query.events
    assert result.plan.query_variants == [query.text]


def test_llm_failure_is_swallowed_not_aborting_pipeline() -> None:
    """An unexpected LLM/translation error must not abort retrieval.

    The deterministic pipeline is expected to catch translation failures and
    fall back to the original text, so a buggy LLM yields results rather than
    a propagated RuntimeError.
    """
    agent = RetrievalAgent(
        make_tools(),
        llm=BuggyLLM(),
    )

    query = Query(
        query_id="q-bug",
        type="kis",
        text="test query",
    )

    # Must NOT raise RuntimeError("programming bug").
    result = agent.run(query)

    assert result is not None
    assert [c.vector_id for c in result.candidates] == [7]


def test_openclip_reload_after_unload_keeps_encode_callable() -> None:
    """Regression for the real NoneType bug: OpenCLIPTextEmbedder.__init__ did
    NOT store self.model_name/self.pretrained, so load() raised AttributeError
    and reload_text_encoder() silently failed -- the next request's retrieve()
    then called encode() on a None model -> 'NoneType' object has no attribute
    'encode_text'. Confirm unload -> load -> encode works with the real encoder.
    """
    from aic2026.embeddings import OpenCLIPTextEmbedder

    enc = OpenCLIPTextEmbedder()
    before = enc.encode("a person speaking")
    assert before.shape[0] == 512
    enc.unload()
    assert enc.model is None
    enc.load()  # previously raised AttributeError: model_name
    assert enc.model is not None
    after = enc.encode("a person presenting")
    assert after.shape[0] == 512
    assert not (before == after).all()


def test_openclip_prompt_ensembling_normalized() -> None:
    """Prompt ensembling must return unit-norm vectors for both single and ensemble modes."""
    from aic2026.embeddings import OpenCLIPTextEmbedder

    enc = OpenCLIPTextEmbedder(use_ensemble=True)
    v_ens = enc.encode("a red car on the highway", ensemble=True)
    assert np.isclose(np.linalg.norm(v_ens), 1.0, atol=1e-5)

    v_single = enc.encode("a red car on the highway", ensemble=False)
    assert np.isclose(np.linalg.norm(v_single), 1.0, atol=1e-5)
    # Ensemble should be closely aligned with single, but smooth out template variance
    cos_sim = float(v_ens @ v_single)
    assert 0.85 < cos_sim <= 1.0001
