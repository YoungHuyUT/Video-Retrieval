"""Reranking lexical — RRF fusion, score normalization, and metadata-aware reranking.

This module implements the lexical/heuristic reranking layer of the retrieval
pipeline.  Its core primitive is **Reciprocal Rank Fusion (RRF)**, a scale-free
method for combining multiple ranked lists (vector search, BM25, late-interaction,
…) into a single consensus ranking.

Design notes
------------
* RRF is intentionally lightweight — no learned model, no LLM — so it can run
  over thousands of candidates in milliseconds.
* Scores from different retrievers live on different scales (RRF 0.0018–0.016,
  CLIP cosine 0.2–0.35, …).  Before blending, ``normalize_scores`` maps every
  candidate onto [0, 1] via **max-division** so magnitudes are comparable and
  no single signal silently dominates.
* ``rerank_with_metadata`` adds a *bounded* keyword-overlap bonus (match ratio
  × weight) on top of the normalized retrieval score, lifting frames whose
  labels or title literally mention query terms.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

import numpy as np

from aic2026.models import Candidate, FrameRecord


def _fold_accents(value: str) -> str:
    """Fold Unicode accents and case so Vietnamese matches are accent-insensitive.

    Applies NFC normalization, casefolds, transliterates ``đ/Đ`` (which Unicode
    NFD does not decompose), then strips combining marks.  This mirrors the
    behavior already used by ``SearchPipeline.filter_videos_by_metadata`` so
    that keyword-overlap matching works for accented Vietnamese queries.
    """
    value = unicodedata.normalize("NFC", value).casefold()
    value = value.translate(str.maketrans({"đ": "d"}))
    return "".join(
        c for c in unicodedata.normalize("NFD", value)
        if unicodedata.category(c) != "Mn"
    )

__all__ = [
    "RRF_K",
    "rrf_fuse",
    "rrf_rank",
    "minmax_normalize",
    "normalize_scores",
    "object_evidence_adjustment",
    "rerank_with_object_evidence",
    "rerank_with_metadata",
]


# Canonical object concepts used by the OpenImages / Faster R-CNN detector.
# The aliases make the query vocabulary forgiving: each concept lists BOTH its
# English detector label(s) (so a detected "Bottle" matches) AND common
# Vietnamese query words (so a query like "chai nước" lights up the concept).
# Without the Vietnamese aliases, a VN query never populated `requested`, so
# object evidence stayed off and CLIP alone decided the top frame.
_OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "person": ("person", "people", "human", "man", "woman", "boy", "girl", "child", "người", "đàn ông", "phụ nữ", "em bé", "trẻ em", "người đàn ông", "người phụ nữ"),
    "crowd": ("crowd", "đám đông", "nhóm người"),
    "turtle": ("turtle", "sea turtle", "tortoise", "rùa"),
    "fish": ("fish", "con cá", "cá"),
    "dog": ("dog", "puppy", "con chó", "chó"),
    "cat": ("cat", "kitten", "con mèo"),
    "car": ("car", "automobile", "vehicle", "xe hơi", "ô tô", "xe ô tô", "xe hơi"),
    "bicycle": ("bicycle", "bike", "xe đạp"),
    "motorcycle": ("motorcycle", "motorbike", "xe máy"),
    "bird": ("bird", "chim"),
    "horse": ("horse", "ngựa"),
    "elephant": ("elephant", "voi"),
    # --- Gia súc lớn: trâu / bò — detector (OpenImages V4) gán nhãn
    # "Bull" / "Cattle" / "Animal" cho khung hình trâu, nên alias phải chứa
    # những nhãn đó để lớp object-evidence rerank nhận diện được. Thiếu entry
    # này, query "buffalo"/"trâu" không kích hoạt object evidence → frame
    # bướm/chim (cosine ~0.287) đứng ngang hàng frame trâu (cosine ~0.30),
    # không bị đẩy xuống → "ảnh chả liên quan" lọt top.
    "buffalo": ("buffalo", "water buffalo", "buffle", "trâu", "con trâu", "bò đực", "bò rừng", "Bull", "Cattle"),
    "cow": ("cow", "cattle", "con bò", "bò"),
    # --- Động vật: mở rộng dựa trên nhãn detector THẬT có trong
    # data/processed/official_manifest.jsonl (513 nhãn). Mỗi entry map query
    # (EN + VI) sang nhãn OpenImages V4 thực tế trong manifest. Thiếu entry
    # → object-evidence rerank không kích hoạt → ảnh không liên quan lọt top
    # (lỗi y hệt "buffalo"). Đã đối chiếu tần suất nhãn thực tế:
    #   pig 24, goat 47, sheep 14, duck 73, chicken 45, rabbit 26, monkey 50,
    #   crab 170, shrimp 773, bee 210, insect 170, deer 41, camel 22, snake 14,
    #   frog 9, turtle/tortoise 30, lion 7, tiger 8, ant 7, butterfly 60,
    #   shellfish 252, goldfish 42, seahorse 9, antelope 11, starfish 1,
    #   jellyfish 80, beetle 5, spider 2, bear 1.
    "pig": ("pig", "con lợn", "con heo", "lợn", "heo"),
    "goat": ("goat", "con dê", "dê"),
    "sheep": ("sheep", "lamb", "con cừu", "cừu"),
    "duck": ("duck", "con vịt", "vịt"),
    "chicken": ("chicken", "con gà", "gà", "gà trống", "rooster"),
    "rabbit": ("rabbit", "bunny", "con thỏ", "thỏ"),
    "monkey": ("monkey", "con khỉ", "khỉ"),
    "crab": ("crab", "con cua", "cua"),
    "shrimp": ("shrimp", "con tôm", "tôm", "prawn"),
    "prawn": ("shrimp", "con tôm", "tôm", "prawn"),
    "bee": ("bee", "honey bee", "con ong", "ong"),
    "insect": ("insect", "con côn trùng", "côn trùng", "bug"),
    "butterfly": ("butterfly", "con bướm", "bướm"),
    "ant": ("ant", "con kiến", "kiến"),
    "beetle": ("beetle", "bọ cánh cứng", "bọ"),
    "spider": ("spider", "con nhện", "nhện"),
    "deer": ("deer", "con nai", "nai", "hart"),
    "antelope": ("antelope", "linh dương", "con linh dương"),
    "camel": ("camel", "con lạc đà", "lạc đà"),
    "snake": ("snake", "con rắn", "rắn"),
    "frog": ("frog", "con ếch", "ếch"),
    "turtle": ("turtle", "sea turtle", "tortoise", "con rùa", "rùa", "rùa biển"),
    "tortoise": ("turtle", "sea turtle", "tortoise", "con rùa", "rùa", "rùa biển"),
    "goldfish": ("goldfish", "cá vàng", "cá chép"),
    "seahorse": ("seahorse", "sea horse", "cá ngựa"),
    "starfish": ("starfish", "sao biển"),
    "jellyfish": ("jellyfish", "con sứa", "sứa"),
    "shellfish": ("shellfish", "động vật thân mềm", "nhuyễn thể"),
    "lion": ("lion", "con sư tử", "sư tử"),
    "tiger": ("tiger", "con hổ", "hổ"),
    "bear": ("bear", "con gấu", "gấu", "polar bear"),
    "mouse": ("mouse", "computer mouse", "con chuột", "chuột"),
    "bus": ("bus", "xe buýt"),
    "train": ("train", "tàu hỏa", "tàu"),
    "boat": ("boat", "ship", "tàu thuyền", "thuyền"),
    # --- Thêm: các vật thể phổ biến thường xuất trong query VN ---
    "bottle": ("bottle", "water bottle", "chai", "chai nước", "cốc nước", "lon"),
    "cup": ("cup", "cái ly", "ly nước", "cốc", "tách"),
    "backpack": ("backpack", "ba lô", "cặp"),
    "handbag": ("handbag", "túi xách", "túi"),
    "suitcase": ("suitcase", "vali"),
    "umbrella": ("umbrella", "cái ô", "ô dù"),
    "traffic light": ("traffic light", "traffic signal", "đèn tín hiệu", "đèn giao thông", "đèn đường"),
    "stop sign": ("stop sign", "biển báo", "biển báo stop"),
    "sign": ("sign", "biển", "biển hiệu", "bảng"),
    "book": ("book", "sách", "quyển sách"),
    "chair": ("chair", "cái ghế", "ghế"),
    "couch": ("couch", "sofa", "ghế sofa"),
    "bed": ("bed", "giường", "cái giường"),
    "dining table": ("dining table", "table", "cái bàn", "bàn ăn", "bàn làm việc"),
    "bench": ("bench", "ghế đá", "ghế dài"),
    "hand": ("hand", "bàn tay"),
    "clock": ("clock", "đồng hồ"),
    "cell phone": ("cell phone", "mobile phone", "phone", "điện thoại", "điện thoại di động"),
    "laptop": ("laptop", "máy tính xách tay", "máy tính"),
    "tv": ("tv", "television", "tivi"),
    "remote": ("remote", "điều khiển"),
    "keyboard": ("keyboard", "bàn phím"),
    "refrigerator": ("refrigerator", "fridge", "tủ lạnh"),
    "microwave": ("microwave", "lò vi sóng"),
    "oven": ("oven", "lò nướng"),
    "sink": ("sink", "bồn rửa", "chậu rửa"),
    "toilet": ("toilet", "bồn cầu"),
    "potted plant": ("potted plant", "plant", "chậu cây", "cây cảnh", "cái cây"),
    "sports ball": ("sports ball", "ball", "quả bóng", "bóng"),
    "teddy bear": ("teddy bear", "búp bê", "thú nhồi bông"),
    "toy": ("toy", "đồ chơi"),
    "skateboard": ("skateboard", "ván trượt"),
    "surfboard": ("surfboard", "ván lướt sóng"),
    "tennis racket": ("tennis racket", "vợt"),
    "wine glass": ("wine glass", "ly rượu"),
    "fork": ("fork", "cái nĩa", "nĩa", "dĩa"),
    "knife": ("knife", "con dao", "cái dao", "dao"),
    "spoon": ("spoon", "cái thìa", "thìa", "muỗng"),
    # Keep the Vietnamese phrase "cái tô" but omit bare "tô": accent-folding
    # turns it into English preposition "to" and falsely detects bowls in
    # relational queries such as "a woman next to a dog".
    "bowl": ("bowl", "cái bát", "bát", "cái tô"),
    "banana": ("banana", "chuối"),
    "apple": ("apple", "quả táo", "táo"),
    "orange": ("orange", "quả cam", "màu cam"),
    "broccoli": ("broccoli", "súp lơ"),
    "carrot": ("carrot", "cà rốt"),
    "pizza": ("pizza", "bánh pizza"),
    "cake": ("cake", "cái bánh", "bánh"),
    "sandwich": ("sandwich", "bánh mì"),
    "vegetable": ("vegetable", "rau", "rau củ", "đồ ăn", "thức ăn"),
    "stage": ("stage", "sân khấu"),
    "scissors": ("scissors", "cái kéo", "kéo"),
    # --- Đồ đeo trên người (thường xuất trong count/attribute query) ---
    "glasses": ("glasses", "eyeglasses", "spectacles", "kính", "kính mắt", "đeo kính"),
    "sunglasses": ("sunglasses", "shades", "kính râm"),
    "hat": ("hat", "cap", "mũ", "nón", "mũ bảo hiểm", "helmet", "nón bảo hiểm"),
    "helmet": ("helmet", "mũ bảo hiểm", "nón bảo hiểm"),
    "tie": ("tie", "cà vạt", "cravat"),
    "glove": ("glove", "găng tay", "găng"),
    "shoe": ("shoe", "giày", "footwear"),
    "áo dài": ("áo dài", "ao dai", "aodai", "aodai", "vietnamese dress", "traditional dress", "truyền thống", "trang phục"),
    # --- Màu sắc (dùng cho constraint verification + object evidence) ---
    "red": ("red", "đỏ", "màu đỏ"),
    "pink": ("pink", "hồng", "màu hồng", "rose", "hường"),
    "blue": ("blue", "xanh", "xanh dương", "xanh lam", "màu xanh"),
    "white": ("white", "trắng", "màu trắng"),
    "black": ("black", "đen", "màu đen"),
    "green": ("green", "xanh lá", "màu xanh lá"),
    "yellow": ("yellow", "vàng", "màu vàng"),
    "purple": ("purple", "tím", "màu tím"),
    "orange color": ("orange", "cam", "màu cam"),
    # --- Trang phục / quần áo ---
    "shirt": ("shirt", "áo", "áo sơ mi", "tshirt", "áo thun"),
    "dress": ("dress", "váy", "cái váy", "váy đầm"),
    "clothes": ("clothes", "quần áo", "trang phục", "trang phục"),
    "skirt": ("skirt", "váy", "chân váy"),
    "pants": ("pants", "quần", "quần dài", "trousers"),
}


def _has_phrase(text: str, phrase: str) -> bool:
    # Word-boundary match so "ca" does NOT match inside "cam" / "cá".  The
    # original code used a raw string r"(?<!\w)" which is literally backslash-w
    # (not the \w class), so the boundary was a no-op and every substring matched
    # ("ca" matched "cam", "o" matched everywhere) — breaking object concept
    # detection for Vietnamese queries.
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))


def _alias_matches(text_folded: str, alias: str) -> bool:
    """Match một alias (đã accent-fold) trong text đã fold.

    Ưu tiên word-boundary (khớp từ nguyên, không lẫn trong từ dài) để:
      - ``"ant"`` KHÔNG khớp ``"elephant"``,
      - ``"bee"`` KHÔNG khớp ``"beetle"``,
      - ``"man"`` KHÔNG khớp ``"woman"`` / ``"human"`` (sửa false-positive cũ).
    Đây là sửa quan trọng: trước đây alias 1-từ dùng substring, nên thêm alias
    ngắn ("ant") sẽ khớp nhầm nhãn dài ("elephant") → sinh bug mới khi mở rộng
    bảng ``_OBJECT_ALIASES``.

    Bridge cả hai chiều khoảng trắng cho alias viết liền ("aodai") và label
    viết liền ("Aodai"): alias có dấu cách thì thử word-boundary chuẩn, SAU
    ĐÓ thử bản bỏ mọi khoảng trắng để bridge sang label liền.
    """
    alias_folded = _fold_accents(alias)
    # Word-boundary luôn an toàn và bắt được cả label viết liền (vd "aodai"
    # là một token trong text đã fold).
    if _has_phrase(text_folded, alias_folded):
        return True
    if " " in alias_folded:
        # alias đa từ ("áo dài") → bridge sang label liền ("aodai"), but
        # require the joined label to be a whole token. Substring matching here
        # made "em bé" (baby/person) spuriously match inside "remember".
        joined = alias_folded.replace(" ", "")
        return joined in text_folded.split()
    # alias 1-từ: KHÔNG fallback substring (tránh ant⊂elephant). Word-boundary
    # ở trên đã bắt được mọi trường hợp hợp lệ (kể cả label viết liền).
    return False


# Animal concepts that benefit from stronger object-evidence penalty.
# CLIP embedding space groups all animals into a broad "animal" manifold, so
# "buffalo" scores near cow, goat, deer, etc.  Boosting the penalty for these
# concepts ensures wrong-animal frames are pushed down harder than for objects.
_ANIMAL_CONCEPTS: frozenset[str] = frozenset({
    "buffalo", "cow", "goat", "sheep", "pig", "horse", "elephant",
    "deer", "dog", "cat", "bird", "snake", "rabbit", "monkey",
    "lion", "tiger", "bear", "frog", "turtle", "duck", "chicken",
    "antelope", "camel", "crab", "shrimp", "fish", "butterfly",
    "bee", "insect", "beetle", "spider", "goldfish", "seahorse",
    "starfish", "jellyfish", "shellfish", "animal",
})
_GENERIC_ANIMAL_ALIASES: tuple[str, ...] = tuple(dict.fromkeys(
    alias
    for concept, aliases in _OBJECT_ALIASES.items()
    if concept in _ANIMAL_CONCEPTS - {"animal"}
    for alias in aliases
))


def _concept_aliases(concept: str) -> tuple[str, ...]:
    if concept == "animal":
        return ("animal", "animals", "creature", "creatures", "wildlife", "con vật", "động vật", *_GENERIC_ANIMAL_ALIASES)
    return _OBJECT_ALIASES.get(concept, ())


_COLOR_CONCEPTS: frozenset[str] = frozenset({
    "red", "pink", "blue", "white", "black", "green", "yellow", "purple", "orange color",
})


def is_animal_query(query: str) -> bool:
    """True if the query mentions an animal concept from _OBJECT_ALIASES."""
    concepts = _requested_concepts(query)
    return bool(concepts & _ANIMAL_CONCEPTS)


def _requested_concepts(query: str) -> set[str]:
    """Concepts the query asks for, per ``_OBJECT_ALIASES`` (empty if none).

    Uses ``vn_synonyms.normalize_entity`` to expand Vietnamese entities
    to English detector labels, so VN queries match EN detector output
    even when the VN term is not in ``_OBJECT_ALIASES`` directly.
    """
    query_text = _fold_accents(query)
    # Fold aliases too: the query is accent-folded, so a Vietnamese alias like
    # "xe máy" must be folded to "xe may" before matching, otherwise it never
    # fires and object evidence stays off for VN queries.
    concepts = {
        concept
        for concept, aliases in _OBJECT_ALIASES.items()
        if any(_alias_matches(query_text, alias) for alias in aliases)
    }

    # Improvement.md Task 5: expand VN entities via vn_synonyms
    # Also check VN→EN mapping for entities not in _OBJECT_ALIASES directly
    try:
        from aic2026.reranking.vn_synonyms import normalize_entity
        # Extract potential VN words from query (2+ char words)
        words = query_text.split()
        for word in words:
            if len(word) >= 2:
                en_labels = normalize_entity(word)
                for label in en_labels:
                    label_folded = _fold_accents(label)
                    for concept, aliases in _OBJECT_ALIASES.items():
                        if any(_alias_matches(label_folded, a) for a in aliases):
                            concepts.add(concept)
    except ImportError:
        pass

    # Treat an explicitly generic animal request as the union of known animal
    # detector labels, but do not weaken a named-species request into that union.
    generic_animal = any(
        _has_phrase(query_text, phrase)
        for phrase in ("animal", "animals", "creature", "creatures", "wildlife", "con vat", "dong vat")
    )
    if generic_animal and not (concepts & (_ANIMAL_CONCEPTS - {"animal"})):
        concepts.add("animal")

    return concepts


def object_evidence_adjustment(
    query: str,
    record: FrameRecord,
    weight: float = 0.03,
    penalty_scale: float = 2.0,
) -> float | None:
    """Return a signed object-evidence adjustment, or ``None`` if no object is asked.

    A full object match gets ``+weight`` and a frame missing every requested
    object gets ``-weight * penalty_scale``.  The penalty is amplified (default
    ``penalty_scale = 2.0``) so that a frame the detector clearly shows lacking
    the asked object is pushed down harder than a matching frame is lifted — this
    directly counters the failure mode where CLIP ranks a frame high for the
    right *scene* but the wrong *object* (e.g. a supermarket aisle with no water
    bottle).  It remains a soft penalty: Faster R-CNN can miss small/occluded
    objects, so vector evidence is never discarded outright.

    **Generic "Animal" penalty:** When a frame carries the generic "Animal" label
    but NO specific species label (e.g. "Bull", "Goat"), and the query asks for
    a specific animal, the frame gets a *half* penalty (``-weight * penalty_scale
    * 0.5``).  This pushes down the 1200+ frames labeled only "Animal" that CLIP
    would otherwise rank high for any animal query — they are "unidentified
    animals" and should not compete with frames that have the correct species
    label.
    """
    requested = _requested_concepts_cached(query)
    if not requested:
        return None

    labels = _folded_labels_for_record(record)
    matched = sum(
        any(_alias_matches(labels, alias) for alias in _concept_aliases(concept))
        for concept in requested
    )
    coverage = matched / len(requested)
    # Symmetrical-ish but asymmetric: full miss penalized harder than full match rewarded.
    if coverage == 0.0:
        # Full penalty: when the query asks for a specific animal,
        # an unlabeled "Animal" frame is unlikely to be the right answer.
        return -weight * penalty_scale
    return weight * (2.0 * coverage - 1.0)


# ---------------------------------------------------------------------------
# Vectorized object-evidence adjustment for TRAKE
# ---------------------------------------------------------------------------
# The legacy path applied object evidence via a per-(event, frame) Python
# closure inside ``RetrievalPipeline.retrieve_trake``. That closure re-derived
# the requested concepts (a 70-concept alias scan) and re-folded the frame's
# object labels on EVERY one of the ~40k calls per query — measured at ~145s
# for a single 3-event TRAKE query. The helpers below precompute the same
# adjustment as a per-video NumPy matrix so the pipeline only needs a cheap
# matrix addition per candidate video.

_REQUESTED_CONCEPT_CACHE: dict[str, frozenset] = {}
_FOLDED_LABEL_CACHE: dict[int, str] = {}


def gate_lion_dance_split(query: str, candidates: list[Candidate]) -> list[Candidate]:
    """Exclude the L24 lion-dance split unless the user explicitly says ``qilin``.

    L24 is visually repetitive and its lion costumes are a high-similarity
    attractor for broad CLIP prompts (people, colour, outdoor scenes).  It must
    therefore never act as a dataset prior.  ``qilin`` is the intentional,
    case-insensitive opt-in codeword requested by the operator; all other
    queries retain every non-L24 candidate in their original order.
    """
    if re.search(r"(?<![a-z0-9])qilin(?![a-z0-9])", _fold_accents(query)):
        return candidates
    return [candidate for candidate in candidates if not candidate.video_id.upper().startswith("L24_")]


def _requested_concepts_cached(query: str) -> frozenset:
    """Cached ``_requested_concepts`` — the per-event scan is identical across
    all frames of a video, so it must be computed ONCE, not 40k times."""

    cached = _REQUESTED_CONCEPT_CACHE.get(query)
    if cached is None:
        cached = frozenset(_requested_concepts(query))
        _REQUESTED_CONCEPT_CACHE[query] = cached
    return cached


def _folded_labels_for_record(record: FrameRecord) -> str:
    """Cached accent-folded, space-joined object labels for one frame."""

    key = record.vector_id
    if key is not None and key in _FOLDED_LABEL_CACHE:
        return _FOLDED_LABEL_CACHE[key]
    folded = _fold_accents(" ".join(record.object_labels or []))
    if key is not None:
        _FOLDED_LABEL_CACHE[key] = folded
    return folded


def filter_candidates_by_requested_objects(
    query: str,
    candidates: list[Candidate],
    records_by_id: dict[int, FrameRecord],
    video_metadata=None,
) -> list[Candidate]:
    """Keep only candidates whose detector labels contain a requested object.

    Queries without a recognized object concept pass through unchanged. For a
    multi-object phrase, every requested concept must be detected in the frame;
    the remaining dense/lexical ranking decides order.
    """
    requested = _requested_concepts_cached(query) - _COLOR_CONCEPTS
    if not requested:
        return candidates

    # In this corpus, lion-dance masks/costumes in L24 receive false-positive
    # ``Dog`` labels from the object detector. A literal label alone is not
    # enough evidence for a dog query there. Use video metadata as a negative
    # context cue, unless the query explicitly asks about lion dance.
    query_folded = _fold_accents(query)
    asks_about_lion_dance = any(
        _has_phrase(query_folded, phrase)
        for phrase in (
            "lan", "mua rong", "mua lan", "lan su rong",
            "lion dance", "dragon dance", "mai hoa thung",
        )
    )
    exclude_lion_dance_dogs = "dog" in requested and not asks_about_lion_dance

    def is_lion_dance_video(video_id: str) -> bool:
        if not exclude_lion_dance_dogs:
            return False
        # The local BTC metadata confirms L24 is the lion-dance competition
        # split; keep this explicit fallback for manifests with no video-level
        # metadata loaded.
        if video_id.upper().startswith("L24_"):
            return True
        if video_metadata is None:
            return False
        meta = video_metadata.get(video_id)
        if meta is None:
            return False
        haystack = _fold_accents(" ".join([
            meta.title or "",
            meta.description or "",
            *(meta.metadata_keywords or []),
        ]))
        return any(
            _has_phrase(haystack, phrase)
            for phrase in (
                "lan su rong", "lansurong", "mua lan", "mua rong",
                "doan lan", "lion dance", "dragon dance", "mai hoa thung",
                "cup cho lon", "cho lon htv",
            )
        )

    # A single-scene query requires all named objects in one frame. For an
    # ordered multi-event query, each frame may satisfy one event's entities;
    # event coverage then scores how well a video covers the full sequence.
    groups = [requested]
    try:
        from aic2026.query.parser import parse_query
        plan = parse_query(query)
        if len(plan.events) > 1:
            event_groups = [
                set(event.entities) - _COLOR_CONCEPTS
                for event in plan.events
            ]
            groups = [group for group in event_groups if group] or groups
    except Exception:
        pass
    result = []
    for candidate in candidates:
        if is_lion_dance_video(candidate.video_id):
            continue
        record = records_by_id.get(candidate.vector_id)
        if record is None:
            continue
        labels = _folded_labels_for_record(record)
        if any(all(
            any(_alias_matches(labels, alias) for alias in _concept_aliases(concept))
            for concept in group
        ) for group in groups):
            result.append(candidate)
    return result


def build_object_adjustment_matrices(
    events: list[str],
    candidate_videos: set[str],
    video_to_manifest_indices: dict[str, np.ndarray],
    manifest: list[FrameRecord],
    weight: float = 0.08,
    penalty_scale: float = 2.0,
) -> dict[str, np.ndarray]:
    """Build a vectorized object-evidence adjustment matrix per candidate video.

    Returns ``{video_id: np.ndarray(shape=(len(events), F), dtype=float32)}``
    where ``F`` is that video's frame count, aligned with the per-video manifest
    order used by ``RetrievalPipeline.retrieve_trake``. Each cell equals the
    signed adjustment ``object_evidence_adjustment`` would have produced for that
    (event, frame) pair, so the pipeline can add it to ``similarity_matrix`` in
    one NumPy op instead of calling Python 40k times.

    Only the coarse-filtered ``candidate_videos`` are processed, and each
    frame's labels are folded once and each concept's membership is computed
    once — collapsing the dominant TRAKE cost from ~145s to a few seconds.
    """

    if not events or not candidate_videos:
        return {}

    event_requested = [_requested_concepts_cached(event) for event in events]

    # Union of all requested concepts across events, with pre-folded aliases.
    union: dict[str, list[str]] = {}
    for requested in event_requested:
        for concept in requested:
            if concept not in union:
                union[concept] = [
                    _fold_accents(alias) for alias in _concept_aliases(concept)
                ]

    matrices: dict[str, np.ndarray] = {}

    for video_id in candidate_videos:
        manifest_indices = video_to_manifest_indices.get(video_id)
        if manifest_indices is None or len(manifest_indices) == 0:
            continue

        frame_count = int(len(manifest_indices))

        # Fold each frame's labels once.
        folded_labels = [
            _folded_labels_for_record(manifest[int(index)])
            for index in manifest_indices
        ]

        # Per-concept membership over the video's frames (frame_count booleans).
        concept_present: dict[str, np.ndarray] = {}
        for concept, aliases in union.items():
            if not aliases:
                continue
            present = np.zeros(frame_count, dtype=np.float32)
            for frame_position in range(frame_count):
                text = folded_labels[frame_position]
                if text and any(
                    alias and _alias_matches(text, alias) for alias in aliases
                ):
                    present[frame_position] = 1.0
            concept_present[concept] = present

        coverage = np.zeros((len(events), frame_count), dtype=np.float32)
        for event_index, requested in enumerate(event_requested):
            if not requested:
                continue
            column = np.zeros(frame_count, dtype=np.float32)
            for concept in requested:
                column += concept_present.get(
                    concept, np.zeros(frame_count, dtype=np.float32)
                )
            coverage[event_index] = column / len(requested)

        adjustment = np.where(
            coverage == 0.0,
            -weight * penalty_scale,
            weight * (2.0 * coverage - 1.0),
        ).astype(np.float32)
        matrices[video_id] = adjustment

    return matrices


def rerank_with_object_evidence(
    query: str,
    candidates: list[Candidate],
    records: dict[int, FrameRecord],
    weight: float = 0.03,
    penalty_scale: float = 2.0,
    drop_empty_object_frames: bool = False,
) -> list[Candidate]:
    """Apply signed Object-detection evidence after vector/BM25 retrieval.

    When ``drop_empty_object_frames`` is True and the query actually asks for an
    object (``object_evidence_adjustment`` returns a non-None *requested* set),
    any candidate whose frame has NO object labels at all (i.e. the ingest step
    marked it as a blurry / no-clear-object frame — no entity reached the 0.4
    present-threshold) is dropped entirely.  This is the hard filter for the
    "blurry frame" case: such frames never enter the final KIS ranking.  It only
    fires when the query asks for an object, so pure scene queries are unaffected.
    If dropping would emptying the whole pool, we fall back to keeping the
    original candidates (avoid returning zero answers).

    **Animal-query auto-boost:** CLIP maps all animals into a broad "animal"
    manifold, so "buffalo" scores near cow, goat, deer.  When the query mentions
    an animal concept, the penalty for missing objects is automatically increased
    (3× instead of 2×) to push wrong-animal frames down harder.
    """
    requested = _requested_concepts_cached(query)
    drop_mode = drop_empty_object_frames and bool(requested)

    # Auto-boost penalty for animal queries: CLIP confuses similar animals,
    # so wrong-animal frames need a stronger push-down.
    if requested & _ANIMAL_CONCEPTS:
        penalty_scale = max(penalty_scale, 4.0)
        weight = max(weight, 0.12)

    kept: list[Candidate] = []
    for item in candidates:
        if drop_mode and item.vector_id is not None:
            record = records.get(item.vector_id)
            # A frame with no object labels while the query asks for an object
            # is the blurry / no-clear-object case -> drop it.
            if record is not None and not (record.object_labels or []):
                continue
        kept.append(item)

    # Fallback: never return an empty pool just because every frame was blurry.
    if drop_mode and not kept:
        kept = list(candidates)

    reranked: list[Candidate] = []
    for item in kept:
        adjustment = None
        if item.vector_id is not None:
            record = records.get(item.vector_id)
            if record is not None:
                adjustment = object_evidence_adjustment(query, record, weight, penalty_scale)
        score = float(item.score) if adjustment is None else float(item.score) + adjustment
        reranked.append(item.model_copy(update={"score": score}))
    return sorted(reranked, key=lambda candidate: candidate.score, reverse=True)

# --- RRF (Reciprocal Rank Fusion) -------------------------------------------------

_RRF_K = 60  # standard constant from reciprocal-rank-fusion literature
RRF_K = _RRF_K  # public alias re-exported via ``__all__``


def rrf_fuse(
    ranked_lists: list[list[int]],
    k: int = _RRF_K,
    weights: list[float] | None = None,
) -> dict[int, float]:
    """Fuse multiple ranked lists using **Reciprocal Rank Fusion**.

    Given several ranked lists of manifest indices (each list from a different
    retrieval method — e.g. vector search, BM25, late interaction), RRF combines
    them by summing reciprocal ranks:

        score(idx) = Σᵢ  weightᵢ / (k + rankᵢ(idx) + 1)

    where *rankᵢ(idx)* is the 0-based position of manifest index ``idx`` in
    list *i* (indices absent from a list contribute 0 to that list's term).

    When ``weights`` is ``None`` (default), all lists get weight 1.0 (standard RRF).
    When ``weights`` is provided, each list ``i`` is scaled by ``weights[i]``.
    This supports weighted expansion variants (Improvement.md Task 4): more
    relevant variants get higher weight in the fusion.

    Parameters
    ----------
    ranked_lists
        A list of ranked manifest-index lists.  Each inner list should be
        ordered from most-relevant to least-relevant for its own retriever.
    k
        The RRF constant that dampens the influence of rank position.
    weights
        Optional per-list weights.  ``None`` = uniform 1.0.  If provided,
        must have the same length as ``ranked_lists``.

    Returns
    -------
    dict[int, float]
        Mapping from manifest index to its fused RRF score.
    """
    fused_scores: dict[int, float] = {}

    for i, ids in enumerate(ranked_lists):
        w = weights[i] if weights is not None and i < len(weights) else 1.0
        for rank, idx in enumerate(ids):
            contribution = w / (k + rank + 1)
            fused_scores[idx] = fused_scores.get(idx, 0.0) + contribution

    return fused_scores


def minmax_normalize(
    values: list[float] | np.ndarray,
    epsilon: float = 1e-9,
) -> list[float]:
    """Min–max normalization (paper Eq. 1) rescaling onto [0, 1].

    Unlike :func:`normalize_scores` (max-division), min–max maps the lowest
    score to 0 and the highest to 1, preserving intra-list ranking.  The paper
    uses this before adaptive modality fusion (Eq. 2) so heterogeneous scores
    (CLIP cosine, OCR BM25, ASR) become comparable on one scale.

    A degenerate all-equal list returns all 1.0 (no division by zero).
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < epsilon:
        return [1.0 for _ in arr]
    return list((arr - lo) / (hi - lo))


def adaptive_modality_fusion(
    modality_ids: dict[str, list[int]],
    modality_scores: dict[str, list[float]],
    weights: dict[str, float],
    epsilon: float = 1e-9,
    renormalize: bool = True,
) -> dict[int, float]:
    """Adaptive multi-modal score fusion — paper Eq.1 (min–max) + Eq.2 (weighted sum).

        s_norm_m(f) = (s_m(f) − min(s_m)) / (max(s_m) − min(s_m) + ε)      (Eq.1)
        S(f)        = Σ_{m ∈ active} w_m · s_norm_m(f)                     (Eq.2a)
                    = Σ_{m ∈ active} w_m · s_norm_m(f) / Σ_{m ∈ active} w_m   (Eq.2b, renormalized)

    Each modality (``visual`` / ``ocr`` / ``asr`` / ``object`` / ``count``)
    contributes a *ranked* list of manifest indices (most→least relevant) with
    parallel similarity scores.  Per-modality scores live on different scales
    (CLIP cosine ≈0.2–0.35, BM25-OCR ≈0–1, object-evidence ≈−0.1–0.1), so each
    is min–max normalized onto [0, 1] (Eq.1, using the existing
    :func:`minmax_normalize`) before the weighted sum (Eq.2).  ``w_m`` are the
    per-modality weights (from either the query planner or a config default) —
    a query about visible text gets ``ocr`` weight ≈1 and ``asr`` ≈0, so only
    the discriminative modality drives the ranking.

    Two failure modes are handled so a *missing* modality never poisons the
    ranking:

    * **Disabled modality** (``w_m <= 0``): skipped entirely (no spurious signal).
    * **Unavailable modality** (``w_m > 0`` but not present in ``modality_ids`` —
      e.g. ASR scores when no ASR sidecar was loaded): its weight is dropped from
      the normalization denominator.  With ``renormalize=True`` (default) the
      final score is divided by the sum of the *actually-used* weights
      (Eq.2b), so the remaining modalities still span the full [0, 1] range
      instead of being squashed into a narrow sub-band.  With ``renormalize=False``
      the raw weighted sum (Eq.2a) is returned, matching legacy behaviour where
      every configured modality was always present.

    A manifest index absent from every active modality gets ``S(f) = 0`` and is
    dropped from the result.

    Returns ``{manifest_idx: S(f)}`` over the union of all active-modality indices.
    The caller (``RetrievalTools.retrieve``) re-scores candidates from this map;
    indices outside it keep their previous score, so recall is never lost.

    This is the paper's *adaptive score fusion* that replaces the equal-weight
    RRF/SRRF used elsewhere — it is opt-in (off by default) so the legacy
    RRF path stays the A/B baseline until benchmarked.
    """

    if not weights:
        return {}

    # Only modalities the caller predicted as discriminative (w_m > 0) AND that
    # actually contributed a ranked list are "active".  A modality configured
    # with w_m > 0 but absent from modality_ids is treated as missing — its
    # weight is simply not counted in the normalization denominator below.
    active = [
        m for m in modality_ids
        if weights.get(m, 0.0) > 0.0 and (modality_ids.get(m) or [])
    ]
    if not active:
        return {}

    active_weight_sum = float(sum(weights[m] for m in active))
    if active_weight_sum <= 0:
        return {}

    fused: dict[int, float] = {}
    for m in active:
        ids = modality_ids.get(m) or []
        scores = modality_scores.get(m) or []
        if len(ids) != len(scores):
            raise ValueError(
                f"modality {m!r}: ids ({len(ids)}) != scores ({len(scores)})"
            )
        if not ids:
            continue
        w_m = float(weights[m])
        s_norm = minmax_normalize(scores, epsilon=epsilon)
        for idx, sn in zip(ids, s_norm):
            manifest_idx = int(idx)
            fused[manifest_idx] = fused.get(manifest_idx, 0.0) + w_m * sn

    # Eq.2b: renormalize by the sum of actually-used weights so a missing
    # modality (e.g. ASR off) does not compress the score range.  When every
    # configured modality is present this is a no-op (divisor == 1.0-equivalent
    # because the raw sum already equals the renormalized value after the
    # division by a full-weight sum of 1.0-equivalent scale — see tests).
    if renormalize and active_weight_sum != 1.0:
        for idx in fused:
            fused[idx] = fused[idx] / active_weight_sum
    return fused


def rrf_rank(
    fused_scores: dict[int, float],
    manifest: list[FrameRecord],
    top_n: int = 100,
) -> list[Candidate]:
    """Convert fused RRF scores into a globally ranked :class:`Candidate` list.

    Parameters
    ----------
    fused_scores
        Output from :func:`rrf_fuse`.
    manifest
        The full manifest of :class:`FrameRecord` so we can look up
        ``video_id``, ``frame_id``, ``keyframe_path``, etc.
    top_n
        How many top candidates to return.

    Returns
    -------
    list[Candidate]
        Globally ranked candidates sorted by RRF score (descending).  Each
        ``Candidate`` carries the RRF score as its ``score`` field.
    """
    if not fused_scores:
        return []

    # Rank manifest indices by fused RRF score (highest first).
    sorted_indices = sorted(
        fused_scores, key=fused_scores.get, reverse=True  # type: ignore[arg-type]
    )[:top_n]

    candidates: list[Candidate] = []
    for manifest_idx in sorted_indices:
        record = manifest[manifest_idx]
        candidates.append(
            Candidate(
                video_id=record.video_id,
                frame_id=record.frame_id,
                score=float(fused_scores[manifest_idx]),
                vector_id=record.vector_id,
                keyframe_path=record.keyframe_path,
            )
        )

    return candidates


# --- Metadata-aware reranking ----------------------------------------------------


def normalize_scores(candidates: list[Candidate]) -> list[float]:
    """Normalize candidate scores into [0, 1] via **max-division**.

    RRF (≈0.0018–0.016), CLIP cosine (≈0.2–0.35), metadata bonus, and video-level
    boosts all live on very different scales.  Before blending any of them we
    rescale to a common [0, 1] range so no single signal silently dominates the
    others.

    We use **max-division** (not min-max) deliberately:

    * It preserves the *magnitude* of the retrieval signal — a strong CLIP
      match (0.34) stays meaningfully higher than a weak one (0.20).
    * Min-max would amplify a tiny 9 % RRF lead into a 100 % gap, letting a
      strong CLIP match dominate even when metadata clearly matches a better
      frame.  Max-division keeps relative magnitudes, so a metadata match can
      nudge mid-ranked frames up without overriding genuinely strong retrieval
      results.

    Parameters
    ----------
    candidates
        List of candidates whose ``score`` attribute holds the raw retrieval
        score (RRF, CLIP, or fused).

    Returns
    -------
    list[float]
        Normalized scores in [0, 1], one per candidate, preserving input order.
    """
    if not candidates:
        return []

    raw = np.asarray([float(c.score) for c in candidates], dtype=np.float64)
    hi = float(raw.max())

    if hi < 1e-9:
        # Degenerate pool: all scores are ~0 — return zeros.
        return [0.0 for _ in candidates]

    return list(raw / hi)


# --- Normalization abstraction (spec §15) ---------------------------------
#
# Spec §15 asks for a single ``normalize_score()`` entry point whose strategy is
# config-driven and A/B-able BEFORE we commit to one scheme.  We keep min-max as
# the default baseline (used by Adaptive Fusion) but expose percentile and
# sigmoid/temperature variants for benchmarking.  No behaviour changes until a
# caller opts into a non-default strategy.
NormalizationStrategy = Literal["minmax", "percentile", "sigmoid", "maxdiv"]


def normalize_score(
    values: list[float] | np.ndarray,
    strategy: NormalizationStrategy = "minmax",
    *,
    percentile: float = 90.0,
    temperature: float = 1.0,
    epsilon: float = 1e-9,
) -> list[float]:
    """Config-driven score normalization onto ~[0, 1] (spec §15).

    Strategies
    -----------
    * ``"minmax"``     — (x - min) / (max - min + ε); the paper's Eq.1, used by
      Adaptive Fusion.  Preserves full dynamic range (worst→best = 0→1).
    * ``"percentile"`` — divide by the ``percentile``-th percentile (default 90)
      of the distribution, then clip to [0, 1].  Robust to a few outliers
      inflating the max (e.g. one CLIP 0.35 vs many 0.22s).
    * ``"sigmoid"``    — 1 / (1 + exp(-x / temperature)); maps any real scale to
      (0, 1) with a tunable steepness.  Good when raw scores are already signed.
    * ``"maxdiv"``     — x / max; the magnitude-preserving max-division used by
      :func:`normalize_scores` (a strong match stays clearly above a weak one).

    All strategies are pure functions of *values* (no candidate/side-effect
    dependency) so they are trivially unit-testable and cacheable.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []

    if strategy == "maxdiv":
        hi = float(arr.max())
        if hi < epsilon:
            return [0.0 for _ in arr]
        return list(arr / hi)

    if strategy == "minmax":
        lo = float(arr.min())
        hi = float(arr.max())
        if hi - lo < epsilon:
            return [1.0 for _ in arr]
        return list((arr - lo) / (hi - lo))

    if strategy == "percentile":
        if arr.size < 2:
            return [1.0 for _ in arr]
        p = float(np.percentile(arr, max(0.0, min(100.0, percentile))))
        if p < epsilon:
            return [1.0 for _ in arr]
        return list(np.clip(arr / p, 0.0, 1.0))

    if strategy == "sigmoid":
        t = temperature if temperature > epsilon else 1.0
        return list(1.0 / (1.0 + np.exp(-arr / t)))

    # Unknown strategy → fall back to min-max so callers never crash.
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < epsilon:
        return [1.0 for _ in arr]
    return list((arr - lo) / (hi - lo))


__all__ = [
    "RRF_K",
    "rrf_fuse",
    "rrf_rank",
    "minmax_normalize",
    "normalize_score",
    "normalize_scores",
    "object_evidence_adjustment",
    "rerank_with_object_evidence",
    "rerank_with_metadata",
]



def rerank_with_metadata(
    query: str,
    candidates: list[Candidate],
    records: dict[int, FrameRecord],
    weight: float = 0.2,
) -> list[Candidate]:
    """Add a bounded keyword-overlap bonus to candidate scores.

    The previous version added a fixed ``0.05`` per matched token directly onto
    the raw retrieval score.  Because RRF scores are only ~0.0018–0.016, a single
    matched object label (0.05) could outweigh the *entire* RRF ranking and
    catapult low-quality frames to the top.  This version instead:

    1. Expresses the bonus as a **match ratio** (matched_query_terms /
       total_query_terms) so it is bounded to [0, 1] and proportional to how
       much of the query the metadata actually covers.
    2. **Normalizes** incoming retrieval scores into [0, 1] (via max-division)
       first, so the metadata bonus is added on the same scale as the rest of
       the pipeline (RRF / CLIP / video-level).
    3. Sorts by the blended score, lifting frames whose labels/titles literally
       mention the query terms above pure vector matches — but now without
       drowning out the retrieval signal.

    Each candidate is **copied** (``model_copy``) before its score is updated so
    the original retrieval scores are preserved for downstream stages.

    Parameters
    ----------
    query
        The user query string.  Tokenized on whitespace; each token is matched
        as a substring against the frame's metadata.
    candidates
        Retrieval candidates (will NOT be mutated in place).
    records
        Mapping of ``vector_id → FrameRecord`` providing ``object_labels`` (entity
        EN) and ``metadata_keywords`` (keyword VN) for keyword matching.
    weight
        Weight of the metadata bonus on the normalized [0, 1] scale.
        Default ``0.2`` keeps the bonus bounded and subdominant.

    Returns
    -------
    list[Candidate]
        New list of candidates sorted by blended score (descending).  The
        original ``candidates`` list and its elements are left untouched.
    """
    # Degenerate cases: no candidates or bonus disabled → keep RRF order.
    if not candidates or weight <= 0:
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    # Tokenize the query once; match ratio = matched_terms / total_terms.
    query_terms = [t for t in query.lower().split() if t]
    term_set = set(query_terms)
    folded_terms = set(_fold_accents(t) for t in query_terms)

    # Direction B: do NOT normalize scores to [0,1]. Normalizing (dividing by the
    # pool max) collapsed the RRF dynamic range to ~1.0 for the top frame of every
    # video, which washed out the genuine RRF ordering and let the metadata bonus
    # (a fixed absolute add) dominate — breaking retrieval vs plain RRF. Instead we
    # keep the RRF score as the base and add only a SMALL bounded bonus on top, so
    # the metadata overlap acts as a gentle tie-breaker, not a re-ranking.
    reranked: list[Candidate] = []
    for item in candidates:
        blended = float(item.score)

        # Add bounded keyword-overlap bonus if metadata is available.
        if term_set and item.vector_id is not None:
            record = records.get(item.vector_id)
            if record is not None:
                # Aggregate all metadata text to search: object_labels (EN entity)
                # + metadata_keywords (VN keyword tóm tắt video). title/description
                # đã bỏ (dư thừa — keywords đã summarize).
                parts: list[str] = list(record.object_labels or [])
                if record.metadata_keywords:
                    parts.extend(record.metadata_keywords)
                haystack = _fold_accents(" ".join(parts))

                # Count how many query terms appear in the metadata.
                matched = sum(1 for term in folded_terms if term in haystack)
                ratio = matched / len(folded_terms) if folded_terms else 0.0
                blended += weight * ratio

        # model_copy preserves all fields; only score is updated.
        reranked.append(item.model_copy(update={"score": blended}))

    return sorted(reranked, key=lambda c: c.score, reverse=True)
