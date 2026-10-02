"""Rule-based QueryPlan parser (III, IV) — deterministic, no LLM.

The parser turns a raw query string into a :class:`QueryPlan`.  It is built
entirely from regular expressions and keyword tables so that the result is
reproducible and debuggable (XXIII).  No network, no model inference — the plan
is produced in microseconds and can be logged verbatim.

Design
------
1. **Language detection** — cheap heuristic (presence of common Vietnamese
   function words).  Used to decide whether translation should run later
   (Task 2 rationale: once ``translate`` is enabled, the parser receives EN).
2. **Event segmentation** — split long queries on discourse markers
   (``begins with`` / ``then`` / ``after`` / ``followed by`` / ``ends with`` …)
   and on ``;`` / ``|`` / ``/`` / ``,`` / newline.  Each segment becomes an
   :class:`Event` with an inferred :class:`TemporalRole` from its marker.
3. **Constraint extraction** — numeric patterns
   (``more than N X``, ``exactly N X``, ``only one X``, ``N people`` …) become
   :class:`Constraint` of kind COUNT (X).
4. **Entity / action / attribute** — reuse the detector vocabulary via
   ``_requested_concepts`` (lexical.py) for entities; actions/attributes are
   captured from constraint subjects and a small keyword table.
5. **Modality gating** — OCR/ASR weights drop to 0 when the query has no text /
   speech signal, keeping retrieval cheap (VIII, IX).

Everything is optional: a short single-sentence query simply yields a plan with
one implicit event and no constraints, which the rest of the pipeline treats
exactly like the raw query today.
"""

from __future__ import annotations

import re

from aic2026.query.plan import (
    Constraint,
    ConstraintKind,
    CountOp,
    Event,
    ModalityWeights,
    QueryPlan,
    RelationKind,
    TemporalRelation,
    TemporalRole,
)
from aic2026.reranking.lexical import _fold_accents, _requested_concepts

# --- Language detection ------------------------------------------------------

_VI_FUNCTION_WORDS = (
    "của", "và", "hoặc", "trong", "ngoài", "trên", "dưới", "một", "những", "các",
    "người", "khi", "sau", "trước", "rồi", "đang", "có", "không", "được", "bằng",
)
_VI_MARKERS = (
    "bắt đầu", "kết thúc", "sau đó", "rồi", "tiếp theo", "cuối cùng", "đầu tiên",
    "xuất hiện", "lần", "một người", "hai người", "ba người",
)
# Folded (accent-stripped) variants — Vietnamese queries are often typed without
# diacritics or in telex, so the parser must match on the folded form too.
_VI_MARKERS_FOLDED = tuple(_fold_accents(w) for w in _VI_MARKERS)
_VI_FUNCTION_FOLDED = tuple(_fold_accents(w) for w in _VI_FUNCTION_WORDS)


def _detect_language(text: str) -> str:
    folded = _fold_accents(text)
    if any(w in folded for w in _VI_MARKERS_FOLDED):
        return "vi"
    toks = set(folded.split())
    if any(w in toks for w in _VI_FUNCTION_FOLDED):
        return "vi"
    # crude English/symbol presence check
    if re.search(r"[a-zA-Z]", text):
        return "en"
    return "unknown"


# --- Discourse / event segmentation -----------------------------------------

# Ordered marker -> temporal role.  Checked in the order listed; first match wins.
_EVENT_MARKERS: list[tuple[str, TemporalRole]] = [
    (r"\bbegins with\b", TemporalRole.START),
    (r"\bstarts with\b", TemporalRole.START),
    (r"\bđầu tiên\b", TemporalRole.START),
    (r"\bbắt đầu\b", TemporalRole.START),
    (r"\bends with\b", TemporalRole.END),
    (r"\bends\b", TemporalRole.END),
    (r"\bconcludes with\b", TemporalRole.END),
    (r"\bkết thúc\b", TemporalRole.END),
    (r"\b cuối cùng\b", TemporalRole.END),
    (r"\bfirst moment\b", TemporalRole.FIRST),
    (r"\bfirst\b", TemporalRole.FIRST),
    (r"\blast moment\b", TemporalRole.LAST),
    (r"\blast\b", TemporalRole.LAST),
]

# Connectors that FOLLOW an event and introduce the NEXT event.  Splitting on
# these yields the segment list.
# NOTE: input to _segment_events is PRE-FOLDED (accents stripped, see parse_query),
# so the Vietnamese connectors must be matched on their FOLDED forms (e.g. "rồi"
# → "roi").  Both accented and folded variants are kept for safety.
_VI_CONNECTORS_FOLDED = (
    _fold_accents(w) for w in ("sau đó", "rồi", "tiếp theo", "kế tiếp", "sau cùng", "cuối cùng")
)
_EVENT_SPLIT_RE = re.compile(
    r"(?:"
    r";|\||/|\.|\n"                               # hard delimiters (incl. sentence ".")
    # "next to" is a spatial relation, not the temporal connector "next".
    r"|\b(?:then|next(?!\s+to)|after that|afterwards|later|followed by|subsequently|finally)\b"
    r"|\b(?:sau đó|rồi|tiếp theo|kế tiếp|sau cùng|cuối cùng|"
    + "|".join(_VI_CONNECTORS_FOLDED)
    + r")\b"
    r")",
    re.IGNORECASE,
)

# Global temporal relations implied by single markers on the whole query.
_FIRST_MARK_RE = re.compile(r"\bfirst moment\b|\bđầu tiên\b", re.IGNORECASE)
_LAST_MARK_RE = re.compile(r"\blast moment\b|\bphút cuối\b", re.IGNORECASE)

# --- Action-chain comma splitting -------------------------------------------
# Lightweight action verb table (the detector vocab is objects, not verbs).
# Used both for event decomposition (comma splitting below) and for the
# ``actions`` field harvest further down.  Defined here so the regex below can
# reference it at import time.
_ACTION_VERBS = (
    "exercise", "touch", "add", "stir", "shake", "walk", "run", "sit", "stand",
    "enter", "exit", "open", "close", "eat", "drink", "cook", "pour", "cut",
    "read", "write", "drive", "ride", "swim", "dance", "sing", "play", "lift",
    "thêm", "khuấy", "lắc", "đi", "chạy", "ngồi", "đứng", "vào", "ra", "mở",
    "đóng", "ăn", "uống", "nấu", "đọc", "viết", "lái", "bơi", "nhảy", "hát",
)

# Splits a query on a comma ONLY when an action verb immediately follows it, so
# a sequential action chain ("enters a room, opens a door, walks to a table")
# becomes separate clauses.  Non-action commas (e.g. "a dog, a cat" or
# "a man, wearing glasses") are left intact because no verb follows, which
# keeps descriptive lists as one event.
# Input is PRE-FOLDED, so VI verbs must be matched on their folded forms too
# (e.g. "mở" → "mo", "vào" → "vao", "ngồi" → "ngoi").
_ACTION_VERBS_FOLDED = tuple(_fold_accents(v) for v in _ACTION_VERBS)
_ACTION_CLAUSE_COMMA_RE = re.compile(
    r",\s*(?=\b(?:"
    + "|".join(re.escape(v) for v in _ACTION_VERBS)
    + "|"
    + "|".join(re.escape(v) for v in _ACTION_VERBS_FOLDED)
    + r")\w*)",
    re.IGNORECASE,
)


_META_PREFIX_CLEAN_RE = re.compile(
    r"^\s*(?:"
    r"(?:the\s+)?(?:video|clip|scene)\s+(?:starts|begins|ends|concludes)\s+with\s+|"
    r"it\s+(?:starts|begins|ends|concludes)\s+with\s+|"
    r"starts\s+with\s+|begins\s+with\s+|ends\s+with\b\s*|"
    r"(?:the\s+)?(?:video|clip|scene)\s+(?:shows|contains|depicts|features)\s+|"
    r"this\s+(?:video|clip|scene)\s+(?:shows|contains|depicts|features)\s+|"
    r"in\s+this\s+(?:video|clip|scene)\b,?\s*|"
    r"video\s+clip\s+showing\s+|a\s+clip\s+of\s+|video\s+of\s+|"
    r"(?:bắt\s+đầu|kết\s+thúc)\s+(?:bằng|với)?\s*|"
    r"đoạn\s+clip\s+(?:bắt\s+đầu|kết\s+thúc)\s+(?:bằng|với)?\s*"
    r")",
    re.IGNORECASE,
)


def _segment_events(text: str) -> list[tuple[str, TemporalRole]]:
    """Split ``text`` into ordered (segment, role) pairs.

    Two passes:
    1. Break *action chains* ("enters, opens, walks") on commas that join two
       verb phrases — i.e. an action verb immediately follows the comma.  This
       isolates each step of a sequential action as its own Event with a
       positional role (start -> middle -> end), WITHOUT splitting descriptive
       lists ("a dog, a cat") or attribute phrases ("a man, wearing glasses").
    2. Further split each clause on discourse markers / connectors.
    """
    action_clauses = _ACTION_CLAUSE_COMMA_RE.split(text)
    n_clauses = len(action_clauses)
    segments: list[tuple[str, TemporalRole]] = []
    for ci, clause in enumerate(action_clauses):
        for raw in _EVENT_SPLIT_RE.split(clause):
            seg = raw.strip(" ,;|/\n\t-")
            if not seg:
                continue
            role = TemporalRole.MIDDLE
            for pat, r in _EVENT_MARKERS:
                if re.search(pat, seg, re.IGNORECASE):
                    role = r
                    break
            # Strip structural meta-prefixes like "starts with", "The clip ends with"
            seg_clean = _META_PREFIX_CLEAN_RE.sub("", seg).strip(" ,;|/\n\t-")
            if seg_clean:
                seg = seg_clean
            # A clause born from an action-comma split with no explicit temporal
            # marker gets a positional role so the chain reads start -> … -> end.
            if role == TemporalRole.MIDDLE and n_clauses > 1:
                if ci == 0:
                    role = TemporalRole.START
                elif ci == n_clauses - 1:
                    role = TemporalRole.END
            segments.append((seg, role))
    return segments


# --- Constraint extraction ---------------------------------------------------

# Patterns that express a count constraint.  Order matters (most specific first).
# VI keywords are listed BOTH with and without diacritics, because Vietnamese
# queries are frequently typed without accents (or in telex) — the folded form
# is what actually matches at runtime.
_COUNT_PATTERNS: list[tuple[re.Pattern, CountOp | None]] = [
    # "exactly N X" / "exactly N red hats"
    (re.compile(r"exactly\s+(\d+)\s+([\wÀ-ỹ\- ]+?)(?:\b|$)", re.IGNORECASE), CountOp.EQ),
    (re.compile(r"chinh xac\s+(\d+)\s+([\wÀ-ỹ\- ]+?)(?:\b|$)", re.IGNORECASE), CountOp.EQ),
    # "more than N X" / "over N X"
    (re.compile(r"(?:more than|over|at least)\s+(\d+)\s+([\wÀ-ỹ\- ]+?)(?:\b|$)", re.IGNORECASE), CountOp.GT),
    (re.compile(r"(?:hon|tren|it nhat)\s+(\d+)\s+([\wÀ-ỹ\- ]+?)(?:\b|$)", re.IGNORECASE), CountOp.GT),
    # "less than N X" / "fewer than N X"
    (re.compile(r"(?:less than|fewer than|under|below)\s+(\d+)\s+([\wÀ-ỹ\- ]+?)(?:\b|$)", re.IGNORECASE), CountOp.LT),
    (re.compile(r"(?:duoi|it hon|nho hon)\s+(\d+)\s+([\wÀ-ỹ\- ]+?)(?:\b|$)", re.IGNORECASE), CountOp.LT),
    # "only one X" / "only N X"
    (re.compile(r"only\s+(?:one|(\d+))\s+([\wÀ-ỹ\- ]+?)(?:\b|$)", re.IGNORECASE), CountOp.EQ),
    (re.compile(r"chi\s+(?:mot|(\d+))\s+([\wÀ-ỹ\- ]+?)(?:\b|$)", re.IGNORECASE), CountOp.EQ),
    # bare "N X" where N is a small integer and X a noun phrase (e.g. "3 red hats")
    # — capture up to 4 words, stopping before a connector word or end-of-string.
    # The connector word (on/in/with/...) is excluded from the subject via the
    # lookahead so "3 red hats on the table" yields subject "red hats", not
    # "red hats on".
    (
        re.compile(
            r"\b(\d+)\s+([\wÀ-ỹ\-]+(?:\s+[\wÀ-ỹ\-]+){0,3}?)"
            r"(?=\s+(?:on|in|with|near|at|the|a|an|of|to|and|that|which|for)\b|\s*$)",
            re.IGNORECASE,
        ),
        CountOp.EQ,
    ),
]

_ONLY_ONE_MARK_RE = re.compile(r"only one|chi mot", re.IGNORECASE)

# Attribute verbs that attach a worn/possessed object to a subject.  We forbid
# the bare preposition "with" because it produces false positives ("with a map",
# "with both hands").  Only possession/garment verbs are kept, and the captured
# subject must be a real noun (>=3 chars, not a determiner).
_ATTRIBUTE_VERB_RE = re.compile(
    r"\b(?:wears?|wearing|has|have|having|đeo|mặc|có|đội|mang)\s+([\wÀ-ỹ\- ]+?)(?:\b|$)",
    re.IGNORECASE,
)
_FILLER_SUBJECT_RE = re.compile(
    r"^(?:a|an|the|cái|chiếc|người|một|both|his|her|their|two|three|four|five)\b",
    re.IGNORECASE,
)


def _clean_subject(subject: str) -> str:
    """Normalize a constraint subject: strip trailing/leading filler words."""
    s = subject.strip().strip(" .,-")
    # drop leading determiners / adjectives that are not part of the object name
    for _ in range(3):
        new = _FILLER_SUBJECT_RE.sub("", s, count=1).strip()
        if new == s:
            break
        s = new
    return s.strip()


def _extract_constraints(text: str) -> list[Constraint]:
    """Extract ATTRIBUTE + COUNT constraints, merging duplicates per subject.

    For counts, multiple patterns may match the same subject (e.g. "more than 5
    people" also contains the bare "5 people").  We merge on (kind, subject) and
    keep the *strongest* operator precedence GT > GE > LT > LE > EQ, so the
    looser "5" never shadows the explicit "more than 5".
    """
    _OP_PRECEDENCE = {
        CountOp.GT: 5,
        CountOp.GE: 4,
        CountOp.LT: 3,
        CountOp.LE: 2,
        CountOp.EQ: 1,
    }

    constraints: list[Constraint] = []
    attr_seen: set[str] = set()
    count_best: dict[str, tuple[int, CountOp, int, str]] = {}

    # attribute-style: "wears glasses" / "đeo kính" (not "with a map")
    for m in _ATTRIBUTE_VERB_RE.finditer(text):
        subj = _clean_subject(m.group(1))
        if len(subj) < 3 or subj.lower() in ("a", "an", "the", "both"):
            continue
        if subj in attr_seen:
            continue
        attr_seen.add(subj)
        constraints.append(
            Constraint(kind=ConstraintKind.ATTRIBUTE, subject=subj, raw=m.group(0).strip())
        )

    for pat, op in _COUNT_PATTERNS:
        for m in pat.finditer(text):
            # Resolve the numeric value.  The "only one X" / "chi mot X" patterns
            # capture the number in group(1) ONLY when it is a digit; the literal
            # words "one"/"mot" live in a non-capturing alternative, so we also
            # scan the whole match for them (they mean exactly 1).
            captured = m.group(1) if m.group(1) else ""
            if captured.isdigit():
                val = int(captured)
            elif captured.lower() in ("one", "mot") or re.search(r"\b(?:one|mot)\b", m.group(0), re.IGNORECASE):
                val = 1
            elif len(m.groups()) >= 2 and m.group(2).isdigit():
                val = int(m.group(2))
            else:
                val = None
            subj = _clean_subject(m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1))
            if not subj or len(subj) < 2 or val is None:
                continue
            # "only one" / "chi mot" forces EQ 1 regardless of captured number
            if _ONLY_ONE_MARK_RE.search(m.group(0)):
                val = 1
                op = CountOp.EQ
            prev = count_best.get(subj)
            strength = _OP_PRECEDENCE[op]
            if prev is None or strength > prev[0]:
                count_best[subj] = (strength, op, val, m.group(0).strip())

    for subj, (_, op, val, raw) in count_best.items():
        constraints.append(
            Constraint(kind=ConstraintKind.COUNT, subject=subj, operator=op, value=val, raw=raw)
        )
    return constraints


# --- Entity / action / attribute harvest -------------------------------------

_ACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(v) for v in _ACTION_VERBS) + r")\b",
    re.IGNORECASE,
)


def _harvest(text: str) -> tuple[list[str], list[str], list[str]]:
    entities = sorted(_requested_concepts(text))  # reuses detector alias table
    actions = sorted(set(a.lower() for a in _ACTION_RE.findall(text)))
    # attributes: colour words + glasses/hat-like modifiers captured loosely
    attr_words = set()
    for c in _COLOR_WORDS:
        if re.search(r"\b" + re.escape(c) + r"\b", text, re.IGNORECASE):
            attr_words.add(c)
    attributes = sorted(attr_words)
    return entities, actions, attributes


_COLOR_WORDS = (
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "black", "white", "gray", "grey", "đỏ", "cam", "vàng", "xanh", "tím",
    "hồng", "nâu", "đen", "trắng", "xám",
)

_SPATIAL_KEYWORDS = (
    "left of", "right of", "above", "below", "behind", "in front of", "next to",
    "near", "inside", "outside", "on top of", "upper-left", "upper-right",
    "lower-left", "lower-right", "center of", "bên trái", "bên phải",
    "phía trước", "phía sau", "cạnh", "gần", "trên", "dưới",
)


def _extract_spatial_relations(text: str) -> list[str]:
    found = set()
    folded = _fold_accents(text)
    for kw in _SPATIAL_KEYWORDS:
        kw_folded = _fold_accents(kw)
        if re.search(r"\b" + re.escape(kw_folded) + r"\b", folded, re.IGNORECASE):
            found.add(kw)
    return sorted(found)


# --- Modality gating ----------------------------------------------------------

_TEXT_SIGNAL_RE = re.compile(
    r"\b(?:sign|banner|title|label|name|text|logo|word|biển|tiêu đề|nhãn|tên|chữ|logo)\b",
    re.IGNORECASE,
)
_SPEECH_SIGNAL_RE = re.compile(
    r"\b(?:say|said|speech|talk|speak|voice|conversation|nói|thoại|giọng|trò chuyện)\b",
    re.IGNORECASE,
)


def _modalities(text: str, constraints: list[Constraint]) -> ModalityWeights:
    mod = ModalityWeights()
    if not _TEXT_SIGNAL_RE.search(text):
        mod.ocr = 0.0
    if not _SPEECH_SIGNAL_RE.search(text):
        mod.asr = 0.0
    if not any(c.kind == ConstraintKind.COUNT for c in constraints):
        mod.count = 0.0
    # object modality is on whenever entities are present
    if not _requested_concepts(text):
        mod.object = 0.0
    # temporal on when query expresses ordering or count repetition
    if not re.search(
        r"\b(?:before|after|then|first|last|begins|ends| followed by|sau đó|trước|sau)\b",
        text,
        re.IGNORECASE,
    ):
        mod.temporal = 0.0
    return mod


# --- Top-level parser --------------------------------------------------------

def parse_query(text: str) -> QueryPlan:
    """Parse ``text`` into a deterministic :class:`QueryPlan` (III, IV)."""
    raw = (text or "").strip()
    lang = _detect_language(raw)
    # Vietnamese queries are frequently typed without diacritics; fold accents so
    # VI patterns (events, counts, markers) match the accent-stripped form.
    folded = _fold_accents(raw)

    segments = _segment_events(folded if lang != "en" else raw)
    events: list[Event] = []
    temporal_relations: list[TemporalRelation] = []
    if segments:
        for i, (seg, role) in enumerate(segments):
            events.append(
                Event(
                    index=i,
                    description=seg,
                    entities=sorted(_requested_concepts(seg)),
                    actions=sorted(set(a.lower() for a in _ACTION_RE.findall(seg))),
                    attributes=_segment_attributes(seg),
                    spatial_relations=_extract_spatial_relations(seg),
                    visual_priority=1.0,
                    asr_priority=1.0 if _SPEECH_SIGNAL_RE.search(seg) else 0.0,
                    temporal_role=role,
                )
            )
        # Build an ordered BEFORE-chain across consecutive events.
        for i in range(1, len(events)):
            temporal_relations.append(
                TemporalRelation(kind=RelationKind.BEFORE, source=i - 1, target=i)
            )
    # Global first/last markers (run on folded text for VI coverage).
    if _FIRST_MARK_RE.search(folded):
        temporal_relations.append(TemporalRelation(kind=RelationKind.FIRST, source=0))
    if _LAST_MARK_RE.search(folded):
        temporal_relations.append(
            TemporalRelation(kind=RelationKind.LAST, source=len(events) - 1 or 0)
        )

    constraints = _extract_constraints(folded)
    entities, actions, attributes = _harvest(raw)
    spatial_relations = _extract_spatial_relations(raw)
    modalities = _modalities(folded, constraints)

    return QueryPlan(
        raw_text=raw,
        global_query=raw,
        language=lang,  # type: ignore[arg-type]
        entities=entities,
        actions=actions,
        attributes=attributes,
        spatial_relations=spatial_relations,
        constraints=constraints,
        events=events,
        temporal_relations=temporal_relations,
        modalities=modalities,
    )


def _segment_attributes(seg: str) -> list[str]:
    return sorted(
        c for c in _COLOR_WORDS if re.search(r"\b" + re.escape(c) + r"\b", seg, re.IGNORECASE)
    )

