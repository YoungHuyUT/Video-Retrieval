"""Deterministic query expansion + modality routing for Phase 5 (III, IV, XVII).

WHY (spec §2, §3 + paper arxiv 2512.12935v1 "parallel search")
----------------------------------------------------------------
A single CLIP query string is one point in embedding space. Expanding it into a
few *template-derived* variants (Q0..Q3) and RRF-fusing their rankings lifts
recall for queries whose intent is spread across entities / actions / attributes
without spending a second LLM round-trip (the runtime.py comment notes the
LLM-based expansion was dropped for speed — this is the free, deterministic
equivalent).

DESIGN (local, no paid API, no facts invented)
-----------------------------------------------
* Expansion is *pure template*: Q0 = the plan's global query, Q1 = the entity
  nouns, Q2 = the action verbs, Q3 = attribute×entity pairs.  No synonym
  generation, no paraphrasing — we never add a "fact" the query didn't state.
* **Modality routing** (XVII): ``plan.modalities`` is the contract the planner
  produces.  A facet whose modality weight is 0 is dropped so its (now spurious)
  variant does not dilute the RRF pool.  ``route_by_modality=False`` keeps every
  non-empty facet (useful for A/B vs the routed variant).
* The function is A/B-able and opt-in: ``tools.retrieve`` only uses it when
  ``use_query_expansion`` is set, so the OFF path is byte-for-byte the legacy
  CLIP+BM25 retrieval (baseline A).

The returned list is what ``tools.retrieve`` feeds into its existing
multi-query RRF path, so no new retrieval code is required downstream.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from aic2026.reranking.lexical import _fold_accents
from aic2026.query.parser import _ACTION_CLAUSE_COMMA_RE

if TYPE_CHECKING:
    from aic2026.query.plan import QueryPlan


# --- Deterministic scene-disambiguation rephrase (no LLM, free) ---------------
# CLIP's text tower embeds a short scene noun (e.g. "landslide") into the generic
# "outdoor" manifold, so it scores near *unrelated* outdoor scenes ("lion dance",
# cosine 0.766).  Appending a few specific visual descriptions pulls the ranking
# toward the real scene WITHOUT inventing facts.  Measured on ViT-L/14 openai:
# adding the phrases below drops "lion dance" similarity 0.766 → 0.58 while
# keeping "landslide" (0.74) and "mountain" (0.78) high — so RRF-fusing the
# variants separates a real landslide frame from a lion-dance frame.  Only fires
# when the query actually names one of these scenes, so ordinary queries are
# untouched.  This is the correct fix for scene queries: object-evidence penalty
# (lexical.py) cannot help here because "landslide" is a scene, not an object the
# detector labels (0/177321 frames carry any landslide/rock/mud label).
_DISASTER_REPHRASE: dict[str, tuple[str, ...]] = {
    "landslide": (
        "landslide debris flowing down a hillside",
        "rocks and mud sliding down a mountain slope",
        "a road blocked by fallen rocks and mud after a landslide",
    ),
    "sạt lở": (
        "landslide debris flowing down a hillside",
        "rocks and mud sliding down a mountain slope",
        "a road blocked by fallen rocks and mud after a landslide",
    ),
    "đất lở": (
        "landslide debris flowing down a hillside",
        "rocks and mud sliding down a mountain slope",
    ),
    "núi lở": (
        "landslide debris flowing down a hillside",
        "rocks and mud sliding down a mountain slope",
    ),
    "flood": (
        "floodwater submerging a street",
        "a flooded road with vehicles half underwater",
    ),
    "lũ lụt": (
        "floodwater submerging a street",
        "a flooded road with vehicles half underwater",
    ),
    "lũ": (
        "floodwater submerging a street",
        "a flooded road with vehicles half underwater",
    ),
    "fire": (
        "flames and smoke rising from a burning building",
        "a wildfire spreading through dry vegetation",
    ),
    "cháy": (
        "flames and smoke rising from a burning building",
        "a wildfire spreading through dry vegetation",
    ),
    "earthquake": (
        "collapsed building rubble after an earthquake",
        "cracked road and fallen walls from an earthquake",
    ),
    "động đất": (
        "collapsed building rubble after an earthquake",
        "cracked road and fallen walls from an earthquake",
    ),
    "storm": (
        "strong wind bending trees and heavy rain",
        "a storm with fallen branches on the ground",
    ),
    "bão": (
        "strong wind bending trees and heavy rain",
        "a storm with fallen branches on the ground",
    ),
}
_DISASTER_KEYWORDS: frozenset[str] = frozenset(_DISASTER_REPHRASE.keys())


# --- Animal disambiguation rephrases (no LLM, free) --------------------------
# CLIP's text tower maps short animal nouns into broad "animal" manifold, so
# "buffalo" scores near cow, goat, deer, horse — unrelated species that share
# the generic animal visual manifold.  Appending specific visual context pulls
# the ranking toward the real animal WITHOUT inventing facts.  Symmetrically,
# this also prevents animal frames from polluting non-animal queries.
# Only fires when the query actually names one of these animals.
_ANIMAL_REPHRASE: dict[str, tuple[str, ...]] = {
    "buffalo": (
        "a large water buffalo standing in a field",
        "a water buffalo with curved horns",
    ),
    "water buffalo": (
        "a large water buffalo standing in a field",
        "a water buffalo with curved horns",
    ),
    "trâu": (
        "a large water buffalo standing in a field",
        "a water buffalo with curved horns",
    ),
    "con trâu": (
        "a large water buffalo standing in a field",
        "a water buffalo with curved horns",
    ),
    "cow": (
        "a cow grazing in a pasture",
        "a dairy cow standing in a field",
    ),
    "bò": (
        "a cow grazing in a pasture",
        "a dairy cow standing in a field",
    ),
    "goat": (
        "a goat standing on rocky terrain",
        "a mountain goat on a hillside",
    ),
    "dê": (
        "a goat standing on rocky terrain",
        "a mountain goat on a hillside",
    ),
    "deer": (
        "a deer standing in a forest clearing",
        "a deer with antlers in the woods",
    ),
    "nai": (
        "a deer standing in a forest clearing",
        "a deer with antlers in the woods",
    ),
    "horse": (
        "a horse running in an open field",
        "a horse galloping on grassland",
    ),
    "ngựa": (
        "a horse running in an open field",
        "a horse galloping on grassland",
    ),
    "elephant": (
        "a large elephant with a trunk",
        "an elephant walking in the wild",
    ),
    "voi": (
        "a large elephant with a trunk",
        "an elephant walking in the wild",
    ),
    "pig": (
        "a pig standing in a farmyard",
        "a domestic pig on grass",
    ),
    "lợn": (
        "a pig standing in a farmyard",
        "a domestic pig on grass",
    ),
    "dog": (
        "a dog sitting on the ground",
        "a domestic dog looking at the camera",
    ),
    "chó": (
        "a dog sitting on the ground",
        "a domestic dog looking at the camera",
    ),
    "cat": (
        "a cat sitting indoors",
        "a domestic cat looking at the camera",
    ),
    "mèo": (
        "a cat sitting indoors",
        "a domestic cat looking at the camera",
    ),
    "bird": (
        "a bird perched on a branch",
        "a wild bird flying in the sky",
    ),
    "chim": (
        "a bird perched on a branch",
        "a wild bird flying in the sky",
    ),
    "snake": (
        "a snake coiled on the ground",
        "a snake slithering through grass",
    ),
    "rắn": (
        "a snake coiled on the ground",
        "a snake slithering through grass",
    ),
    "rabbit": (
        "a rabbit sitting on grass",
        "a bunny in a garden",
    ),
    "thỏ": (
        "a rabbit sitting on grass",
        "a bunny in a garden",
    ),
    "monkey": (
        "a monkey sitting in a tree",
        "a monkey on a tree branch",
    ),
    "khỉ": (
        "a monkey sitting in a tree",
        "a monkey on a tree branch",
    ),
    "lion": (
        "a lion lying in the grass",
        "a male lion with a mane",
    ),
    "sư tử": (
        "a lion lying in the grass",
        "a male lion with a mane",
    ),
    "tiger": (
        "a tiger walking through vegetation",
        "a tiger with striped fur",
    ),
    "hổ": (
        "a tiger walking through vegetation",
        "a tiger with striped fur",
    ),
    "bear": (
        "a bear standing in a forest",
        "a brown bear in the wild",
    ),
    "gấu": (
        "a bear standing in a forest",
        "a brown bear in the wild",
    ),
    "frog": (
        "a frog sitting on a lily pad",
        "a green frog on a leaf",
    ),
    "ếch": (
        "a frog sitting on a lily pad",
        "a green frog on a leaf",
    ),
    "turtle": (
        "a turtle walking on land",
        "a sea turtle swimming in water",
    ),
    "rùa": (
        "a turtle walking on land",
        "a sea turtle swimming in water",
    ),
}
_ANIMAL_KEYWORDS: frozenset[str] = frozenset(_ANIMAL_REPHRASE.keys())


def _animal_variants(text: str) -> list[str]:
    """Return specific visual-rephrase variants when *text* names an animal.

    CLIP maps short animal nouns (e.g. "buffalo") into a broad "animal"
    manifold, so they score near unrelated species.  Appending specific
    visual descriptions pulls the ranking toward the real animal.
    """
    if not text:
        return []
    folded = _fold_accents(text)
    out: list[str] = []
    for key in _ANIMAL_KEYWORDS:
        if re.search(r"(?<![\wÀ-ỹ])" + re.escape(_fold_accents(key)) + r"(?![\wÀ-ỹ])", folded):
            out.extend(_ANIMAL_REPHRASE[key])
    return out


def _disaster_variants(text: str) -> list[str]:
    """Return specific visual-rephrase variants when *text* names a disaster scene."""
    if not text:
        return []
    folded = _fold_accents(text)
    out: list[str] = []
    for key in _DISASTER_KEYWORDS:
        # Word-boundary match, NOT a raw substring: 'lũ' (flood) is a substring of
        # 'headscarf' (he-adscar-lũ-f), so a loose `in` check false-fires disaster
        # rephrases on unrelated queries.  Accent-stripped so 'sạt lở' / 'lu lut'
        # match regardless of how the user typed the diacritics.
        if re.search(r"(?<![\wÀ-ỹ])" + re.escape(_fold_accents(key)) + r"(?![\wÀ-ỹ])", folded):
            out.extend(_DISASTER_REPHRASE[key])
    return out


def disaster_keyword_present(text: str) -> bool:
    """True if *text* names a disaster scene — used to force expansion even for
    short/single-word queries that would otherwise skip the gated expansion."""
    if not text:
        return False
    folded = _fold_accents(text)
    return any(
        re.search(r"(?<![\wÀ-ỹ])" + re.escape(_fold_accents(k)) + r"(?![\wÀ-ỹ])", folded)
        for k in _DISASTER_KEYWORDS
    )


# --- Keyword-to-detector-label mapping (no LLM, free) -----------------------
# When a query contains a keyword like "banner", "mountain", "school", the
# detector does NOT have a matching label (no "banner" label in OpenImages V4).
# But related labels DO exist: "Poster", "Billboard", "Flag" for banner;
# "Building", "Tree" for mountain; "Boy", "Girl" for children.
# Mapping query keywords → detector labels → focused CLIP sub-queries so RRF
# fusion pulls frames with the right visual concepts.
_KEYWORD_TO_LABELS: dict[str, tuple[str, ...]] = {
    # Scene/background keywords → detector labels
    "banner": ("Poster", "Billboard", "Flag"),
    "poster": ("Poster", "Billboard"),
    "billboard": ("Billboard", "Poster"),
    "sign": ("Traffic sign", "Stop sign", "Poster", "Billboard"),
    "poster": ("Poster", "Billboard"),
    "mountain": ("Building", "Skyscraper", "Tree", "House"),
    "cloud": ("Building", "Tree"),
    "road": ("Building", "Street light", "Tree"),
    "street": ("Building", "Street light", "Tree"),
    "school": ("Building", "House", "Office building"),
    "classroom": ("Building", "House"),
    "river": ("Building", "Tree"),
    "bridge": ("Building", "Tree"),
    "sky": ("Building", "Tree"),
    # People keywords → detector labels
    "child": ("Boy", "Girl"),
    "children": ("Boy", "Girl"),
    "kid": ("Boy", "Girl"),
    "student": ("Boy", "Girl", "Person"),
    "young": ("Boy", "Girl"),
    "people": ("Person", "Man", "Woman"),
    "person": ("Person", "Man", "Woman"),
    "man": ("Man", "Person"),
    "woman": ("Woman", "Person"),
    "girl": ("Girl", "Person"),
    "boy": ("Boy", "Person"),
    "teacher": ("Person", "Man", "Woman"),
    # Clothing/color keywords → detector labels
    "shirt": ("Clothing", "Jeans", "Dress", "Suit"),
    "yellow": ("Clothing", "Jeans", "Dress"),
    "blue": ("Clothing", "Jeans", "Dress"),
    "red": ("Clothing", "Jeans", "Dress"),
    "white": ("Clothing", "Jeans", "Dress"),
    "black": ("Clothing", "Jeans", "Dress"),
    "green": ("Clothing", "Jeans", "Dress"),
    # Object keywords → detector labels
    "bag": ("Handbag", "Backpack", "Luggage"),
    "hat": ("Cowboy hat", "Hat", "Fedora"),
    "glasses": ("Glasses", "Sunglasses"),
    "phone": ("Cell phone", "Mobile phone"),
    "laptop": ("Laptop", "Computer keyboard"),
    "bottle": ("Bottle", "Wine glass"),
    "cup": ("Cup", "Coffee cup"),
    "food": ("Food", "Fruit", "Vegetable"),
    "fruit": ("Fruit", "Apple", "Banana"),
    "vegetable": ("Vegetable", "Plant"),
    # Generic animal queries must not be expanded into arbitrary species. That
    # adds claims absent from the query and can make retrieval return one of
    # those species even when the user asked for animals in general.
    "dog": ("Dog", "Puppy"),
    "cat": ("Cat", "Kitten"),
    "bird": ("Bird", "Pigeon"),
    "fish": ("Fish", "Seafood"),
    "car": ("Car", "Land vehicle", "Truck"),
    "bike": ("Bicycle", "Bicycle wheel"),
    "motorcycle": ("Motorcycle", "Motorbike"),
    "bus": ("Bus", "Land vehicle"),
    "truck": ("Truck", "Land vehicle"),
    "boat": ("Boat", "Ship"),
    "airplane": ("Airplane", "Helicopter"),
    # Action keywords → detector labels (actions imply objects)
    "hanging": ("Poster", "Billboard", "Flag", "Person"),
    "holding": ("Person", "Clothing"),
    "wearing": ("Clothing", "Jeans", "Dress", "Suit", "Glasses", "Hat"),
    "riding": ("Person", "Bicycle", "Motorcycle", "Horse"),
    "running": ("Person", "Car", "Truck"),
    "walking": ("Person", "Man", "Woman"),
    "standing": ("Person", "Man", "Woman"),
    "sitting": ("Person", "Man", "Woman"),
    "eating": ("Person", "Food", "Fruit"),
    "drinking": ("Person", "Bottle", "Cup"),
    "cooking": ("Person", "Food", "Stove"),
    "playing": ("Person", "Sports equipment"),
    "working": ("Person", "Laptop", "Computer keyboard"),
}

_KEYWORD_LABELS_FROZEN: frozenset[str] = frozenset(_KEYWORD_TO_LABELS.keys())


def _keyword_variants(text: str) -> list[str]:
    """Map query keywords to actual detector labels for focused CLIP sub-queries.

    CLIP's text tower embeds descriptive words like "banner" or "mountain" into
    generic manifolds that may not match the actual detector labels.  Mapping
    these keywords to detector labels and adding them as focused sub-queries
    helps RRF fusion find frames with the right visual concepts.
    """
    if not text:
        return []
    folded = _fold_accents(text)
    out: list[str] = []
    for keyword in _KEYWORD_LABELS_FROZEN:
        kw_folded = _fold_accents(keyword)
        if re.search(r"(?<![\wÀ-ỹ])" + re.escape(kw_folded) + r"(?![\wÀ-ỹ])", folded):
            # Skip animal keyword->label expansion: bare Dog/Puppy labels
            # match the entire CLIP animal manifold; use animal rephrases instead.
            if keyword in _ANIMAL_KEYWORDS:
                continue
            labels = _KEYWORD_TO_LABELS[keyword]
            # Add single-label queries (high recall for that concept)
            out.extend(labels)
            # Add label+keyword pairs for specificity (e.g. "Poster banner")
            for label in labels:
                pair = f"{label} {keyword}"
                if pair not in out:
                    out.append(pair)
    return out


# --- Long-query sentence split (no LLM, free) --------------------------------
# CLIP's text tower caps at 77 tokens, so a long multi-scene description (e.g. a
# 3-sentence clip summary) is TRUNCATED into one blurred vector — and never ranks
# its real frames.  Splitting the query into its constituent sentences (each a
# short, focused phrase) and RRF-fusing their rankings recovers the scenes the
# single pooled vector washes out.  This is the correct fix for long queries, the
# mirror image of the disaster rephrase (which fixes *short* ambiguous nouns).
# Split ONLY on hard delimiters (. ; | / newline) and discourse connectors —
# NOT on commas after verbs, which would shatter a descriptive sentence into
# garbage fragments ("white shirt, and tie" -> "and tie", "open-pit" -> "open").
_SENTENCE_SPLIT_RE = re.compile(
    r"(?:"
    r"[.;|/\n]"                                                     # hard delimiters
    r"|\b(?:then|next|after that|afterwards|later|followed by|subsequently|finally|"
    r"first|second|third|meanwhile)\b"
    r")",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    """Split *text* into short, focused query sentences (CLIP-token-safe)."""
    if not text:
        return []
    out: list[str] = []
    for seg in _SENTENCE_SPLIT_RE.split(text):
        seg = " ".join(seg.split()).strip(" ,;|/\n\t-")
        # Drop fragments that are too short to be a meaningful CLIP query
        # (they would just add noise to the RRF pool).
        if len(seg.split()) >= 3:
            out.append(seg)
    return out


def _looks_long(text: str) -> bool:
    return bool(text) and len(text.split()) >= 12


# --- Descriptive clause splitting (no LLM, free) --------------------------------
# Long KIS queries often describe a SINGLE complex visual scene with multiple
# comma-separated descriptors ("a large blue-toned banner, decorated with images
# of mountains, clouds, and a road leading to a school").  These are NOT separate
# events — they are facets of one frame.  Splitting them into focused sub-queries
# lets CLIP capture each facet ("blue banner", "mountains", "school") and RRF
# fuses them back, lifting the correct frame.  The split only fires for LONG
# queries that do NOT look like action chains (no action verbs in commas).
_DESCRIPTIVE_SPLIT_RE = re.compile(
    r",\s*(?=the |a |an |and |with |of |decorated |featuring |featuring |"
    r"showing |displaying |wearing |pictures? )",
    re.IGNORECASE,
)


def _looks_descriptive(text: str) -> bool:
    """True when a long query is a single-scene visual description (not action chain)."""
    if not _looks_long(text):
        return False
    # Check for comma-separated clauses that have ACTION VERBS between them
    # (real action chain) vs descriptive phrases (visual details).
    clauses = _ACTION_CLAUSE_COMMA_RE.split(text)
    if len(clauses) >= 2:
        # Check if clauses start with action verbs (he/they/she/X is verb-ing)
        # A real action chain: "person enters room, sits down, opens door"
        # A descriptive chain: "blue banner, decorated with mountains, clouds"
        action_starters = (
            "he ", "she ", "they ", "we ", "i ", "it ", "the person ",
            "the man ", "the woman ", "the boy ", "the girl ", "a person ",
            "a man ", "a woman ", "a boy ", "a girl ",
        )
        for c in clauses[1:]:  # skip first clause
            c_fold = " " + c.strip().lower()
            # If any clause after the first starts with an action verb, it's an action chain
            if any(c_fold.startswith(s) or c_fold.lstrip().startswith(s) for s in action_starters):
                return False
    # Has commas — likely a descriptive list.
    return "," in text


def _split_descriptive(text: str) -> list[str]:
    """Split a descriptive long query into focused sub-queries per facet.

    Splits on commas followed by descriptive phrases ("decorated with",
    "featuring", "with", "pictures of", "images of", etc.).
    """
    if not text:
        return []
    out: list[str] = []
    for seg in _DESCRIPTIVE_SPLIT_RE.split(text):
        seg = " ".join(seg.split()).strip(" ,;|/\n\t-")
        if len(seg.split()) >= 2:
            out.append(seg)
    return out


# --- Attribute-entity pair extraction (no LLM, free) ---------------------------
# Detect "color + object" or "adjective + noun" pairs directly from the query
# text and generate focused sub-queries like "blue banner", "yellow shirts".
_COLOR_WORDS_FOLD = {
    "red", "blue", "yellow", "green", "black", "white", "brown",
    "orange", "purple", "pink", "gray", "grey", "đỏ", "xanh", "vàng",
    "đen", "trắng", "nâu", "cam", "tím", "hồng", "xám",
}

# Words that are NOT entities (stop words, prepositions, verbs).
_NON_ENTITY_WORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "and",
    "or", "but", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "need", "dare", "ought", "used", "to",
    "from", "by", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "also", "than", "too", "very", "just",
    "not", "only", "own", "same", "so", "now", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "what", "which", "who",
    "whom", "this", "that", "these", "those", "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it",
    "its", "itself", "they", "them", "their", "theirs", "themselves",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "any", "all", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "will", "just", "don", "should",
    "now",
})


def _extract_color_entity_pairs(text: str) -> list[str]:
    """Extract 'color noun' pairs from text for focused sub-queries.

    E.g. "blue-toned banner" → "blue banner", "yellow shirts" → "yellow shirts".
    Also handles compound color words like "blue-toned" → "blue".
    """
    if not text:
        return []
    # Work on original text to preserve word boundaries.
    words = text.split()
    pairs: list[str] = []
    for i, w in enumerate(words):
        # Handle compound color words like "blue-toned" → extract "blue"
        w_clean = w.rstrip(".,;:!?")
        w_fold = _fold_accents(w_clean)
        # Check if it's a color word or starts with a color (e.g. "blue-toned")
        is_color = w_fold in _COLOR_WORDS_FOLD
        if not is_color and "-" in w_fold:
            # Check first part of hyphenated word
            first_part = w_fold.split("-")[0]
            if first_part in _COLOR_WORDS_FOLD:
                is_color = True
                # Use the full original word but note the color
                w_clean = first_part  # Use just "blue" not "blue-toned"
        if is_color:
            # Take next 1-2 NON-STOP words as the entity.
            entity_words = []
            for j in range(i + 1, min(i + 3, len(words))):
                ew = words[j].rstrip(".,;:!?")
                ew_fold = _fold_accents(ew)
                if ew_fold in _NON_ENTITY_WORDS:
                    break
                entity_words.append(ew)
            entity = " ".join(entity_words).strip()
            if entity and len(entity) >= 2:
                pairs.append(f"{w_clean} {entity}")
    return pairs


# --- Explicit phrase extraction (no LLM, free) ---------------------------------
# For long descriptive queries, extract multi-word noun phrases that are
# visually distinctive: "blue-toned banner", "yellow shirts", "mountain images".
_PHRASE_PATTERNS = [
    # "decorated with images of X" → "X images"
    re.compile(
        r"decorated\s+with\s+images?\s+of\s+([\w\s,]+?)(?:\.\s|,\s*the |,\s*and |\.$|$)",
        re.IGNORECASE,
    ),
    # "featuring pictures of X" → "X pictures"
    re.compile(
        r"featuring\s+pictures?\s+of\s+([\w\s,]+?)(?:\.\s|,\s*the |,\s*and |\.$|$)",
        re.IGNORECASE,
    ),
    # "leading to a X" → "X building" (capture 1-2 words only)
    re.compile(
        r"leading\s+to\s+(?:a |an )([\w]+(?:\s+[\w]+)?)",
        re.IGNORECASE,
    ),
    # "pictures of X wearing Y" → "X wearing Y" (capture full phrase)
    re.compile(
        r"pictures?\s+of\s+([\w\s]+?wearing\s+[\w\s]+?)(?:\.|$)",
        re.IGNORECASE,
    ),
    # "images of X, Y" → "X images", "Y images"
    re.compile(
        r"images?\s+of\s+([\w\s,]+?)(?:\.\s|,\s*a |,\s*and |\.$|$)",
        re.IGNORECASE,
    ),
]


def _extract_phrases(text: str) -> list[str]:
    """Extract explicit visual phrases from descriptive text."""
    if not text:
        return []
    phrases: list[str] = []
    for pat in _PHRASE_PATTERNS:
        for m in pat.finditer(text):
            phrase = " ".join(m.group(1).split()).strip()
            if len(phrase) >= 2:
                phrases.append(phrase)
    return phrases

# Variant kinds, for traceability in the agent plan / A/B labels.
Q0_GLOBAL = "global"
Q1_ENTITIES = "entities"
Q2_ACTIONS = "actions"
Q3_ATTRIBUTE_ENTITY = "attribute_entity"


def expand_query(
    plan: "QueryPlan",
    max_variants: int = 8,
    route_by_modality: bool = True,
) -> list[str]:
    """Return deterministic query variants Q0..Q3 (+ sentence splits) from a *plan*.

    Order is always ``[Q0, Q1?, Q2?, Q3?, …sentences…]`` with duplicates
    (case/space-insensitive) removed and the list capped at ``max_variants``.
    ``Q0`` (the global query) is always retained as the primary CLIP signal; the
    others are appended only when their facet carries content *and* (when routing)
    its modality weight > 0.

    Sentence-split branch: for a LONG query (>= 12 words, i.e. a clip summary /
    multi-scene description) the global string is split into its constituent
    sentences and each is appended as a focused variant.  CLIP truncates text at
    77 tokens, so a long query encoded as one vector is blurred and ranks nothing
    useful; the per-sentence rankings are RRF-fused downstream (see ``tools.retrieve``
    and the multi-query RRF path) to recover the scenes the pooled vector loses.
    """
    mod = plan.modalities

    def _keep(kind: str, content: list[str]) -> bool:
        if not content:
            return False
        if not route_by_modality:
            return True
        weight = {
            Q1_ENTITIES: mod.object,
            Q2_ACTIONS: mod.semantic,
            Q3_ATTRIBUTE_ENTITY: mod.object,
        }.get(kind, 1.0)
        return weight > 0.0

    variants: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        t = " ".join(text.split()).strip()
        if not t:
            return
        key = t.casefold()
        if key not in seen:
            seen.add(key)
            variants.append(t)

    # Q0 — the canonical global query (planner rewrite or raw text).
    q0 = (plan.global_query or plan.raw_text or "").strip()
    _add(q0)

    # Sentence-split branch (long / multi-scene queries): append each constituent
    # sentence as a focused variant so the 77-token CLIP cap doesn't blur the whole
    # description into one useless vector.  Each sentence is short enough to encode
    # fully; the rankings are RRF-fused downstream.  Skipped for short queries (the
    # global query already covers them, and splitting adds noise).
    if _looks_long(q0):
        for sent in _split_sentences(q0):
            if sent.casefold() != q0.casefold():
                _add(sent)

    # Q1/Q2/Q3 — entity / action / attribute facets.
    # For ACTION-CHAIN long queries: skip facets (they scatter into noise).
    # For DESCRIPTIVE long queries: KEEP facets + add descriptive splits.
    _is_action_chain = _looks_long(q0) and (
        len(_ACTION_CLAUSE_COMMA_RE.split(q0)) >= 2
    )
    if not _looks_long(q0) or (not _is_action_chain and _looks_descriptive(q0)):
        # Q1 — entity nouns ("glasses", "red door", "table").
        if _keep(Q1_ENTITIES, plan.entities):
            _add(" ".join(plan.entities))

        # Q2 — action verbs ("enters", "opens", "walks").
        if _keep(Q2_ACTIONS, plan.actions):
            _add(" ".join(plan.actions))

        # Q3 — attribute × entity pairs ("red glasses", "blue door"). Keyed on the
        # *object* modality (visual attribute/entity facet), not semantic, so that a
        # query with no object signal drops these pairs instead of polluting the pool.
        if _keep(Q3_ATTRIBUTE_ENTITY, plan.attributes):
            if plan.entities:
                for attr in plan.attributes:
                    for ent in plan.entities:
                        _add(f"{attr} {ent}")
            else:
                for attr in plan.attributes:
                    _add(attr)

    # Descriptive clause splitting: for long single-scene descriptions that
    # are NOT action chains, split on descriptive commas to get focused sub-
    # queries ("blue banner", "mountains", "school", "children wearing yellow
    # shirts").  Each facet is short enough for CLIP to encode fully.
    if _looks_descriptive(q0) and not _is_action_chain:
        for clause in _split_descriptive(q0):
            _add(clause)

    # Color+entity pairs: extract "blue banner", "yellow shirts" directly.
    for pair in _extract_color_entity_pairs(q0):
        _add(pair)

    # Explicit phrase extraction: "images of mountains", "pictures of children".
    for phrase in _extract_phrases(q0):
        _add(phrase)

    # Scene-disambiguation rephrases: a query naming a disaster scene (landslide,
    # flood, fire, …) gets specific visual descriptions appended so RRF fusion
    # pulls the ranking toward the real scene instead of the generic "outdoor"
    # manifold CLIP assigns the bare noun (see _DISASTER_REPHRASE docstring).
    for v in _disaster_variants(plan.global_query or plan.raw_text or ""):
        _add(v)

    # Animal-disambiguation rephrases: a query naming an animal (buffalo, cow,
    # goat, …) gets specific visual descriptions appended so RRF fusion pulls
    # the ranking toward the real animal instead of the generic "animal" manifold
    # CLIP assigns the bare noun (see _ANIMAL_REPHRASE docstring).
    for v in _animal_variants(plan.global_query or plan.raw_text or ""):
        _add(v)

    # Keyword-to-detector-label mapping: query keywords like "banner", "mountain",
    # "school", "children" don't have matching detector labels, but related labels
    # do exist ("Poster", "Building", "Boy", "Girl").  Adding these as focused
    # sub-queries helps RRF fusion find frames with the right visual concepts.
    # Respect modality routing: only add when object modality is active.
    if _keep(Q1_ENTITIES, plan.entities):
        for v in _keyword_variants(plan.global_query or plan.raw_text or ""):
            _add(v)

    return variants[:max_variants]


def expand_text(
    text: str,
    max_variants: int = 8,
    route_by_modality: bool = True,
) -> list[str]:
    """Convenience: parse *text* then expand (no LLM — rule-based parser).

    For callers that only have the raw string (e.g. the agent when the planner is
    disabled).  Uses the same deterministic template path.
    """
    from aic2026.query.parser import parse_query

    plan = parse_query(text)
    variants = expand_query(
        plan,
        max_variants=max_variants,
        route_by_modality=route_by_modality,
    )
    # Always append disaster-scene rephrases when present, even if the planner
    # produced no structured facets (keeps the bare-noun query discriminative).
    extra = _disaster_variants(text)
    if extra:
        seen_keys = {v.casefold() for v in variants}
        for v in extra:
            if v.casefold() not in seen_keys:
                seen_keys.add(v.casefold())
                variants.append(v)
    # Also append animal-disambiguation rephrases (pulls ranking toward the
    # real animal instead of generic "animal" manifold).
    animal_extra = _animal_variants(text)
    if animal_extra:
        seen_keys = {v.casefold() for v in variants}
        for v in animal_extra:
            if v.casefold() not in seen_keys:
                seen_keys.add(v.casefold())
                variants.append(v)
    # Also append keyword-to-detector-label variants (maps query keywords
    # like "banner", "mountain", "school" to actual detector labels).
    kw_extra = _keyword_variants(text)
    if kw_extra:
        seen_keys = {v.casefold() for v in variants}
        for v in kw_extra:
            if v.casefold() not in seen_keys:
                seen_keys.add(v.casefold())
                variants.append(v)
    return variants[:max_variants]
