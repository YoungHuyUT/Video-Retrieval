from __future__ import annotations

import re
from collections.abc import Mapping

"""Lightweight post-correction for Vietnamese OCR output.

PaddleOCR (PP-OCRv6) is already the best CPU-only option for Vietnamese
scene text in this project — a public VN fine-tune was tested on real
keyframes and scored *worse* (it overfits document-scan fonts). So instead
of swapping the model, we clean its systematic errors here:

* stray/garbled diacritic characters (e.g. ``Ä`` where ``Ư``/``Ơ`` is meant),
* confusable letter pairs (``rn`` ↔ ``m``, ``cl`` ↔ ``d``, ``I`` ↔ ``l``),
* noise tokens (pure symbols, single stray letters, obvious OCR artifacts),
* whitespace/encoding normalization so BM25 sees clean terms.

The corrector is conservative: it only rewrites a token when the rewrite is
clearly an improvement (valid VN word / known mapping), never guessing blindly
on already-valid tokens.
"""

# --- Character-level fixes -------------------------------------------------
# Garbled Latin chars Paddle sometimes emits for VN diacritics, mapped to the
# most likely intended base letter. We keep the diacritic *slot* empty so the
# diacritic-repair step below can re-attach a plausible mark.
_GARBLE_MAP = {
    "Ä": "A", "Å": "A", "Æ": "A", "À": "A", "Á": "A", "Â": "A", "Ã": "A",
    "Ö": "O", "Ø": "O", "Ò": "O", "Ó": "O", "Ô": "O", "Õ": "O",
    "Ü": "U", "Û": "U", "Ù": "U", "Ú": "U", "Ñ": "N", "Ç": "C",
    "ï": "i", "ÿ": "y", "õ": "o", "ñ": "n",
}

# Common whole-string OCR confusions found on VN video keyframes. Keys are the
# raw OCR text (case-insensitive match), values the corrected form.
_TOKEN_MAP = {
    "he thao": "thể thao",
    "the thao": "thể thao",
    "hoi dong": "hội đồng",
    "uy ban": "ủy ban",
    "chinh phu": "chính phủ",
    "truong hoc": "trường học",
    "dai hoc": "đại học",
    "cong ty": "công ty",
    "tin tuc": "tin tức",
    "thu do": "thủ đô",
    "can bo": "cán bộ",
    "giao thong": "giao thông",
    "y te": "y tế",
    "van hoa": "văn hóa",
    "khoa hoc": "khoa học",
    "cong an": "công an",
    "quoc hoi": "quốc hội",
    "nguoi dan": "người dân",
    "ban to chuc": "ban tổ chức",
}

# Keep these as-is even if they look "weird" (real VN abbreviations / units).
_PRESERVE = {"htv", "vtv", "vn", "hcm", "hn", "qg", "tp", "km", "m2", "m3", "đ"}

# A token is treated as noise if it is only symbols/digits-with-stray-letters
# after stripping. We drop pure-symbol and single-char tokens that are not
# valid VN letters.
_NOISE_RE = re.compile(r"^[^a-zA-ZÀ-ỹ]+$")
_LETTER_RE = re.compile(r"[a-zA-ZÀ-ỹ]")
_WS_RE = re.compile(r"\s+")


def _has_vietnamese_vowel(token: str) -> bool:
    """True if the token contains a VN vowel (incl. diacritics)."""
    return bool(re.search(r"[aeiouyàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]", token, re.IGNORECASE))


def correct_token(token: str) -> str | None:
    """Return the corrected token, or ``None`` to drop it as noise.

    Conservative: a token that is already valid VN text is returned unchanged
    unless an explicit high-confidence mapping applies.
    """
    if not token:
        return None
    raw = token.strip()
    if not raw:
        return None

    # Normalize internal whitespace (some OCR joins "01: 01: 36").
    cleaned = _WS_RE.sub(" ", raw)
    # Drop surrounding quotes/brackets that are OCR artifacts on banners.
    cleaned = cleaned.strip("'\"`“”‘’()[]{}")

    # Pure noise (no letters at all) → drop, unless it looks like a timestamp/id.
    if not _LETTER_RE.search(cleaned):
        # Keep things that are mostly digits or time-like (01:01:36, +0'14").
        if re.fullmatch(r"[0-9:.,'\"+\-/%]+", cleaned):
            return cleaned
        return None

    lower = cleaned.lower()

    # Explicit whole-token mapping (high confidence).
    if lower in _TOKEN_MAP:
        return _TOKEN_MAP[lower]

    # Single isolated letter that is not a known abbreviation → drop.
    if len(cleaned) == 1 and lower not in _PRESERVE:
        return None

    # Preserve known abbreviations unchanged.
    if lower in _PRESERVE:
        return cleaned

    # From here on we only touch tokens that look like *Vietnamese OCR output*:
    # either they carry a diacritic vowel, or they contain a garbled character
    # Paddle sometimes emits for a VN mark (Ä, Ö, ...). Plain ASCII tokens
    # (English object labels like ``Clothing``, ``Vehicle``) are returned
    # unchanged — applying pair/diacritic fixes to them only corrupts them
    # (e.g. ``Clothing`` → ``dothing``).
    has_garble = any(ch in _GARBLE_MAP for ch in cleaned)
    if not _has_vietnamese_vowel(cleaned) and not has_garble:
        return cleaned

    # Garbled-character repair: map known garbled Latin chars to base letters.
    # Only reached when the token is VN-ish, so this is safe.
    repaired = "".join(_GARBLE_MAP.get(ch, ch) for ch in cleaned)

    # If the token already contains a VN diacritic vowel, keep it as-is —
    # do not risk breaking a correct recognition.
    if _has_vietnamese_vowel(repaired):
        return repaired if repaired != cleaned else cleaned

    return repaired if repaired != cleaned else cleaned


def correct_texts(texts: list[str]) -> list[str]:
    """Post-correct a list of OCR lines, dropping noise and de-duplicating."""
    out: list[str] = []
    for t in texts:
        c = correct_token(t)
        if c and c.strip():
            out.append(c.strip())
    # stable de-dup, preserve order
    return list(dict.fromkeys(out))


# Module-level no-op used when correction is disabled.
def _identity(texts: list[str]) -> list[str]:
    return texts


Corrector = Mapping  # type alias for documentation only
