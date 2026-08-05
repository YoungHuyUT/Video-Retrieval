from __future__ import annotations

import re
import unicodedata


def normalize_answer(answer: str) -> str:
    """Safe normalization before comparing VLM and human answers; preserves Vietnamese accents."""
    answer = unicodedata.normalize("NFC", answer).lower().strip()
    answer = re.sub(r"^[\s.,:;!?]+|[\s.,:;!?]+$", "", answer)
    return re.sub(r"\s+", " ", answer)
