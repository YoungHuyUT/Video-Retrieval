from __future__ import annotations

import re
import unicodedata
from pathlib import Path


def resolve_keyframe_path(keyframe_path: str | Path | None) -> Path | None:
    """Resolve a manifest-relative or cross-machine keyframe path against actual filesystem locations."""
    if not keyframe_path:
        return None
    candidate = Path(keyframe_path)
    if candidate.exists():
        return candidate

    parts = candidate.parts
    rel_candidates: list[Path] = []
    if "Keyframes" in parts:
        kf_idx = parts.index("Keyframes")
        rel_candidates.append(Path(*parts[kf_idx + 1 :]))
        rel_candidates.append(Path(*parts[kf_idx:]))
    elif len(parts) >= 2:
        rel_candidates.append(Path(parts[-2]) / parts[-1])

    rel_candidates.append(candidate)
    if len(parts) >= 1:
        rel_candidates.append(Path(parts[-1]))

    root = Path.cwd()
    for rel in rel_candidates:
        for probe in (
            Path(r"D:\aichallenge\data\extracted\Keyframes") / rel,
            Path(r"D:\aichallenge\data\extracted") / rel,
            Path(r"D:\aichallenge\data\raw\Keyframes") / rel,
            root / rel,
            root / "data" / rel,
            root / "data" / "raw" / rel,
            root / "data" / "extracted" / rel,
            root / "data" / "raw" / "Keyframes" / rel,
            root / "data" / "extracted" / "Keyframes" / rel,
        ):
            if probe.exists():
                return probe
    return None


_VI_PATTERN = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵđèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]",
    re.IGNORECASE,
)

_VQA_VI_EN_MAP = [
    (r"\bcó bao nhiêu\b|\bmấy\b|\bbao nhiêu\b|\bsố lượng\b|\bđếm\b", "how many"),
    (r"\bmàu gì\b|\bmàu sắc gì\b|\bmàu nào\b", "what color"),
    (r"\bở đâu\b|\btại đâu\b|\bvị trí nào\b", "where"),
    (r"\blà ai\b|\bai\b|\bngười nào\b", "who"),
    (r"\bđang làm gì\b|\blàm gì\b|\bhành động gì\b", "what action is happening"),
    (r"\bchữ gì\b|\bviết gì\b|\bnội dung gì\b", "what text"),
    (r"\blà gì\b|\bcái gì\b|\bvật gì\b", "what is"),
    (r"\bngười\b", "person"),
    (r"\bngười đàn ông\b|\bđàn ông\b", "man"),
    (r"\bngười phụ nữ\b|\bphụ nữ\b", "woman"),
    (r"\btrẻ em\b|\bđứa trẻ\b|\bem bé\b", "child"),
    (r"\báo sơ mi\b|\báo\b", "shirt"),
    (r"\bquần\b", "pants"),
    (r"\bváy\b", "dress"),
    (r"\bmũ\b|\bnón\b", "hat"),
    (r"\bkính\b|\bmắt kính\b", "glasses"),
    (r"\bcà vạt\b|\bcaravat\b", "tie"),
    (r"\bxe hơi\b|\bxe ô tô\b|\bô tô\b|\bxe\b", "car"),
    (r"\bxe máy\b|\bxe mô tô\b", "motorbike"),
    (r"\bxe đạp\b", "bicycle"),
    (r"\bxe buýt\b|\bxe bus\b", "bus"),
    (r"\bxe cứu thương\b", "ambulance"),
    (r"\bxe cảnh sát\b", "police car"),
    (r"\bbiển số xe\b|\bbiển số\b", "license plate"),
    (r"\bbiển hiệu\b|\bbảng hiệu\b|\bbiển báo\b", "signboard"),
    (r"\bphòng\b", "room"),
    (r"\bbàn\b", "table"),
    (r"\bghế\b", "chair"),
    (r"\bđiện thoại\b", "phone"),
    (r"\bmáy tính\b|\blaptop\b", "laptop"),
    (r"\bchó\b|\bcon chó\b", "dog"),
    (r"\bmèo\b|\bcon mèo\b", "cat"),
    (r"\bcây\b", "tree"),
    (r"\bhoa\b", "flower"),
    (r"\btrong ảnh\b|\btrong hình\b|\btrong video\b", "in the image"),
]


def translate_vqa_question(question: str) -> str:
    """Translate or normalize Vietnamese VQA question to English for models like Florence-2."""
    if not question:
        return ""
    q = question.strip()
    if not _VI_PATTERN.search(q):
        return q

    # Apply phrase mappings
    norm_q = q.lower()
    for pattern, repl in _VQA_VI_EN_MAP:
        norm_q = re.sub(pattern, repl, norm_q, flags=re.IGNORECASE)

    # Ensure clean structure
    norm_q = re.sub(r"\s+", " ", norm_q).strip(" ?.,!")
    return f"{norm_q}?"


def clean_vqa_answer(answer: str | None) -> str | None:
    """Clean and standardize raw VLM output into a concise competition-ready answer."""
    if not answer:
        return None
    ans = answer.strip()
    # Strip prompt wrappers
    ans = re.sub(r"^<VQA>.*?>", "", ans)
    ans = re.sub(r"^QA>", "", ans)
    # Strip special tokens and bounding box tokens
    ans = re.sub(r"<loc_\d+>", " ", ans)
    ans = re.sub(r"<[^>]+>", " ", ans)
    ans = ans.replace("<s>", "").replace("</s>", "")
    ans = re.sub(r"[\s\n\r\t]+", " ", ans).strip(" .,;!?")
    if not ans:
        return None
    # Capitalize first letter
    return ans[:1].upper() + ans[1:] if len(ans) > 1 else ans.upper()


def normalize_answer(answer: str) -> str:
    """Safe normalization before comparing VLM and human answers; preserves Vietnamese accents."""
    answer = unicodedata.normalize("NFC", answer).lower().strip()
    answer = re.sub(r"^[\s.,:;!?]+|[\s.,:;!?]+$", "", answer)
    return re.sub(r"\s+", " ", answer)
