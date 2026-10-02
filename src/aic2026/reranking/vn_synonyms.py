"""Vietnamese→English object-label synonyms (Improvement.md Task 5).

Maps Vietnamese query entities to their English detector labels so that
object_evidence_adjustment() can match VN queries against EN detector output.

Usage::

    from aic2026.reranking.vn_synonyms import normalize_entity
    labels = normalize_entity("xe máy")  # → ["motorcycle", "motorbike"]
    labels = normalize_entity("người")    # → ["person", "people", "human", ...]
"""
from __future__ import annotations

# Vietnamese → English detector labels.
# Keys are lowercase Vietnamese terms; values are lists of English labels
# that the Faster R-CNN / OpenImages detector outputs.
VN_TO_EN: dict[str, list[str]] = {
    # con người
    "người": ["person", "people", "human", "man", "woman"],
    "đàn ông": ["man", "person"],
    "phụ nữ": ["woman", "person"],
    "em bé": ["baby", "child", "person"],
    "trẻ em": ["child", "person"],
    "người đàn ông": ["man", "person"],
    "người phụ nữ": ["woman", "person"],
    "nhóm người": ["crowd", "person"],
    "đám đông": ["crowd", "person"],
    # động vật
    "chó": ["dog", "puppy"],
    "mèo": ["cat", "kitten"],
    "cá": ["fish"],
    "chim": ["bird"],
    "ngựa": ["horse"],
    "voi": ["elephant"],
    "rùa": ["turtle", "sea turtle"],
    "bò": ["cow", "cattle", "bull"],
    "trâu": ["bull", "cattle", "animal"],
    "lợn": ["pig"],
    "gà": ["chicken", "bird"],
    "vịt": ["duck", "bird"],
    # phương tiện
    "xe hơi": ["car", "automobile"],
    "ô tô": ["car", "automobile"],
    "xe ô tô": ["car", "automobile"],
    "xe máy": ["motorcycle", "motorbike"],
    "xe đạp": ["bicycle", "bike"],
    "xe tải": ["truck"],
    "xe buýt": ["bus"],
    "tàu": ["boat", "ship"],
    "thuyền": ["boat", "ship"],
    "máy bay": ["airplane"],
    "tàu hỏa": ["train"],
    # đồ vật
    "điện thoại": ["cell phone", "mobile phone", "phone"],
    "máy tính": ["laptop", "computer"],
    "máy tính xách tay": ["laptop"],
    "bình nước": ["bottle", "water bottle"],
    "chai nước": ["bottle", "water bottle"],
    "ly": ["cup", "wine glass"],
    "cốc": ["cup", "wine glass"],
    "bát": ["bowl"],
    "đĩa": ["plate"],
    "dao": ["knife"],
    "nĩa": ["fork"],
    "thìa": ["spoon"],
    "chảo": ["pan", "pot"],
    "nồi": ["pot"],
    # thực phẩm
    "bánh mì": ["bread"],
    "cơm": ["rice"],
    "trái cây": ["apple", "banana", "orange"],
    "rau": ["broccoli", "vegetable"],
    "thịt": ["meat"],
    "cá (thực phẩm)": ["fish"],
    # quần áo
    "áo": ["shirt"],
    "quần": ["pants", "jeans"],
    "váy": ["dress", "skirt"],
    "giày": ["shoe", "sneaker"],
    "dép": ["sandal", "slipper"],
    "mũ": ["hat", "cap"],
    "nón": ["hat", "cap"],
    "kính": ["sunglasses", "glasses"],
    "túi": ["handbag", "backpack"],
    # cơ thể
    "mặt": ["face"],
    "tay": ["hand"],
    "chân": ["foot", "leg"],
    "mắt": ["eye"],
    "miệng": ["mouth"],
    "tóc": ["hair"],
    #家务 / nhà cửa
    "bàn": ["dining table", "table"],
    "ghế": ["chair"],
    "giường": ["bed"],
    "tủ": ["cabinet", "refrigerator"],
    "tivi": ["tv", "television"],
    "đèn": ["lamp", "light"],
    "cửa": ["door", "window"],
    "thang": ["stairs"],
    "sân": ["tennis court"],
    "bể bơi": ["swimming pool"],
}

# Build reverse index: EN label → VN terms (for bi-directional lookup)
_EN_TO_VN: dict[str, list[str]] = {}
for _vn, _ens in VN_TO_EN.items():
    for _en in _ens:
        _EN_TO_VN.setdefault(_en.lower(), []).append(_vn)


def normalize_entity(entity: str) -> list[str]:
    """Map a Vietnamese (or English) entity to English detector labels.

    Args:
        entity: A query entity string, possibly Vietnamese.

    Returns:
        List of English detector labels. Returns [entity] unchanged if no
        mapping exists (entity is already English or unknown).
    """
    key = entity.strip().lower()
    if not key:
        return [entity]

    # Direct Vietnamese lookup
    if key in VN_TO_EN:
        return VN_TO_EN[key]

    # Already an English detector label?
    if key in _EN_TO_VN:
        return [key]

    # Partial match: "xe máy Shark" → try longest VN prefix
    for vn_key in sorted(VN_TO_EN.keys(), key=len, reverse=True):
        if key.startswith(vn_key) or vn_key.startswith(key):
            return VN_TO_EN[vn_key]

    # Unknown: return as-is (might be an English label)
    return [entity]


def normalize_entities(entities: list[str]) -> list[str]:
    """Map multiple entities to English detector labels, deduplicating.

    Args:
        entities: List of query entity strings.

    Returns:
        Deduplicated list of English detector labels.
    """
    seen: set[str] = set()
    result: list[str] = []
    for entity in entities:
        for label in normalize_entity(entity):
            if label not in seen:
                seen.add(label)
                result.append(label)
    return result
