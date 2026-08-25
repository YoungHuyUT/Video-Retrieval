from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import TYPE_CHECKING

from pydantic import BaseModel

from aic2026.agent.types import ModalityDecomposition

if TYPE_CHECKING:
    from aic2026.agent.local_llm import LocalLLM

logger = logging.getLogger(__name__)


# Đoạn nằm trong các loại dấu ngoặc dưới đây được GIỮ NGUYÊN (không dịch):
#   (...)  [ ... ]  { ... }  "..." (ASCII)  "..." (smart)  '...' (smart)
# Dùng cho trường hợp "từ đang tìm kiếm" / tên riêng / chữ trên biển báo.
_BRACKET_RE = re.compile(
    r"\([^()]*\)"  # ( )
    r"|\[[^\]]*\]"  # [ ]
    r"|\{[^{}]*\}"  # { }
    r'|"[^"]*"'  # " "
    r"|“[^”]*”"  # " "
    r"|‘[^’]*’",  # ' '
    re.UNICODE,
)

# Phát hiện ký tự tiếng Việt có dấu → chỉ dịch khi thực sự có tiếng Việt.
_VI_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵđèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]",
    re.IGNORECASE,
)

# Placeholder thay thế tạm thời cho đoạn trong ngoặc, yêu cầu LLM giữ nguyên.
_KEEP_RE = re.compile(r"<<<KEEP_(\d+)>>>", re.IGNORECASE)


class Translation(BaseModel):
    """Schema đầu ra bắt buộc của LLM khi dịch."""

    text: str


# Prompt CHUẨN HÓA: dịch tiếng Việt → tiếng Anh VÀ sửa lỗi chính tả tiếng Anh.
# Khác với "translator" thuần: nếu input đã là tiếng Anh nhưng sai chính tả
# (vd "a man holding a water bottel"), LLM vẫn phải trả về bản đúng ("bottle").
# Quan trọng: output PHẢI là một JSON object duy nhất {"text": "..."} (không dùng
# structured-output format của Ollama vì một số model reasoning — qwen3/qwen3.5 —
# trả rỗng khi bị ép schema; ta parse resilient ở tầng client).
_SYSTEM_PROMPT = (
    "You are a text normalizer for a video retrieval system. "
    "Produce a single faithful, literal English query that: "
    "(1) translates the user's text from Vietnamese to English if needed, and "
    "(2) corrects only obvious English spelling or grammar mistakes. "
    "Do NOT summarize, add details, remove details, reinterpret the search intent, "
    "or replace an object/action/text with a related concept. "
    "For example, translate 'Hình ảnh tấm bảng có chữ <<<KEEP_0>>>' as "
    "'an image of a sign with the text <<<KEEP_0>>>'. "
    "If the text is already correct English, return it unchanged. "
    "Preserve every placeholder token of the form <<<KEEP_N>>> exactly and "
    "verbatim — do not translate, rename, or drop them. "
    "Reply with ONLY a single JSON object of the form {\"text\": \"...\"} "
    "and nothing else — no prose, no markdown, no thinking block."
)

# Prompt dùng khi gộp nhiều phần (text + question + events) vào 1 call.
_MULTI_SYSTEM_PROMPT = (
    "You are a text normalizer for a video retrieval system. For each numbered "
    "section, produce a single clean English query that translates Vietnamese "
    "to English (if needed) and corrects any English spelling/grammar mistakes. "
    "Sections are separated by lines of the form '[N]' (N = 0,1,2,...). Reply "
    "with the SAME sections in the SAME order, each introduced by its '[N]' "
    "marker, with the normalized text on the following line(s). Do not merge or "
    "renumber sections. If a section is already correct English, return it "
    "unchanged. Preserve every placeholder token of the form <<<KEEP_N>>> "
    "exactly and verbatim — do not translate, rename, or drop them. "
    "Reply with ONLY a single JSON object of the form {\"text\": \"...\"} "
    "and nothing else — no prose, no markdown, no thinking block."
)


def _split_protect(text: str) -> tuple[str, list[str]]:
    """Thay thế các đoạn trong ngoặc bằng placeholder, trả về (phần còn lại, danh sách gốc)."""
    protected: list[str] = []

    def _repl(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"<<<KEEP_{len(protected) - 1}>>>"

    remainder = _BRACKET_RE.sub(_repl, text)
    return remainder, protected


def _restore(remainder: str, protected: list[str]) -> str:
    """Khôi phục lại các đoạn trong ngoặc nguyên bản."""

    def _repl(match: re.Match[str]) -> str:
        return protected[int(match.group(1))]

    return _KEEP_RE.sub(_repl, remainder)


def _translate_one(text: str, llm: LocalLLM, system_prompt: str = _SYSTEM_PROMPT) -> str:
    """Dịch một chuỗi (đã bóc ngoặc), gọi Ollama 1 lần.

    Hàm này KHÔNG bắt lỗi: nếu LLM fail (timeout, connection, JSON rỗng…) nó ném
    lên cho caller. Lý do quan trọng: ``_cached_translate`` dùng ``lru_cache``, và
    lru_cache **không cache exception** — nên nếu dịch hỏng, lần gọi sau cùng query
    sẽ THỬ LẠI thay vì dùng kết quả lỗi bị "đóng băng" suốt phiên (lỗi cũ từng làm
    câu tiếng Việt gốc bị cache rồi đưa thẳng vào CLIP gây ra frame sai).
    Việc fallback "giữ nguyên text gốc" nằm ở caller (translate_vi_to_en /
    translate_query_fields).
    """
    return llm.structured(system_prompt, text, Translation).text


def _is_vietnamese(text: str) -> bool:
    return bool(_VI_RE.search(text or ""))


# ---------------------------------------------------------------------------
# OFFLINE Vietnamese -> English translator (không cần LLM).
#
# Dùng làm translator CHÍNH khi không cấu hình LLM, và làm fallback an toàn
# khi cuộc gọi LLM thất bại — nên query tiếng Việt như "Con voi" vẫn ra
# tiếng Anh ("a elephant"/"elephant") cho CLIP text encoder MÀ KHÔNG BAO GIỜ
# báo "LLM error". Tra từ điển lightweight, O(số từ), chạy trên CPU tẹt.
# ---------------------------------------------------------------------------

# Từ điển hạt giống: token tiếng Việt (đã bỏ dấu, viết thường) -> tiếng Anh.
_VN_EN_TOKENS: dict[str, str] = {
    # Mạo từ / đại từ / phân loại
    "con": "a", "chu": "a", "cai": "a", "chiec": "a", "mot": "a", "buc": "a", "tam": "a",
    # Con người / nhân vật / vai trò
    "nguoi": "person", "dan": "people", "nam": "man", "nu": "woman",
    "dan ong": "man", "phu nu": "woman", "con trai": "boy", "con gai": "girl",
    "tre": "child", "em": "child", "tre em": "children", "em be": "baby",
    "nguoi gia": "elderly person", "ong lao": "old man", "ba lao": "old woman",
    "ba": "grandmother", "me": "mother", "bo": "father", "anh": "brother", "chi": "sister",
    # Nghề nghiệp chuyên biệt (AIC / thời sự / phóng sự)
    "phat thanh vien": "news anchor in studio", "bien tap vien": "news editor",
    "phong vien": "reporter with microphone", "nha bao": "journalist",
    "nguoi dan chuong trinh": "host MC", "mc": "host",
    "canh sat": "police officer", "canh sat giao thong": "traffic police officer",
    "cong an": "police officer", "bo doi": "soldier", "quan doi": "military soldiers",
    "linh": "soldier", "bao ve": "security guard", "linh cuu hoa": "firefighter",
    "bac si": "doctor", "y ta": "nurse", "benh nhan": "patient", "duoc si": "pharmacist",
    "hoc sinh": "student in uniform", "sinh vien": "college student",
    "giao vien": "teacher", "thay giao": "male teacher", "co giao": "female teacher",
    "cong nhan": "factory worker in uniform", "tho xay": "construction worker with helmet",
    "nong dan": "farmer", "ngu dan": "fisherman", "dau bep": "chef cook",
    "ca si": "singer on stage", "dien vien": "actor", "nghe si": "artist",
    "vu cong": "dancer", "khan gia": "audience spectators", "trong tai": "referee",
    "cau thu": "football soccer player", "van dong vien": "athlete",
    # Động vật
    "voi": "elephant", "cho": "dog", "meo": "cat", "trau": "buffalo",
    "ngua": "horse", "ho": "tiger", "khi": "monkey", "chim": "bird",
    "ca": "fish", "bo": "cow", "de": "goat", "lon": "pig", "heo": "pig",
    "ga": "chicken", "vit": "duck", "ran": "snake", "ca sau": "crocodile",
    # Hành động / Hoạt động
    "noi": "speak", "noi chuyen": "talk", "phong van": "interview",
    "thuyet trinh": "giving presentation", "phat bieu": "giving speech",
    "hat": "sing", "ca hat": "singing", "nhay": "dance", "nhay mua": "dancing",
    "chay": "run", "chay bo": "jogging", "di": "walk", "di bo": "walking",
    "ngoi": "sit", "dung": "stand", "dam": "stomp", "nhay len": "jump",
    "an": "eat", "an uong": "eating", "uong": "drink", "choi": "play",
    "lam": "do", "cam": "hold", "nam": "hold", "cuoi": "smile", "khoc": "cry",
    "ngu": "sleep", "doc": "read", "viet": "write", "xem": "watch", "nhin": "look",
    "nghe": "listen", "chup": "take photo", "quay": "film", "chup anh": "taking photo",
    "quay phim": "filming", "dua": "hug", "om": "hug embrace", "bat tay": "handshake",
    "cui chao": "bowing", "vo tay": "clapping", "vay tay": "waving",
    "chi": "point", "dem": "carry", "khieng": "carry", "vac": "carry",
    "lai xe": "driving", "chay xe": "riding driving", "dap xe": "cycling",
    "qua duong": "crossing street", "bang qua duong": "crossing street",
    "cheo thuyen": "rowing boat", "cau ca": "fishing", "tha luoi": "casting fishing net",
    "kham benh": "examining patient", "tiem thuoc": "giving injection", "phau thuat": "surgery",
    "chua chay": "extinguishing fire", "dap lua": "putting out fire", "cuu ho": "rescuing",
    "bat giu": "arresting", "truy duoi": "chasing", "kham xet": "searching",
    "mua sam": "shopping", "ban hang": "selling goods", "trao giai": "awarding trophy",
    "da bong": "playing football", "tap gym": "workout gym",
    # Phương tiện giao thông
    "xe": "vehicle", "oto": "car", "xe hoi": "car", "xe oto": "car automobile",
    "xe may": "motorbike scooter", "xe moto": "motorcycle", "xe tay ga": "scooter",
    "xe om": "motorbike taxi", "xe dap": "bicycle", "xe dap dien": "electric bicycle",
    "xe ba gac": "three-wheeled cargo motorcycle", "xe xich lo": "cycle rickshaw cyclo",
    "xe cap cuu": "ambulance", "xe cuu thuong": "ambulance", "xe cuu hoa": "fire engine truck",
    "xe canh sat": "police patrol car", "xe taxi": "taxi cab",
    "xe buyt": "city bus", "xe bus": "bus", "xe khach": "passenger coach bus",
    "xe tai": "cargo truck", "xe container": "container trailer truck",
    "may bay": "airplane", "may bay truc thang": "helicopter", "truc thang": "helicopter",
    "tau hoa": "train", "tau lua": "train", "metro": "metro subway",
    "thuyen": "boat", "tau": "ship", "tau thuy": "ship", "cano": "speed motorboat",
    "thuyen buom": "sailboat", "pha": "ferry", "ghe": "small wooden boat", "xuong": "small canoe",
    # Trang phục & Phụ kiện
    "ao": "shirt", "ao dai": "traditional Vietnamese Ao Dai dress",
    "ao ba ba": "traditional Vietnamese Ao Ba Ba shirt",
    "ao so mi": "buttoned shirt", "ao thun": "t-shirt", "ao phong": "t-shirt",
    "ao khoac": "jacket coat", "ao vest": "suit jacket", "ao len": "sweater",
    "ao mua": "raincoat", "ao phan quang": "reflective vest", "ao blouse": "medical lab coat",
    "dong phuc": "uniform", "quan": "pants", "quan jean": "jeans", "quan bo": "jeans",
    "quan dui": "shorts", "quan short": "shorts", "vay": "skirt", "dam": "dress",
    "non": "hat", "mu": "hat", "non la": "traditional Vietnamese conical leaf hat",
    "mu bao hiem": "motorcycle helmet", "non bao hiem": "safety helmet",
    "mu luoi trai": "cap", "khau trang": "face mask", "kinh": "glasses",
    "kinh ram": "sunglasses", "kinh mat": "glasses", "gang tay": "gloves",
    "giay": "shoes", "dep": "sandals", "ca vat": "necktie", "balo": "backpack", "tui": "bag",
    # Địa điểm / Bối cảnh / Công trình
    "nha": "house", "toa nha": "building", "cao oc": "skyscraper",
    "truong quay": "television news studio", "phong thu": "studio",
    "phong hop": "meeting room", "hoi truong": "auditorium hall", "san khau": "stage",
    "benh vien": "hospital", "phong kham": "clinic", "nha thuoc": "pharmacy",
    "truong": "school", "truong hoc": "school", "lop hoc": "classroom", "thu vien": "library",
    "cho": "market", "cho noi": "floating river market", "sieu thi": "supermarket",
    "trung tam thuong mai": "shopping mall", "cua hang": "shop",
    "quan an": "restaurant", "quan ca phe": "coffee shop", "khach san": "hotel",
    "nha tho": "cathedral church", "chua": "buddhist pagoda temple",
    "nga tu": "street intersection", "nga ba": "three-way junction",
    "vong xoay": "roundabout", "cau vuot": "overpass flyover bridge",
    "ham chui": "underpass tunnel", "cau": "bridge", "via he": "sidewalk",
    "duong": "road", "duong pho": "street", "lan duong": "traffic lane",
    "ben xe": "bus station", "san bay": "airport", "ga tau": "railway station", "ben cang": "harbor port",
    "cong vien": "park garden", "quang truong": "city square", "san van dong": "sports stadium",
    "song": "river", "dong song": "river", "bien": "beach ocean", "bai bien": "beach",
    "dong lua": "rice paddy field", "canh dong": "field", "nui": "mountain", "rung": "forest",
    # Thời tiết / Sự cố / Hiện tượng
    "ngap nuoc": "flooded street with water", "ngap lut": "flood flooding",
    "trieu cuong": "high tide flooding", "mua": "rain", "mua bao": "storm rain",
    "bao": "typhoon storm", "sam set": "lightning",
    "khoi lua": "smoke and fire", "dam chay": "building fire blaze", "chay nha": "house fire",
    "hoa hoan": "fire incident", "khoi den": "black smoke",
    "tai nan": "traffic accident collision", "tai nan giao thong": "traffic crash accident",
    "ket xe": "traffic jam congestion", "un tac": "traffic congestion", "dong duc": "crowded",
    "ban ngay": "daytime daylight", "ban dem": "nighttime darkness", "hoang hon": "sunset", "binh minh": "sunrise",
    # Màu sắc
    "mau": "color", "do": "red", "xanh": "green", "xanh duong": "blue",
    "xanh bien": "blue", "xanh la": "green", "vang": "yellow", "trang": "white",
    "den": "black", "tim": "purple", "cam": "orange", "hong": "pink",
    "nau": "brown", "xam": "gray",
    # Đồ vật & Văn bản
    "nuoc": "water", "lua": "fire", "bong": "ball", "sach": "book",
    "dien thoai": "phone", "may tinh": "computer", "ti vi": "television TV",
    "ban": "table", "ghe": "chair", "cua": "door", "cua so": "window",
    "cay": "tree", "hoa": "flower", "la": "leaf", "co": "grass",
    "bien bao": "traffic road sign", "bang hieu": "storefront sign",
    "bien so xe": "license plate", "logo": "logo emblem", "dong chu": "written text",
}

# Cụm truy vấn thường gặp phải được dịch theo cả nghĩa, không ghép từng từ.
_FIXED_PHRASES: list[tuple[str, str]] = [
    ("hinh anh tam bang co chu", "an image of a sign with the text"),
    ("tam bang co chu", "a sign with the text"),
    ("bang hieu co chu", "a storefront sign with the text"),
    ("bien bao co chu", "a traffic sign with the text"),
    ("bang co chu", "a sign with the text"),
    ("bien co chu", "a sign with the text"),
    ("co chu", "with the text"),
    ("dong chu tren ao", "text written on shirt"),
    ("dong chu tren xe", "text written on vehicle"),
    ("dong chu", "written text"),
    ("chu viet", "written text"),
    ("hinh anh", "an image of"),
    ("tim hinh anh", "find an image of"),
    ("tim video", "find a video of"),
    ("doan video", "a video clip showing"),
    ("canh quay", "a scene showing"),
    ("trong truong quay", "inside news studio"),
    ("tren duong pho", "on the city street"),
    ("ngoai duong", "on the street"),
]


# Ánh xạ riêng cho ký tự tiếng Việt KHÔNG có decomposition Unicode (NFD không tách
# được). Quan trọng nhất là đ/Đ (U+0111/U+0110) — NFD không thể phân rã nó thành
# "d" + dấu gạch ngang, nên bước lọc Mn không bao giờ bỏ được. Thiếu nó khiến từ
# chứa "đ" (vd "đi", "đỏ", "đường") không khớp từ điển và giữ nguyên tiếng Việt.
_STRIP_MAP = str.maketrans(
    {
        "đ": "d",
        "Đ": "D",
    }
)


def _strip_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt để tra từ điển bất kể có dấu hay không.

    NFD + lọc Mn bỏ được phần lớn dấu (à/á/ả…). Riêng đ/Đ (U+0111/U+0110) là ký tự
    precomposed KHÔNG có decomposition nên phải transliterate riêng qua ``_STRIP_MAP``.
    """
    import unicodedata

    text = text.translate(_STRIP_MAP)
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def offline_translate(text: str) -> str:
    """Dịch Việt -> Anh dựa trên từ điển có sẵn. Không cần LLM, không bao giờ lỗi.

    Luôn tra từ điển (bất kể có dấu hay không dấu) — tiếng Việt không dấu
    (vd "con voi") vẫn được dịch. Token nào không có trong từ điển (tiếng Anh
    tự nhiên như "person", "speaking") được giữ nguyên, nên query tiếng Anh
    không bị biến đổi. Ưu tiên thay cụm nhiều từ trước, rồi từng token.
    Kết quả đủ tốt cho CLIP ViT-B/32 (tiếng Anh) so khớp với ảnh.
    """
    if not text:
        return text

    # Giữ nguyên các đoạn trong ngoặc (tên riêng / chữ trên biển báo).
    remainder, protected = _split_protect(text)
    stripped = _strip_diacritics(remainder)
    lowered = stripped.lower()

    # 1) Thay cụm nhiều từ trước bằng placeholder tạm để không bị step 2 dịch đè.
    phrase_placeholders: list[tuple[str, str]] = []
    phrase_replaced = False
    for phrase, en in _PHRASE_TABLE:
        if phrase in lowered:
            phrase_replaced = True
            placeholder = f"___phrase_{len(phrase_placeholders)}___"
            phrase_placeholders.append((placeholder, en))
            lowered = lowered.replace(phrase, f" {placeholder} ")

    # 2) Thay từng token (lookup theo dạng đã bỏ dấu). Token không có trong
    #    từ điển tiếng Việt được giữ nguyên (tiếng Anh).
    out = []
    seen_vietnamese = False
    for tok in lowered.split():
        if tok.startswith("___phrase_") and tok.endswith("___"):
            out.append(tok)
            continue
        key = _strip_diacritics(tok)
        en = _VN_EN_TOKENS.get(key)
        if en is not None:
            seen_vietnamese = True
            out.append(en)
        else:
            out.append(tok)
    translated = " ".join(out).strip()
    for placeholder, en in phrase_placeholders:
        translated = translated.replace(placeholder, en)

    # Nếu không có token tiếng Việt nào được dịch thì input vốn đã là tiếng Anh
    # (hoặc UNKNOWN-token thuần) — trả về nguyên văn, giữ nguyên hoa/thường và dấu
    # câu (quan trọng cho TRAKE events: "The person..." phải được giữ y hệt).
    if not seen_vietnamese and not phrase_replaced:
        return _restore(remainder, protected).strip()

    return _restore(translated, protected).strip()


def _is_literal_text_query(text: str) -> bool:
    """Whether a query asks for text visibly written on an object.

    A quoted string is an exact retrieval constraint.  A small LLM can easily
    paraphrase its surrounding intent (sign/banner/logo), which is worse than a
    literal dictionary translation for this OCR-oriented query type.
    """
    remainder, protected = _split_protect(text)
    normalized = _strip_diacritics(remainder).lower()
    return bool(protected) and any(
        marker in normalized
        for marker in ("co chu", "dong chu", "viet", "text")
    )


# Bảng cụm từ (xây từ _VN_EN_TOKENS, ưu tiên chuỗi dài) - tính 1 lần.
_PHRASE_TABLE: list[tuple[str, str]] = sorted(
    [*_FIXED_PHRASES, *((k, v) for k, v in _VN_EN_TOKENS.items() if " " in k)],
    key=lambda kv: -len(kv[0]),
)


# ---------------------------------------------------------------------------
# CACHE: query giống nhau (hoặc đề BTC lặp) chỉ dịch 1 lần. Key = (system_prompt, text)
# đã bóc ngoặc (vì phần ngoặc không đổi nghĩa dịch). LRU 1024 đủ cho 1 phiên.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1024)
def _cached_translate(system_prompt: str, text: str) -> str:
    return _TRANSLATE_FN(system_prompt, text)


# Biến module-level giữ hàm gọi LLM thực (được set bởi translate_query_fields).
_TRANSLATE_FN = lambda system, text: _translate_one(text, None)  # placeholder


def translate_vi_to_en(
    text: str | None,
    llm: LocalLLM | None,
    correct_english: bool = True,
) -> str:
    """Chuẩn hóa ``text`` thành tiếng Anh sạch: dịch VI→EN và sửa lỗi chính tả EN.

    Khác với bản cũ chỉ dịch khi phát hiện tiếng Việt, hàm này theo mặc định
    (``correct_english=True``) cũng gọi LLM khi input đã là tiếng Anh nhưng có
    lỗi (vd "a man holding a water bottel" → "a man holding a water bottle").

    Fallback an toàn (trả về ``text`` gốc) khi: rỗng, thiếu LLM, hoặc LLM lỗi —
    không bao giờ làm hỏng retrieval. Khi ``correct_english=False``, giữ nguyên
    hành vi cũ: chỉ dịch nếu có tiếng Việt.
    """
    if not text or llm is None:
        return text or ""

    remainder, protected = _split_protect(text)

    # Chế độ tương thích: chỉ dịch khi có tiếng Việt.
    if not correct_english and not _is_vietnamese(remainder):
        return text

    global _TRANSLATE_FN
    saved = _TRANSLATE_FN
    _TRANSLATE_FN = lambda system, r, _llm=llm: _translate_one(r, _llm, system)
    try:
        translated = _cached_translate(_SYSTEM_PROMPT, remainder)
    except Exception:  # noqa: BLE001 — dịch hỏng thì giữ nguyên, retrieval vẫn chạy được.
        logger.warning(
            "Translation failed for query (LLM unavailable/invalid); "
            "using original text. Query: %r",
            text,
        )
        translated = text  # fallback: giữ nguyên text gốc (có dấu ngoặc)
    finally:
        _TRANSLATE_FN = saved

    result = _restore(translated or "", protected).strip()
    if result and result != text.strip():
        logger.info("Normalized query text: %r -> %r", text, result)
    return result


def translate_query_fields(
    text: str,
    question: str | None,
    events: list[str] | None,
    llm: LocalLLM | None,
    correct_english: bool = True,
    use_llm: bool = False,
) -> tuple[str, str | None, list[str], bool, str]:
    """Chuẩn hóa cả 3 trường text của một query (dịch VI→EN + sửa lỗi EN).

    Trả về ``(text, question, events, changed, source)``; ``source`` là
    ``offline``, ``llm`` hoặc ``offline_fallback`` để UI hiển thị đúng trạng thái.

    Mặc định dịch **OFFLINE** (từ điển, không cần LLM) nên không bao giờ báo
    "LLM error" và chạy tức thì trên CPU. Chỉ gọi LLM khi ``use_llm=True``
    (bật rõ ràng, ví dụ muốn sửa lỗi chính tả tiếng Anh tinh vi hơn); nếu LLM
    lỗi/thiếu, tự động fallback về bản offline thay vì giữ nguyên text gốc.

    Đoạn trong ngoặc (tên riêng / chữ trên biển báo) được giữ nguyên.
    """
    # 1) Dịch offline trước — luôn thành công, không phụ thuộc LLM.
    off_text = offline_translate(text)
    off_question = offline_translate(question) if question else question
    off_events = [offline_translate(ev) for ev in (events or [])]

    # Với truy vấn tìm chữ trên ảnh, ưu tiên bản dịch literal/offline để giữ
    # nguyên intent "sign with the text …"; không để LLM diễn giải thành cảnh
    # hoặc vật thể khác. Question/events (nếu có) vẫn được dịch offline.
    if _is_literal_text_query(text):
        changed = (
            off_text != text
            or (off_question or "") != (question or "")
            or list(events or []) != off_events
        )
        return off_text, off_question, off_events, changed, "offline"

    # Nếu không yêu cầu LLM, dùng luôn bản offline.
    if not use_llm:
        changed = (
            off_text != text
            or (off_question or "") != (question or "")
            or list(events or []) != off_events
        )
        if changed:
            logger.info(
                "Offline-normalized query: text=%r question=%r events=%r",
                (text, off_text),
                (question, off_question),
                (events, off_events),
            )
        return off_text, off_question, off_events, changed, "offline"

    # 2) use_llm=True: thử LLM, fallback offline nếu lỗi.
    if llm is None:
        logger.warning("use_llm=True nhưng không có LLM; dùng offline translator.")
        return off_text, off_question, off_events, (
            off_text != text
            or (off_question or "") != (question or "")
            or list(events or []) != off_events
        ), "offline_fallback"

    new_text = off_text
    new_question = off_question
    new_events = list(off_events)
    had_llm_failure = False
    had_llm_success = False

    global _TRANSLATE_FN
    saved = _TRANSLATE_FN
    _TRANSLATE_FN = lambda system, r, _llm=llm: _translate_one(r, _llm, system)
    try:
        rem_t, prot_t = _split_protect(text)
        if _is_vietnamese(rem_t) or correct_english:
            try:
                translated_text = _cached_translate(_SYSTEM_PROMPT, rem_t)
                new_text = _restore(translated_text, prot_t).strip()
                had_llm_success = True
            except Exception:  # noqa: BLE001 — fallback offline đã tính.
                logger.warning(
                    "LLM translate failed for query.text; using offline. Query: %r",
                    text,
                )
                new_text = off_text
                had_llm_failure = True

        if question:
            rem_q, prot_q = _split_protect(question)
            if _is_vietnamese(rem_q) or correct_english:
                try:
                    new_question = _restore(
                        _cached_translate(_SYSTEM_PROMPT, rem_q), prot_q
                    ).strip()
                    had_llm_success = True
                except Exception:  # noqa: BLE001 — fallback offline.
                    logger.warning(
                        "LLM translate failed for query.question; using offline. "
                        "Question: %r",
                        question,
                    )
                    new_question = off_question
                    had_llm_failure = True

        for i, ev in enumerate(events or []):
            rem_e, prot_e = _split_protect(ev)
            if _is_vietnamese(rem_e) or correct_english:
                try:
                    new_events[i] = _restore(
                        _cached_translate(_SYSTEM_PROMPT, rem_e), prot_e
                    ).strip()
                    had_llm_success = True
                except Exception:  # noqa: BLE001 — fallback offline.
                    logger.warning(
                        "LLM translate failed for event[%d]; using offline. Event: %r",
                        i,
                        ev,
                    )
                    new_events[i] = off_events[i]
                    had_llm_failure = True
    finally:
        _TRANSLATE_FN = saved

    changed = (
        new_text != text
        or (new_question or "") != (question or "")
        or list(events or []) != new_events
    )
    source = "offline_fallback" if had_llm_failure else ("llm" if had_llm_success else "offline")
    return new_text, new_question, new_events, changed, source


_DECOMPOSITION_SYSTEM_PROMPT = (
    "You are a multimodal query routing and decomposition agent for video retrieval (AAAI 2026). "
    "Given a user query (in Vietnamese or English), decompose it into 3 modality-specific sub-queries and assign importance weights (w_vis, w_ocr, w_asr) summing to 1.0:\n"
    "1. visual_query (w_vis): visual concepts, actions, scenes, objects, colors seen on screen.\n"
    "2. ocr_query (w_ocr): textual cues read on screen, banners, jersey names, signs, logos.\n"
    "3. asr_query (w_asr): spoken dialogue, speech keywords, commentary, narration heard in audio.\n\n"
    "Rules:\n"
    "- If query specifically seeks on-screen text, signs, logos, or quoted text, assign high w_ocr (e.g. 0.5 - 0.7).\n"
    "- If query seeks spoken statements, news anchor speech, broadcast dialogue, assign high w_asr (e.g. 0.4 - 0.6).\n"
    "- For standard visual scene descriptions, visual weight w_vis should dominate (e.g. 0.6 - 0.8).\n"
    "- visual_query MUST be translated into clean English.\n"
    "- Reply ONLY with a valid JSON object matching the schema:\n"
    '{"visual_query": "...", "ocr_query": "...", "asr_query": "...", "w_vis": 0.6, "w_ocr": 0.2, "w_asr": 0.2, "reason": "..."}'
)


def offline_decompose_modalities(query_text: str) -> ModalityDecomposition:
    """Deterministic rule-based modality decomposition when LLM is unavailable."""
    text = (query_text or "").strip()
    text_lower = text.lower()
    translated_vis = offline_translate(text)

    # Extract quoted text if any
    quotes = _BRACKET_RE.findall(text)
    quoted_str = " ".join([q.strip('"\'«»“”[]()') for q in quotes if len(q) > 2])

    # OCR indicators
    ocr_keywords = [
        "chữ", "biển", "bảng", "logo", "text", "banner", "sign",
        "tấm bảng", "tiêu đề", "dòng chữ", "áo số", "jersey", "tên hiệu",
        "ghi chữ", "có chữ", "biển hiệu", "bảng hiệu", "khẩu hiệu",
    ]
    has_ocr_cue = bool(quotes) or any(kw in text_lower for kw in ocr_keywords)

    # ASR speech indicators
    asr_keywords = [
        "phát thanh viên", "nói", "nói rằng", "thông báo", "bản tin",
        "lời thoại", "phỏng vấn", "hát", "giọng", "speech", "saying",
        "dialogue", "anchor", "tuyên bố", "phát biểu", "kể về", "chia sẻ",
        "nói về", "trò chuyện", "thảo luận",
    ]
    has_asr_cue = any(kw in text_lower for kw in asr_keywords)

    if has_ocr_cue and not has_asr_cue:
        ocr_q = quoted_str if quoted_str else text
        return ModalityDecomposition(
            visual_query=translated_vis,
            ocr_query=ocr_q,
            asr_query="",
            w_vis=0.35,
            w_ocr=0.55,
            w_asr=0.10,
            reason="Detected text/sign/quoted keyword indicating on-screen OCR importance.",
        )
    elif has_asr_cue and not has_ocr_cue:
        return ModalityDecomposition(
            visual_query=translated_vis,
            ocr_query="",
            asr_query=text,
            w_vis=0.35,
            w_ocr=0.10,
            w_asr=0.55,
            reason="Detected speech/news/dialogue keywords indicating audio ASR importance.",
        )
    elif has_ocr_cue and has_asr_cue:
        return ModalityDecomposition(
            visual_query=translated_vis,
            ocr_query=quoted_str or text,
            asr_query=text,
            w_vis=0.30,
            w_ocr=0.35,
            w_asr=0.35,
            reason="Detected both text and speech cues.",
        )
    else:
        return ModalityDecomposition(
            visual_query=translated_vis,
            ocr_query=text,
            asr_query=text,
            w_vis=0.60,
            w_ocr=0.20,
            w_asr=0.20,
            reason="Default visual-dominant multimodal query with translated English visual text.",
        )


def decompose_query_modalities(
    query_text: str,
    llm: LocalLLM | None = None,
    use_llm: bool = True,
) -> ModalityDecomposition:
    """Decompose query into modality sub-queries & weights (w_vis, w_ocr, w_asr).

    Uses LLM when available; falls back to deterministic rule-based heuristic.
    """
    if not query_text:
        return ModalityDecomposition(visual_query="")

    if use_llm and llm is not None:
        try:
            res = llm.structured(_DECOMPOSITION_SYSTEM_PROMPT, query_text, ModalityDecomposition)
            # Normalize weights so sum is 1.0
            total_w = max(1e-6, res.w_vis + res.w_ocr + res.w_asr)
            res.w_vis = round(res.w_vis / total_w, 3)
            res.w_ocr = round(res.w_ocr / total_w, 3)
            res.w_asr = round(res.w_asr / total_w, 3)
            if not res.visual_query:
                res.visual_query = query_text
            return res
        except Exception as exc:
            logger.warning("LLM query decomposition failed (%s); using offline heuristic.", exc)

    return offline_decompose_modalities(query_text)

