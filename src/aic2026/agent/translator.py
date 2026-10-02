from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import TYPE_CHECKING

from pydantic import BaseModel

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
# IGNORECASE: bản thân placeholder có thể bị lower() (do bước chuẩn hóa tiếng
# Việt), nên regex phải match cả <<<keep_0>>> lẫn <<<KEEP_0>>>.
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
# Không chứa từ đa nghĩa trùng nhau (vd "cho" chỉ lấy 1 nghĩa) để tránh nhiễu.
_VN_EN_TOKENS: dict[str, str] = {
    # người / động vật
    "con": "a", "chu": "a", "cai": "a", "chiec": "a", "mu": "hat",
    "nguoi": "person", "dan": "people", "nam": "man", "nu": "woman",
    "tre": "child", "em": "child", "ba": "grandfather", "me": "mother",
    "bo": "father", "anh": "brother", "chi": "sister",
    "voi": "elephant", "cho": "dog", "meo": "cat", "trau": "buffalo",
    "ngua": "horse", "ho": "tiger", "khi": "monkey", "chim": "bird",
    "ca": "fish", "bo": "cow", "de": "goat", "lon": "pig",
    # hành động
    "noi": "speak", "noi chuyen": "talk", "hat": "sing", "nhay": "dance",
    "chay": "run", "di": "walk", "ngoi": "sit", "dam": "stomp",
    "nhay": "jump", "an": "eat", "uong": "drink", "choi": "play",
    "lam": "do", "cam": "hold", "cuoi": "smile", "khoc": "cry",
    "ngu": "sleep", "doc": "read", "viet": "write", "xem": "watch",
    "nghe": "listen", "chup": "take", "quay": "film", "dua": "hug",
    "vet": "wave", "chi": "point", "dem": "carry",
    # sequential-action vocabulary — critical for TRAKE queries, where an
    # untranslated verb can make one event vector unrelated to every frame.
    "mo": "open", "dong": "close", "vao": "enter", "ra": "exit",
    "do vao": "pour", "rot": "pour", "khuay": "stir", "lac": "shake",
    "nhat": "pick up", "dat": "put down", "dua len": "raise",
    "lay": "take", "nem": "throw", "rua": "wash", "cat": "cut",
    # địa điểm / cảnh
    "nha": "house", "cong": "factory", "truong": "school",
    "benh": "hospital", "cho": "market", "cua hang": "shop",
    "cong vien": "park", "bien": "beach", "nui": "mountain",
    "song": "river", "duong": "road", "san": "field", "san khau": "stage",
    "san van dong": "stadium", "cho": "square", "duong pho": "street",
    # màu sắc
    "mau": "color", "do": "red", "xanh": "green", "vang": "yellow", "trang": "white",
    "den": "black", "tim": "purple", "cam": "orange", "hong": "pink",
    "nau": "brown", "xam": "gray",
    # đồ vật
    "nuoc": "water", "lua": "fire", "bong": "ball", "sach": "book",
    "dien thoai": "phone", "may tinh": "computer", "tui": "bag",
    "ao": "shirt", "quan": "pants", "giay": "shoes", "non": "hat",
    "ban": "table", "ghe": "chair", "cua": "door", "cua so": "window",
    "xe": "vehicle", "oto": "car", "xe may": "motorbike",
    "may bay": "airplane", "tau": "train", "thuyen": "boat",
    # sự kiện / thể thao
    "video": "video", "phim": "film", "su kien": "event", "le": "ceremony",
    "trao giai": "award", "hoi nghi": "conference", "the thao": "sport",
    "banh": "ball", "bong da": "football", "cau long": "badminton",
    "tennis": "tennis", "boi": "swim", "dua": "race", "thi": "compete",
    "tap": "practice", "huan luyen": "train", "chien thang": "win",
    "that bai": "lose", "ghi ban": "score", "van dong": "athlete",
    "khi leu": "clown", "nhac cong": "orchestra", "ca si": "singer",
    "khan gia": "audience", "dao dien": "director", "dien vien": "actor",
    "hoa mi": "flower", "cay": "tree", "bong hoa": "flower",
    # Whisper ASR common spelling corrections (diacritics-removed forms)
    "oong": "ong",      # "ông" misheard
    "aang": "ang",      # "ã" misheard
    "eeng": "eng",      # "ề" misheard
    "oang": "oang",
    "uong": "uong",
    "ueng": "ueng",
    "gi": "di",         # "gi" → "di" (northern dialect)
    "gh": "g",          # "gh" → "g"
    "kh": "k",          # "kh" → "k"
    "ngh": "ng",        # "ngh" → "ng"
    "nh": "n",          # "nh" → "n"
    "ph": "f",          # "ph" → "f"
    "th": "t",          # "th" → "t"
    "tr": "tr",         # "tr" → "tr"
    "ch": "c",          # "ch" → "c"
    "con cho": "con cho",
    "con meo": "con meo",
    "nguoi": "nguoi",
    "nguoi dan": "nguoi dan",
    "xa hoi": "xa hoi",
    "phat trien": "phat trien",
    "cong nghe": "cong nghe",
    "khoa hoc": "khoa hoc",
    "giao duc": "giao duc",
    "y te": "y te",
    "van hoa": "van hoa",
    "lich su": "lich su",
    "dia ly": "dia ly",
    "toan hoc": "toan hoc",
    "vat ly": "vat ly",
    "hoa hoc": "hoa hoc",
    "sinh hoc": "sinh hoc",
}

# Cụm truy vấn thường gặp phải được dịch theo cả nghĩa, không ghép từng từ.
# Đặc biệt hữu ích cho OCR/textual KIS (biển hiệu, bảng chỉ dẫn, logo).
_FIXED_PHRASES: list[tuple[str, str]] = [
    ("xanh duong", "blue"),
    ("xanh la cay", "green"),
    ("xanh la", "green"),
    ("bong den", "lamp"),
    ("hinh anh tam bang co chu", "an image of a sign with the text"),
    ("tam bang co chu", "a sign with the text"),
    ("bang co chu", "a sign with the text"),
    ("co chu", "with the text"),
    ("hinh anh", "an image of"),
    ("tim hinh anh", "find an image of"),
    ("tim video", "find a video of"),
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

    # 1) Thay cụm nhiều từ trước (chuỗi dài hơn ưu tiên). Cụm được thay bằng
    #    placeholder BẢO VỆ (<<<P_N>>>) rồi khôi phục SAU bước dịch từng token,
    #    để các từ tiếng Anh do cụm chèn vào (vd "an" trong "an image of") KHÔNG
    #    bị từ điển tiếng Việt dịch nhầm (ví dụ "an" -> "eat" do "ăn" bỏ dấu
    #    thành "an"). _PHRASE_TABLE đã sắp xếp chuỗi dài trước, nên cụm dài thắng.
    protected_phrases: list[str] = []
    for phrase, en in _PHRASE_TABLE:
        if phrase in lowered:
            placeholder = f"<<<P_{len(protected_phrases)}>>>"
            lowered = lowered.replace(phrase, " " + placeholder + " ")
            protected_phrases.append(en)  # lưu bản TIẾNG ANH để khôi phục sau
    phrase_replaced = bool(protected_phrases)

    # 2) Thay từng token (lookup theo dạng đã bỏ dấu). Token không có trong
    #    từ điển tiếng Việt được giữ nguyên (tiếng Anh). Placeholder <<<P_N>>>
    #    là 1 token nguyên khối, không khớp từ điển, nên được giữ nguyên.
    out = []
    seen_vietnamese = False
    for tok in lowered.split():
        key = _strip_diacritics(tok)
        en = _VN_EN_TOKENS.get(key)
        if en is not None:
            seen_vietnamese = True
        out.append(en if en is not None else tok)
    translated = " ".join(out).strip()

    # Khôi phục các cụm tiếng Anh đã bảo vệ (placeholder -> English gốc).
    for i, en in enumerate(protected_phrases):
        translated = translated.replace(f"<<<P_{i}>>>", en)

    # Nếu không có token tiếng Việt nào được dịch thì input vốn đã là tiếng Anh
    # (hoặc UNKNOWN-token thuần) — trả về nguyên văn, giữ nguyên hoa/thường và dấu
    # câu (quan trọng cho TRAKE events: "The person..." phải được giữ y hệt).
    if not seen_vietnamese and not phrase_replaced:
        return _restore(remainder, protected).strip()

    return _restore(translated, protected).strip()


# ---------------------------------------------------------------------------
# OFFLINE English -> Vietnamese translator (mirror của offline_translate VI->EN).
#
# Dùng cho NHÁNH BM25 tiếng Việt: index BM25 chứa ASR/OCR/metadata TIẾNG VIỆT
# (vd transcript "Botswana phát hiện viên kim cương thô..."), nên query tiếng Anh
# chạy thẳng vào BM25 sẽ KHÔNG khớp ("gemstone" != "kim cương"). Hàm này dịch các
# từ KHÓA domain (khai khoáng, thiên tai, con người, hành động, vật thể...) sang
# tiếng Việt để BM25 bắt được transcript, còn từ không có trong từ điển (vd "a",
# "the", "his") được GIỮ NGUYÊN — vô hại vì không khớp index nhưng cũng không
# làm nhiễu. Chạy trên CPU, O(số từ), không cần LLM.
#
# CHỈ dịch những từ CÓ khả năng xuất hiện trong transcript/metadata tiếng Việt;
# không dịch đại mọi từ (tránh sinh rác). Từ không có trong dict => giữ nguyên EN.
# ---------------------------------------------------------------------------
_EN_VN_TOKENS: dict[str, str] = {
    # khai khoáng / đá quý (domain hay bị lỗi nhất)
    "gemstone": "đá quý", "gem": "đá quý", "diamond": "kim cương",
    "rough": "thô", "raw": "thô", "mine": "mỏ", "mining": "khai thác mỏ",
    "quarry": "mỏ đá", "open-pit": "lộ thiên", "pit": "hố", "ore": "quặng",
    "excavation": "đào", "excavator": "máy xúc", "jewel": "trang sức",
    "precious": "quý", "stone": "đá", "rock": "đá", "mineral": "khoáng sản",
    # thiên tai / cảnh ngoài trời
    "flood": "lũ", "lũ": "lũ", "landslide": "sạt lở", "avalanche": "lở núi",
    "fire": "cháy", "earthquake": "động đất", "storm": "bão",
    "typhoon": "bão", "cyclone": "bão", "drought": "hạn hán",
    "erosion": "xói mòn", "disaster": "thảm họa", "tsunami": "sóng thần",
    "volcano": "núi lửa", "whirlpool": "xoáy nước", "sinkhole": "sụt lún",
    # con người
    "man": "nam", "men": "nam", "woman": "nữ", "women": "nữ",
    "child": "trẻ em", "children": "trẻ em", "person": "người",
    "people": "người dân", "soldier": "lính", "police": "cảnh sát",
    "student": "học sinh", "worker": "công nhân", "farmer": "nông dân",
    "doctor": "bác sĩ", "rescuer": "người cứu hộ", "crowd": "đám đông",
    # hành động
    "hold": "cầm", "holding": "cầm", "carry": "mang", "carrying": "mang",
    "sit": "ngồi", "sitting": "ngồi", "stand": "đứng", "standing": "đứng",
    "run": "chạy", "running": "chạy", "walk": "đi", "dig": "đào",
    "digging": "đào", "explode": "nổ", "explosion": "vụ nổ",
    "rescue": "cứu hộ", "smile": "mỉm cười", "lift": "nâng", "examine": "xem xét",
    # địa điểm / cảnh
    "river": "sông", "mountain": "núi", "beach": "bãi biển", "sea": "biển",
    "road": "đường", "street": "đường phố", "factory": "nhà máy",
    "field": "cánh đồng", "forest": "rừng", "village": "làng", "city": "thành phố",
    "aerial": "trên không", "panoramic": "toàn cảnh", "view": "cảnh",
    # vật thể / trang phục
    "suit": "vest", "tie": "cà vạt", "shirt": "áo sơ mi", "helmet": "mũ bảo hộ",
    "truck": "xe tải", "vehicle": "xe", "car": "ô tô", "boat": "thuyền",
    "chair": "ghế", "table": "bàn", "scarf": "khăn choàng đầu", "headscarf": "khăn choàng đầu",
    # màu sắc (giữ đồng bộ với VI->EN)
    "blue": "xanh", "green": "xanh", "red": "đỏ", "yellow": "vàng",
    "white": "trắng", "black": "đen", "purple": "tím", "pink": "hồng",
    "brown": "nâu", "gray": "xám", "grey": "xám",
}


def offline_translate_en_to_vi(text: str) -> str:
    """Dịch Anh -> Việt (từ điển domain) cho nhánh BM25 tiếng Việt.

    Chỉ thay các từ KHÓA có trong ``_EN_VN_TOKENS``; từ còn lại giữ nguyên EN.
    Trả về chuỗi đã dịch. Nếu KHÔNG có token nào được dịch (query thuần EN không
    thuộc domain, hoặc vốn đã là tiếng Việt) thì trả về bản gốc để caller biết
    không cần chạy nhánh BM25-VI riêng.
    """
    if not text:
        return text
    out = []
    changed = False
    for tok in text.split():
        # tách punctuation đơn giản để "mine." -> "mine" match dict, nhưng giữ lại
        core = tok.strip(".,;:!?\"'()[]{}")
        if not core:
            out.append(tok)
            continue
        vi = _EN_VN_TOKENS.get(core.lower())
        if vi is not None:
            out.append(vi)
            changed = True
        else:
            out.append(tok)
    return " ".join(out).strip() if changed else text


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
