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
_KEEP_RE = re.compile(r"<<<KEEP_(\d+)>>>")


class Translation(BaseModel):
    """Schema đầu ra bắt buộc của LLM khi dịch."""

    text: str


# Prompt CHUẨN HÓA: dịch tiếng Việt → tiếng Anh VÀ sửa lỗi chính tả tiếng Anh.
# Khác với "translator" thuần: nếu input đã là tiếng Anh nhưng sai chính tả
# (vd "a man holding a water bottel"), LLM vẫn phải trả về bản đúng ("bottle").
_SYSTEM_PROMPT = (
    "You are a text normalizer for a video retrieval system. "
    "Produce a single clean English query that: "
    "(1) translates the user's text from Vietnamese to English if needed, and "
    "(2) corrects any English spelling or grammar mistakes in the input. "
    "If the text is already correct English, return it unchanged. "
    "Preserve every placeholder token of the form <<<KEEP_N>>> exactly and "
    "verbatim — do not translate, rename, or drop them. "
    "Return only the normalized text in the 'text' field."
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
    "exactly and verbatim — do not translate, rename, or drop them."
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
    """Dịch một chuỗi (đã bóc ngoặc), gọi Ollama 1 lần. Fallback = giữ nguyên."""
    try:
        result = llm.structured(system_prompt, text, Translation)
        return result.text
    except Exception:
        # Dịch hỏng thì giữ nguyên, retrieval vẫn chạy được.
        return text


def _is_vietnamese(text: str) -> bool:
    return bool(_VI_RE.search(text or ""))


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
    _TRANSLATE_FN = lambda system, r, _llm=llm: _translate_one(r, _llm, system)  # noqa: E731
    try:
        translated = _cached_translate(_SYSTEM_PROMPT, remainder)
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
) -> tuple[str, str | None, list[str], bool]:
    """Chuẩn hóa cả 3 trường text của một query (dịch VI→EN + sửa lỗi EN).

    Trả về ``(text, question, events, changed)``.

    Khác bản cũ (chỉ dịch khi có tiếng Việt), theo mặc định ``correct_english``
    bật nên HỆ THỐNG CŨNG SỬA tiếng Anh đã có lỗi chính tả (vd
    "holding a water bottel" → "holding a water bottle"). Các đoạn trong ngoặc
    được bảo vệ riêng cho từng phần. Fallback an toàn: giữ nguyên phần gốc nếu
    LLM lỗi/thiếu. Khi ``correct_english=False``, chỉ dịch nếu có tiếng Việt.
    """
    if llm is None:
        return text, question, list(events or []), False

    new_text = text
    new_question = question
    new_events: list[str] = list(events or [])

    global _TRANSLATE_FN
    saved = _TRANSLATE_FN
    _TRANSLATE_FN = lambda system, r, _llm=llm: _translate_one(r, _llm, system)  # noqa: E731
    try:
        rem_t, prot_t = _split_protect(text)
        # Chỉ gọi LLM khi thực sự cần (có tiếng Việt, hoặc đang bật sửa EN).
        if _is_vietnamese(rem_t) or correct_english:
            translated_text = _cached_translate(_SYSTEM_PROMPT, rem_t)
            new_text = _restore(translated_text, prot_t).strip()
            if new_text and new_text != text.strip():
                logger.info("Normalized query.text: %r -> %r", text, new_text)

        if question:
            rem_q, prot_q = _split_protect(question)
            if _is_vietnamese(rem_q) or correct_english:
                new_question = _restore(
                    _cached_translate(_SYSTEM_PROMPT, rem_q), prot_q
                ).strip()
                if new_question and new_question != question.strip():
                    logger.info(
                        "Normalized query.question: %r -> %r", question, new_question
                    )

        for i, ev in enumerate(new_events):
            rem_e, prot_e = _split_protect(ev)
            if _is_vietnamese(rem_e) or correct_english:
                new_events[i] = _restore(
                    _cached_translate(_SYSTEM_PROMPT, rem_e), prot_e
                ).strip()
                if new_events[i] and new_events[i] != ev.strip():
                    logger.info("Normalized event[%d]: %r -> %r", i, ev, new_events[i])
    finally:
        _TRANSLATE_FN = saved

    changed = (
        new_text != text
        or (new_question or "") != (question or "")
        or list(events or []) != new_events
    )
    return new_text, new_question, new_events, changed
