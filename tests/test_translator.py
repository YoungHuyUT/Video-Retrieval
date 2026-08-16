from __future__ import annotations

from aic2026.agent.local_llm import _strip_to_json
from aic2026.agent.translator import (
    Translation,
    translate_query_fields,
)


class _FakeJSONLLM:
    """Mô phỏng một reasoning model: bọc JSON trong <think> + markdown fence."""

    def __init__(self, reply: str) -> None:
        self._reply = reply

    def structured(self, system: str, user: str, schema: type) -> Translation:
        # Giả lập Ollama: trả content thô (có nhiễu), client tự parse.
        return schema.model_validate_json(_strip_to_json(self._reply))


def test_translate_query_fields_translates_vietnamese() -> None:
    llm = _FakeJSONLLM(
        '<think>dịch sang tiếng Anh</think>\n'
        '```json\n{"text": "a speaker in a red shirt giving a speech"}\n```'
    )
    text, _question, _events, changed, source = translate_query_fields(
        "một diễn giả mặc áo đỏ đang phát biểu",
        None,
        None,
        llm,
        use_llm=True,  # LLM chỉ được gọi khi bật rõ ràng (tương ứng nút "Dịch VI→EN")
    )
    assert changed is True
    assert source == "llm"
    assert text == "a speaker in a red shirt giving a speech"


def test_translate_query_fields_offline_runs_without_llm() -> None:
    # The offline dictionary translator always runs, so a Vietnamese query is
    # normalized to English even with no LLM — CLIP text encoder needs English.
    text, _question, _events, changed, source = translate_query_fields(
        "một diễn giả mặc áo đỏ",
        None,
        None,
        None,
    )
    assert changed is True
    assert source == "offline"
    assert text != "một diễn giả mặc áo đỏ"
    assert "speaker" in text.lower() or "red" in text.lower()


def test_offline_translation_handles_sign_text_query() -> None:
    text, _question, _events, _changed, source = translate_query_fields(
        'Hình ảnh tấm bảng có chữ "BENVENUTI"', None, None, None
    )
    assert text == 'an image of a sign with the text "BENVENUTI"'
    assert source == "offline"


def test_offline_translation_keeps_hinh_anh_as_one_phrase() -> None:
    text, _question, _events, _changed, source = translate_query_fields(
        "Hình ảnh con trâu", None, None, None
    )
    assert text == "an image of a buffalo"
    assert source == "offline"


def test_literal_sign_query_does_not_allow_llm_to_paraphrase_intent() -> None:
    llm = _FakeJSONLLM('{"text": "a welcome sign"}')
    text, _question, _events, _changed, source = translate_query_fields(
        'Hình ảnh tấm bảng có chữ "BENVENUTI"', None, None, llm, use_llm=True
    )
    assert text == 'an image of a sign with the text "BENVENUTI"'
    assert source == "offline"



def test_strip_to_json_handles_reasoning_think_tag() -> None:
    content = '<think>Tôi sẽ dịch câu này sang tiếng Anh.</think>\n{"text": "A red speaker"}'
    assert _strip_to_json(content) == '{"text": "A red speaker"}'


def test_strip_to_json_handles_markdown_fence() -> None:
    content = '```json\n{"text": "A red speaker"}\n```'
    assert _strip_to_json(content) == '{"text": "A red speaker"}'


def test_strip_to_json_handles_leading_prose() -> None:
    content = 'Here is the translation:\n{"text": "A red speaker"}'
    assert _strip_to_json(content) == '{"text": "A red speaker"}'


def test_strip_to_json_passthrough_clean_json() -> None:
    assert _strip_to_json('{"text": "A red speaker"}') == '{"text": "A red speaker"}'


def test_translation_schema_parses_noisy_output() -> None:
    noisy = '<think>...</think>\n```json\n{"text": "A red speaker giving a speech"}\n```'
    parsed = _strip_to_json(noisy)
    assert Translation.model_validate_json(parsed).text == "A red speaker giving a speech"
