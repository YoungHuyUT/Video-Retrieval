from __future__ import annotations

from aic2026.qa.ocr import OCRTextExtractor


class _V3Result:
    def __init__(self, payload: dict) -> None:
        self.json = payload


def test_extract_v3_reads_rec_texts_from_result_payload() -> None:
    results = [
        _V3Result({"res": {"rec_texts": ["BENVENUTI", "Xin chào", "BENVENUTI"]}})
    ]

    assert OCRTextExtractor._extract_v3(results) == ["BENVENUTI", "Xin chào"]


class _BatchedV3OCR:
    def predict(self, paths: list[str]) -> list[_V3Result]:
        return [_V3Result({"res": {"rec_texts": [path]}}) for path in paths]


def test_extract_many_uses_v3_batch_prediction() -> None:
    extractor = OCRTextExtractor()
    extractor._ocr = _BatchedV3OCR()

    assert extractor.extract_many(["one.jpg", "two.jpg"], batch_size=2) == [
        ["one.jpg"],
        ["two.jpg"],
    ]
