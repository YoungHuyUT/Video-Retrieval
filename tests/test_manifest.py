from pathlib import Path

from aic2026.ingestion.manifest import build_manifest


def test_build_manifest_accepts_keyframes_in_processed_dir(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = tmp_path / "processed"
    keyframes_dir = processed_dir / "keyframes" / "L21_V001"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    (keyframes_dir / "000000001.jpg").write_bytes(b"fake")

    output = tmp_path / "manifest.jsonl"
    count = build_manifest(raw_dir, output)

    assert count == 1
    assert output.exists()
    assert "L21_V001" in output.read_text(encoding="utf-8")
