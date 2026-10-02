from __future__ import annotations

from aic2026.video_archive import archive_key_for_video


def test_archive_key_routes_only_known_btc_video_ids() -> None:
    assert archive_key_for_video("M02_V031") == "M02"
    assert archive_key_for_video("N017-V004") == "N011-N020"
    assert archive_key_for_video("S01-V011") == "S01"
    assert archive_key_for_video("L24_V001") is None
    assert archive_key_for_video("../../not-a-video") is None
