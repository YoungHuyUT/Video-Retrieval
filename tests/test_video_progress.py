from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np


class FakeImage:
    def __init__(self, array):
        self.array = array

    def save(self, path, quality=None, optimize=None):
        Path(path).write_bytes(b"fake")


class FakeVideoCapture:
    def __init__(self, path):
        self.path = path
        self._frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]
        self._index = 0

    def isOpened(self):
        return True

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def get(self, prop):
        if prop == 0:
            return self._index
        if prop == 1:
            return len(self._frames)
        return 0

    def release(self):
        return None


class FakeEncoder:
    def encode_images(self, images):
        return np.ones((len(images), 2), dtype=np.float32)


class FakeCv2Module(types.SimpleNamespace):
    CAP_PROP_POS_FRAMES = 0
    CAP_PROP_FRAME_COUNT = 1
    COLOR_BGR2RGB = 0


class FakeImageModule(types.SimpleNamespace):
    @staticmethod
    def fromarray(array):
        return FakeImage(array)


def test_extract_deduplicated_keyframes_reports_progress(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "cv2", FakeCv2Module())
    monkeypatch.setitem(sys.modules, "PIL", types.ModuleType("PIL"))
    sys.modules["PIL"].Image = FakeImageModule()

    from aic2026.data_platform.video_frames import extract_deduplicated_keyframes

    progress_updates: list[tuple[int, int, int]] = []

    report = extract_deduplicated_keyframes(
        video_path=tmp_path / "demo.mp4",
        keyframes_root=tmp_path / "keyframes",
        features_root=tmp_path / "features",
        encoder=FakeEncoder(),
        cosine_threshold=0.985,
        progress_callback=lambda current, total, retained: progress_updates.append((current, total, retained)),
    )

    assert report.decoded_frames == 3
    assert progress_updates[-1][0] == 3
    assert progress_updates[-1][1] == 3
    assert progress_updates[-1][2] >= 1
