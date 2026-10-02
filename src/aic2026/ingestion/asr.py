"""Local ASR (Automatic Speech Recognition) for video retrieval (Phase 4).

WHY THIS EXISTS (spec §6 + paper arxiv 2512.12935v1, "ASR" branch)
------------------------------------------------------------------
The brief lists ASR as a first-class retrieval modality: a query like
*A woman explains the word "remember"* should match a video where someone
*said* that word, even if no on-screen text or visual cue exists.  The paper
runs Whisper Large-v3; we run the **local, free** ``faster-whisper`` family
(the ``tiny``/``base`` checkpoints are <150MB and CPU-friendly) — no paid API,
no cloud inference, satisfying the hard constraints.

DESIGN (coarse-to-fine, no re-run per query)
--------------------------------------------
* Transcription is **offline ingestion**, not per-query.  Each source video is
  transcribed ONCE into a sidecar ``.jsonl`` (one line per video: video_id +
  segments with text + start/end seconds).  The retrieval path only reads the
  sidecar.
* Per-video transcript text is indexed into the same BM25 index as OCR (paper
  treats OCR + ASR as lexical branches fused via RRF).  Optionally, each ASR
  segment is mapped to the nearest keyframe so a spoken keyword can surface the
  keyframe shown while it was said (useful for KIS/QA evidence localization).
* Only ``faster-whisper`` is imported lazily inside :meth:`ASRTranscriber.load`
  so a machine without it still imports this module (degrades gracefully).

The ``tiny`` model is the default because the dev box is CPU-only; pass
``model_size="base"`` or ``"small"`` on a CUDA machine for higher accuracy.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ASRSegment:
    """One transcribed segment with its time span (seconds).

    ``frame`` is the nearest BTC keyframe *video-frame index* for ``start``,
    computed as ``round(start * fps)`` using the *source video's own* fps
    (read from the video file, never assumed).  BTC keyframe filenames ARE the
    original 0-based video frame index, so this lets a spoken keyword surface
    the exact frame it was said on.  ``frame`` is optional — older sidecars
    without it simply fall back to whole-video (BM25) matching.
    """

    text: str
    start: float
    end: float
    frame: int | None = None

    def to_dict(self) -> dict:
        d = {"text": self.text, "start": self.start, "end": self.end}
        if self.frame is not None:
            d["frame"] = self.frame
        return d


@dataclass
class VideoTranscript:
    """Full transcript for one video."""

    video_id: str
    segments: list[ASRSegment]

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments).strip()

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "text": self.full_text,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "VideoTranscript":
        return cls(
            video_id=payload["video_id"],
            segments=[
                ASRSegment(
                    text=s["text"],
                    start=float(s.get("start", 0.0)),
                    end=float(s.get("end", 0.0)),
                    frame=int(s["frame"]) if s.get("frame") is not None else None,
                )
                for s in payload.get("segments", [])
            ],
        )


class ASRTranscriber:
    """Local Whisper transcription via ``faster-whisper``.

    Build once per ingestion run; call :meth:`transcribe_video` per source video.
    All heavy imports are lazy so importing this module never pulls torch on a
    machine that will only read precomputed sidecars.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = "vi",  # AIC 2026 videos are Vietnamese-narrated per memory
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        # AIC 2026 videos are Vietnamese-narrated; pin language for speed/accuracy.
        # ``None`` lets Whisper auto-detect (slower, use for mixed-language sets).
        self.language = language
        self.vad_filter = False
        self._model = None
        self._load_failed = False

    def load(self) -> bool:
        """Load the Whisper model. Returns True on success, False on failure."""
        if self._model is not None or self._load_failed:
            return self._model is not None
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            logger.warning(
                "ASRTranscriber: faster-whisper not installed (%s). "
                "Install with `uv sync --extra asr` or `pip install faster-whisper`. "
                "ASR retrieval will be unavailable (vector + OCR still work).",
                exc,
            )
            self._load_failed = True
            return False
        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            return True
        except Exception as exc:  # noqa: BLE001 — ASR must degrade, not crash
            logger.warning("ASRTranscriber: failed to load Whisper %s: %s",
                           self.model_size, exc)
            self._load_failed = True
            return False

    def transcribe_video(self, video_path: str | Path) -> VideoTranscript:
        """Transcribe one video file into a :class:`VideoTranscript`."""
        video_path = Path(video_path)
        if not self.load() or self._model is None:
            return VideoTranscript(video_id=video_path.stem, segments=[])
        try:
            segments_iter, _info = self._model.transcribe(
                str(video_path),
                language=self.language,
                beam_size=5,
                vad_filter=self.vad_filter,
            )
            segments = [
                ASRSegment(text=s.text.strip(), start=float(s.start), end=float(s.end))
                for s in segments_iter
                if s.text and s.text.strip()
            ]
            return VideoTranscript(video_id=video_path.stem, segments=segments)
        except Exception as exc:  # noqa: BLE001 — one bad video must not abort batch
            logger.warning("ASRTranscriber: error on %s: %s", video_path, exc)
            return VideoTranscript(video_id=video_path.stem, segments=[])

    def unload(self) -> None:
        """Free the model weights (e.g. before a VLM loads on low-RAM machines)."""
        self._model = None


# ---------------------------------------------------------------------------
# Sidecar persistence (one JSONL file, one line per video — append-friendly)
# ---------------------------------------------------------------------------

def save_transcripts_sidecar(
    transcripts: list[VideoTranscript],
    sidecar_path: str | Path,
) -> None:
    """Append/write all *transcripts* to a ``.jsonl`` sidecar (one line/video)."""
    sidecar_path = Path(sidecar_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with sidecar_path.open("w", encoding="utf-8") as handle:
        for t in transcripts:
            handle.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")


def load_transcripts_sidecar(
    sidecar_path: str | Path,
) -> dict[str, VideoTranscript]:
    """Load a ``.jsonl`` ASR sidecar into ``{video_id: VideoTranscript}``."""
    out: dict[str, VideoTranscript] = {}
    # Guard: empty/blank path → return empty (Path("") resolves to "." on Windows
    # and .exists() returns True, which would then crash on .open()).
    raw = (sidecar_path or "").strip() if isinstance(sidecar_path, str) else str(sidecar_path)
    if not raw:
        return out
    sidecar_path = Path(raw)
    if not sidecar_path.exists() or not sidecar_path.is_file():
        return out
    with sidecar_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = VideoTranscript.from_dict(payload)
            out[t.video_id] = t
    return out


def read_video_fps(video_path: "str | Path") -> float:
    """Read the *actual* fps of a source video via OpenCV.

    BTC videos are NOT uniformly 25/30 fps, so the per-video fps must be read
    from the file itself — assuming a single global fps would misplace frames.
    Returns 0.0 if the file cannot be opened / has no fps metadata (callers
    should then skip frame mapping for that video).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return 0.0
    try:
        import cv2
    except ImportError:
        logger.warning("read_video_fps: opencv not installed; cannot read fps.")
        return 0.0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return float(fps or 0.0)
    finally:
        cap.release()


def enrich_sidecar_with_frames(
    sidecar_path: "str | Path",
    video_dir: "str | Path",
    *,
    overwrite: bool = True,
) -> int:
    """Add a ``frame`` field (nearest BTC keyframe index) to every ASR segment.

    For each video in the sidecar we read that video's OWN fps from the source
    file and compute ``frame = round(start * fps)`` for every segment.  The
    result is written back into the same ``.jsonl`` (start/end/text untouched),
    so a spoken keyword can later be mapped to the exact frame it was said on.

    ``video_dir`` must contain the source videos named ``<video_id>.*`` (e.g.
    ``data/raw/Videos``).  Videos whose fps cannot be read are left with
    ``frame = None`` (whole-video fallback) instead of being dropped.

    Returns the number of segments that received a valid ``frame``.
    """
    sidecar_path = Path(sidecar_path)
    video_dir = Path(video_dir)
    transcripts = load_transcripts_sidecar(sidecar_path)
    if not transcripts:
        logger.warning("enrich_sidecar_with_frames: sidecar empty/absent.")
        return 0

    mapped = 0
    for vid, t in transcripts.items():
        video_path = _find_video(video_dir, vid)
        fps = read_video_fps(video_path) if video_path else 0.0
        if fps <= 0:
            logger.warning(
                "enrich: no fps for %s (video not found or unreadable) — "
                "leaving frame=None.",
                vid,
            )
            continue
        for seg in t.segments:
            seg.frame = int(round(seg.start * fps))
            mapped += 1

    if overwrite:
        save_transcripts_sidecar(list(transcripts.values()), sidecar_path)
    logger.info(
        "enrich_sidecar_with_frames: mapped %d segments across %d videos.",
        mapped,
        len(transcripts),
    )
    return mapped


def _find_video(video_dir: Path, video_id: str) -> "Path | None":
    """Locate ``<video_id>.<ext>`` under ``video_dir`` (recursive).

    BTC videos live in per-split subfolders (e.g.
    ``data/raw/Videos/Videos_L21_a/video/L21_V001.mp4``), so we must search
    recursively rather than just the top level.
    """
    exts = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts", ".m4v", ".flv")
    if not video_dir.exists():
        return None
    # Recursive scan, one pass, matching the stem exactly.
    for f in video_dir.rglob("*"):
        if f.is_file() and f.stem == video_id and f.suffix.lower() in exts:
            return f
    return None


__all__ = [
    "ASRSegment",
    "VideoTranscript",
    "ASRTranscriber",
    "save_transcripts_sidecar",
    "load_transcripts_sidecar",
    "read_video_fps",
    "enrich_sidecar_with_frames",
]
