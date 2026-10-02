"""Shared timestamp <-> frame-number resolution for the video player.

Mục đích (giống MPC-BE Ctrl+G "Show Current Frame Number"):
khi user pause video tại một timestamp bất kỳ, hệ thống phải báo
"bạn đang ở frame số mấy" (0-based, khớp với `frame_idx` của BTC keyframe
map và `frame_id` của retrieval).

QUY ƯỚC DUY NHẤT TRONG TOÀN HỆ THỐNG (phải giữ nhất quán):
- frame_number là 0-based: frame 0 nằm tại timestamp = 0.0 giây.
- Điều này khớp với `frame_idx` trong CSV map BTC và `frame_id` của retrieval,
  nên nếu user mở video từ một kết quả retrieval rồi pause + Ctrl+G, số frame
  trả về sẽ khớp với frame_id của kết quả đó.

CÁCH TÍNH (tránh timestamp × FPS mù quáng):
- Ưu tiên fps THỰC của file video gốc qua cv2 (CAP_PROP_FPS). Đây là fps mà
  HTML5 <video> dùng để đánh chỉ số frame khi playback — nên Ctrl+G pause tại
  ts bất kỳ phải ra frame = round(ts × fps_thực) (CFR, chuẩn MPC-BE).
- Nếu có bản đồ keyframe BTC (CSV có cột n, pts_time, fps, frame_idx):
  thì KHI timestamp yêu cầu KHỚP GẦN ĐÚNG một entry (cùng 1 frame, diff <
  nửa khoảng frame) → lấy `frame_idx` trực tiếp của entry đó (xử lý VFR exact),
  và `actual_frame_timestamp` = pts_time thực của entry. Nếu KHÔNG khớp entry
  nào (pause ở giây lẻ giữa hai keyframe) → vẫn tính CFR round(ts × fps) với
  fps từ map, báo vfr=False.
- Nếu KHÔNG có cả hai: fallback fps_hint (do caller biết) rồi CFR.
  Trường hợp này `actual_frame_timestamp` ước lượng = frame_number / fps.

Kết quả luôn trả về đủ 3 trường theo yêu cầu #4:
- requested_timestamp: timestamp user đưa vào (giây)
- actual_frame_timestamp: timestamp thực tế của frame được chọn (giây)
- frame_number: chỉ số frame 0-based
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
from functools import lru_cache
from pathlib import Path

# CSV map-keyframe BTC batch 2 then batch 1: map-keyframes/{video_id}.csv
# Cột: n, pts_time, fps, frame_idx
_KEYFRAME_MAP_DIR = Path("data/raw/map-keyframes-aic25-b1/map-keyframes")
_KEYFRAME_MAP_DIRS = (
    Path("data/raw/aic26-b2-map-keyframes/map-keyframes"),
    _KEYFRAME_MAP_DIR,
)
_VIDEOS_DIR = Path("data/raw/Videos")


@dataclass
class FrameInfo:
    """Kết quả resolve frame tại một timestamp."""

    video_id: str
    requested_timestamp: float
    actual_frame_timestamp: float
    frame_number: int  # 0-based
    fps: float
    vfr: bool  # True nếu dùng bản đồ thực tế (xử lý VFR), False nếu fallback CFR
    source: str  # 'btc-map' | 'cv2-fps' | 'unknown'


def format_timestamp(seconds: float) -> str:
    """HH:MM:SS.mmm từ giây (MPC-BE style)."""
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        seconds += 1.0
        ms = 0
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _keyframe_csv_path(video_id: str) -> Path | None:
    for directory in _KEYFRAME_MAP_DIRS:
        path = directory / f"{video_id}.csv"
        if path.exists():
            return path
    return None


@lru_cache(maxsize=256)
def _keyframe_pts_by_frame(video_id: str) -> dict[int, float]:
    """Cached exact frame_idx -> PTS seconds mapping for result cards."""
    path = _keyframe_csv_path(video_id)
    if path is None:
        return {}
    mapping: dict[int, float] = {}
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    mapping[int(row["frame_idx"])] = float(row["pts_time"])
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return {}
    return mapping


def frame_timestamp_seconds(
    video_id: str,
    frame_id: int,
    *,
    frame_unit: str | None = None,
    fps_hint: float | None = None,
) -> float | None:
    """Return the frame's playback time, using exact map PTS when available."""
    if frame_unit == "milliseconds":
        return max(0, int(frame_id)) / 1000.0
    pts = _keyframe_pts_by_frame(video_id).get(int(frame_id))
    if pts is not None:
        return pts
    fps = fps_hint or video_fps(video_id)
    if fps and fps > 0:
        return max(0, int(frame_id)) / fps
    return None


@lru_cache(maxsize=4096)
def video_fps_info(video_id: str) -> tuple[float | None, str | None]:
    """Return cached per-video FPS and its source, preferring BTC map metadata.

    The BTC map's ``fps`` is the nominal/base stream rate. For VFR N videos,
    use manifest/map PTS for the exact time; this FPS is descriptive and must
    not be used to replace those timestamps.
    """
    csv_path = _keyframe_csv_path(video_id)
    if csv_path is not None:
        try:
            with csv_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        fps = float(row.get("fps", 0) or 0)
                    except (TypeError, ValueError):
                        continue
                    if fps > 0:
                        return fps, "map-keyframes"
        except OSError:
            pass
    try:
        fps = _video_fps(video_id)
    except (ImportError, OSError, ValueError):
        fps = None
    return (fps, "video-file") if fps and fps > 0 else (None, None)


def video_fps(video_id: str) -> float | None:
    """Cached convenience accessor for per-video nominal FPS."""
    return video_fps_info(video_id)[0]


@lru_cache(maxsize=4096)
def _video_fps(video_id: str) -> float | None:
    """Đọc fps thực tế từ file video qua cv2 (fallback khi thiếu map)."""
    import cv2

    candidates = sorted(_VIDEOS_DIR.glob(f"**/{video_id}.mp4"))
    if not candidates:
        return None
    cap = cv2.VideoCapture(str(candidates[0]))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return float(fps) if fps and fps > 0 else None


def resolve_frame_at_timestamp(
    video_id: str, timestamp: float, *, fps_hint: float | None = None
) -> FrameInfo:
    """Resolve frame number tại `timestamp` (giây) cho video `video_id`.

    Ưu tiên cv2 fps của file gốc (CFR, chuẩn MPC-BE: frame = round(ts × fps)).
    Nếu có bản đồ BTC và timestamp khớp GẦN ĐÚNG 1 entry keyframe (cùng 1
    frame) thì lấy frame_idx thực của entry đó (xử lý VFR exact). Nếu không có
    gì cả → fallback fps_hint rồi CFR.
    Hàm này là utility DUY NHẤT dùng chung cho: Ctrl+G, retrieval result,
    debug, evaluation, frame display.
    """
    ts = max(0.0, float(timestamp))

    # 1) Bản đồ keyframe BTC: dùng KHI tìm được entry khớp gần đúng 1 frame.
    csv_path = _keyframe_csv_path(video_id)
    if csv_path is not None:
        try:
            with csv_path.open(encoding="utf-8", newline="") as handle:
                map_rows = list(csv.DictReader(handle))
        except (OSError, ValueError):
            map_rows = []
        if map_rows:
            map_fps = 0.0
            for r in map_rows:
                try:
                    map_fps = float(r.get("fps", 0) or 0)
                except ValueError:
                    map_fps = 0.0
                if map_fps > 0:
                    break
            if map_fps > 0:
                # ngưỡng: khớp entry nếu |ts - pts| < nửa khoảng frame
                half = 0.5 / map_fps
                best_diff = float("inf")
                best_frame = 0
                best_pts = 0.0
                for r in map_rows:
                    try:
                        pts = float(r["pts_time"])
                        fidx = int(r["frame_idx"])
                    except (KeyError, ValueError):
                        continue
                    diff = abs(pts - ts)
                    if diff < best_diff:
                        best_diff = diff
                        best_frame = fidx
                        best_pts = pts
                if best_diff <= half:
                    # Khớp exact 1 frame → dùng frame_idx thực (VFR-safe).
                    return FrameInfo(
                        video_id=video_id,
                        requested_timestamp=ts,
                        actual_frame_timestamp=best_pts,
                        frame_number=best_frame,
                        fps=map_fps,
                        vfr=True,
                        source="btc-map",
                    )
                # Không khớp entry → dùng CFR với fps map, báo vfr=False.
                fn = int(round(ts * map_fps))
                return FrameInfo(
                    video_id=video_id,
                    requested_timestamp=ts,
                    actual_frame_timestamp=fn / map_fps,
                    frame_number=fn,
                    fps=map_fps,
                    vfr=False,
                    source="btc-map-cfr",
                )

    # 2) Fallback: fps từ file video gốc (cv2) hoặc fps_hint.
    fps = fps_hint or _video_fps(video_id) or 0.0
    if fps <= 0:
        return FrameInfo(
            video_id=video_id,
            requested_timestamp=ts,
            actual_frame_timestamp=ts,
            frame_number=0,
            fps=0.0,
            vfr=False,
            source="unknown",
        )
    frame_number = int(round(ts * fps))
    return FrameInfo(
        video_id=video_id,
        requested_timestamp=ts,
        actual_frame_timestamp=frame_number / fps,
        frame_number=frame_number,
        fps=fps,
        vfr=False,
        source="cv2-fps",
    )


# Alias theo tên gọi trong spec (#6).
def timestamp_to_frame(video_path: str, timestamp: float, fps: float | None = None) -> FrameInfo:
    """Alias of :func:`resolve_frame_at_timestamp` dùng chung cho mọi nơi."""
    return resolve_frame_at_timestamp(video_path, timestamp, fps_hint=fps)
