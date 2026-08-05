import sys
from pathlib import Path

# Add src to sys.path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aic2026.data_platform import build_derived_artifacts, extract_deduplicated_keyframes
from aic2026.embeddings import SigLIPEncoder


def main() -> None:
    videos_dir = Path("data/raw/Videos")
    keyframes_dir = Path("data/processed/keyframes")
    features_dir = Path("data/processed/clip_features")
    manifest_out = Path("data/processed/derived_manifest.jsonl")
    features_out = Path("data/processed/derived_features.npy")

    if not videos_dir.exists():
        print(f"LỖI: Thư mục {videos_dir} không tồn tại!")
        print("Vui lòng tạo thư mục data/raw/Videos và chép các file .mp4 vào đó.")
        return

    videos = sorted(list(videos_dir.glob("*.mp4")))
    if not videos:
        print(f"LỖI: Không tìm thấy file .mp4 nào trong {videos_dir}")
        return

    print(f"=== TÌM THẤY {len(videos)} VIDEO ===")
    print("Đang tải mô hình SigLIP2...")
    encoder = SigLIPEncoder("google/siglip2-base-patch16-224")

    for idx, video in enumerate(videos, start=1):
        print(f"[{idx}/{len(videos)}] Đang xử lý: {video.name} ...")
        last_percent = -1

        def progress_callback(current: int, total: int, retained: int) -> None:
            nonlocal last_percent
            if total <= 0:
                return
            percent = int(round(current / total * 100))
            if current == total or percent >= last_percent + 5:
                print(f"   -> tiến độ {current}/{total} ({percent}%) - giữ {retained} frame(s)")
                last_percent = percent

        try:
            report = extract_deduplicated_keyframes(
                video_path=video,
                keyframes_root=keyframes_dir,
                features_root=features_dir,
                encoder=encoder,
                cosine_threshold=0.985,
                progress_callback=progress_callback,
            )
            print(f"   -> Giữ lại {report.kept_frames}/{report.decoded_frames} frames")
        except Exception as exc:
            print(f"   -> LỖI khi xử lý {video.name}: {exc}")

    print("\n=== ĐÓNG GÓI INDEX KẾT QUẢ ===")
    records = build_derived_artifacts(
        keyframes_root=keyframes_dir,
        features_root=features_dir,
        manifest_output=manifest_out,
        features_output=features_out,
    )
    print(f"Thành công! Đã ghi {records} records vào {manifest_out}")
    print("Bây giờ bạn có thể bật Streamlit UI để tìm kiếm!")


if __name__ == "__main__":
    main()
