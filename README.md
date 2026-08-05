# VideoTime-Agent

## Quick Start
```bash
uv sync --extra retrieval --extra agent --extra models --extra video --extra dev
```
### Bước 2: Bật Ollama LLM ( Hoặc thay thế bằng API ): Chạy trên terminal khác
```bash
ollama pull qwen2.5:7b
ollama serve
```
### Bước 3: Trích xuất dữ liệu & Tạo Index

- **Cách A: Nếu bạn CHỈ CÓ các file video `.mp4` trong `data/raw/Videos/`:**
  Chạy script tự động (tương thích mọi hệ điều hành Windows/Mac/Linux):
  ```bash
  python scripts/extract_all_videos.py
  ```

- **Cách B: Nếu bạn đã có keyframes, hãy đặt dưới `data/raw/Keyframes/` trước.**
  Đây là vị trí chuẩn trong project này. Các asset BTC khác như `Videos`, `Objects`, `Metadata`, `CLIP features` cũng nên được đặt dưới `data/raw/` theo cấu trúc thư mục tương ứng.

  Lệnh này chỉ tạo manifest JSONL, không tạo file embedding `.npy` cho UI:
  ```bash
  aic2026 prepare --raw-dir data/raw --output data/processed/manifest.jsonl
  ```
  Nếu bạn muốn có cả file embedding cho UI, hãy chạy tiếp lệnh sau:
  ```bash
  aic2026 embed-keyframes --keyframes-dir data/raw/Keyframes --features-dir data/processed/clip_features --manifest data/processed/derived_manifest.jsonl --features data/processed/derived_features.npy
  ```
### Chạy cho 1 video trước, ví dụ L21_V001, cho test thử
```
python -m aic2026.cli embed-keyframes --keyframes-dir data/raw/Keyframes --features-dir data/processed/clip_features --manifest data/processed/derived_manifest.jsonl --features data/processed/derived_features.npy --video-id L21_V001 --batch-size 2
```

### Bước 4: Khởi chạy Giao diện UI
```bash
python -m aic2026.cli serve ( chạy trên terminal khác )
streamlit run src/aic2026/app/ui.py
```
