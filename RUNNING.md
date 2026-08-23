# AIC 2026 — Hướng dẫn Vận hành Hệ thống Video Retrieval (CLIP + ASR + BM25)

> **Dành cho:** Thành viên/người dùng đã được bàn giao sẵn thư mục **`data/processed/asr/`** (chứa các file lời thoại JSON có mốc thời gian) cùng file **`official_manifest.jsonl`** và **`official_features.npy`**.

---

## 📑 Mục lục
1. [Cài đặt môi trường & Thư viện](#1-cài-đặt-môi-trường--thư-viện)
2. [Cấu trúc thư mục dữ liệu](#2-cấu-trúc-thư-mục-dữ-liệu)
3. [Ánh xạ ASR & Khởi tạo Index](#3-ánh-xạ-asr--khởi-tạo-index)
4. [Khởi động Web UI PEGASUS (Giao diện tìm kiếm)](#4-khởi-động-web-ui-pegasus-giao-diện-tìm-kiếm)
5. [Chạy truy vấn bằng CLI & Xuất file nộp bài BTC](#5-chạy-truy-vấn-bằng-cli--xuất-file-nộp-bài-btc)
6. [Tự trích xuất ASR cho video mới bằng GPU (Nâng cao)](#6-tự-trích-xuất-asr-cho-video-mới-bằng-gpu-nâng-cao)
7. [Bảng xử lý sự cố thường gặp (Troubleshooting)](#7-bảng-xử-lý-sự-cố-thường-gặp-troubleshooting)

---

## 1. Cài đặt môi trường & Thư viện

### Yêu cầu tiên quyết:
* **Hệ điều hành:** Windows 10/11 hoặc Linux / macOS
* **Python:** `3.10` – `3.13` (khuyên dùng môi trường Conda hoặc venv)
* **GPU NVIDIA (Khuyên dùng):** GTX 1060/1070/1080 hoặc RTX-series (8GB VRAM) để tăng tốc ASR và CLIP.

### Các bước cài đặt:

Mở terminal tại thư mục gốc của dự án:

#### ⚡ Cách 1: Dùng `uv` (Khuyên dùng — Cực nhanh ~2-5 giây)
```powershell
# Kích hoạt môi trường ảo
.venv\Scripts\Activate

# Tự động đồng bộ toàn bộ thư viện & extras (retrieval, models, video, asr, dev)
uv sync --extra retrieval --extra models --extra video --extra asr --extra dev
```

#### 📦 Cách 2: Dùng `pip` tiêu chuẩn
```powershell
# 1. Kích hoạt môi trường ảo (ví dụ conda hoặc venv)
# conda activate aic2026  HOẶC  .venv\Scripts\Activate

# 2. Cài đặt trọn gói aic2026 cùng toàn bộ dependencies
pip install -e ".[retrieval,models,video,asr,dev]"

# 3. (Nếu dùng GPU NVIDIA trên Windows) Cài đặt thư viện CUDA runtime cho Whisper
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

> [!NOTE]
> **Kiểm tra FAISS & Cuda GPU:**
> ```powershell
> python -c "import faiss; print('FAISS version:', faiss.__version__)"
> python -c "import ctranslate2; print('CUDA Devices:', ctranslate2.get_cuda_device_count())"
> ```
> *(Nếu máy chưa có FAISS, hệ thống sẽ tự động dùng fallback NumPy Vector Search mà không gây lỗi).*

---

## 2. Cấu trúc thư mục dữ liệu

Hãy sắp xếp các file dữ liệu được cung cấp vào đúng vị trí như sơ đồ bên dưới:

```text
Video-Retrieval/
├── src/                          # Mã nguồn hệ thống
├── tests/                        # Bộ kiểm thử tự động
├── data/
│   ├── raw/                      # Dữ liệu gốc (hoặc trỏ tới thư mục extracted)
│   │   ├── Keyframes/            # Thư mục ảnh keyframe (vd: Keyframes/L21_V001/001.jpg)
│   │   ├── Objects/              # File JSON nhãn vật thể Faster R-CNN BTC
│   │   ├── Metadata/             # File JSON thông tin video BTC
│   │   ├── map-keyframes/        # File CSV ánh xạ keyframe -> frame_idx gốc
│   │   └── Videos/               # File video .mp4 (dùng cho TRAKE dense / VLM)
│   └── processed/                # DỮ LIỆU ĐÃ XỬ LÝ (QUAN TRỌNG NHẤT)
│       ├── official_manifest.jsonl   # Manifest chính (chứa vector_id, frame_id, labels, asr_text)
│       ├── official_features.npy     # Ma trận CLIP vector tương ứng
│       └── asr/                      # THƯ MỤC ASR ĐÃ ĐƯỢC CUNG CẤP
│           ├── L21_V001.json         # Lời thoại theo timestamp: [{"start": 0.0, "end": 4.5, "text": "..."}]
│           ├── L21_V002.json
│           └── ...
```

---

## 3. Ánh xạ ASR & Khởi tạo Index

### 3.1. Ánh xạ lời thoại ASR vào Manifest
Nếu bạn vừa nhận thư mục `data/processed/asr/` và muốn nạp toàn bộ lời thoại vào file `official_manifest.jsonl`:

```powershell
python -m aic2026.cli asr-manifest --manifest "data/processed/official_manifest.jsonl" --asr-dir "data/processed/asr"
```
*Lệnh này sẽ tự động đọc mốc thời gian trong từng file ASR JSON và gắn các câu thoại vào đúng keyframe tương ứng theo khoảng thời gian $T \in [\text{start}-1.5s, \text{end}+1.5s]$.*

### 3.2. Tiền tính toán đặc trưng màu sắc (Tùy chọn - Giúp truy vấn màu sắc siêu nhanh)
```powershell
python -m aic2026.cli compute-colour-features --manifest "data/processed/official_manifest.jsonl" --raw-dir "data/raw" --output "data/processed/colour_features.jsonl"
```

---

## 4. Khởi động Web UI PEGASUS (Giao diện tìm kiếm)

Hệ thống hỗ trợ 2 cách vận hành:

### 🟢 Cách 1: Chạy trực tiếp Streamlit (Đơn giản & Nhanh nhất)
Chỉ cần chạy 1 câu lệnh:
```powershell
streamlit run src/aic2026/app/ui.py
```
* Mở trình duyệt web tại: **`http://localhost:8501`**
* Giao diện sẽ tự động nạp `official_manifest.jsonl`, `official_features.npy`, BM25 và ASR.

---

### 🔵 Cách 2: Chạy tách biệt Backend API + Frontend UI (Khuyên dùng cho thi đấu)
Mở **2 cửa sổ Terminal**:

* **Terminal 1 — Khởi động Backend API Server (giữ mở liên tục để cache RAM):**
  ```powershell
  python -m aic2026.cli serve
  ```
  *(API chạy tại `http://127.0.0.1:8000`)*

* **Terminal 2 — Khởi động Frontend Web Streamlit:**
  ```powershell
  streamlit run src/aic2026/app/ui.py
  ```

---

### 💡 Hướng dẫn tìm kiếm trên giao diện:
1. **Tìm kiếm KIS (Known-Item Search):**
   * Nhập câu mô tả nội dung hình ảnh hoặc lời thoại: ví dụ *"phát thanh viên bản tin 60 giây"*, *"sạt lở bờ sông Cửu Long"*, *"xe cứu thương"*.
   * Hệ thống tự động kích hoạt **Hybrid RRF** (Vector CLIP + Từ khóa BM25 + Lời thoại ASR).
   * Các kết quả có khớp lời thoại sẽ hiển thị huy hiệu: `🗣️ Lời thoại: "..."`.
2. **Tìm kiếm Q&A:**
   * Nhập câu truy vấn ngữ cảnh và câu hỏi chi tiết.
3. **Tìm kiếm TRAKE:**
   * Nhập mô tả chuỗi hành động và danh sách các sự kiện theo thứ tự thời gian.

---

## 5. Chạy truy vấn bằng CLI & Xuất file nộp bài BTC

### 5.1. Chạy truy vấn KIS
Tạo file `query_kis.json`:
```json
{
  "query_id": "kis_001",
  "type": "kis",
  "text": "bản tin thời sự thông báo sạt lở bờ sông"
}
```

Chạy tìm kiếm:
```powershell
python -m aic2026.cli agent-query --query query_kis.json --backend faiss --output outputs/result_kis.json
```

### 5.2. Xuất file CSV định dạng nộp bài cho BTC
Chuyển kết quả JSON sang định dạng CSV chuẩn của BTC (cách nhau bởi `", "`):
```powershell
python -m aic2026.cli export-submission --result outputs/result_kis.json --query query_kis.json --output outputs/submission_kis.csv
```

---

## 6. Tự trích xuất ASR cho video mới bằng GPU (Nâng cao)

Nếu bạn có thêm các file video `.mp4` mới và muốn tự động trích xuất thêm ASR bằng GPU NVIDIA:

```powershell
# Chạy với model 'small' (rất nhanh, độ chính xác cao)
python -m aic2026.cli extract-asr --videos-dir "data/raw/Videos" --output-dir "data/processed/asr" --model-size "small" --device "cuda" --lang "vi" --resume
```

```powershell
# Chạy với model 'large-v3-turbo' (chính xác tối đa cho địa danh/thuật ngữ)
python -m aic2026.cli extract-asr --videos-dir "data/raw/Videos" --output-dir "data/processed/asr" --model-size "large-v3-turbo" --device "cuda" --lang "vi" --resume
```

> **Sau khi trích xuất xong:** Hãy chạy lại lệnh `asr-manifest` (ở Mục 3.1) để cập nhật lời thoại mới vào `official_manifest.jsonl`.

---

## 7. Bảng xử lý sự cố thường gặp (Troubleshooting)

| Triệu chứng / Lỗi | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| **Lỗi OpenMP `libiomp5md.dll already initialized`** | Trùng thư viện runtime giữa Anaconda và PyTorch trên Windows. | Chạy lệnh `$env:KMP_DUPLICATE_LIB_OK="TRUE"` trong PowerShell trước khi chạy lệnh Python. |
| **Lỗi thiếu CUDA DLL (`cublas64_12.dll`)** | Thiếu gói CUDA runtime trên Windows. | Chạy `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`. |
| **Query đầu tiên chạy chậm** | Hệ thống đang nạp model CLIP và xây dựng cây tìm kiếm BM25 lên RAM. | Đây là hiện tượng bình thường; các query tiếp theo sẽ phản hồi tức thì. |
| **Không hiển thị ảnh trên Web UI** | Đường dẫn ảnh trong manifest không khớp với thư mục máy bạn. | Vào **Cấu hình nâng cao** trên thanh bên Streamlit, chỉnh lại đường dẫn thư mục `Keyframes` cho đúng vị trí trên máy bạn. |
| **Kiểm tra tính toàn vẹn hệ thống** | Đảm bảo mã nguồn hoạt động chính xác. | Chạy `pytest` — hệ thống có sẵn 118 unit tests bao phủ 100% các module. |

---

*Hệ thống được phát triển và tối ưu cho cuộc thi AI Challenge 2026.*
