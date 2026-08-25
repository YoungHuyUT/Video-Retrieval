# 🚀 AIC 2026 — HƯỚNG DẪN VẬN HÀNH HỆ THỐNG VIDEO RETRIEVAL (PEGASUS)

> **Tài liệu hướng dẫn dành cho toàn bộ thành viên trong đội.**
> Hệ thống hỗ trợ tìm kiếm đa phương thức: **CLIP / SigLIP Vector Search + BM25 Từ khóa vật thể + Lời thoại ASR (Whisper) + Rerank Màu sắc + VLM Q&A**.

---

## 📑 MỤC LỤC
1. [Cài đặt môi trường từ đầu (Dành cho máy mới)](#1-cài-đặt-môi-trường-từ-đầu-dành-cho-máy-mới)
2. [Cấu trúc dữ liệu yêu cầu](#2-cấu-trúc-dữ-liệu-yêu-cầu)
3. [Tự Encode Keyframes bằng GPU (Mô hình mới)](#3-tự-encode-keyframes-bằng-gpu-mô-hình-mới)
4. [Ánh xạ Lời thoại ASR & Màu sắc vào Manifest](#4-ánh-xạ-lời-thoại-asr--màu-sắc-vào-manifest)
5. [Khởi động Giao diện Web PEGASUS](#5-khởi-động-giao-diện-web-pegasus)
6. [Hướng dẫn Thao tác Tìm kiếm & Xuất file nộp bài BTC](#6-hướng-dẫn-thao-tác-tìm-kiếm--xuất-file-nộp-bài-btc)
7. [Chạy bằng Dòng lệnh (CLI) & Đánh giá Ground-Truth](#7-chạy-bằng-dòng-lệnh-cli--đánh-giá-ground-truth)
8. [Xử lý sự cố thường gặp (Troubleshooting)](#8-xử-lý-sự-cố-thường-gặp-troubleshooting)

---

## 1. Cài đặt môi trường từ đầu (Dành cho máy mới)

### 📌 Yêu cầu:
* **Hệ điều hành:** Windows 10/11 hoặc Linux / macOS.
* **Python:** `3.10` – `3.12` (khuyên dùng Python 3.11).
* **Công cụ quản lý gói (Khuyên dùng):** `uv` (nhanh hơn pip gấp 10 lần).
* **GPU (Khuyên dùng):** NVIDIA GTX 1060 trở lên hoặc RTX Series (để tăng tốc CLIP và Whisper).

### 🛠️ Các bước cài đặt:

Mở **PowerShell** tại thư mục gốc dự án:

```powershell
# 1. Mở quyền chạy script cho PowerShell (nếu gặp lỗi PSSecurityException)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 2. Tạo môi trường ảo với Python 3.11
uv venv --python 3.11 .venv

# 3. Kích hoạt môi trường ảo
.venv\Scripts\activate

# 4. Cài đặt toàn bộ dependencies của dự án
uv sync --extra retrieval --extra models --extra video --extra asr --extra dev

# 5. Cài PyTorch hỗ trợ GPU NVIDIA (CUDA 12.4)
uv pip install --reinstall-package torch --reinstall-package torchvision torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124
```

> **Kiểm tra GPU đã nhận diện:**
> ```powershell
> python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
> ```

---

## 2. Cấu trúc dữ liệu yêu cầu

Dữ liệu giải nén để tại `D:\aichallenge\data\extracted` (hoặc thư mục tương tự trên máy bạn) gồm:

```text
D:\aichallenge\data\extracted\
├── Keyframes/            # Chứa 873 thư mục video (L21_V001, L21_V002...), mỗi thư mục chứa các file ảnh (001.jpg, 002.jpg...)
├── map-keyframes/        # Chứa 873 file CSV (L21_V001.csv...) ánh xạ frame_idx gốc
├── media-info/           # Chứa 873 file JSON (L21_V001.json...) tiêu đề & từ khóa YouTube
├── objects/              # Chứa 873 thư mục JSON nhãn vật thể Faster R-CNN
├── clip-features-32/     # (Tùy chọn) 873 file .npy CLIP ViT-B-32 gốc của BTC
└── Videos/               # (Tùy chọn) Video .mp4 gốc
```

---

## 3. Tự Encode Keyframes bằng GPU (Mô hình mới)

Nếu bạn muốn sử dụng mô hình CLIP / SigLIP mạnh hơn (ví dụ `ViT-L-14`, `ViT-SO400M-14-SigLIP-384`, `ViT-B-16-SigLIP`):

### Chạy encode toàn bộ 873 video bằng GPU:
```powershell
python -m aic2026.cli embed-keyframes `
  --keyframes-dir "D:\aichallenge\data\extracted\Keyframes" `
  --map-keyframes-dir "D:\aichallenge\data\extracted\map-keyframes" `
  --features-dir "data/processed/my_features" `
  --manifest "data/processed/derived_manifest.jsonl" `
  --features "data/processed/derived_features.npy" `
  --device cuda `
  --batch-size 64 `
  --clip-model "ViT-L-14" `
  --clip-pretrained "openai"
```

* **Thời gian:** Khoảng 3 – 5 phút trên RTX 4070 / RTX 3060 cho toàn bộ 177,000 ảnh.
* **Cơ chế Resume:** Nếu bị gián đoạn, khi chạy lại sẽ **tự động tiếp tục từ video chưa xong**.
* **Đầu ra:** Tạo ra 2 file chính là `data/processed/derived_manifest.jsonl` và `data/processed/derived_features.npy`.

---

## 4. Ánh xạ Lời thoại ASR & Màu sắc vào Manifest

### 4.1. Nạp dữ liệu ASR (Lời thoại) vào Manifest
*(Đã có sẵn thư mục ASR tại `D:\aichallenge\Video-Retrieval\data\processed\asr`)*:

```powershell
python -m aic2026.cli asr-manifest `
  --manifest "data/processed/derived_manifest.jsonl" `
  --asr-dir "D:\aichallenge\Video-Retrieval\data\processed\asr"
```
*Lệnh này gắn lời thoại theo mốc thời gian vào đúng từng keyframe, giúp tìm kiếm được cả những câu thoại nhân vật nói trong video.*

### 4.2. Tiền trích xuất đặc trưng màu sắc (Tùy chọn - Tăng tốc Rerank màu)
```powershell
python -m aic2026.cli compute-colour-features `
  --manifest "data/processed/derived_manifest.jsonl" `
  --raw-dir "D:\aichallenge\data\extracted" `
  --output "data/processed/colour_features.jsonl"
```

---

## 5. Khởi động Giao diện Web PEGASUS

### 🟢 Cách 1: Chạy trực tiếp Streamlit (Nhanh & Đơn giản nhất)
```powershell
streamlit run src/aic2026/app/ui.py
```
* Mở trình duyệt tại: **`http://localhost:8501`**

---

### 🔵 Cách 2: Chạy tách biệt Backend API + Frontend (Khuyên dùng khi thi đấu)
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

## 6. Hướng dẫn Thao tác Tìm kiếm & Xuất file nộp bài BTC

### 6.1. Cấu hình mô hình trên Web UI (Khi dùng vector tự encode)
Trên thanh bên trái (Sidebar) của Web UI:
1. Mở mục **"Cấu hình nâng cao"**.
2. Kiểm tra/chỉnh lại đường dẫn:
   * **Manifest Path:** `data/processed/derived_manifest.jsonl`
   * **Features Path:** `data/processed/derived_features.npy`
   * **CLIP Model:** `ViT-L-14` *(hoặc model bạn đã chọn lúc encode)*
   * **CLIP Pretrained:** `openai` *(hoặc checkpoint tương ứng)*
   * **Keyframes Dir:** `D:\aichallenge\data\extracted\Keyframes`

---

### 6.2. Các chế độ tìm kiếm:
1. **Tìm kiếm KIS (Known-Item Search):**
   * Nhập câu mô tả nội dung hình ảnh hoặc lời thoại tiếng Việt: ví dụ *"bản tin thời sự 60 giây"*, *"sạt lở bờ sông Cửu Long"*, *"xe cảnh sát dẫn đường"*.
   * Hệ thống tự động kích hoạt **Hybrid RRF Search** (kết hợp CLIP Vector + BM25 Nhãn vật thể + BM25 Lời thoại ASR).
   * Keyframe khớp lời thoại sẽ hiển thị huy hiệu: `🗣️ Lời thoại: "..."`.
2. **Tìm kiếm Q&A:**
   * Nhập câu truy vấn ngữ cảnh và câu hỏi chi tiết.
3. **Tìm kiếm TRAKE:**
   * Nhập chuỗi hành động / danh sách các sự kiện theo trình tự thời gian.

---

### 6.3. Chọn kết quả & Xuất file nộp BTC
1. Click trực tiếp lên ảnh các keyframe đúng với mô tả câu hỏi.
2. Các ảnh được chọn sẽ có viền hồng đánh dấu.
3. Bấm **"Xuất CSV nộp bài"** trên giao diện để tải về file CSV chuẩn định dạng nộp cho BTC.

---

## 7. Chạy bằng Dòng lệnh (CLI) & Đánh giá Ground-Truth

### 7.1. Chạy truy vấn KIS bằng CLI:
Tạo file `query_kis.json`:
```json
{
  "query_id": "kis_001",
  "type": "kis",
  "text": "bản tin thời sự thông báo sạt lở bờ sông"
}
```

Thực hiện tìm kiếm:
```powershell
python -m aic2026.cli agent-query --query query_kis.json --manifest "data/processed/derived_manifest.jsonl" --features "data/processed/derived_features.npy" --clip-model "ViT-L-14" --clip-pretrained "openai" --output outputs/result_kis.json
```

### 7.2. Xuất CSV từ kết quả JSON:
```powershell
python -m aic2026.cli export-submission --result outputs/result_kis.json --query query_kis.json --output outputs/submission_kis.csv
```

---

## 8. Xử lý sự cố thường gặp (Troubleshooting)

| Vấn đề / Lỗi | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| **`activate.ps1 cannot be loaded...`** | PowerShell chặn script mặc định. | Chạy lệnh: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` |
| **`CUDA Available: False`** | Chưa cài bản PyTorch có CUDA. | Chạy: `uv pip install --reinstall-package torch --reinstall-package torchvision torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124` |
| **Ảnh không hiển thị trên Web UI** | Đường dẫn ảnh trong manifest không khớp. | Vào mục **Cấu hình nâng cao** trên Sidebar Web UI, sửa lại đường dẫn thư mục `Keyframes` cho khớp với máy bạn (`D:\aichallenge\data\extracted\Keyframes`). |
| **Lỗi OpenMP `libiomp5md.dll`** | Trùng runtime OpenMP giữa PyTorch và thư viện khác. | Trong PowerShell chạy: `$env:KMP_DUPLICATE_LIB_OK="TRUE"` trước khi khởi động. |

---
*Chúc đội thi AIC 2026 đạt kết quả cao nhất!*
