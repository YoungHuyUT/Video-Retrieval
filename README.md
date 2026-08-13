# AIC 2026 — Video Retrieval (KIS / Q&A / TRAKE)

Pipeline cho 3 task vòng sơ tuyển AIC 2026, dựa trên **CLIP ViT-B-32** (index chính) + **BM25** (lexical từ Objects/Metadata). Bỏ SigLIP2. Chạy được **CPU-only**; VLM/Q&A GPU làm giai đoạn sau.

## Mục lục
1. [LƯU Ý WINDOWS — cách chạy lệnh](#1-lưu-ý-windows--cách-chạy-lệnh)
2. [Cài đặt](#2-cài-đặt)
3. [Cấu trúc dữ liệu](#3-cấu-trúc-dữ-liệu)
4. [Cách chạy — Chế độ A: chỉ có Keyframes](#4-cách-chạy--chế-độ-a-chỉ-có-keyframes)
5. [Cách chạy — Chế độ B: có đầy đủ data BTC](#5-cách-chạy--chế-độ-b-có-đầy-đủ-data-btc)
6. [Chạy agent (KIS / Q&A / TRAKE)](#6-chạy-agent-kis--qa--trake)
7. [Đánh giá (evaluate)](#7-đánh-giá-evaluate)
8. [Giao diện UI](#8-giao-diện-ui)
9. [Điểm quan trọng về frame_id và metadata](#9-điểm-quan-trọng-về-frame_id-và-metadata)

---

## 1. LƯU Ý WINDOWS — cách chạy lệnh

> ⚠️ **Quan trọng nhất:** bạn đang dùng **Windows (cmd / PowerShell)** thì **KHÔNG dùng dấu `\` xuống dòng** — đó là cú pháp Linux/Mac, cmd báo `'python' is not recognized` hoặc tương tự. Mỗi lệnh phải gõ **trên một dòng duy nhất**:

```bat
:: ✅ ĐÚNG — tất cả trên 1 dòng, cmd chạy được
python -m aic2026.cli embed-keyframes --keyframes-dir data/raw/Keyframes --features-dir data/processed/clip_features --manifest data/processed/derived_manifest.jsonl --features data/processed/derived_features.npy --video-id L21_V001


- Lệnh dài thì **copy nguyên cụm 1 dòng** từ README dán vào terminal là được.
- PowerShell muốn xuống dòng thì dùng `` ` `` (backtick) — nhưng khuyến nghị cứ gõ 1 dòng cho chắc.
- Tất cả code block dạng `bat` trong README này đã được viết **sẵn trên 1 dòng**, copy nguyên văn là chạy được.

---

## 2. Cài đặt

> ⚠️ Nếu bạn **đã có `.venv`** rồi (đang chạy được code), **KHÔNG chạy lại `python -m venv .venv`** — lệnh đó chỉ cần lần đầu.

```bash
# Python 3.11+
python -m venv .venv        # CHỈ LẦN ĐẦU khi chưa có .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# Dùng uv (nhanh, khuyến nghị) — nhớ --extra agent (langgraph cho pipeline agent)
uv sync --extra retrieval --extra models --extra video --extra agent --extra dev
# Hoặc pip
pip install -e ".[retrieval,models,video,agent,dev]"
```

> **Nếu `uv sync` báo "Access denied" (zstandard .pyd):** có tiến trình python đang giữ file. Đóng hết cửa sổ dùng `.venv` (Streamlit, server agent), rồi chạy lại `uv sync`.

Cài Ollama (cho planner/judge) — **model nào cũng được** (qwen3.5:4b, qwen3:8b, llama3.1...). Mặc định code dùng `qwen3.5:4b`; đổi qua flag `--llm-model` nếu cần:
```bash
ollama pull qwen3.5:4b
ollama serve   # chạy ở terminal khác
```

---

## 3. Cấu trúc dữ liệu

```text
data/
├── raw/
│   ├── Videos/        # .mp4 gốc (BTC) — KHÔNG bắt buộc
│   ├── Keyframes/     # keyframe BTC (bắt buộc) — L01_V001/0000.jpg, ...
│   ├── Objects/       # JSON Faster R-CNN (tùy chọn, nên có)
│   ├── Metadata/      # JSON YouTube (tùy chọn, nên có)
│   └── CLIP features/ # 1 file .npy CLIP ViT-B-32 (tùy chọn, nên có)
├── downloads/         # ZIP hỗ trợ BTC (clip-features, map-keyframes, media-info, objects)
└── processed/         # Code tự sinh: manifest, features, index
```

**Nguyên tắc:** pipeline chạy tốt **chỉ với Keyframes** (Chế độ A). Các nhóm còn lại (Objects/Metadata/CLIP features) làm cho retrieval **chính xác hơn** — điền vào sau càng tốt, code tự phát hiện và dùng.

> **Nếu bạn đã tự extract keyframe bằng CLI (`extract-keyframes`)** thì thư mục là `data/raw/keyframes` (chữ thường) thay vì `data/raw/Keyframes`. Các lệnh dưới đây viết theo tên chuẩn BTC (`Keyframes`); bạn đổi cho khớp với tên thư mục thực tế của mình.

---

## 4. Cách chạy — Chế độ A: chỉ có Keyframes

Nếu bạn **chỉ có thư mục Keyframes** (chưa có Objects/Metadata/CLIP features):

```bat
:: 1. Encode toàn bộ keyframe bằng CLIP ViT-B-32 (tạo features + manifest + index)
::    Bỏ --video-id để encode hết; hoặc chỉ 1 video để test nhanh (1 dòng duy nhất)
python -m aic2026.cli embed-keyframes --keyframes-dir data/raw/Keyframes --features-dir data/processed/clip_features --manifest data/processed/derived_manifest.jsonl --features data/processed/derived_features.npy --video-id L21_V001

:: 2. Kiểm tra index khớp (số feature = số manifest)
python -m aic2026.cli check-features data/processed/derived_features.npy data/processed/derived_manifest.jsonl
```

> **Lưu ý:** ở chế độ này `frame_id` = thứ tự (ordinal) trong tên file keyframe (vì chưa có Metadata để map). Chạy KIS/TRAKE vẫn được, nhưng khi bạn có Metadata BTC, hãy chuyển sang Chế độ B để map `frame_id` thật.

---

## 5. Cách chạy — Chế độ B: có đầy đủ data BTC

Nếu bạn có **Keyframes + Objects + Metadata + CLIP features** (đầy đủ 5 nhóm):

### Bước 1 — Giải nén ZIP hỗ trợ (nếu BTC cấp dạng ZIP)
```bat
python -m aic2026.cli unpack-support-assets --archives-dir data/downloads --destination data/raw/support
```

### Bước 2 — Kiểm tra nhận diện 5 nhóm asset
```bat
python -m aic2026.cli inspect-data --raw-dir data/raw
```

### Bước 3 — Probe file CLIP features chính thức (xác định thứ tự hàng trong .npy)
> Nếu tên thư mục có khoảng trắng như `CLIP features`, bọc đường dẫn trong dấu `"..."` — 1 dòng duy nhất:
```bat
python -m aic2026.cli probe-official-features --features "data/raw/CLIP features/clip_feature.npy" --raw-dir data/raw --metadata-dir data/raw/Metadata
```

### Bước 4 — Build manifest + index khớp đúng frame_id thật
```bat
python -m aic2026.cli prepare-official --features "data/raw/CLIP features/clip_feature.npy" --raw-dir data/raw --output-manifest data/processed/official_manifest.jsonl --output-features data/processed/official_features.npy

:: Kiểm tra khớp
python -m aic2026.cli check-features data/processed/official_features.npy data/processed/official_manifest.jsonl
```

### Bước 5 — (Tùy chọn) Nếu không có CLIP features BTC, tự encode từ Keyframes
```bat
python -m aic2026.cli embed-keyframes --keyframes-dir data/raw/Keyframes --features-dir data/processed/clip_features --manifest data/processed/derived_manifest.jsonl --features data/processed/derived_features.npy
```

---

## 6. Chạy agent (KIS / Q&A / TRAKE)

Tạo file query JSON theo schema `Query` (`query_id`, `type`, `text`, `question?`, `events?`):

**KIS** (`query_kis.json`):
```json
{
  "query_id": "q_kis_1",
  "type": "kis",
  "text": "Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời, phía sau có nhiều cây xanh."
}
```

**Q&A** (`query_qa.json`):
```json
{
  "query_id": "q_qa_1",
  "type": "qa",
  "text": "Trong video về lễ trao giải thưởng âm nhạc",
  "question": "Có bao nhiêu người lên sân khấu để nhận giải thưởng lớn nhất?"
}
```

**TRAKE** (`query_trake.json`):
```json
{
  "query_id": "q_trake_1",
  "type": "trake",
  "text": "Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy",
  "events": ["Giậm nhảy", "Bay qua xà", "Tiếp đất", "Đứng dậy"]
}
```

Chạy:
```bat
python -m aic2026.cli agent-query --query query_kis.json --features data/processed/derived_features.npy --manifest data/processed/derived_manifest.jsonl --output outputs/result_kis.json
```

**Lưu ý Q&A — VLM vision qua Ollama (khuyến nghị, CPU-friendly):**
Q&A gắn `answer` qua model vision chạy trên **Ollama** (`qwen2.5vl:3b`), không cần torch/transformers cho bước này:
```bat
:: Pull model vision một lần
ollama pull qwen2.5vl:3b

:: Chạy Q&A với backend VLM mặc định = ollama
python -m aic2026.cli agent-query --query query_qa.json --features data/processed/derived_features.npy --manifest data/processed/derived_manifest.jsonl --vlm-backend ollama --vlm-model qwen2.5vl:3b --output outputs/result_qa.json
```
Các option: `--vlm-backend ollama|transformers|none`, `--vlm-model` (tên model trên Ollama), `--vlm-timeout`. Backend `transformers` (Qwen2.5-VL-3B) nặng và cần GPU — chỉ dùng nếu bạn có VRAM. Nếu VLM không trả lời được, pipeline vẫn chạy nhưng candidate Q&A sẽ có `answer` rỗng → submission không hợp lệ.

### Các bước pipeline (KIS/Q&A)
```
plan (LLM) → retrieve (CLIP vector + BM25 → RRF fusion) → rerank (metadata keyword bonus) → judge (LLM chọn evidence) → finalize (VLM cho Q&A)
```
- **retrieve**: fusion qua RRF hai bảng xếp hạng (vector CLIP và BM25 lexical) — xem `hybrid_retrieve_raw`.
- **rerank**: `rerank_with_metadata` cộng điểm thưởng nhỏ (mặc định `weight=0.05`) cho candidate có `object_labels`/`title`/`description` chứa từ khóa query. Đây là **heuristic đếm từ khóa, không phải model**. TRAKE không qua bước này (dùng DP alignment riêng).
- **judge**: text LLM (`qwen3.5:4b`) chọn `selected_vector_ids` từ evidence đã retrieval.

---

## 7. Đánh giá (evaluate)

Đánh giá 1 query với ground-truth theo đúng công thức R@k + Final Score của BTC.

> ⚠️ **Lệnh `evaluate` nhận 3 argument vị trí (`query candidates ground_truth`), KHÔNG phải flag `--query/--candidates/--ground-truth`.**
> Đồng thời, `agent-query` ghi file kết quả dạng object `{candidates, plan, trace}` — nhưng `evaluate` cần **một JSON array** `[{...candidate}, ...]`. Vì vậy phải trích mảng `candidates` ra trước:

```bat
:: 1. Trích mảng candidates từ kết quả agent-query (dùng JQ nếu có, hoặc script nhỏ bên dưới)
jq ".candidates" outputs/result_kis.json > outputs/candidates_kis.json

:: 2. Đánh giá — lưu ý 3 argument vị trí, không có --query/--candidates
python -m aic2026.cli evaluate query_kis.json outputs/candidates_kis.json ground_truth_kis.json
```

Nếu không có `jq`, tạo `outputs/candidates_kis.json` bằng lệnh sau:
```bat
python -c "import json; d=json.load(open('outputs/result_kis.json', encoding='utf-8')); json.dump(d['candidates'], open('outputs/candidates_kis.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)"
```

**Ground truth** theo schema `GroundTruth` (`video_id`, `ranges: [[start,end], ...]`, `answer?`):
```json
{
  "video_id": "L01_V001",
  "ranges": [[500, 510]]
}
```

> **Q&A:** đánh giá **ngữ nghĩa** (đúng quy định BTC) — không so chuỗi chính xác. "màu xanh" vs "xanh", "xanh lá cây" vs "màu xanh lá" đều tính là khớp.

---

## 8. Giao diện UI

> ⚠️ Chạy **từ thư mục gốc dự án** (`D:\MinhHuy\AIC2026`) — đường dẫn tương đối trong code (`data/...`, `src/...`) được tính theo thư mục làm việc hiện tại.

```bat
:: Backend FastAPI (terminal 1) — mở trong thư mục gốc dự án
python -m aic2026.cli serve

:: Giao diện Streamlit (terminal 2)
streamlit run src/aic2026/app/ui.py
```

Mở `http://127.0.0.1:8000` (API docs) và `http://127.0.0.1:8501` (UI).

---

## 9. Điểm quan trọng về frame_id và metadata

1. **frame_id là frame index thật của video**, không phải thứ tự keyframe. Khi có Metadata BTC (field `frame_indices`/`keyframe_indices`), `prepare-official` sẽ map đúng. Ở Chế độ A (chỉ Keyframes) thì tạm dùng ordinal.
2. **Metadata filter tăng precision:** với query như "người đi bộ trong cửa hàng", pipeline có thể **lọc video theo metadata** ("cửa hàng") trước khi CLIP search — giảm frame sai, tăng precision. Code đã có sẵn `filter_videos_by_metadata`.
3. **Videos (.mp4) không bắt buộc** — chỉ cần khi re-extract keyframe fallback.
4. **Submission tối đa 100 answers/query**, chỉ xuất field đúng chuẩn BTC (`video_id`, `frame_id`, `answer`, `frame_ids`).

---

## Cấu hình

Cấu hình chạy được truyền trực tiếp qua CLI (`aic2026 --help`) và qua UI "Cấu hình nâng cao" (Backend API, Manifest, Feature, CLIP pretrained, Ollama model/URL, Metadata/Object filter, `coarse_top_k`). Không dùng file config YAML.

## Test

```bash
python -m pytest tests/ -q
```
