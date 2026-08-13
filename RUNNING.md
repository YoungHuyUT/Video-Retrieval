# 🚀 CÁCH CHẠY THỬ AIC 2026 (PEGASUS)

---

## 0. Chuẩn bị môi trường (chỉ lần đầu)

Mở **terminal (PowerShell hoặc cmd)**, chuyển vào thư mục gốc dự án:

Kích hoạt virtualenv (đã có sẵn `.venv`, **không** tạo lại):

```bat
.venv\Scripts\Activate
```


---

## 0b2. Bố trí thư mục data (QUAN TRỌNG — đọc trước khi chạy)

Toàn bộ đường dẫn trong mọi lệnh (`data/...`, `src/...`) đều tính **tương đối từ thư mục gốc
`D:\MinhHuy\AIC2026`**. Cấu trúc chuẩn bạn cần tạo/đặt đúng như sau:

```
D:\MinhHuy\AIC2026\
├── data\
│   ├── raw\                      ← Dữ liệu BTC thô (đặt đúng 5 nhóm asset ở đây)
│   │   ├── Videos\               ← video .mp4 (không bắt buộc, chỉ dùng khi re-extract)
│   │   │   ├── L21_V001.mp4
│   │   │   └── ...
│   │   ├── Keyframes\            ← ảnh keyframe .jpg do BTC cấp (BẮT BUỘC)
│   │   │   ├── L21_V001\
│   │   │   │   ├── 001.jpg
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── Objects\              ← Faster R-CNN labels .json (CÓ → BM25 sống)
│   │   │   ├── L21_V001\
│   │   │   │   ├── 001.json
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── Metadata\            ← YouTube title/description .json (CÓ → BM25 sống)
│   │   │   ├── L21_V001.json
│   │   │   └── ...
│   │   └── clip_features\       ← CLIP .npy của BTC (chỉ cần cho prepare-official)
│   │       └── *.npy
│   ├── processed\                ← kết quả build (manifest + features), tự sinh
│   │   ├── derived_manifest.jsonl      (luồng A: KHÔNG có Objects/Metadata)
│   │   ├── derived_features.npy
│   │   ├── official_manifest.jsonl     (luồng B: CÓ Objects/Metadata)
│   │   ├── official_features.npy
│   │   └── clip_features\        ← .npz tạm khi tự encode
│   ├── indexes\chroma\           ← Chroma vector index (tự sinh)
│   └── downloads\                ← ZIP BTC tải về (cho unpack-support-assets)
├── src\aic2026\...
└── outputs\                      ← kết quả agent/evaluate (tự sinh)
```
---

## 0c. Nạp data BTC chính thức (QUAN TRỌNG cho chất lượng retrieval)

> 💻 **Về ký tự nối dòng:** các lệnh dưới dùng `^` (chuẩn **cmd.exe**). Nếu bạn chạy bằng
> **PowerShell**, hãy đổi `^` thành dấu backtick `` ` `` (hoặc gõ 1 dòng duy nhất không xuống dòng).
> Lệnh `python -m aic2026.cli ...` thì không cần nối dòng (gõ 1 dòng là chạy được).

### Tại sao retrieval bị "phẳng" nếu không nạp data đúng cách

Hệ thống retrieval là **hybrid**: `CLIP vector` + `BM25 lexical` (trên Objects/Metadata) → RRF fusion,
cộng thêm `metadata rerank`. Nhưng BM25 và metadata rerank **chỉ lấy text từ 3 trường**
`object_labels` / `title` / `description` của mỗi frame.

- Nếu manifest được build bằng `build-derived-index-input` / `embed-keyframes`
  (`build_derived_artifacts`) → **3 trường trên đều rỗng** (`object_labels=[]`, `title=null`).
- → **BM25 index rỗng** → hybrid chỉ còn chạy mỗi CLIP vector (vốn phẳng trên full-frame video).
- → metadata rerank cộng 0 cho mọi frame.
- → **Kết quả: score các frame xấp xỉ nhau**, không tách bạch được frame đúng/sai.

**Cố quá chỉnh sửa ở tầng rerank không giải quyết được gốc rễ.** Phải cấp text cho BM25.

### Hai trạng thái manifest (tự động thích ứng)

Code đã được sửa để **tự động**: có data thì BM25 chạy, không có data thì bỏ qua (chạy vector thuần):

- `tools.retrieve` chỉ vào hybrid path khi `bm25_index` tồn tại **và không rỗng**
  (`bm25_index.is_empty == False`). Manifest rỗng → bỏ qua BM25, log cảnh báo.
- Khi bạn nạp data và build manifest đúng → BM25 tự động truy xuất, không cần sửa code.

### Cách build manifest CÓ Objects + Metadata (dùng `prepare`)

BTC cung cấp: `Keyframes/`, `Objects/` (Faster R-CNN labels), `Metadata/` (YouTube title/desc),
`CLIP features/` (.npy). Đặt đúng cấu trúc dưới `data/raw/` rồi:

```bat
:: 1) Kiểm tra 5 nhóm asset đã được nhận diện
python -m aic2026.cli inspect-data

:: 2) Build manifest CHÍNH CHỦ — đọc Objects + Metadata (KHÔNG dùng build-derived-index-input)
python -m aic2026.cli prepare --raw-dir data/raw --output data/processed/manifest.jsonl
```


### Dùng luôn CLIP features của BTC (khuyên dùng)

Đừng tự encode lại — BTC đã cho sẵn `.npy` CLIP ViT-B-32 khớp thứ tự keyframe. Dùng
`prepare-official` để vừa lấy features BTC, vừa sinh manifest có Objects/Metadata:

```bat
:: Sinh manifest + .npy đã align với CLIP features BTC
python -m aic2026.cli prepare-official ^
    --features data/raw/<path>/clip_features.npy ^
    --raw-dir data/raw ^
    --output-manifest data/processed/official_manifest.jsonl ^
    --output-features data/processed/official_features.npy
```

Sau đó build index (Chroma hoặc FAISS) từ manifest này:

```bat
python -m aic2026.cli build-chroma-index ^
    --features data/processed/official_features.npy ^
    --manifest data/processed/official_manifest.jsonl ^
    --chroma-dir data/indexes/chroma
```

### Khi chạy agent, trỏ vào manifest có data

```bat
python -m aic2026.cli agent-query ^
    --query query_kis.json ^
    --features data/processed/official_features.npy ^
    --manifest data/processed/official_manifest.jsonl ^
    --output outputs/result_kis.json
```

Hoặc trên UI / backend: đổi biến `manifest` + `features` tương ứng.

### Tóm tắt luồng nạp data (3 bước)

| Bước | Lệnh | Tác dụng |
|---|---|---|
| 1. Kiểm tra | `inspect-data` | Xác nhận 5 nhóm asset (Video/Keyframes/Objects/CLIP/Metadata) có đủ |
| 2. Build | `prepare` hoặc `prepare-official` | Sinh manifest **có Objects + Metadata** (BM25 sống lại) |
| 3. Index | `build-chroma-index` | Build vector index từ manifest đó |

> ⚠️ **Sai lầm hay gặp:** build manifest bằng `build-derived-index-input` (ra `derived_manifest.jsonl`)
> rồi dùng nó cho retrieval → BM25 rỗng → score phẳng. Luôn dùng `prepare`/`prepare-official`.

---

## 1. Kiểm tra data mẫu hiện có

Project đã có sẵn data mẫu để chạy thử ngay:

- `data/raw/Keyframes/L21_V001/*.jpg` — 307 keyframe (ảnh)
- `data/processed/derived_manifest.jsonl` — 307 records (đã có sẵn)
- `data/indexes/chroma/chroma.sqlite3` — Chroma vector index (đã có sẵn)

> ⚠️ `derived_manifest.jsonl` này được build bởi `build-derived-index-input` nên
> **không có Objects/Metadata** → BM25 rỗng, retrieval chỉ chạy CLIP thuần (score dễ bị phẳng).
> Nó dùng được để **chạy thử luồng**, nhưng để có chất lượng thật hãy nạp data BTC và làm theo **mục 0c**.
>
> 📁 **Hiện trạng `data/raw/` trên máy này:** chỉ có `Keyframes/` và `Videos/` (chưa có `Objects/`,
> `Metadata/`, `clip_features/`). Chạy `python -m aic2026.cli inspect-data` sẽ báo thiếu 3 nhóm
> asset đó — bình thường, vì data mẫu chỉ đủ để chạy thử luồng A. Muốn có BM25 (luồng B) thì
> đặt thêm `Objects/`, `Metadata/`, `clip_features/` theo sơ đồ ở **mục 0b2**.

**Thiếu 1 thứ:** file đặc trưng CLIP `data/processed/derived_features.npy`.
Nếu chưa có (hoặc muốn encode lại), chạy bước 2. Nếu đã có rồi thì bỏ qua.

Kiểm tra nhanh:
```bat
dir data\processed\derived_features.npy
```
Có file là xong bước 2.

---

## 0e. Metadata ảnh hưởng thế nào đến KIS / QA / TRAKE

Sau khi có manifest (luồng A hoặc B), metadata được dùng **không đồng đều** giữa 3 task.
Hiểu rõ để biết tại sao kết quả mỗi task khác nhau:

| Cách dùng metadata | KIS | QA | TRAKE |
|---|:---:|:---:|:---:|
| **Pre-filter** (lọc video theo từ khoá object/title) | ✅ | ✅ | ✅ |
| **BM25 lexical** (tìm frame qua text Objects/Metadata) | ✅ (nếu có data) | ✅ (nếu có data) | ❌ không dùng |
| **Rerank bonus** (cộng điểm frame khớp từ khoá) | ✅ | ✅ | ❌ không dùng |
| VLM đọc text metadata khi trả lời | — | ❌ chỉ nhìn ảnh | — |

**Giải thích:**

- **KIS** dùng metadata mạnh nhất: có BM25 + rerank + video-level rerank. Luồng B (có data) sẽ
  vượt trội luồng A (chỉ CLIP).
- **QA** truy xuất giống KIS (luồng B có BM25), nhưng khi sinh đáp án, VLM **chỉ nhận ảnh**,
  không nhận text object/metadata → metadata không giúp trực tiếp vào câu trả lời.
- **TRAKE** chỉ dùng metadata ở bước **pre-filter** (loại video không chứa từ khoá). Bước căn
  chỉnh (DP alignment) luôn chạy CLIP thuần — BM25 và rerank **không áp dụng**. Nên với TRAKE,
  có hay không có Objects/Metadata hầu như không đổi chất lượng alignment.

> ⚠️ **Silent fallback của pre-filter:** nếu bạn gõ từ khoá metadata mà **không video nào khớp**,
> hệ thống sẽ **bỏ qua filter** (trả về mọi video) để không mất recall — và **không báo lỗi**.
> Nên nếu thấy filter "như không tác dụng", hãy kiểm tra lại nhãn object có đúng trong manifest
> (`inspect-data` để xem asset, mở `*.json` trong `Objects/` để xem label).

---

## 2. Sinh đặc trưng CLIP (chỉ khi chưa có features)

```bat
python -m aic2026.cli embed-keyframes --keyframes-dir data/raw/Keyframes --features-dir data/processed/clip_features --manifest data/processed/derived_manifest.jsonl --features data/processed/derived_features.npy --video-id L21_V001
```

Kết quả đúng sẽ in:
`Created 1 feature archives ... wrote 307 records to ... derived_manifest.jsonl and ... derived_features.npy`

> Lần đầu chạy sẽ tải model CLIP từ HuggingFace (có warning HF_TOKEN, bỏ qua được).
> Mất khoảng 1–3 phút cho 307 ảnh.

---

## 3. Cài & chạy Ollama (bắt buộc để agent suy luận)

Agent dùng LLM (`qwen3.5:4b`) để plan/judge và VLM (`qwen2.5vl:3b`) để trả lời Q&A.
Ollama **đã cài** trên máy này (`C:\Users\duong\AppData\Local\Programs\Ollama`),
chỉ cần **bật lên** và **tải model**.

### 3.1 Bật Ollama (terminal riêng, để nguyên)
```bat
ollama serve
```
Giữ terminal này mở. Kiểm tra bằng:
```bat
curl http://127.0.0.1:11434/api/tags
```
Trả JSON là đã chạy.

### 3.2 Tải model (chạy 1 lần)
```bat
ollama pull qwen3.5:4b
ollama pull qwen2.5vl:3b
```
> Không có model → agent báo lỗi kết nối, nhưng UI vẫn hiện được (chỉ không có kết quả).

---

## 4. Chạy Backend (FastAPI) — Terminal 1

```bat
python -m aic2026.cli serve
```
Mở `http://127.0.0.1:8000/health` → trả `{"status":"ok",...}`.
Kiểm tra sẵn sàng data: `http://127.0.0.1:8000/health/ready`
→ `ready: true` khi có cả manifest và features.

> Backend này load index + model 1 lần, UI gọi qua HTTP.

---

## 5. Chạy Giao diện (Streamlit) — Terminal 2

```bat
streamlit run src/aic2026/app/ui.py
```
Mở `http://127.0.0.1:8501`.

## 6. Chạy thử một vòng end-to-end

1. Ở Terminal 2 (UI), chọn **Luồng = kis**.
2. Dán đề vào ô query, ví dụ:
   ```
   Tìm video về một diễn giả mặc áo phát biểu tại sự kiện ngoài trời.
   ```
3. Bấm **Chạy Agent** (có thể bấm ngay khi đang gõ, không cần click ra ngoài).
4. Đợi vài chục giây (agent gọi Ollama plan → retrieve → judge).
5. Gallery hiện các keyframe → **tick** vào ảnh muốn chọn.
6. Bấm **📥 Tải JSON chuẩn BTC** để xuất đáp án.

> Nếu backend chưa chạy → UI báo "Không thể kết nối tới backend".
> Nếu Ollama chưa có model → backend báo lỗi, UI hiện thông báo lỗi tương ứng.

---

## 7. Chạy agent qua CLI (không cần UI) — tùy chọn

Tạo file `query_kis.json`:
```json
{
  "query_id": "q_kis_1",
  "type": "kis",
  "text": "Tìm video về một diễn giả mặc áo phát biểu tại sự kiện ngoài trời."
}
```
Chạy:
```bat
python -m aic2026.cli agent-query --query query_kis.json --features data/processed/derived_features.npy --manifest data/processed/derived_manifest.jsonl --output outputs/result_kis.json
```

Q&A (cần thêm `question`, và Ollama VLM):
```json
{ "query_id":"q_qa_1", "type":"qa", "text":"Video lễ trao giải", "question":"Có bao nhiêu người lên sân khấu?" }
```
```bat
python -m aic2026.cli agent-query --query query_qa.json --features data/processed/derived_features.npy --manifest data/processed/derived_manifest.jsonl --vlm-backend ollama --vlm-model qwen2.5vl:3b --output outputs/result_qa.json
```

TRAKE (cần `events` theo thứ tự):
```json
{ "query_id":"q_trake_1", "type":"trake", "text":"Tìm 4 khoảnh khắc cú nhảy", "events":["Giậm nhảy","Bay qua xà","Tiếp đất","Đứng dậy"] }
```
```bat
python -m aic2026.cli agent-query --query query_trake.json --features data/processed/derived_features.npy --manifest data/processed/derived_manifest.jsonl --output outputs/result_trake.json
```

---

## 8. Đánh giá (evaluate) — tùy chọn

`evaluate` nhận **3 argument vị trí** (không phải flag):
```bat
python -m aic2026.cli evaluate query_kis.json outputs/candidates_kis.json ground_truth_kis.json
```
(Lấy mảng `candidates` từ kết quả agent trước khi đánh giá.)

---

## 9. Troubleshoot nhanh

| Triệu chứng | Nguyên nhân | Xử lý |
|---|---|---|
| UI báo "Không thể kết nối backend" | Backend chưa chạy | Chạy Terminal 1 (`python -m aic2026.cli serve`) |
| Backend báo thiếu features | Chưa encode CLIP | Chạy bước 2 |
| Agent lỗi / timeout | Ollama chưa `serve` hoặc thiếu model | Bước 3 (bật `ollama serve`, `ollama pull`) |
| Nút "Chạy Agent" xám | Chưa gõ đề vào ô query | Gõ đề vào ô trước |
| Ảnh gallery không hiện | Sai đường dẫn keyframe | Kiểm tra `keyframe_path` trong manifest trỏ đúng `data/raw/Keyframes/...` |
| Warning HF_TOKEN | Bình thường, bỏ qua | — |

---

## 9b. QA — cách chọn frame cho VLM (temporal-diverse sampling)

> **Tại sao không lấy top-K frame theo score?** Nghiên cứu chỉ ra rằng chọn frame chỉ
> theo retrieval score dễ **bỏ sót frame chứa đáp án** và lấy phải các frame gần trùng
> nhau (cùng 1 cảnh) → bổ sung thông tin = 0.

Hệ thống Q&A (cho cả backend Ollama `vlm_ollama.py` và transformers `vlm.py`) trả lời
mỗi video bằng cách tổng hợp **top-K frame (mặc định K=4)** trong **1 lần gọi VLM**
(không tăng số lần gọi so với bản cũ 1 frame). Quan trọng: **cách chọn 4 frame** đã được
đổi từ "top-4 liền kề theo score" sang **temporal-diverse sampling** (inspired by QCA):

1. Chia các candidate của video thành `K` bin theo trục thời gian (`frame_id`).
2. Mỗi bin chọn frame có **score cao nhất** trong bin đó (anchor on relevance).
3. Diversity guard: không bao giờ trùng `frame_id` — nếu trùng thì nhảy sang frame kế tiếp
   trong bin, đảm bảo 4 frame phủ đều các giai đoạn của video.
4. Fallback: nếu video có < K candidate, hoặc thiếu `frame_id`, dùng top-score.

→ VLM nhìn được đầu/cuối/đỉnh của sự kiện thay vì 4 khung na ná nhau → tổng hợp đúng hơn
với câu cần nhiều khung ("có bao nhiêu người", "ai làm X rồi Y").

### Tham khảo (papers)

- **QCA: Query- and Content-Aware Keyframe Selection for Long Video Understanding**
  (Peng et al., 2026, arXiv:2607.00983) — chia temporal segments, anchor trên frame relevance
  cao nhất rồi thêm frame tối đa hóa diversity; SOTA trên LongVideoBench với ít frame hơn.
  Code: https://github.com/hktk07/QCA
- **Self-Adaptive Sampling for Efficient Video QA on Image–Text Models** (Han et al.,
  NAACL 2024, Findings) — đề xuất MIF/MDF; chỉ ra sampling đơn giản "may miss key frames
  that offer answer clues", và "question-aware sampling is not necessary" (đôi khi chọn
  frame theo nội dung dominate tốt hơn theo query). Code: https://github.com/declare-lab/Sealing

> Để đổi số frame K: sửa tham số `top_k_frames` khi gọi `answer_question`
> (`aic2026/qa/vlm_ollama.py` / `aic2026/qa/vlm.py`).

---

## 10. Thứ tự terminal tóm tắt

```
Terminal 1:  ollama serve
Terminal 2:  python -m aic2026.cli serve          (backend :8000)
Terminal 3:  streamlit run src/aic2026/app/ui.py  (UI :8501)
```
Mở `http://127.0.0.1:8501` → dán đề → Chạy Agent → tick → tải JSON.
