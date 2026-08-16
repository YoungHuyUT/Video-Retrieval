# AIC 2026 — Hướng dẫn chạy hệ thống

Hệ thống chạy KIS, Q&A và TRAKE bằng **FAISS + CLIP features BTC + BM25**. Luồng chuẩn không dùng Chroma.

## 1. Cài đặt


```bat
.venv\Scripts\Activate
uv sync --extra retrieval --extra models --extra video --extra dev
```

Không dùng `uv` thì:

```bat
pip install -e ".[retrieval,models,video,dev]"

```

FAISS phải hoạt động; nếu thiếu, hệ thống fallback NumPy và sẽ chậm hơn:
```bat
python -c "import faiss; print(faiss.__version__)"
```

## 2. Dữ liệu và file quan trọng

```text
data/
├── raw/
│   ├── Keyframes/       # JPG: Keyframes/L01_V001/001.jpg
│   ├── Objects/         # object labels BTC
│   ├── Metadata/        # metadata video BTC
│   ├── CLIP features/   # file hoặc thư mục .npy BTC
│   └── Videos/          # tùy chọn
└── processed/
    ├── official_features.npy
    ├── official_manifest.jsonl
    └── official_manifest_ocr.jsonl   # chỉ có sau OCR
```

- `frame_id` là vị trí frame thật trong video gốc, nên không cần liên tiếp.
- Giữ `official_features.npy` và `official_manifest.jsonl`; đây là index chính.
- Keyframes chỉ cần cho gallery, OCR và VLM Q&A. Không có ảnh vẫn retrieval CLIP được, nhưng không OCR/VLM được frame đó.

## 3. Chuẩn bị index chính thức

Kiểm tra asset BTC:

```bat
python -m aic2026.cli inspect-data --raw-dir data/raw
```

Tạo feature matrix và manifest căn đúng `frame_id`:

```bat
python -m aic2026.cli prepare-official --features "data/raw/CLIP features" --raw-dir data/raw --output-manifest data/processed/official_manifest.jsonl --output-features data/processed/official_features.npy
```

Kiểm tra số vector khớp manifest:

```bat
python -m aic2026.cli check-features data/processed/official_features.npy data/processed/official_manifest.jsonl
```

Chỉ chạy mục này khi nhận/thay dữ liệu BTC, không chạy lại trước mỗi query.

### Khi chỉ có Keyframes

Nếu chưa có CLIP features BTC, tự encode. Thử một video trước:

```bat
python -m aic2026.cli embed-keyframes --keyframes-dir data/raw/Keyframes --features-dir data/processed/clip_features --manifest data/processed/derived_manifest.jsonl --features data/processed/derived_features.npy --video-id L21_V001
```
Bỏ `--video-id` để chạy toàn bộ. Dùng rõ `derived_*` ở UI/CLI. Luồng này thiếu Objects/Metadata nên BM25 thường yếu hơn `official_*`.

## 4. Chạy UI

Mở hai terminal đã activate `.venv`.

Terminal 1 — backend (giữ mở để cache index, BM25 và CLIP):

```bat
python -m aic2026.cli serve
```

Terminal 2 — UI:

```bat
streamlit run src/aic2026/app/ui.py
```

Mở `http://127.0.0.1:8501`.
- KIS: nhập mô tả cảnh.
- Q&A: nhập mô tả retrieval và câu hỏi.
- TRAKE: nhập mô tả và event theo đúng thứ tự thời gian.
- Mặc định UI dùng `official_manifest.jsonl` và `official_features.npy`.
- Không bật **Dịch VI→EN** nếu đã nhập tiếng Anh. Khi bật, UI hiển thị bản dịch và báo rõ khi LLM fallback.

Query đầu sau khi start backend chậm vì phải load features/manifest/BM25. Query KIS tiếp theo phải nhanh hơn; không restart backend giữa các query.

## 5. Chạy CLI

```bat
mkdir outputs
```

### KIS

`query_kis.json`:

```json
{"query_id":"kis_001","type":"kis","text":"a speaker giving a speech at an outdoor event"}
``````bat
python -m aic2026.cli agent-query --query query_kis.json --backend faiss --output outputs/result_kis.json
```

### Q&A

Q&A cần Ollama vision model:

```bat
ollama pull qwen2.5vl:3b
ollama serve
```

`query_qa.json`:

```json
{"query_id":"qa_001","type":"qa","text":"an award ceremony on a stage","question":"How many people are on the stage?"}
```

```bat
python -m aic2026.cli agent-query --query query_qa.json --backend faiss --vlm-backend ollama --vlm-model qwen2.5vl:3b --output outputs/result_qa.json
```

Q&A chỉ sinh `answer` khi candidate có file JPG để VLM đọc.

### TRAKE
`query_trake.json`:

```json
{"query_id":"trake_001","type":"trake","text":"high jump sequence","events":["running toward the bar","jumping over the bar","landing on the mat"]}
```

Chạy thử nhanh với coarse filter 50 video:

```bat
python -m aic2026.cli agent-query --query query_trake.json --backend faiss --coarse-top-k 50 --output outputs/result_trake.json
```

Tăng `--coarse-top-k` để tăng recall, nhưng TRAKE chậm hơn vì DP căn chỉnh nhiều video hơn.

### Dịch Việt → Anh

Mặc định tắt. Bật khi query là tiếng Việt:
```bat
python -m aic2026.cli agent-query --query query_kis.json --backend faiss --translate --output outputs/result_kis.json
```

Với query tìm chữ, đặt text đích trong ngoặc kép:

```text
Hình ảnh tấm bảng có chữ "BENVENUTI"
```

Hệ thống giữ nguyên `"BENVENUTI"` và dùng truy vấn literal `an image of a sign with the text "BENVENUTI"`.

### Lọc Object/Metadata

Chỉ dùng khi chắc từ khóa có trong dữ liệu BTC; filter quá chặt có thể mất recall:

```bat
python -m aic2026.cli agent-query --query query_kis.json --backend faiss --metadata-filter "concert,outdoor" --output outputs/result_kis.json
```

### Giới hạn theo tập dữ liệu (L25/L26)

Hai luồng **KIS** và **Q&A** luôn tìm kiếm trên **TOÀN BỘ corpus** (không giới
hạn split): KIS là retrieval nguyên video theo bản chất, còn Q&A đọc text OCR
nằm rải rác khắp các tập. Retrieval KHÔNG bị scope.

**TRAKE** cũng tìm trên toàn bộ corpus, nhưng **ưu tiên mềm (soft bias)** tập
**L26** — tức là video L26 được cộng một điểm nhỏ để xếp trên các video tương
đương không phải L26, nhưng KHÔNG bao giờ bị loại trừ các tập khác. Lý do: query
event của BTC rất tổng quát (vd "bước vào phòng / ngồi xuống") và khớp nhiều
split, còn L26 chủ yếu là nội dung nấu ăn/lifestyle — ép cứng TRAKE→L26 sẽ trả
về **0 ứng viên** cho hầu hết query hành động. Ưu tiên mềm giữ được recall mà
vẫn đẩy L26 lên khi nó thực sự khớp (đã kiểm chứng: query "cooking food" → 100%
ứng viên L26).

OCR thì ngược lại: **chỉ OCR trên L25** (khóa học online, nhiều bảng biểu/logo)
để tiết kiệm thời gian máy — xem mục "Chỉ OCR một tập (ví dụ L25)" ở §6. Các
video không khớp vẫn giữ nguyên trong manifest, thứ tự gốc bảo toàn nên
`vector_id` khớp file `.npy`.

Ghi đè ưu tiên L26 của TRAKE:

```bat
:: TRAKE ưu tiên L25 thay vì mặc định L26
python -m aic2026.cli agent-query --query query_trake.json --backend faiss --trake-preferred-prefixes L25 --output outputs/result_trake.json

:: Tắt ưu tiên (xét hết mọi tập, không thiên vị): truyền dấu chấm
python -m aic2026.cli agent-query --query query_trake.json --backend faiss --trake-preferred-prefixes . --output outputs/result_trake.json
```

`--trake-preferred-prefixes` nhận nhiều tiền tố phẩy-ngăn-cách; để trống = dùng
mặc định (ưu tiên L26), truyền `.` = không ưu tiên. KIS/Q&A không có flag scope
vì luôn xét toàn bộ corpus.

## 6. OCR — tìm chữ trên biển/bảng/logo

OCR không tự chạy và không nằm trong CLIP feature. Nó ghi text nhận dạng vào `object_labels` của manifest OCR để BM25 tìm khớp chính xác.
Chỉ OCR được frame có JPG trong `data/raw/Keyframes`.

Cài PaddleOCR một lần:

```bat
:: Cài PaddlePaddle CPU từ source Windows chính thức
python -m pip uninstall -y paddlepaddle paddleocr
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install --upgrade paddleocr

:: Phải in được version trước khi chạy OCR
python -c "import paddle; print(paddle.__version__)"
```

> **Kiểm tra bắt buộc.** Không chạy OCR nếu lệnh import trên lỗi. Khi OCR không
> khởi động được, pipeline vẫn hoàn tất nhưng không thêm được dòng chữ nào vào
> manifest. Nếu báo `DLL load failed`/`libpaddle.pyd`, xem hướng dẫn bên dưới.

Code OCR tự tắt các model phụ dành cho tài liệu (xoay trang, nắn ảnh, hướng dòng
text) để giảm download/RAM; keyframe video chỉ tải model detection + recognition cần thiết.

Nếu lệnh import báo lỗi `DLL load failed`/`libpaddle.pyd`, đóng mọi Python/Streamlit,
cài hoặc Repair **Microsoft Visual C++ 2015–2022 Redistributable (x64)**, rồi chạy lại
lệnh cài CPU ở trên. Không chạy `ocr-manifest` cho đến khi `import paddle` thành công.

Mỗi lần chạy hiện dùng một model ngôn ngữ. Để nhận tốt cả tiếng Việt lẫn tiếng Anh,
chạy hai lượt: lượt hai nhận manifest của lượt một làm input nên text được **gộp** vào
`object_labels`, không bị mất.

```bat
:: Lượt 1: tiếng Việt (cũng đọc được ký tự Latin cơ bản)
python -m aic2026.cli ocr-manifest --manifest data/processed/official_manifest.jsonl --output data/processed/official_manifest_ocr_vi.jsonl --keyframes-root data/raw/Keyframes --lang vi --batch-size 16

:: Lượt 2: tiếng Anh/Latin, gộp thêm vào kết quả lượt 1
python -m aic2026.cli ocr-manifest --manifest data/processed/official_manifest_ocr_vi.jsonl --output data/processed/official_manifest_ocr.jsonl --keyframes-root data/raw/Keyframes --lang en --batch-size 16
```

Lệnh trên **không ghi đè** `official_manifest.jsonl` gốc. CPU OCR toàn bộ
177k keyframe có thể mất nhiều giờ. Nếu thiếu RAM, giảm `--batch-size 16`
thành `--batch-size 8`.

### Chỉ OCR một tập (ví dụ L25)

OCR chủ yếu có giá trị trên tập **L25** (khóa học online, nhiều bảng biểu/logo),
nên **chỉ OCR tập L25** để chạy nhanh và tiết kiệm. (TRAKE thì tìm toàn bộ
corpus, chỉ ưu tiên mềm L26 — xem §5.) Dùng `--video-prefix L25` để **chỉ OCR**
các video có `video_id` bắt đầu `L25` (88 video / 37.445 frames).

> Các video **không** khớp (L26, L21, ...) vẫn được **giữ nguyên** trong file
> output (không OCR, không drop), và thứ tự dòng manifest gốc được bảo toàn —
> nên `vector_id` vẫn khớp file `official_features.npy`. TRAKE vẫn retrieval
> bình thường trên toàn bộ corpus (gồm cả L26) sau khi OCR.

```bat
:: Chỉ OCR L25, hai lượt vi/en, gộp text vào cùng 1 manifest đầu ra
python -m aic2026.cli ocr-manifest --manifest data/processed/official_manifest.jsonl --output data/processed/official_manifest_ocr_vi.jsonl --keyframes-root data/raw/Keyframes --video-prefix L25 --lang vi --batch-size 16

python -m aic2026.cli ocr-manifest --manifest data/processed/official_manifest_ocr_vi.jsonl --output data/processed/official_manifest_ocr.jsonl --keyframes-root data/raw/Keyframes --video-prefix L25 --lang en --batch-size 16
```

Muốn OCR thêm một tập khác (vd L26) sau này, chạy tiếp `--video-prefix L26`
ghi đè lên cùng file — text cũ của L25 được giữ (gộp không trùng).

`--video-prefix` nhận nhiều tiền tố cách nhau dấu phẩy: `--video-prefix L25,L21`.
Để trống = OCR hết (như hai lượt ở trên).


OCR lưu tại:

```text
data/processed/official_manifest_ocr.jsonl
```

Kiểm tra text đã được OCR:

```bat
findstr /I /C:"BENVENUTI" data\processed\official_manifest_ocr.jsonl
```

Nếu không có dòng nào, OCR không nhận được chữ đó hoặc ảnh chứa chữ chưa có trên máy.

Kiểm tra tổng quát hơn rằng manifest OCR **thực sự khác** manifest gốc (đếm số
frame có text/label được OCR thêm vào):

```powershell
python -c "import json; from pathlib import Path; a=Path('data/processed/official_manifest.jsonl').open(encoding='utf8'); b=Path('data/processed/official_manifest_ocr.jsonl').open(encoding='utf8'); d=[any(v not in x.get('object_labels',[]) for v in y.get('object_labels',[])) for x,y in ((json.loads(i),json.loads(j)) for i,j in zip(a,b))]; print(f'frames with added OCR text: {sum(d):,} / {len(d):,}')"
```

Nếu kết quả là `0`, OCR chưa sinh text: kiểm tra log `PaddleOCR` ở lúc chạy,
đảm bảo `data/raw/Keyframes` có JPG, rồi chạy lại hai lượt OCR. Sau OCR:

- UI: trong **Cấu hình nâng cao → Manifest**, đặt `data/processed/official_manifest_ocr.jsonl`, rồi restart backend.
- CLI: thêm `--manifest data/processed/official_manifest_ocr.jsonl`.

Ví dụ:

```bat
python -m aic2026.cli agent-query --query query_kis.json --backend faiss --manifest data/processed/official_manifest_ocr.jsonl --output outputs/result_benvenuti.json
```

Không cần build Chroma hoặc tạo lại CLIP features sau OCR.

## 7. Đánh giá

`agent-query` ghi object có `candidates`, `plan`, `trace`; `evaluate` cần JSON array candidates:

```bat
python -c "import json; d=json.load(open('outputs/result_kis.json', encoding='utf-8')); json.dump(d['candidates'], open('outputs/candidates_kis.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)"
```Ground truth `ground_truth_kis.json` ví dụ:

```json
{"video_id":"L01_V001","ranges":[[500,510]]}
```

```bat
python -m aic2026.cli evaluate query_kis.json outputs/candidates_kis.json ground_truth_kis.json
```

## 8. Khắc phục lỗi

| Triệu chứng | Xử lý |
|---|---|
| Query đầu tiên chậm | Bình thường: backend đang load index, CLIP và BM25. Giữ `serve` chạy. |
| Mọi query chậm | Kiểm tra `import faiss`; không restart backend mỗi query. |
| `200 OK` nhưng UI không hiện | Backend đã trả lời, UI/browser có thể reset kết nối. Refresh UI và thử lại khi backend vẫn chạy. |
| `LLM translate failed` | Ollama/model dịch lỗi; dùng query tiếng Anh hoặc để UI dùng fallback. |
| Không tìm `BENVENUTI` | Chưa chạy OCR, chưa dùng manifest OCR, OCR không đọc được chữ, hoặc ảnh keyframe chưa có. |
| Q&A không có `answer` | Ollama VLM chưa chạy/model chưa tải/candidate không có ảnh. |
| Thiếu manifest/features | Chạy lại mục 3 bằng `prepare-official`. |

## 9. Dọn dữ liệu

Giữ:
```text
data/raw/
data/processed/official_features.npy
data/processed/official_manifest.jsonl
data/processed/official_manifest_ocr.jsonl  # nếu đã chạy OCR
```

Có thể xóa vì sinh lại được:

```text
outputs/
.pytest_cache/
.ruff_cache/
data/processed/clip_features/
```

Luồng chuẩn không có Chroma.
