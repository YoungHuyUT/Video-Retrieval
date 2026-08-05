# AIC 2026 - Kế hoạch và khung mã nguồn

## Mục tiêu

Khung triển khai cho Textual KIS, Q&A và TRAKE, ưu tiên Colab/Kaggle chi phí thấp.
Dữ liệu gốc và index lớn nằm ngoài Git; manifest, cấu hình và script đảm bảo tái lập.

## Thành phần đã triển khai

- `src/aic2026/ingestion`: phát hiện Keyframes, Objects, Metadata và tạo JSONL manifest.
- `src/aic2026/retrieval`: tìm kiếm cosine đã chuẩn hóa, đa dạng hóa candidate theo video.
- `src/aic2026/evaluation`: tính R-Score, R@k và Final Score đúng theo đề.
- `src/aic2026/qa`, `temporal`, `reranking`: các điểm mở rộng baseline an toàn.
- `src/aic2026/submission`: kiểm tra tối đa 100 đáp án và ghi JSONL.
- `src/aic2026/app`: FastAPI và Streamlit inspector nội bộ.
- `src/aic2026/data_platform`: kiểm tra đúng 5 nhóm asset BTC cung cấp.
- `src/aic2026/tasks`: registry cho KIS, Q&A, TRAKE và query type thứ tư trong tương lai.

## Bám sát dữ liệu và truy vấn BTC

- **Video là nguồn chính thức:** `extract-keyframes` tự decode tất cả frame, tạo CLIP embedding
  theo batch, sau đó loại near-duplicate theo cosine với frame giữ gần nhất. Nó xuất frame ID gốc,
  `.npz` theo từng video, rồi `build-derived-index-input` gộp thành manifest + `.npy` cho FAISS.
- **Archive hỗ trợ BTC:** nhận `clip-features`, `map-keyframes`, `media-info`, `objects` dưới dạng
  ZIP riêng trong `data/downloads`; lệnh `unpack-support-assets` giải nén an toàn, tách provenance
  khỏi index tạo từ Video.
- **Embedding đa kiến trúc:** feature CLIP BTC là baseline/index riêng; video tự xử lý mặc định
  dùng SigLIP2 Base Patch16/224. Chỉ fusion ở cấp score/rank sau benchmark, không trộn vector.
- **Textual KIS:** CLIP feature là retrieval chính; Objects/Metadata dùng để rerank và filter.
- **Q&A:** retrieval xác định video/frame trước, sau đó mới gọi VLM local trên candidate;
  text LLM không được tự trả lời nội dung hình ảnh.
- **TRAKE:** retrieval từng event để lọc video, sau đó DP căn chỉnh theo thứ tự thời gian.
- **Mở rộng query type:** thêm handler vào `aic2026.tasks.default_registry`; contract chung vẫn là
  `Query -> Candidate[] -> evaluator/submission`, nên không phải sửa agent graph hay index.

## Kiến trúc Agent local (đã cập nhật)

Kế thừa ba ý tưởng hiệu quả từ code AIC 2025: retrieval + metadata tách lớp, query
rewrite và dynamic programming căn chỉnh event. Khác biệt chính là một agent cục bộ,
được triển khai tại `src/aic2026/agent`, hoạt động như bộ điều phối có giới hạn:

```text
Truy vấn -> Local LLM planner (JSON) -> nhiều truy vấn CLIP -> FAISS/cosine retrieval
        -> Local LLM judge (chỉ chọn vector_id có evidence) -> TRAKE DP / Q&A VLM -> validator -> submission
```

- Provider mặc định: Ollama local, API localhost, không cần API key hay cloud.
- Planner tách event TRAKE, tạo ít biến thể Việt/Anh; judge chỉ được chọn `vector_id`
  trong evidence tool trả về.
- Agent bị giới hạn 4 lần retrieval và 100 đáp án; `video_id`/`frame_id` luôn lấy từ
  manifest. Đây là hàng rào chống hallucination và giữ latency/VRAM ở mức sinh viên.
- Q&A cần thêm VLM local ở bước sau. Không dùng text LLM để nhìn ảnh hoặc bịa câu trả lời.
- `LangGraphRetrievalAgent` điều phối các node `plan → retrieve → judge → finalize`;
  các node retrieval, validation và submission vẫn deterministic.

## Công nghệ khuyến nghị

Python 3.11 + uv, NumPy, Pydantic, Typer, FastAPI, Streamlit. Các extra tùy chọn
bổ sung FAISS CPU, PyTorch, OpenCLIP, Transformers và Ollama local. Baseline kiểm tra
được trên CPU; chỉ cài model extra khi chạy Colab/Kaggle GPU.

## Việc cần làm khi có dữ liệu

1. Đặt Batch 1 vào `data/raw`, chạy `aic2026 prepare` và kiểm tra feature-manifest.
2. Xác nhận checkpoint OpenCLIP khớp chính xác feature CLIP của BTC; benchmark Recall@k.
3. Tích hợp FAISS persistence và VLM local cho Q&A/reranking sau khi có tập gán nhãn.
4. Cập nhật `submission.writer` khi BTC công bố schema output.

## Kiểm tra

`pytest` kiểm tra KIS, Q&A, TRAKE (kể cả video sai bằng 0), rank cutoff, submission
và ràng buộc chống hallucination của agent.

Prompt theo task và output adapter không chứa field nội bộ được mô tả tại
`docs/ai-prompts-and-output.md`.
