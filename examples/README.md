# Cách nhập event cho TRAKE

TRAKE tìm một **chuỗi sự kiện có thứ tự thời gian** trong video — khác KIS (tìm
1 khung hình) và Q&A (trả lời câu hỏi). Bạn nhập event qua trường `"events"` trong
file query JSON.

## Quy tắc nhập event

1. Mỗi phần tử của `"events"` là **một sự kiện**, viết theo thứ tự xuất hiện
   trong video (từ sớm → muộn).
2. Số lượng event = số frame sẽ xuất ra trong CSV nộp bài. Ví dụ 3 event →
   `L26_V001,1200,1850,2100` (video_id + 3 frame).
3. Mỗi event nên là 1 câu ngắn mô tả hành động, tiếng Anh (CLIP là model EN).
   Dùng danh từ + động từ cụ thể: `"a person opens the fridge"`,
   `"pours water into a glass"`.
4. Không dùng dấu chấm phẩy / newline để ngăn cách — mỗi event là 1 phần tử
   riêng trong mảng JSON.

## Template

```json
{
  "query_id": "trake_001",
  "type": "trake",
  "text": "Mô tả ngắn toàn bộ chuỗi (tùy chọn, để tham khảo)",
  "events": [
    "Event 1 xảy ra đầu tiên",
    "Event 2 xảy ra kế tiếp",
    "Event 3 cuối cùng"
  ]
}
```

## Chạy & xuất kết quả (quan trọng: tránh format 'như KIS')

`agent-query` chỉ xuất **JSON** (có `event_frames` bên trong). Để có file CSV
nộp bài đúng định dạng TRAKE (`video_id,f1,f2,f3`), bạn PHẢI chạy thêm
`export-submission`:

```bash
# 1) Chạy retrieval (TRAKE) -> JSON
aic2026 agent-query --query examples/query_trake.json \
    --backend faiss --coarse-top-k 50 \
    --output outputs/result_trake.json

# 2) Chuyển JSON -> CSV nộp bài (dòng TRAKE, KHÔNG phải KIS)
aic2026 export-submission --result outputs/result_trake.json \
    --query examples/query_trake.json \
    --output outputs/result_trake.csv
```

Nội dung `outputs/result_trake.csv` sẽ là (ví dụ):

```
L26_V001,1200,1850,2100
```

— đúng 1 dòng, đầu tiên là `video_id`, theo sau là N frame theo đúng thứ tự event.
Còn nếu bạn chỉ lấy JSON hoặc in ra, bạn sẽ thấy `frame_id` đơn → trông "như KIS",
đó là do chưa chạy `export-submission`.
