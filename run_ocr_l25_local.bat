@echo off
REM Chạy OCR L25 local (Windows) - ghi tăng dần tung record, resume an toan.
REM Yeu cau: .venv da install paddleocr (da sua thieu aistudio_sdk + aiohttp).
REM Duong dan keyframe trong manifest la relative nen resolve dung tu project root.
cd /d %~dp0

.venv\Scripts\python.exe -m aic2026.cli ocr-manifest ^
  --manifest data/processed/official_manifest.jsonl ^
  --output data/processed/official_manifest_ocr_vi.jsonl ^
  --keyframes-root data/raw/Keyframes ^
  --batch-size 16 ^
  --lang vi ^
  --model-size medium ^
  --video-prefix L25 ^
  --resume

REM Neu lan dau bi ngat giua chung, chay LAI cung lenh nay (--resume tu dong bo qua
REM nhung frame da xong, chi OCR tiep phan con lai).
pause
