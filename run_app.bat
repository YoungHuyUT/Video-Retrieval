@echo off
title PEGASUS - AI Challenge 2026 Launcher
cd /d "%~dp0"

echo ================================================================
echo    PEGASUS - Video Retrieval AI Challenge 2026
echo    Dang khoi dong ung dung Web UI...
echo ================================================================
echo.

where uv >nul 2>&1
if %errorlevel% equ 0 (
    echo [*] Dang chay bang 'uv'...
    uv run streamlit run src/aic2026/app/ui.py --server.port 8501 --server.headless false
    goto end
)

if exist ".venv\Scripts\streamlit.exe" (
    echo [*] Dang chay bang moi truong ao .venv...
    .venv\Scripts\streamlit.exe run src/aic2026/app/ui.py --server.port 8501 --server.headless false
    goto end
)

echo [*] Dang chay bang python he thong...
python -m streamlit run src/aic2026/app/ui.py --server.port 8501 --server.headless false

:end
if %errorlevel% neq 0 (
    echo.
    echo [!] Co loi xay ra khi chay chuong trinh.
    pause
)
