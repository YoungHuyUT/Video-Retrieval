@echo off
title PEGASUS - AI Challenge 2026 Launcher
cd /d "%~dp0"

echo ================================================================
echo    PEGASUS - Video Retrieval AI Challenge 2026
echo    Dang khoi dong he thong (Backend ^& Web UI)...
echo ================================================================
echo.

:: 1. Kiem tra va khoi dong FastAPI Backend (Port 8000) trong cua so an
where uv >nul 2>&1
if %errorlevel% equ 0 (
    echo [*] Dang khoi dong FastAPI Backend (Port 8000)...
    start "PEGASUS Backend" /min cmd /c "uv run uvicorn aic2026.app.api:app --host 127.0.0.1 --port 8000"
    timeout /t 2 /nobreak >nul
    echo [*] Dang khoi dong Streamlit Web UI (Port 8501)...
    uv run streamlit run src/aic2026/app/ui.py --server.port 8501 --server.headless false
    goto end
)

if exist ".venv\Scripts\streamlit.exe" (
    echo [*] Dang khoi dong FastAPI Backend bang .venv...
    start "PEGASUS Backend" /min cmd /c ".venv\Scripts\uvicorn.exe aic2026.app.api:app --host 127.0.0.1 --port 8000"
    timeout /t 2 /nobreak >nul
    echo [*] Dang khoi dong Streamlit Web UI...
    .venv\Scripts\streamlit.exe run src/aic2026/app/ui.py --server.port 8501 --server.headless false
    goto end
)

echo [*] Dang khoi dong bang Python he thong...
start "PEGASUS Backend" /min cmd /c "python -m uvicorn aic2026.app.api:app --host 127.0.0.1 --port 8000"
timeout /t 2 /nobreak >nul
python -m streamlit run src/aic2026/app/ui.py --server.port 8501 --server.headless false

:end
if %errorlevel% neq 0 (
    echo.
    echo [!] Co loi xay ra khi chay chuong trinh.
    pause
)
