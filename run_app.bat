@echo off
title PEGASUS AI Challenge 2026
cd /d "%~dp0"

echo ================================================================
echo    PEGASUS - Video Retrieval AI Challenge 2026
echo    Dang khoi dong ung dung Web UI...
echo ================================================================
echo.

set "PATH=%PATH%;E:\Anaconda3\Scripts;E:\Anaconda3;%USERPROFILE%\.cargo\bin"

:: 1. Uu tien dung uv run
where uv >nul 2>&1
if %errorlevel% equ 0 (
    echo [*] Khoi dong bang uv...
    uv run streamlit run src/aic2026/app/ui.py --server.port 8501 --server.headless false
    goto finish
)

:: 2. Dung moi truong ao .venv
if exist ".venv\Scripts\streamlit.exe" (
    echo [*] Khoi dong bang .venv...
    .venv\Scripts\streamlit.exe run src/aic2026/app/ui.py --server.port 8501 --server.headless false
    goto finish
)

:: 3. Dung python he thong
echo [*] Khoi dong bang python he thong...
python -m streamlit run src/aic2026/app/ui.py --server.port 8501 --server.headless false

:finish
if %errorlevel% neq 0 (
    echo.
    echo [!] Chuong trinh dung voi ma loi: %errorlevel%
    pause
)
