# ===========================================================================
# Dockerfile - AIC2026 Video Retrieval
#
# GIANG: Moi dong IN HOA la mot chi thi. Docker doc tu tren xuong,
# moi RUN/COPY/ADD tao mot LAYER (tam mong xep chong). Layer giong nhau
# giua 2 lan build -> Docker dung CACHE -> build lan 2 chi nhung gi thay doi.
#
# Chien luoc cache (quan trong):
#   1. COPY file deps TRUOC (pyproject.toml + uv.lock)
#   2. RUN cai deps
#   3. COPY src/ SAU
#   -> Sua code != cai lai torch/transformers (von mat nhieu phut).
# ===========================================================================

# ---------------------------------------------------------------------------
# ARG + FROM - chon khoon nen (base image).
# python:3.11-slim = Python 3.11 toi gian (Debian). KHONG dung 'latest'
# (latest doi bat ngo -> build ngay mai co the fail). Pin version = reproducible.
#
# GPU: Docker KHONG co GPU free. GPU phai la GPU vat ly cua HOST +
# NVIDIA Container Toolkit. Khi co GPU:
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 .
# ---------------------------------------------------------------------------
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# ENV - bien moi truong cho container (nhu set trong terminal).
# PYTHONPATH=/app/src -> import aic2026.* khi source van o /app/src.
# HF_HOME -> HuggingFace cache se mount qua volume, KHONG copy model vao image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/app/.cache/huggingface \
    PYTHONPATH=/app/src \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# WORKDIR - thu muc lam viec mac dinh cho cac lenh sau.
# api.py dung duong dan tuong doi data/processed/... -> WORKDIR phai la project root.
WORKDIR /app

# RUN - cai system packages (library C ma wheel Python can).
# libgl1/libglib2.0-0 cho opencv; curl cho healthcheck; ca-certificates cho HTTPS HF.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# RUN - cai uv (package manager nhanh cua Astral, lockfile giong npm).
RUN pip install --no-cache-dir uv

# COPY deps TRUOC -> layer cache tot: sua src khong rebuild lai torch.
COPY pyproject.toml uv.lock ./

# Torch CPU truoc: tren Linux, torch tu PyPI keo CUDA wheels (nvidia-*, hang GB).
# May khong GPU -> dung CPU wheel de image nho va chay duoc.
RUN uv pip install --system "torch>=2.3" \
        --index-url https://download.pytorch.org/whl/cpu

# Cai cac goi runtime can thiet (fastapi, streamlit, faiss, transformers, opencv...).
RUN uv pip install --system \
        "fastapi>=0.115" \
        "uvicorn>=0.30" \
        "streamlit>=1.36" \
        "httpx>=0.27" \
        "rank-bm25>=0.2" \
        "diskcache>=5.6" \
        "python-dotenv>=1.0" \
        "pydantic>=2.7" \
        "pyyaml>=6.0" \
        "typer>=0.12" \
        "numpy>=1.26" \
        "faiss-cpu>=1.8" \
        "chromadb>=0.5" \
        "transformers>=4.49" \
        "sentencepiece>=0.2" \
        "sentence-transformers>=3.0" \
        "open-clip-torch>=2.24" \
        "qwen-vl-utils>=0.0.8" \
        "opencv-python-headless>=4.10" \
        "pillow>=10.0" \
        "scenedetect[opencv]>=0.6.4"

# COPY source SAU cung -> sua code chi rebuild tu buoc nay.
COPY src/ ./src/

# Install editable package de CLI entry point `aic2026` + import path dung.
RUN uv pip install --system -e . --no-deps

# User khong-root (best practice). Data mount qua volume nen khong can write host.
RUN useradd --create-home --uid 1000 --shell /bin/bash app \
    && mkdir -p /home/app/.cache/huggingface /app/outputs \
    && chown -R app:app /home/app /app/outputs

# EXPOSE - chi ghi chu de document; KHONG mo port ra internet.
# Compose se map host:container khi can.
EXPOSE 8000 8501

# HEALTHCHECK - container tu bao healthy/unhealthy.
# /health API tra {"status":"ok"} khi heavy assets preload xong.
# start-period 90s: preload SigLIP2+FAISS+BM25 ~20s+, lan dau tai model co the lau.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# CMD - lenh mac dinh khi `docker run` khong truyen command rieng.
# Compose OVERRIDE lenh nay cho tung service (api vs ui).
CMD ["uvicorn", "aic2026.app.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ===========================================================================
# GPU (khi can):
#   1. Host can NVIDIA driver + nvidia-container-toolkit
#   2. docker run --gpus all ...
#   3. Base image doi sang nvidia/cuda runtime + cai python3.11
#   4. PyTorch can wheel cu12x khop CUDA runtime
# Docker khong cho GPU free - khong co driver host thi van CPU.
# ===========================================================================
