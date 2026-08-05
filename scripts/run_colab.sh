#!/usr/bin/env bash
set -euo pipefail
uv sync --extra retrieval --extra models --extra dev
aic2026 prepare --raw-dir data/raw --output data/processed/manifest.jsonl
echo "Manifest complete. Add model/index configuration before retrieval benchmark."
