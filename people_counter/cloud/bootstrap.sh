#!/usr/bin/env bash
# Runs ON the cloud VM (or any new Linux box): venv + deps + benchmark. Usage: bootstrap.sh <video> [extra benchmark args]
set -euo pipefail
cd "$(dirname "$0")/.."
VIDEO="$1"; shift || true
python3 -m venv .venv && . .venv/bin/activate
pip -q install --upgrade pip
if command -v nvidia-smi >/dev/null; then pip -q install torch torchvision; else pip -q install torch torchvision --index-url https://download.pytorch.org/whl/cpu; fi
pip -q install -r <(grep -vE '^(torch|torchvision)' requirements.txt)
python -m peoplecount hardware | head -30
python -m peoplecount benchmark "$VIDEO" --out results "$@"
