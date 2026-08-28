#!/usr/bin/env bash

set -e

python -m pip install --upgrade pip

pip install -e ".[dev]"

echo ""
echo "=========================================="
echo " RHS Adaptive Beamforming NAS Environment "
echo "=========================================="

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY