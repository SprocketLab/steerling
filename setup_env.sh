#!/usr/bin/env bash
# Bootstrap a Conda environment with the packages needed by train.py
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-steerling}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "error: ${CONDA_BIN} is not available; ensure Conda/Miniconda is installed and on PATH" >&2
  exit 1
fi

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Creating Conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
else
  echo "Conda environment '${ENV_NAME}' already exists; skipping creation."
fi

TORCH_CUDA_CHANNEL="${TORCH_CUDA_CHANNEL:-https://download.pytorch.org/whl/cu121}"

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
"${CONDA_BIN}" run -n "${ENV_NAME}" pip install --extra-index-url "${TORCH_CUDA_CHANNEL}" \
  "torch>=2.2" \
  "torchvision>=0.17" \
  "torchaudio>=2.2"

"${CONDA_BIN}" run -n "${ENV_NAME}" pip install \
  "transformers>=4.40" \
  "accelerate>=0.26" \
  "datasets>=2.19" \
  "numpy>=1.26" \
  "tqdm>=4.66" \
  "wandb>=0.16" \
  "safetensors>=0.4"

echo
echo "Environment ready."
echo "Activate it with: conda activate ${ENV_NAME}"
