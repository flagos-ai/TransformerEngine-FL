#!/usr/bin/env bash
# MetaX Backend Environment Setup

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PLATFORM="${PLATFORM:-metax}"
export TE_PATH="${TE_PATH:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd)}"
export TE_LIB_PATH="${TE_LIB_PATH:-$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)/transformer_engine}"
export PYTHONPATH="${TE_PATH}${PYTHONPATH:+:$PYTHONPATH}"
export TE_FL_SKIP_CUDA="${TE_FL_SKIP_CUDA:-1}"
export TE_FL_PREFER="${TE_FL_PREFER:-vendor}"
export NVTE_WITH_MACA="${NVTE_WITH_MACA:-1}"
export NVTE_WITH_NCCL_EP="${NVTE_WITH_NCCL_EP:-0}"
export CUDA_HOME="${CUDA_HOME:-/opt/maca}"
export MACA_HOME="${MACA_HOME:-/opt/maca}"
export PATH="${MACA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export MCORE_USE_CUDA_ENV_DEFAULTS="${MCORE_USE_CUDA_ENV_DEFAULTS:-0}"
export NUM_LAYERS="${NUM_LAYERS:-2}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
export NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-4}"
export SEQ_LENGTH="${SEQ_LENGTH:-128}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
export ENABLE_DIAGNOSTICS="${ENABLE_DIAGNOSTICS:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
