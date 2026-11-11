#!/usr/bin/env bash
# CUDA Backend Environment Setup

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TE_PATH="${TE_PATH:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd)}"
export TE_LIB_PATH="${TE_LIB_PATH:-$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)/transformer_engine}"
export PYTHONPATH="${TE_PATH}${PYTHONPATH:+:$PYTHONPATH}"
export TE_FL_SKIP_CUDA="${TE_FL_SKIP_CUDA:-0}"
export SKIP_CUDA_BUILD="${SKIP_CUDA_BUILD:-0}"
export NVTE_WITH_CUDA="${NVTE_WITH_CUDA:-1}"
export NVTE_WITH_MACA="${NVTE_WITH_MACA:-0}"
export TE_WITH_NCCL="${TE_WITH_NCCL:-1}"
export NVTE_WITH_NCCL_EP="${NVTE_WITH_NCCL_EP:-0}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-80;90}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
