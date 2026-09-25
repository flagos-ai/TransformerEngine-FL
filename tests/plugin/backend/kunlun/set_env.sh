#!/usr/bin/env bash
# KunlunXin Backend Environment Setup

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TE_PATH="${TE_PATH:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd)}"
export TE_LIB_PATH="${TE_LIB_PATH:-$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)/transformer_engine}"
export PYTHONPATH="${TE_PATH}${PYTHONPATH:+:$PYTHONPATH}"
export PLATFORM="${PLATFORM:-kunlunxin}"
export TE_FL_SKIP_CUDA="${TE_FL_SKIP_CUDA:-1}"
export SKIP_CUDA_BUILD="${SKIP_CUDA_BUILD:-1}"
export NVTE_WITH_CUDA="${NVTE_WITH_CUDA:-0}"
export NVTE_WITH_MACA="${NVTE_WITH_MACA:-0}"
export TE_WITH_NCCL="${TE_WITH_NCCL:-0}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export TE_FL_PREFER="${TE_FL_PREFER:-vendor}"
export DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-0}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export NVTE_UNFUSED_ATTN="${NVTE_UNFUSED_ATTN:-1}"
export NVTE_TEST_NVINSPECT_FEATURE_DIRS="${NVTE_TEST_NVINSPECT_FEATURE_DIRS:-$TE_PATH/transformer_engine/debug/features}"
export NVTE_TEST_NVINSPECT_CONFIGS_DIR="${NVTE_TEST_NVINSPECT_CONFIGS_DIR:-$TE_PATH/tests/pytorch/debug/test_configs/}"
