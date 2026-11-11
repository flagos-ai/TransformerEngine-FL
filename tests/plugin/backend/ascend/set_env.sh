#!/usr/bin/env bash
# Ascend Backend Environment Setup

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TE_PATH="${TE_PATH:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd)}"
export TE_LIB_PATH="${TE_LIB_PATH:-$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)/transformer_engine}"
export PYTHONPATH="${TE_PATH}${PYTHONPATH:+:$PYTHONPATH}"
export PLATFORM="${PLATFORM:-ascend}"
export TE_FL_SKIP_CUDA="${TE_FL_SKIP_CUDA:-1}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export NVTE_WITH_CUDA="${NVTE_WITH_CUDA:-0}"
export NVTE_WITH_MACA="${NVTE_WITH_MACA:-0}"
export NVTE_WITH_NCCL_EP="${NVTE_WITH_NCCL_EP:-0}"
export TE_WITH_NCCL="${TE_WITH_NCCL:-0}"
export TE_FL_REQUIRE_NPU_VENDOR="${TE_FL_REQUIRE_NPU_VENDOR:-1}"
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0,1,2,3}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
