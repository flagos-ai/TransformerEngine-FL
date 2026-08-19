#!/usr/bin/env bash
set -euo pipefail

# Hygon/DTK-specific environment for plugin QA workflows.
# Keep chip/runtime details here so common QA entrypoints do not need
# Hygon-specific branches.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

if [ -f "${DTK_ENV_SH:-/opt/dtk/env.sh}" ]; then
    # DTK owns Hygon runtime paths; keep them out of common workflow logic.
    source "${DTK_ENV_SH:-/opt/dtk/env.sh}"
fi

export TE_PATH="${TE_PATH:-$REPO_ROOT}"
export XML_LOG_DIR="${XML_LOG_DIR:-$TE_PATH/logs}"
export TE_FL_SKIP_CUDA="${TE_FL_SKIP_CUDA:-1}"
export TE_FL_PREFER="${TE_FL_PREFER:-reference}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPATH="$TE_PATH${PYTHONPATH:+:$PYTHONPATH}"

# The current DTK reference/vendor CI path should avoid fused CUDA attention
# assumptions.
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-0}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export NVTE_UNFUSED_ATTN="${NVTE_UNFUSED_ATTN:-1}"

# ONNX export tests can emulate FP8 attention when no native backend is
# available.
export NVTE_UnfusedDPA_Emulate_FP8="${NVTE_UnfusedDPA_Emulate_FP8:-1}"

mkdir -p "$XML_LOG_DIR"
