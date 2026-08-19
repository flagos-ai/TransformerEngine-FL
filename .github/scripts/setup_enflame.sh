#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

export PLATFORM="${PLATFORM:-enflame}"
source "$REPO_ROOT/tests/plugin/backend/enflame/set_env.sh"

if [ -n "${GITHUB_ENV:-}" ]; then
    {
        echo "PATH=$PATH"
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
        echo "PYTHONPATH=$PYTHONPATH"
        echo "TE_PATH=$TE_PATH"
        echo "XML_LOG_DIR=$XML_LOG_DIR"
        echo "PLATFORM=$PLATFORM"
        echo "TE_FL_SKIP_CUDA=$TE_FL_SKIP_CUDA"
        echo "TE_FL_PREFER=$TE_FL_PREFER"
        echo "NVTE_FRAMEWORK=$NVTE_FRAMEWORK"
        echo "PYTHON_BIN=$PYTHON_BIN"
        echo "NVTE_FLASH_ATTN=$NVTE_FLASH_ATTN"
        echo "NVTE_FUSED_ATTN=$NVTE_FUSED_ATTN"
        echo "NVTE_UNFUSED_ATTN=$NVTE_UNFUSED_ATTN"
        echo "NVTE_UnfusedDPA_Emulate_FP8=$NVTE_UnfusedDPA_Emulate_FP8"
        echo "ENFLAME_CONFIG_FILE=$ENFLAME_CONFIG_FILE"
    } >> "$GITHUB_ENV"
fi
