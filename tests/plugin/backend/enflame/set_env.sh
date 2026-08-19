#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

export TE_PATH="${TE_PATH:-$REPO_ROOT}"
export XML_LOG_DIR="${XML_LOG_DIR:-${RUNNER_TEMP:-/tmp}/te-fl-enflame-logs}"
export TE_FL_SKIP_CUDA="${TE_FL_SKIP_CUDA:-1}"
export TE_FL_PREFER="${TE_FL_PREFER:-reference}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export ENFLAME_CONFIG_FILE="${ENFLAME_CONFIG_FILE:-$REPO_ROOT/.github/configs/enflame.yml}"

export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-0}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export NVTE_UNFUSED_ATTN="${NVTE_UNFUSED_ATTN:-1}"
export NVTE_UnfusedDPA_Emulate_FP8="${NVTE_UnfusedDPA_Emulate_FP8:-1}"

export PYTHONPATH="$TE_PATH${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$XML_LOG_DIR"
