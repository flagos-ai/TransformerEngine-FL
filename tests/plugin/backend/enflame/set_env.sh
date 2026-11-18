#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

export TE_PATH="${TE_PATH:-$REPO_ROOT}"
export TE_LIB_PATH="${TE_LIB_PATH:-$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)/transformer_engine}"
export XML_LOG_DIR="${XML_LOG_DIR:-${RUNNER_TEMP:-/tmp}/te-fl-enflame-logs}"
export TE_FL_SKIP_CUDA="${TE_FL_SKIP_CUDA:-1}"
# TE-FL uses one dispatch order on every platform: flagos -> vendor ->
# reference (policy.py PREFER_DEFAULT). Do not pin a different preference
# here; quarantine the failing cases in the platform config instead.
export TE_FL_PREFER="${TE_FL_PREFER:-flagos}"
# FlagGems layernorm does not compile on the GCU backend (arith.extsi cannot be
# legalized), so layernorm_bwd falls back while layernorm_fwd succeeds. A
# forward/backward pair from different implementations produced NaN losses in
# mcore training (run 43798250037), so keep the pair together on reference
# until the FlagGems Enflame kernel compiles.
export TE_FL_PER_OP="${TE_FL_PER_OP:-layernorm_fwd=reference|flagos|vendor;layernorm_bwd=reference|flagos|vendor}"
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
