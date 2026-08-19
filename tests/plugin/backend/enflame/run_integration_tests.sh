#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

source "$SCRIPT_DIR/set_env.sh"

export ENFLAME_FORCE_MEGATRON_PLATFORM="${ENFLAME_FORCE_MEGATRON_PLATFORM:-1}"
export TE_FL_PREFER="${TE_FL_PREFER:-reference}"
export ENABLE_DIAGNOSTICS="${ENABLE_DIAGNOSTICS:-0}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export NVTE_TORCH_COMPILE="${NVTE_TORCH_COMPILE:-0}"

exec bash "$REPO_ROOT/qa/L1_pytorch_mcore_integration/test.sh"
