#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

source "$SCRIPT_DIR/set_env.sh"

export PLATFORM="ppu"
export TE_FL_PREFER="${TE_FL_PREFER:-flagos}"
export DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"
export MCORE_USE_CUDA_ENV_DEFAULTS="${MCORE_USE_CUDA_ENV_DEFAULTS:-0}"
export MCORE_FP8_MODE="${MCORE_FP8_MODE:-off}"
export MCORE_PATH="${MCORE_PATH:-/workspace/Megatron-LM-FL}"
export MCORE_REF="${MCORE_REF:-175ae90ec92a9e6fea2d74ccd24d6a1835d3ae82}"
export NUM_LAYERS="${NUM_LAYERS:-2}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
export NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-4}"
export SEQ_LENGTH="${SEQ_LENGTH:-128}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
export ENABLE_DIAGNOSTICS="${ENABLE_DIAGNOSTICS:-0}"

exec bash "$REPO_ROOT/qa/L1_pytorch_mcore_integration/test.sh"
