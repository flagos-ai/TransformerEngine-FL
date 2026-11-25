#!/usr/bin/env bash
# Iluvatar Backend Integration Tests Entry Point

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"

# Platform runtime adaptation stays here; test execution is shared QA.
export PLATFORM=iluvatar
export TE_FL_PREFER="${TE_FL_PREFER:-flagos}"
# Use the shared MCore checkout and four-step BF16 mock-data training.
# Avoid CUDA-only environment defaults; keep the initial model within BI-V150 memory.
export MCORE_USE_CUDA_ENV_DEFAULTS=0
export MCORE_FP8_MODE="${MCORE_FP8_MODE:-off}"
export DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"
export NUM_LAYERS="${NUM_LAYERS:-2}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
export NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-4}"
export SEQ_LENGTH="${SEQ_LENGTH:-128}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
export ENABLE_DIAGNOSTICS="${ENABLE_DIAGNOSTICS:-0}"

echo "Running ${PLATFORM_DISPLAY_NAME} integration tests"
exec bash "$TE_PATH/$PLATFORM_INTEGRATION_TEST_SCRIPT"
