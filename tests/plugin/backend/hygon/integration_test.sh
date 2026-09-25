#!/usr/bin/env bash
# Hygon-specific MCore integration implementation.

set -euo pipefail

export PLATFORM="hygon"
export TE_FL_PREFER="reference"
export MCORE_REPO_URL="${MCORE_REPO_URL:-https://github.com/flagos-ai/Megatron-LM-FL.git}"
export MCORE_REF="${MCORE_REF:-175ae90ec92a9e6fea2d74ccd24d6a1835d3ae82}"
export DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"
export NUM_LAYERS="${NUM_LAYERS:-2}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
export NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-4}"
export SEQ_LENGTH="${SEQ_LENGTH:-128}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
export ENABLE_DIAGNOSTICS="${ENABLE_DIAGNOSTICS:-0}"

exec bash "$TE_PATH/qa/L1_pytorch_mcore_integration/test.sh"
