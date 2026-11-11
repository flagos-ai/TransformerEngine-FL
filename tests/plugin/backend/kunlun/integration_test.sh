#!/usr/bin/env bash
# KunlunXin-specific MCore integration implementation.

set -euo pipefail

MCORE_PATH="${MCORE_PATH:-/workspace/Megatron-LM-FL}"
MCORE_WRAPPER_DIR="${TE_PATH}/qa/L1_pytorch_mcore_integration/output"
MCORE_ENTRYPOINT="${MCORE_WRAPPER_DIR}/pretrain_gpt_kunlun_wrapper.py"

export NUM_LAYERS="${NUM_LAYERS:-2}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
export NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-4}"
export SEQ_LENGTH="${SEQ_LENGTH:-128}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
export ENABLE_DIAGNOSTICS="${ENABLE_DIAGNOSTICS:-0}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export MCORE_PATH
export MCORE_REPO_URL="${MCORE_REPO_URL:-https://github.com/flagos-ai/Megatron-LM-FL.git}"
export MCORE_REF="${MCORE_REF:-175ae90ec92a9e6fea2d74ccd24d6a1835d3ae82}"
export MCORE_ENTRYPOINT
export MCORE_USE_CUDA_ENV_DEFAULTS=0

mkdir -p "$MCORE_WRAPPER_DIR"
cat > "$MCORE_ENTRYPOINT" <<'PY'
import os
import runpy
import sys
from pathlib import Path

mcore_path = Path(os.environ["MCORE_PATH"])
pretrain_path = mcore_path / "pretrain_gpt.py"
sys.path.insert(0, str(mcore_path))

import megatron.training as training

original_get_args = training.get_args


def get_args_with_kunlun_defaults(*args, **kwargs):
    args_namespace = original_get_args(*args, **kwargs)
    if not hasattr(args_namespace, "no_shared_fs"):
        args_namespace.no_shared_fs = False
    return args_namespace


training.get_args = get_args_with_kunlun_defaults
try:
    import megatron.training.global_vars as global_vars

    global_vars.get_args = get_args_with_kunlun_defaults
except ImportError:
    pass

runpy.run_path(str(pretrain_path), run_name="__main__")
PY

exec bash "$TE_PATH/qa/L1_pytorch_mcore_integration/test.sh"
