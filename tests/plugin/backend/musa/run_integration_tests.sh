#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${TE_PATH:=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)}"

export PLATFORM=mthreads
export TE_FL_PREFER="${TE_FL_PREFER:-vendor}"
export TE_FL_PER_OP="${TE_FL_PER_OP:-layernorm_fwd=reference|flagos|vendor;layernorm_bwd=reference|flagos|vendor}"
export DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-mccl}"
export PYTHONPATH="${TE_PATH}:${PYTHONPATH:-}"
export NUM_LAYERS="${NUM_LAYERS:-2}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
export NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-4}"
export SEQ_LENGTH="${SEQ_LENGTH:-128}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
export ENABLE_DIAGNOSTICS="${ENABLE_DIAGNOSTICS:-0}"
export MCORE_PREPARE_SCRIPT="${MCORE_PREPARE_SCRIPT:-${SCRIPT_DIR}/patch_megatron_mccl.py}"

timeout "${MUSA_MCORE_BACKEND_CHECK_TIMEOUT:-15}s" python3 - <<'PY'
import os
import tempfile

import torch
import torch_musa  # noqa: F401
import torch.distributed as dist

backend = os.environ["DISTRIBUTED_BACKEND"]
if backend not in {"mccl", "nccl", "gloo"}:
    raise RuntimeError(
        f"MUSA integration launcher accepts mccl/nccl/gloo, not {backend!r}"
    )
if backend == "nccl" and not dist.is_nccl_available():
    raise RuntimeError("NCCL is not available in the current MUSA torch image")

with tempfile.TemporaryDirectory(prefix="te_mcore_musa_") as temp_dir:
    try:
        dist.init_process_group(
            backend=backend,
            init_method=f"file://{temp_dir}/store",
            rank=0,
            world_size=1,
        )
        tensor = torch.ones(1, device="musa")
        dist.all_reduce(tensor)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

print(f"MUSA collective backend is usable: {backend}")
PY

exec bash "${TE_PATH}/qa/L1_pytorch_mcore_integration/test.sh"
