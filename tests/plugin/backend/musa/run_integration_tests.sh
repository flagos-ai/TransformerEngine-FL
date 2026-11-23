#!/usr/bin/env bash
# MUSA Backend Integration Tests Entry Point

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"

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

echo "MUSA collective backend check: timeout=${MUSA_MCORE_BACKEND_CHECK_TIMEOUT:-120}s"
timeout "${MUSA_MCORE_BACKEND_CHECK_TIMEOUT:-120}s" python3 -u - <<'PY'
import os
import tempfile
import time

started_at = time.monotonic()


def log_stage(message):
    elapsed = time.monotonic() - started_at
    print(f"[MUSA backend check +{elapsed:.2f}s] {message}", flush=True)


log_stage("Importing torch")
import torch

log_stage("Imported torch; importing torch_musa")
import torch_musa  # noqa: F401

log_stage("Imported torch_musa; importing torch.distributed")
import torch.distributed as dist

log_stage("Imported torch.distributed")
backend = os.environ["DISTRIBUTED_BACKEND"]
if backend not in {"mccl", "nccl", "gloo"}:
    raise RuntimeError(
        f"MUSA integration launcher accepts mccl/nccl/gloo, not {backend!r}"
    )
if backend == "nccl" and not dist.is_nccl_available():
    raise RuntimeError("NCCL is not available in the current MUSA torch image")

with tempfile.TemporaryDirectory(prefix="te_mcore_musa_") as temp_dir:
    try:
        log_stage(f"Initializing process group: backend={backend}, world_size=1")
        dist.init_process_group(
            backend=backend,
            init_method=f"file://{temp_dir}/store",
            rank=0,
            world_size=1,
        )
        log_stage("Initialized process group")
        log_stage("Creating MUSA tensor")
        tensor = torch.ones(1, device="musa")
        log_stage("Created MUSA tensor")
        log_stage("Calling all_reduce")
        dist.all_reduce(tensor)
        log_stage("all_reduce returned")
    finally:
        if dist.is_initialized():
            log_stage("Destroying process group")
            dist.destroy_process_group()
            log_stage("Destroyed process group")

log_stage(f"MUSA collective backend is usable: {backend}")
PY

echo "Running ${PLATFORM_DISPLAY_NAME} integration tests"
exec bash "$TE_PATH/$PLATFORM_INTEGRATION_TEST_SCRIPT"
