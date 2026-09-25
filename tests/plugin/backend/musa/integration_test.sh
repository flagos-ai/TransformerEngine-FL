#!/usr/bin/env bash
# MUSA-specific MCore integration implementation.

set -euo pipefail

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

exec bash "$TE_PATH/qa/L1_pytorch_mcore_integration/test.sh"
