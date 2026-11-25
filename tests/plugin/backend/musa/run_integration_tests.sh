#!/usr/bin/env bash
# MUSA Backend Integration Tests Entry Point

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"

# Platform runtime adaptation stays here; test execution is shared QA.
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

echo "Running ${PLATFORM_DISPLAY_NAME} integration tests"
exec bash "$TE_PATH/$PLATFORM_INTEGRATION_TEST_SCRIPT"
