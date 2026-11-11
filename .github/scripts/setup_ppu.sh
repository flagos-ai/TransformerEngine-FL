#!/usr/bin/env bash
# Source this script before invoking the common CI dispatcher.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export TE_PATH="${GITHUB_WORKSPACE:-$REPO_ROOT}"
source "$REPO_ROOT/tests/plugin/backend/ppu/set_env.sh"
if [ -n "${GITHUB_WORKSPACE:-}" ]; then
    export XML_LOG_DIR="$GITHUB_WORKSPACE/logs/ppu"
fi
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import torch
import transformer_engine.pytorch
from transformer_engine.plugin.core.manager import get_default_manager

if not torch.cuda.is_available():
    raise SystemExit("PPU runtime is unavailable through torch.cuda")
device_count = torch.cuda.device_count()
if device_count < 1:
    raise SystemExit("torch.cuda reports zero PPU devices")
source = Path(transformer_engine.__file__).resolve()
if not source.is_relative_to(Path(os.environ["TE_PATH"]).resolve()):
    raise SystemExit(f"TransformerEngine was imported from the wrong checkout: {source}")
device = torch.device("cuda")
lhs = torch.ones((2, 2), device=device)
rhs = torch.full((2, 2), 2.0, device=device)
torch.testing.assert_close((lhs @ rhs).cpu(), torch.full((2, 2), 4.0))
print("PPU runtime:", torch.__version__, torch.cuda.get_device_name(0), device_count)
print("TransformerEngine source:", source)
print("GEMM implementation:", get_default_manager().get_selected_impl_id("generic_gemm"))
PY
if [ -n "${GITHUB_ENV:-}" ]; then
    for key in TE_PATH TE_LIB_PATH PYTHONPATH PLATFORM TE_FL_SKIP_CUDA \
        NVTE_FRAMEWORK TE_FL_PREFER NVTE_FLASH_ATTN NVTE_FUSED_ATTN \
        NVTE_UNFUSED_ATTN NVTE_TORCH_COMPILE PYTHON_BIN PYTHONDONTWRITEBYTECODE \
        XML_LOG_DIR; do
        printf '%s=%s\n' "$key" "${!key}" >> "$GITHUB_ENV"
    done
fi
