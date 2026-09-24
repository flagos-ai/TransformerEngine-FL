#!/usr/bin/env bash
# Source this script before invoking the common CI dispatcher.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/tests/plugin/backend/ppu/set_env.sh"
if [ -n "${GITHUB_WORKSPACE:-}" ]; then
    export XML_LOG_DIR="$GITHUB_WORKSPACE/logs/ppu"
fi
python3 - <<'PY'
import os
from pathlib import Path

import torch
import transformer_engine.pytorch
from transformer_engine.plugin.core.manager import get_default_manager

if not torch.cuda.is_available() or "PPU" not in torch.cuda.get_device_name():
    raise SystemExit("PPU runtime is unavailable")
source = Path(transformer_engine.__file__).resolve()
if not source.is_relative_to(Path(os.environ["TE_PATH"]).resolve()):
    raise SystemExit(f"TransformerEngine was imported from the wrong checkout: {source}")
torch.testing.assert_close((torch.ones(4, device="cuda") + 1).cpu(), torch.full((4,), 2.))
print("PPU runtime:", torch.__version__, torch.cuda.get_device_name(), torch.cuda.device_count())
print("TransformerEngine source:", source)
print("GEMM implementation:", get_default_manager().get_selected_impl_id("generic_gemm"))
PY
if [ -n "${GITHUB_ENV:-}" ]; then
    for key in TE_PATH PYTHONPATH XML_LOG_DIR; do
        printf '%s=%s\n' "$key" "${!key}" >> "$GITHUB_ENV"
    done
fi
