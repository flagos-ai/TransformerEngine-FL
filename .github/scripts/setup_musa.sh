#!/usr/bin/env bash
# MUSA Platform Environment Setup Script
# Called by unit_tests_common.yml / integration_tests_common.yml for MUSA platforms.
set -euo pipefail

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
source "$WORKSPACE/tests/plugin/backend/musa/set_env.sh"

echo "===== Step 0: Base Environment ====="
echo "Python: $(which python3) ($(python3 --version 2>&1))"

if [ -n "${GITHUB_ENV:-}" ]; then
    {
        echo "PATH=$PATH"
        echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
        echo "MUSA_HOME=$MUSA_HOME"
        echo "CUDA_HOME=$CUDA_HOME"
        echo "PLATFORM=$PLATFORM"
        echo "TE_FL_SKIP_CUDA=$TE_FL_SKIP_CUDA"
        echo "SKIP_CUDA_BUILD=$SKIP_CUDA_BUILD"
        echo "NVTE_WITH_CUDA=$NVTE_WITH_CUDA"
        echo "NVTE_WITH_MACA=$NVTE_WITH_MACA"
        echo "NVTE_WITH_NCCL_EP=$NVTE_WITH_NCCL_EP"
        echo "NVTE_FRAMEWORK=$NVTE_FRAMEWORK"
        echo "TE_FL_ENABLE_MUSA_CUDA_COMPAT=$TE_FL_ENABLE_MUSA_CUDA_COMPAT"
        echo "TORCH_DEVICE_BACKEND_AUTOLOAD=$TORCH_DEVICE_BACKEND_AUTOLOAD"
        echo "TE_FL_PREFER=$TE_FL_PREFER"
    } >> "$GITHUB_ENV"
fi

echo "===== Step 1: Verify Image Dependencies ====="
python3 - <<'PY'
from importlib import metadata

required = (
    "pytest",
    "expecttest",
    "nvdlfw-inspect",
    "onnxruntime",
    "onnxruntime-extensions",
)
missing = []
for package in required:
    try:
        print(f"{package}=={metadata.version(package)}")
    except metadata.PackageNotFoundError:
        missing.append(package)

if missing:
    raise RuntimeError(f"Missing MUSA CI image dependencies: {', '.join(missing)}")
PY

echo "===== Step 2: Verify Checked-out TransformerEngine-FL Python Layer ====="
cd "${GITHUB_WORKSPACE}"
python3 -c "import transformer_engine; print('transformer_engine:', transformer_engine.__file__)"

echo "===== Step 3: Verify MUSA Runtime ====="
python3 - <<'PY'
import importlib
import os
import subprocess
import sys

import torch
import torch_musa  # noqa: F401  # Registers torch.musa in the base image.

if not hasattr(torch, "musa") or not torch.musa.is_available():
    raise RuntimeError("MUSA runtime is not available in the current CI container")

# Check the image's installed TE in an isolated interpreter so the checked-out
# TE-FL package on PYTHONPATH cannot shadow or mix with its package modules.
extension_check = r'''
import importlib, sys
import torch, torch_musa
import transformer_engine
if not hasattr(torch, "musa") or not torch.musa.is_available():
    raise RuntimeError("MUSA runtime is unavailable")
try:
    package = importlib.import_module("transformer_engine_musa")
    tex = importlib.import_module("transformer_engine_musa_torch")
    layout = "standalone"
except ModuleNotFoundError as exc:
    if exc.name not in {"transformer_engine_musa", "transformer_engine_musa_torch"}:
        raise
    package = importlib.import_module("transformer_engine.musa")
    importlib.import_module("transformer_engine.musa.pytorch")
    tex = sys.modules.get("transformer_engine_torch") or importlib.import_module("transformer_engine_torch")
    layout = "transformer_engine.musa (bundled TE 2.x extension)"
required = ("multi_tensor_scale", "multi_tensor_compute_scale_and_scale_inv")
missing = [name for name in required if not hasattr(tex, name)]
if missing:
    raise RuntimeError("MUSA torch extension is missing APIs: " + ", ".join(missing))
print("installed transformer_engine:", transformer_engine.__file__)
print("MUSA package:", package.__file__)
print("MUSA layout:", layout)
print("MUSA torch extension:", tex.__file__)
print("required MUSA APIs:", ", ".join(required))
'''
clean_env = os.environ.copy()
clean_env.pop("PYTHONPATH", None)
subprocess.run([sys.executable, "-c", extension_check], cwd="/tmp", env=clean_env, check=True)

from transformer_engine.plugin.core.backends.vendor.musa.musa import MUSABackend
from transformer_engine.plugin.core.manager import OpManager

backend = MUSABackend()
if not backend.is_available():
    raise RuntimeError("transformer_engine vendor.musa backend is not available")

# The dispatch order is flagos -> vendor -> reference on every platform, so the
# selected implementation depends on what FlagOS registers. Record it instead of
# pinning one: vendor.musa availability is already asserted above.
selected_impl = OpManager().get_selected_impl_id("generic_gemm")

print("required MUSA backend APIs verified in installed TE extension")
print("vendor.musa backend is available")
print("generic_gemm selected implementation:", selected_impl)
PY

echo "===== MUSA Environment Setup Complete ====="
