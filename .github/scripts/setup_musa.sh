#!/usr/bin/env bash
# MUSA Platform Environment Setup Script
# Called by unit_tests_common.yml / integration_tests_common.yml for MUSA platforms.
set -euo pipefail

echo "===== Step 0: Base Environment ====="
echo "Python: $(which python3) ($(python3 --version 2>&1))"
export PATH=/usr/local/musa/bin:${PATH}
export LD_LIBRARY_PATH=/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/musa/lib:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}
export MUSA_HOME=${MUSA_HOME:-/usr/local/musa}
export CUDA_HOME=${CUDA_HOME:-/usr/local/musa}
export PLATFORM="${PLATFORM:-mthreads}"
export TE_FL_SKIP_CUDA="${TE_FL_SKIP_CUDA:-1}"
export SKIP_CUDA_BUILD="${SKIP_CUDA_BUILD:-1}"
export NVTE_WITH_CUDA="${NVTE_WITH_CUDA:-0}"
export NVTE_WITH_MACA="${NVTE_WITH_MACA:-0}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export TE_FL_ENABLE_MUSA_CUDA_COMPAT="${TE_FL_ENABLE_MUSA_CUDA_COMPAT:-1}"
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
export TE_FL_PREFER="${TE_FL_PREFER:-vendor}"

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

import torch
import transformer_engine

if not hasattr(torch, "musa") or not torch.musa.is_available():
    raise RuntimeError("MUSA runtime is not available in the current CI container")

import transformer_engine_musa  # noqa: F401
tex = importlib.import_module("transformer_engine_musa_torch")
print("transformer_engine:", transformer_engine.__file__)
print("transformer_engine_musa_torch:", tex.__file__)
required_symbols = (
    "multi_tensor_scale",
    "multi_tensor_compute_scale_and_scale_inv",
)
missing_symbols = [name for name in required_symbols if not hasattr(tex, name)]
if missing_symbols:
    raise RuntimeError(
        "transformer_engine_musa_torch is missing required APIs: "
        + ", ".join(missing_symbols)
    )

from transformer_engine.plugin.core.backends.vendor.musa.musa import MUSABackend
from transformer_engine.plugin.core.manager import OpManager

backend = MUSABackend()
if not backend.is_available():
    raise RuntimeError("transformer_engine vendor.musa backend is not available")

selected_impl = OpManager().get_selected_impl_id("generic_gemm")
if selected_impl != "vendor.musa":
    raise RuntimeError(
        "generic_gemm did not select vendor.musa; selected " + repr(selected_impl)
    )

print("required MUSA backend APIs:", ", ".join(required_symbols))
print("vendor.musa backend is available")
print("generic_gemm selected implementation:", selected_impl)
PY

echo "===== MUSA Environment Setup Complete ====="
