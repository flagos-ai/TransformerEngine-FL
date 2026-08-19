#!/usr/bin/env bash
# KunlunXin XPU environment setup for TransformerEngine-FL.
set -euo pipefail

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"

export PLATFORM="${PLATFORM:-kunlunxin}"
export TE_FL_SKIP_CUDA="${TE_FL_SKIP_CUDA:-1}"
export SKIP_CUDA_BUILD="${SKIP_CUDA_BUILD:-1}"
export NVTE_WITH_CUDA="${NVTE_WITH_CUDA:-0}"
export NVTE_WITH_MACA="${NVTE_WITH_MACA:-0}"
export TE_WITH_NCCL="${TE_WITH_NCCL:-0}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export TE_FL_PREFER="${TE_FL_PREFER:-vendor}"
export DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-0}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export NVTE_UNFUSED_ATTN="${NVTE_UNFUSED_ATTN:-1}"

echo "===== Activate KunlunXin Python environment ====="
if [ -f /root/miniconda/etc/profile.d/conda.sh ]; then
    source /root/miniconda/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-python310_torch29_cuda}"
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-base}"
elif [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-base}"
else
    echo "WARNING: No supported conda installation found; using current environment"
fi

echo "===== Configure KunlunXin runtime ====="
if [ -n "${XPU_HOME:-}" ] && [ -d "${XPU_HOME}/lib" ]; then
    export LD_LIBRARY_PATH="${XPU_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi
if [ -d /opt/kunlunxin/lib ]; then
    export LD_LIBRARY_PATH="/opt/kunlunxin/lib:${LD_LIBRARY_PATH:-}"
fi

if [ -n "${GITHUB_ENV:-}" ]; then
    {
        echo "PLATFORM=$PLATFORM"
        echo "TE_FL_SKIP_CUDA=$TE_FL_SKIP_CUDA"
        echo "SKIP_CUDA_BUILD=$SKIP_CUDA_BUILD"
        echo "NVTE_WITH_CUDA=$NVTE_WITH_CUDA"
        echo "NVTE_WITH_MACA=$NVTE_WITH_MACA"
        echo "TE_WITH_NCCL=$TE_WITH_NCCL"
        echo "NVTE_FRAMEWORK=$NVTE_FRAMEWORK"
        echo "TE_FL_PREFER=$TE_FL_PREFER"
        echo "DISTRIBUTED_BACKEND=$DISTRIBUTED_BACKEND"
        echo "NVTE_FLASH_ATTN=$NVTE_FLASH_ATTN"
        echo "NVTE_FUSED_ATTN=$NVTE_FUSED_ATTN"
        echo "NVTE_UNFUSED_ATTN=$NVTE_UNFUSED_ATTN"
        echo "PATH=$PATH"
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    } >> "$GITHUB_ENV"
fi

echo "Python: $(which python3) ($(python3 --version 2>&1))"
echo "XPU devices: $(find /dev -maxdepth 1 -name 'xpu[0-9]*' -printf '%f ' 2>/dev/null || true)"

echo "===== Install test dependencies ====="
python3 -m pip install --disable-pip-version-check \
    pytest==8.2.1 expecttest coverage pytest-cov \
    onnxruntime onnxruntime_extensions

echo "===== Install TransformerEngine-FL Python/plugin layer ====="
cd "$WORKSPACE"
python3 -m pip uninstall -y transformer_engine transformer_engine_torch || true
TE_FL_SKIP_CUDA=1 SKIP_CUDA_BUILD=1 python3 setup.py install

echo "===== Verify KunlunXin environment ====="
python3 - <<'PY'
import importlib.metadata as metadata
import os

import torch
import transformer_engine_klx_torch

from transformer_engine.plugin.core.backends.vendor.kunlunxin.kunlunxin import (
    KunLunXinBackend,
)
from transformer_engine.plugin.core.manager import get_default_manager

print("torch:", torch.__version__)
print("pytest:", metadata.version("pytest"))
print("coverage:", metadata.version("coverage"))
print("pytest-cov:", metadata.version("pytest-cov"))
print("onnxruntime:", metadata.version("onnxruntime"))

if not os.path.exists("/dev/xpu0"):
    raise SystemExit("KunlunXin XPU device is not available")

backend = KunLunXinBackend()
if not backend.is_available():
    raise SystemExit("vendor.kunlunxin backend is not available")

selected_impl = get_default_manager().get_selected_impl_id("generic_gemm")
if selected_impl != "vendor.kunlunxin":
    raise SystemExit(
        "generic_gemm did not select vendor.kunlunxin; selected "
        + repr(selected_impl)
    )

print("transformer_engine_klx_torch:", transformer_engine_klx_torch.__file__)
print("vendor.kunlunxin backend is available")
print("generic_gemm selected implementation:", selected_impl)
PY
python3 tests/pytorch/test_sanity_import.py

echo "===== KunlunXin environment setup complete ====="
