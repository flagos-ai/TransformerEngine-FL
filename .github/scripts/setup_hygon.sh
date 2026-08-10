#!/usr/bin/env bash
# Hygon/DTK environment setup for TransformerEngine-FL plugin QA.
set -euo pipefail

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"

echo "===== Load Hygon/DTK runtime environment ====="
source "$WORKSPACE/tests/plugin/backend/hygon/set_env.sh"

# Hygon CI is a reference-backend baseline. Force the selection policy here so
# inherited shell state cannot silently fall back to FlagOS.
export TE_FL_SKIP_CUDA=1
export TE_FL_PREFER=reference
export NVTE_FRAMEWORK=pytorch
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=1
export NVTE_UnfusedDPA_Emulate_FP8=1

echo "===== Verify Hygon device visibility ====="
if [ "${HYGON_REQUIRE_DEVICE:-1}" = "1" ] && ! command -v hy-smi >/dev/null 2>&1; then
    echo "ERROR: hy-smi is unavailable in the Hygon CI image" >&2
    exit 1
elif command -v hy-smi >/dev/null 2>&1; then
    hy-smi
else
    echo "WARNING: hy-smi is unavailable; device verification is disabled"
fi

echo "===== Verify Python runtime ====="
"$PYTHON_BIN" - <<'PY'
import os
import sys

print("python:", sys.executable)
print("version:", sys.version)

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(f"PyTorch is required in the Hygon CI image: {exc}") from exc

print("torch:", torch.__version__)

if os.environ.get("HYGON_REQUIRE_DEVICE", "1") == "1":
    if not torch.cuda.is_available():
        raise SystemExit("Hygon DCU is not visible through torch.cuda")

    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise SystemExit("torch.cuda reports zero Hygon devices")

    device = torch.device("cuda")
    lhs = torch.ones((2, 2), device=device)
    rhs = torch.full((2, 2), 2.0, device=device)
    result = lhs @ rhs
    if not torch.allclose(result.cpu(), torch.full((2, 2), 4.0)):
        raise SystemExit("Hygon DCU matrix-multiplication smoke test failed")

    print("cuda_device_count:", device_count)
    print("cuda_device_name:", torch.cuda.get_device_name(0))
    print("matmul_smoke: passed")
PY

echo "===== Verify reference backend selection ====="
"$PYTHON_BIN" - <<'PY'
from transformer_engine.plugin.core import get_manager

manager = get_manager()
selected_impl = manager.get_selected_impl_id("generic_gemm")
if selected_impl != "reference.torch":
    raise SystemExit(
        f"Expected generic_gemm to use reference.torch, selected {selected_impl!r}"
    )
print("generic_gemm_impl:", selected_impl)
PY

echo "===== Install Hygon QA dependencies ====="
if [ "${HYGON_SKIP_DEP_INSTALL:-0}" = "1" ]; then
    echo "Skipping Python dependency installation because HYGON_SKIP_DEP_INSTALL=1"
else
    missing_modules=()
    for module_name in pytest expecttest coverage pytest_cov; do
        if ! "$PYTHON_BIN" -c "import importlib; importlib.import_module('$module_name')" >/dev/null 2>&1; then
            missing_modules+=("$module_name")
        fi
    done

    if [ "${#missing_modules[@]}" -gt 0 ]; then
        echo "Missing Hygon QA modules: ${missing_modules[*]}"
        "$PYTHON_BIN" -m pip install pytest==8.2.1 expecttest coverage pytest-cov
    else
        echo "Hygon QA dependencies are already available in the image"
    fi

    # ONNX dependencies are installed only by the ONNX test group.
fi

"$PYTHON_BIN" -c "import coverage, pytest_cov; print('coverage dependencies: ready')"

if [ "${HYGON_INSTALL_TE:-0}" = "1" ]; then
    echo "===== Install TransformerEngine-FL Python layer ====="
    cd "$WORKSPACE"
    TE_FL_SKIP_CUDA=1 "$PYTHON_BIN" setup.py install
else
    echo "Skipping TransformerEngine-FL install; tests run from source via PYTHONPATH"
fi

if [ -n "${GITHUB_ENV:-}" ]; then
    {
        echo "PATH=$PATH"
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
        echo "PYTHONPATH=$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
        echo "TE_PATH=$WORKSPACE"
        echo "XML_LOG_DIR=$WORKSPACE/logs"
        echo "PLATFORM=$PLATFORM"
        echo "TE_FL_SKIP_CUDA=$TE_FL_SKIP_CUDA"
        echo "TE_FL_PREFER=$TE_FL_PREFER"
        echo "NVTE_FRAMEWORK=$NVTE_FRAMEWORK"
        echo "PYTHON_BIN=$PYTHON_BIN"
        echo "NVTE_FLASH_ATTN=$NVTE_FLASH_ATTN"
        echo "NVTE_FUSED_ATTN=$NVTE_FUSED_ATTN"
        echo "NVTE_UNFUSED_ATTN=$NVTE_UNFUSED_ATTN"
        echo "NVTE_UnfusedDPA_Emulate_FP8=$NVTE_UnfusedDPA_Emulate_FP8"
        echo "HYGON_REQUIRE_DEVICE=${HYGON_REQUIRE_DEVICE:-1}"
    } >> "$GITHUB_ENV"
fi

echo "===== Hygon environment setup complete ====="
