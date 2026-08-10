#!/usr/bin/env bash
# Huawei Ascend NPU environment setup for TransformerEngine-FL.
set -euo pipefail

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"

export PLATFORM="${PLATFORM:-ascend}"
export TE_FL_SKIP_CUDA="${TE_FL_SKIP_CUDA:-1}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export NVTE_WITH_CUDA="${NVTE_WITH_CUDA:-0}"
export NVTE_WITH_MACA="${NVTE_WITH_MACA:-0}"
export TE_WITH_NCCL="${TE_WITH_NCCL:-0}"
export TE_FL_REQUIRE_NPU_VENDOR="${TE_FL_REQUIRE_NPU_VENDOR:-1}"
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0,1,2,3}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

echo "===== Activate Python environment ====="
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-base}"
elif [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-flagscale-train}"
else
    echo "WARNING: No supported conda installation found; using current environment"
fi

echo "===== Load Ascend runtime environment ====="
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f /usr/local/Ascend/latest/set_env.sh ]; then
    source /usr/local/Ascend/latest/set_env.sh
fi

if [ -n "${GITHUB_ENV:-}" ]; then
    # Persist the active runtime and pytest bootstrap for subsequent CI steps.
    {
        echo "PLATFORM=$PLATFORM"
        echo "TE_FL_SKIP_CUDA=$TE_FL_SKIP_CUDA"
        echo "NVTE_FRAMEWORK=$NVTE_FRAMEWORK"
        echo "NVTE_WITH_CUDA=$NVTE_WITH_CUDA"
        echo "NVTE_WITH_MACA=$NVTE_WITH_MACA"
        echo "TE_WITH_NCCL=$TE_WITH_NCCL"
        echo "TE_FL_REQUIRE_NPU_VENDOR=$TE_FL_REQUIRE_NPU_VENDOR"
        echo "ASCEND_VISIBLE_DEVICES=$ASCEND_VISIBLE_DEVICES"
        echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
        echo "PYTORCH_NPU_ALLOC_CONF=$PYTORCH_NPU_ALLOC_CONF"
        echo "PATH=$PATH"
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    } >> "$GITHUB_ENV"
fi

echo "===== Verify Ascend PyTorch runtime ====="
python3 - <<'PY'
import torch
import torch_npu  # noqa: F401

print("torch:", torch.__version__)

if not hasattr(torch, "npu"):
    raise SystemExit("PyTorch NPU API is unavailable")

if not torch.npu.is_available():
    raise SystemExit("Ascend NPU is not available")

print("NPU device count:", torch.npu.device_count())
PY

echo "===== Install test dependencies ====="
python3 -m pip install nvdlfw-inspect --quiet

echo "===== Ensure TransformerEngineNPU vendor wheel ====="
if python3 - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("transformer_engine_npu") else 1)
PY
then
    echo "transformer_engine_npu is already installed"
elif [ -n "${TRANSFORMER_ENGINE_NPU_WHEEL:-}" ]; then
    python3 -m pip install "$TRANSFORMER_ENGINE_NPU_WHEEL"
elif [ -n "${TRANSFORMER_ENGINE_NPU_WHEEL_DIR:-}" ] && [ -d "$TRANSFORMER_ENGINE_NPU_WHEEL_DIR" ]; then
    shopt -s nullglob
    npu_wheels=("$TRANSFORMER_ENGINE_NPU_WHEEL_DIR"/transformer_engine_npu*.whl)
    shopt -u nullglob
    if [ "${#npu_wheels[@]}" -eq 0 ]; then
        echo "No transformer_engine_npu wheel found in TRANSFORMER_ENGINE_NPU_WHEEL_DIR=$TRANSFORMER_ENGINE_NPU_WHEEL_DIR" >&2
        exit 1
    fi
    python3 -m pip install "${npu_wheels[0]}"
elif [ "$TE_FL_REQUIRE_NPU_VENDOR" = "1" ]; then
    echo "transformer_engine_npu is required for Ascend vendor.npu CI but is not installed." >&2
    echo "Install it in the CI image, or provide TRANSFORMER_ENGINE_NPU_WHEEL / TRANSFORMER_ENGINE_NPU_WHEEL_DIR." >&2
    exit 1
else
    echo "WARNING: transformer_engine_npu is not installed; vendor.npu tests may be skipped or fail."
fi

echo "===== Install TransformerEngine-FL Python/plugin layer ====="
cd "$WORKSPACE"
python3 -m pip uninstall -y transformer_engine transformer_engine_torch || true
TE_FL_SKIP_CUDA=1 python3 setup.py install

echo "===== Verify TransformerEngine installation ====="
python3 tests/pytorch/test_sanity_import.py

echo "===== Verify Ascend vendor.npu backend ====="
if python3 - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("transformer_engine_npu") else 1)
PY
then
    python3 - <<'PY'
import torch
import torch_npu  # noqa: F401
import transformer_engine_npu  # noqa: F401

from transformer_engine.plugin.core.backends.vendor.npu.npu import NPUBackend

backend = NPUBackend()
if not torch.npu.is_available():
    raise SystemExit("Ascend NPU is not available")
if not backend.is_available():
    raise SystemExit("vendor.npu backend is not available")

print("vendor.npu backend is available")
PY
elif [ "$TE_FL_REQUIRE_NPU_VENDOR" = "1" ]; then
    echo "transformer_engine_npu is required for Ascend vendor.npu CI but is not installed." >&2
    exit 1
else
    echo "WARNING: skipped vendor.npu verification because transformer_engine_npu is not installed."
fi

echo "===== Ascend environment setup complete ====="
