#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
source "$WORKSPACE/tests/plugin/backend/iluvatar/config.sh"
command -v "${PYTHON_BIN:-python3}" >/dev/null || {
    echo "Use a prepared Iluvatar QA image; the CoreX base image has no Python." >&2
    exit 1
}
source "$WORKSPACE/tests/plugin/backend/iluvatar/set_env.sh"
# CI prefers FlagOS, with the per-op reference selections from set_env.sh.
export TE_FL_PREFER=flagos
export TE_FL_ALLOW_VENDORS=
command -v ixsmi >/dev/null
ixsmi
"$PYTHON_BIN" - <<'CHECK'
import os
import sys
import torch
import flag_gems

assert sys.version_info >= (3, 10), "Python 3.10+ is required"
assert torch.cuda.is_available(), "Iluvatar device unavailable"
required = int(os.environ["ILUVATAR_NPROC_PER_NODE"])
assert torch.cuda.device_count() >= required, f"Iluvatar CI requires {required} visible devices"
name = torch.cuda.get_device_name(0)
assert "BI-V150" in name, f"Expected BI-V150, found {name}"
x = torch.ones((16, 16), device="cuda")
torch.testing.assert_close((x @ x).cpu(), torch.full((16, 16), 16.0))
torch.cuda.synchronize()
print("Python:", sys.version, "PyTorch:", torch.__version__, "Device:", name)
CHECK

# Install this checkout without replacing the vendor Torch or compiling CUDA.
cd "$WORKSPACE"
"$PYTHON_BIN" -m pip install --no-deps --no-build-isolation .
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" tests/pytorch/test_sanity_import.py
"$PYTHON_BIN" - <<'CHECK'
from transformer_engine.plugin.core import get_manager
manager = get_manager()
for op in ("generic_gemm", "layernorm_fwd", "layernorm_bwd", "rmsnorm_fwd", "rmsnorm_bwd"):
    selected = manager.get_selected_impl_id(op)
    expected = "reference.torch" if op == "generic_gemm" else "default.flagos"
    if selected != expected:
        raise RuntimeError(f"{op}: expected {expected}, selected {selected!r}")
    print(f"{op}: {selected}")
selected = manager.get_selected_impl_id("multi_tensor_unscale_l2norm")
if selected != "reference.torch":
    raise RuntimeError(f"multi_tensor_unscale_l2norm: expected reference.torch, selected {selected!r}")
print(f"multi_tensor_unscale_l2norm: {selected}")
CHECK
"$PYTHON_BIN" - <<'CHECK'
import importlib

# The complete QA matrix requires these dependencies in the prepared image.
for module in ("pytest", "coverage", "pytest_cov", "yaml", "expecttest",
               "nvdlfw_inspect.api", "onnx", "onnxscript", "onnxruntime",
               "onnxruntime_extensions", "transformers", "torchvision", "scipy",
               "pybind11", "tensorboard", "regex", "six", "psutil"):
    importlib.import_module(module)
    print(f"QA dependency ready: {module}")
CHECK

if [ -n "${GITHUB_ENV:-}" ]; then
    for key in PATH LD_LIBRARY_PATH PYTHONPATH TE_PATH PLATFORM IX_HOME CUDA_HOME \
        TE_FL_SKIP_CUDA NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD NVTE_FRAMEWORK \
        NVTE_WITH_NCCL_EP TE_FL_PREFER TE_FL_PER_OP TE_FL_ALLOW_VENDORS CUDA_VISIBLE_DEVICES TE_FL_STRICT \
        NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN PYTHON_BIN TE_LIB_PATH XML_LOG_DIR \
        PYTHONDONTWRITEBYTECODE ILUVATAR_NPROC_PER_NODE NVTE_TEST_CHECKPOINT_ARTIFACT_PATH; do
        printf '%s=%s\n' "$key" "${!key}" >> "$GITHUB_ENV"
    done
fi
