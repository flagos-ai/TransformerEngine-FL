#!/usr/bin/env bash
# Install the checked-out TE-FL plugin layer into the MetaX CI environment.
set -euo pipefail

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"

echo "===== Activate Python environment ====="
if [ -x /opt/te-ci/bin/python3 ]; then
    export PATH="/opt/te-ci/bin:$PATH"
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
fi
source "$WORKSPACE/tests/plugin/backend/metax/set_env.sh"
echo "Python: $(command -v python3) ($(python3 --version 2>&1))"

if [ -n "${GITHUB_ENV:-}" ]; then
    {
        echo "TE_FL_SKIP_CUDA=$TE_FL_SKIP_CUDA"
        echo "TE_FL_PREFER=$TE_FL_PREFER"
        echo "NVTE_WITH_MACA=$NVTE_WITH_MACA"
        echo "NVTE_WITH_NCCL_EP=$NVTE_WITH_NCCL_EP"
        echo "CUDA_HOME=$CUDA_HOME"
        echo "MACA_HOME=$MACA_HOME"
        echo "TE_METAX_TE_HOME=${TE_METAX_TE_HOME:-}"
        echo "PATH=$PATH"
        echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    } >> "$GITHUB_ENV"
fi

echo "===== Verify image dependencies ====="
python3 - <<'CHECK'
from importlib import metadata

for package in (
    "torch", "triton", "pytest", "coverage", "pytest-cov", "expecttest",
    "nvdlfw-inspect", "onnxruntime", "onnxruntime-extensions", "onnxscript", "onnx-ir",
    "flash-attn", "apex",
):
    print(f"{package}=={metadata.version(package)}")
# Fail before running tests if a runner reused the previous image tag.
expected = {
    "torch": "2.8.0+metax3.7.2.0",
    "triton": "3.0.0+metax3.7.2.0",
    "flash-attn": "2.6.3+metax3.7.2.0torch2.8",
    "apex": "0.1+metax3.7.2.0",
    "onnxscript": "0.7.2",
    "onnx-ir": "1.1.0",
}
for package, version in expected.items():
    actual = metadata.version(package)
    if actual != version:
        raise RuntimeError(f"Unexpected MetaX image dependency: {package}=={actual}; expected {version}")
CHECK

echo "===== Install checked-out TE-FL Python/plugin layer ====="
# Run outside the source tree so generated egg-info cannot shadow the image's
# installed distribution during uninstall. The isolated vendor TE is retained.
(cd /tmp && PYTHONPATH= python3 -m pip uninstall -y transformer_engine)
cd "$WORKSPACE"
TE_FL_SKIP_CUDA=1 NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1 \
    python3 -m pip install --no-deps --no-build-isolation .

echo "===== Verify MetaX runtime ====="
python3 - <<'CHECK'
import torch
import sys
from transformer_engine.plugin.core.backends.vendor.metax.metax import MetaxBackend
import transformer_engine_torch

print("torch:", torch.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("MetaX GPU runtime is not available")
if not MetaxBackend().is_available():
    raise RuntimeError("TransformerEngine MetaX native backend is not available")
from transformer_engine_metax.pytorch.attention.dot_product_attention.backends import FlashAttention
assert sys.modules["transformer_engine_torch"] is transformer_engine_torch
print("MetaX attention:", FlashAttention.__module__)
print("MetaX device:", torch.cuda.get_device_name())
CHECK
python3 tests/pytorch/test_sanity_import.py

echo "===== MetaX environment setup complete ====="
