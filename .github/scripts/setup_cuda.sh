#!/usr/bin/env bash
# CUDA Platform Environment Setup Script
# Called by unit_tests_common.yml for CUDA platforms (A100, H100, etc.)
set -euo pipefail

export TE_FL_SKIP_CUDA="${TE_FL_SKIP_CUDA:-0}"
export SKIP_CUDA_BUILD="${SKIP_CUDA_BUILD:-0}"
export NVTE_WITH_CUDA="${NVTE_WITH_CUDA:-1}"
export NVTE_WITH_MACA="${NVTE_WITH_MACA:-0}"
export TE_WITH_NCCL="${TE_WITH_NCCL:-1}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-80;90}"

echo "===== Step 0: Activate Python environment ====="
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate flagscale-train
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
{
    echo "TE_FL_SKIP_CUDA=$TE_FL_SKIP_CUDA"
    echo "SKIP_CUDA_BUILD=$SKIP_CUDA_BUILD"
    echo "NVTE_WITH_CUDA=$NVTE_WITH_CUDA"
    echo "NVTE_WITH_MACA=$NVTE_WITH_MACA"
    echo "TE_WITH_NCCL=$TE_WITH_NCCL"
    echo "NVTE_FRAMEWORK=$NVTE_FRAMEWORK"
    echo "CUDA_HOME=$CUDA_HOME"
    echo "NVCC=$NVCC"
    echo "NVTE_CUDA_ARCHS=$NVTE_CUDA_ARCHS"
    echo "PATH=$PATH"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
} >> "$GITHUB_ENV"
echo "Python: $(which python3) ($(python3 --version 2>&1))"

echo "===== Step 1: Remove Existing TransformerEngine ====="
pip uninstall transformer_engine transformer_engine_torch -y || true

echo "===== Step 2: Install TransformerEngine ====="
cd "$GITHUB_WORKSPACE"

pip install nvdlfw-inspect --quiet
pip install expecttest --quiet

if [ -n "${TE_CI_ARTIFACT_DIR:-}" ]; then
    if [[ "$TE_CI_ARTIFACT_DIR" != /* ]]; then
        TE_CI_ARTIFACT_DIR="$GITHUB_WORKSPACE/$TE_CI_ARTIFACT_DIR"
    fi
    shopt -s nullglob
    wheels=("$TE_CI_ARTIFACT_DIR"/*.whl)
    shopt -u nullglob
    if [ "${#wheels[@]}" -ne 1 ]; then
        echo "Expected exactly one CI wheel in $TE_CI_ARTIFACT_DIR, found ${#wheels[@]}" >&2
        exit 1
    fi
    echo "Installing prebuilt CI wheel: ${wheels[0]}"
    pip install "${wheels[0]}" --force-reinstall --no-deps
else
    echo "No prebuilt CI wheel was provided; building from source"
    git submodule update --init --recursive
    pip install . -v --no-deps --no-build-isolation
fi

echo "===== Step 3: Verify Installation ====="
python3 tests/pytorch/test_sanity_import.py

echo "===== Environment Setup Complete ====="
