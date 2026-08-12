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
    echo "PATH=$PATH"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
} >> "$GITHUB_ENV"
echo "Python: $(which python3) ($(python3 --version 2>&1))"

echo "===== Step 1: Remove Existing TransformerEngine ====="
pip uninstall transformer_engine transformer_engine_torch -y || true

echo "===== Step 2: Build & Install TransformerEngine ====="
cd $GITHUB_WORKSPACE
git submodule update --init --recursive

pip install nvdlfw-inspect --quiet
pip install expecttest --quiet
pip install . -v --no-deps --no-build-isolation

echo "===== Step 3: Verify Installation ====="
python3 tests/pytorch/test_sanity_import.py

echo "===== Environment Setup Complete ====="
