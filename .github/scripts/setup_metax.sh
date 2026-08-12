#!/usr/bin/env bash
# Metax Platform Environment Setup Script
# Called by unit_tests_common.yml for Metax platforms (C500, etc.)
set -euo pipefail

export TE_FL_SKIP_CUDA="${TE_FL_SKIP_CUDA:-1}"
export NVTE_WITH_MACA="${NVTE_WITH_MACA:-1}"
export CUDA_HOME="${CUDA_HOME:-/opt/maca}"
export MACA_HOME="${MACA_HOME:-/opt/maca}"

echo "===== Step 0: Activate Python environment ====="
source /opt/conda/etc/profile.d/conda.sh
conda activate base
echo "Python: $(which python3) ($(python3 --version 2>&1))"

echo "===== Step 1: Base Environment Setup ====="
# Configure MACA toolchain paths
export PATH="${MACA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${MACA_HOME}/lib:${LD_LIBRARY_PATH:-}"
{
    echo "TE_FL_SKIP_CUDA=$TE_FL_SKIP_CUDA"
    echo "NVTE_WITH_MACA=$NVTE_WITH_MACA"
    echo "CUDA_HOME=$CUDA_HOME"
    echo "MACA_HOME=$MACA_HOME"
    echo "PATH=$PATH"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
} >> "$GITHUB_ENV"
service ssh restart

echo "===== Step 2: Create nvcc Symlink (cucc -> nvcc) ====="
# TransformerEngine expects nvcc, but MACA provides cucc
ln -sf /opt/maca/tools/cu-bridge/bin/cucc /opt/maca/tools/cu-bridge/bin/nvcc
which nvcc || true

echo "===== Step 3: Install Required System Tools ====="
# Use apt to install git, curl
sed -i 's|http://mirrors.aliyun.com/ubuntu|http://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list
apt-get update -qq || true
apt-get install -y -qq git curl
# Install cmake and ninja via pip (more reliable than apt in this env)
python3 -m pip install cmake ninja torch --no-cache-dir

echo "===== Step 4: Remove Existing TransformerEngine ====="
# Prevent conflicts with preinstalled or incompatible versions
python3 -m pip uninstall transformer_engine -y || true
python3 -m pip install nvdlfw-inspect --no-deps || true

echo "===== Step 5: Install TE-FL Plugin Layer ====="
# Install TransformerEngine-FL Python layer (plugin logic)
cd $GITHUB_WORKSPACE
git submodule update --init --recursive
TE_FL_SKIP_CUDA=1 python3 setup.py install

echo "===== Step 6: Final Verification ====="
python3 tests/pytorch/test_sanity_import.py

echo "===== Environment Setup Complete ====="
