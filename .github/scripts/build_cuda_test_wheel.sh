#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
: "${TE_CI_ARTIFACT_DIR:?TE_CI_ARTIFACT_DIR is required}"

source /flagos/bin/activate

export TE_FL_SKIP_CUDA=0
export SKIP_CUDA_BUILD=0
export NVTE_WITH_CUDA=1
export NVTE_WITH_MACA=0
export TE_WITH_NCCL=1
export NVTE_WITH_NCCL_EP="${NVTE_WITH_NCCL_EP:-0}"
export NVTE_FRAMEWORK=pytorch
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-80;90}"
export MAX_JOBS="${MAX_JOBS:-48}"
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

cd "$GITHUB_WORKSPACE"
git submodule update --init --recursive
mkdir -p "$TE_CI_ARTIFACT_DIR"
find "$TE_CI_ARTIFACT_DIR" -maxdepth 1 -type f -name '*.whl' -delete

echo "Building CUDA test wheel with:"
echo "  NVTE_CUDA_ARCHS=$NVTE_CUDA_ARCHS"
echo "  NVTE_WITH_NCCL_EP=$NVTE_WITH_NCCL_EP"
echo "  MAX_JOBS=$MAX_JOBS"
pip wheel . \
    --wheel-dir "$TE_CI_ARTIFACT_DIR" \
    --no-deps \
    --no-build-isolation \
    -v

shopt -s nullglob
wheels=("$TE_CI_ARTIFACT_DIR"/*.whl)
shopt -u nullglob
if [ "${#wheels[@]}" -ne 1 ]; then
    echo "Expected exactly one built wheel, found ${#wheels[@]}" >&2
    exit 1
fi

echo "Built CI wheel: ${wheels[0]}"
