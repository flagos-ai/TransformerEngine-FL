#!/usr/bin/env bash
set -euo pipefail

ILUVATAR_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$ILUVATAR_SCRIPT_DIR/../../../.." && pwd)"
export TE_PATH="${TE_PATH:-$REPO_ROOT}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export XML_LOG_DIR="${XML_LOG_DIR:-${RUNNER_TEMP:-$TE_PATH/logs}/iluvatar}"
export TE_LIB_PATH="${TE_LIB_PATH:-$($PYTHON_BIN -c 'import site; print(site.getsitepackages()[0])')/transformer_engine}"
export PLATFORM=iluvatar
export IX_HOME="${IX_HOME:-/usr/local/corex}"
export CUDA_HOME="$IX_HOME"
export PATH="$IX_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$IX_HOME/lib:$IX_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$TE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export TE_FL_SKIP_CUDA=1
export NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1
export NVTE_FRAMEWORK=pytorch
export NVTE_WITH_NCCL_EP=0
export TE_FL_PREFER="${TE_FL_PREFER:-flagos}"
# Keep the FlagOS baseline, with explicit reference implementations for the
# GEMM compatibility cases and the unscale/L2 norm op missing from IXTE.
export TE_FL_PER_OP="${TE_FL_PER_OP-generic_gemm=reference;multi_tensor_unscale_l2norm=reference}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# OpManager.call uses 0 to propagate backend exceptions without runtime fallback.
export TE_FL_STRICT=0
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=1
export PYTHONDONTWRITEBYTECODE=1
export ILUVATAR_NPROC_PER_NODE="${PLATFORM_NPROC_PER_NODE:-8}"
export NVTE_TEST_CHECKPOINT_ARTIFACT_PATH="${NVTE_TEST_CHECKPOINT_ARTIFACT_PATH:-$TE_PATH/artifacts/tests/pytorch/test_checkpoint}"
