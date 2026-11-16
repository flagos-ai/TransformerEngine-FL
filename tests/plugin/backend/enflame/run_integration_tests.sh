#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"

export ENFLAME_FORCE_MEGATRON_PLATFORM="${ENFLAME_FORCE_MEGATRON_PLATFORM:-1}"
# This job tests the reference backend explicitly. Under the shared flagos order
# the training loss turns NaN at iteration 3 (run 43804752380), while the same
# image trained normally with every op on reference.torch (run 43759073663).
# Candidates still unresolved: generic_gemm, multi_tensor_adam/l2norm/
# scale_tensor, scaled_masked_softmax_forward/backward (owner: FlagGems/FlagTree
# Enflame). The job env already carries TE_FL_PREFER=flagos, so override it here.
# Reproduce the FlagOS path with ENFLAME_MCORE_PREFER=flagos; return to the
# shared order once that run finishes without NaN.
export TE_FL_PREFER="${ENFLAME_MCORE_PREFER:-reference}"
echo "Enflame MCore integration: TE_FL_PREFER=$TE_FL_PREFER"
export ENABLE_DIAGNOSTICS="${ENABLE_DIAGNOSTICS:-0}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export NVTE_TORCH_COMPILE="${NVTE_TORCH_COMPILE:-0}"

exec bash "$REPO_ROOT/$PLATFORM_INTEGRATION_TEST_SCRIPT"
