#!/usr/bin/env bash
# CUDA Backend Integration Tests Entry Point

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"

# Diagnostics off, as on every other platform's launcher (Ascend defaults to
# off in qa/L1_pytorch_mcore_integration/test.sh). The shared script defaults
# ENABLE_DIAGNOSTICS to 1 for non-Ascend, which adds --log-params-norm.
# Turn it back on for all platforms together once this is fixed:
# --log-params-norm makes Megatron's calc_params_l2_norm
# (megatron/training/utils/common_utils.py:158, norm_2 += sharded_norm_2) add
# its own torch.zeros((1,)) in place to the squared multi_tensor_l2norm result.
# The native kernel returns that result with shape [1]
# (transformer_engine/pytorch/csrc/extensions/multi_tensor/l2norm.cpp:21), but
# the FlagOS and reference implementations return a 0-dim scalar, so the job
# died at iteration 1 with both (runs 43899242175 and 44108177180):
#   RuntimeError: output with shape [] doesn't match the broadcast shape [1]
# The grad-norm clip calls the same op without that in-place add, so training
# itself is unaffected.
# Owner: FlagOS/reference multi_tensor_l2norm (return shape [1] plus a
# rank-parity test against the native kernel).
export ENABLE_DIAGNOSTICS="${ENABLE_DIAGNOSTICS:-0}"

echo "Running ${PLATFORM_DISPLAY_NAME} integration tests"
exec bash "$TE_PATH/$PLATFORM_INTEGRATION_TEST_SCRIPT"
