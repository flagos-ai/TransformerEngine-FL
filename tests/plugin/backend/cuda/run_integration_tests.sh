#!/usr/bin/env bash
# CUDA Backend Integration Tests Entry Point

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"

# FlagOS multi_tensor_l2norm returns a shape-[1] total norm where every other
# backend returns a 0-dim scalar, so Megatron's calc_params_l2_norm dies at
# iteration 1 on "norm_2 += sharded_norm_2":
#   megatron/training/utils/common_utils.py:158
#   RuntimeError: output with shape [] doesn't match the broadcast shape [1]
# transformer_engine/plugin/core/backends/flagos/impl/multi_tensor.py:37 builds
# the norm with flag_gems.sum, which keeps a length-1 dimension, while
# reference/impl/optimizer.py:72 uses torch.sum and reduces to 0-dim. Both
# attempts of run 43899242175 failed identically, so this is deterministic.
# Owner: FlagOS multi_tensor_l2norm. Remove this pin once total_norm is 0-dim.
# Only this op is pinned; every other op stays on the shared flagos order.
# Why no other platform sees it: qa/L1_pytorch_mcore_integration/test.sh:73
# defaults ENABLE_DIAGNOSTICS to 1 for every non-Ascend platform, which is what
# adds --log-params-norm, and all seven other launchers override it back to 0.
# A100 is the only platform that runs this diagnostic path at all, and its seven
# green siblings are not evidence that the path works. Keep diagnostics on here
# so the coverage exists somewhere; unifying the other launchers is a follow-up.
export TE_FL_PER_OP="${TE_FL_PER_OP-multi_tensor_l2norm=reference}"
echo "CUDA MCore integration: TE_FL_PER_OP=$TE_FL_PER_OP"

echo "Running ${PLATFORM_DISPLAY_NAME} integration tests"
exec bash "$TE_PATH/$PLATFORM_INTEGRATION_TEST_SCRIPT"
