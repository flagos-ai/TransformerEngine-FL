#!/usr/bin/env bash

# Enflame/GCU workflow configuration.
# Keep chip-specific skips and runner knobs here instead of common workflows.

ENFLAME_CONFIG_FILE="${ENFLAME_CONFIG_FILE:-${GITHUB_WORKSPACE:-$(pwd)}/.github/configs/enflame.yml}"
if [ -z "${ENFLAME_NPROC_PER_NODE:-}" ]; then
    ENFLAME_NPROC_PER_NODE="$(
        # Use the stdlib-only interpreter so vendor .pth bootstrap hooks cannot
        # write diagnostics into the command-substitution result.
        ENFLAME_CONFIG_FILE="$ENFLAME_CONFIG_FILE" python3 -S - <<'PY'
import os
import re
from pathlib import Path

config_file = Path(os.environ["ENFLAME_CONFIG_FILE"])
match = re.search(r"^\s*nproc_per_node\s*:\s*(\d+)\s*$", config_file.read_text(), re.MULTILINE)
print(match.group(1) if match else 2)
PY
    )"
fi
export ENFLAME_NPROC_PER_NODE

ENFLAME_UNITTEST_SKIP_FUSED_OPTIMIZER=(
    "test_float"
    "test_half"
    "test_grad_scaler_capturable"
    "test_grad_scaler_capturable_master"
    # This case exhausts/overruns the S60 GCU kernel loader and can crash the
    # vendor driver (loadSegments failed in alloc mem), rather than reporting
    # a test failure that Transformer Engine can handle.
    "test_multi_params"
)

ENFLAME_UNITTEST_SKIP_HF_INTEGRATION=(
    "test_save_and_load_hf_model"
)

ENFLAME_DISTRIBUTED_SKIP_FILES=(
    "tests/pytorch/distributed/test_numerics.py"
    "tests/pytorch/distributed/test_numerics_exact.py"
    "tests/pytorch/distributed/test_torch_fsdp2.py"
)

ENFLAME_ONNX_SKIP_GROUPS=(
    "test_export_linear"
    "test_export_layernorm_linear"
    "test_export_layernorm_mlp"
    "test_export_core_attention"
    "test_export_transformer_layer"
    "test_export_multihead_attention"
    "test_export_gpt_generation"
)
