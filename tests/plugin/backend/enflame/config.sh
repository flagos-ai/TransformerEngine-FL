#!/usr/bin/env bash
# Enflame Backend Test Configuration

set -euo pipefail

readonly PLATFORM_NAME="enflame"
readonly PLATFORM_DISPLAY_NAME="Enflame GCU"
readonly PLATFORM_TYPE="shell_launcher"
readonly PLATFORM_NPROC_PER_NODE=2
readonly PLATFORM_DEVICE_ENV_VAR="GCU_VISIBLE_DEVICES"
readonly PLATFORM_UNIT_TIMEOUT=14400
readonly PLATFORM_INTEGRATION_TIMEOUT=1800
readonly PLATFORM_UNIT_TEST_PATHS=(
    "tests/plugin/backend/enflame"
    "tests/plugin/backend/reference"
    "tests/plugin/backend/flagos"
)
readonly PLATFORM_INTEGRATION_TEST_SCRIPT="qa/L1_pytorch_mcore_integration/test.sh"
readonly PLATFORM_PYTEST_UNIT_MARKERS="not slow and not integration"
readonly PLATFORM_PYTEST_INTEGRATION_MARKERS="integration"
readonly PLATFORM_PYTEST_EXTRA_ARGS="--tb=short --verbose"
readonly PLATFORM_PYTEST_DESELECT=""
readonly PLATFORM_COVERAGE_ENABLED=true

readonly -a ENFLAME_UNITTEST_SKIP_FUSED_OPTIMIZER=(
    "test_float"
    "test_half"
    "test_grad_scaler_capturable"
    "test_grad_scaler_capturable_master"
)

readonly -a ENFLAME_UNITTEST_SKIP_HF_INTEGRATION=(
    "test_save_and_load_hf_model"
)

readonly -a ENFLAME_DISTRIBUTED_SKIP_FILES=(
    "tests/pytorch/distributed/test_numerics.py"
    "tests/pytorch/distributed/test_numerics_exact.py"
    "tests/pytorch/distributed/test_torch_fsdp2.py"
)

readonly -a ENFLAME_ONNX_SKIP_GROUPS=(
    "test_export_linear"
    "test_export_layernorm_linear"
    "test_export_layernorm_mlp"
    "test_export_core_attention"
    "test_export_transformer_layer"
    "test_export_multihead_attention"
    "test_export_gpt_generation"
)
