#!/usr/bin/env bash
# Hygon Backend Test Configuration

set -euo pipefail

readonly PLATFORM_NAME="hygon"
readonly PLATFORM_DISPLAY_NAME="Hygon DCU"
readonly PLATFORM_TYPE="shell_launcher"
readonly PLATFORM_NPROC_PER_NODE=8
readonly PLATFORM_DEVICE_ENV_VAR="CUDA_VISIBLE_DEVICES"
readonly PLATFORM_UNIT_TIMEOUT=14400
readonly PLATFORM_INTEGRATION_TIMEOUT=1800
readonly PLATFORM_UNIT_TEST_PATHS=(
    "tests/plugin/backend/hygon"
    "tests/plugin/backend/reference"
    "tests/plugin/backend/flagos"
)
readonly PLATFORM_INTEGRATION_TEST_SCRIPT="tests/plugin/backend/hygon/integration_test.sh"
readonly PLATFORM_PYTEST_UNIT_MARKERS="not slow and not integration"
readonly PLATFORM_PYTEST_INTEGRATION_MARKERS="integration"
readonly PLATFORM_PYTEST_EXTRA_ARGS="--tb=short --verbose"
readonly PLATFORM_PYTEST_DESELECT=""
readonly PLATFORM_COVERAGE_ENABLED=true

readonly -a HYGON_ONNX_SKIP_GROUPS=(
    "test_export_linear"
    "test_export_layernorm_linear"
    "test_export_layernorm_mlp"
    "test_export_core_attention"
    "test_export_transformer_layer"
    "test_export_multihead_attention"
    "test_export_gpt_generation"
)
