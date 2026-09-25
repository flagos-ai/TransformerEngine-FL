#!/usr/bin/env bash
# Ascend NPU Backend Test Configuration

set -euo pipefail

readonly PLATFORM_NAME="npu"
readonly PLATFORM_DISPLAY_NAME="Ascend NPU"
readonly PLATFORM_TYPE="python_wrapper"
readonly PLATFORM_NPROC_PER_NODE=4
readonly PLATFORM_DEVICE_ENV_VAR="ASCEND_RT_VISIBLE_DEVICES"
readonly PLATFORM_UNIT_TIMEOUT=14400
readonly PLATFORM_INTEGRATION_TIMEOUT=1800
readonly PLATFORM_UNIT_TEST_PATHS=(
    "tests/plugin/backend/npu"
    "tests/plugin/backend/reference"
    "tests/plugin/backend/flagos"
)
readonly PLATFORM_INTEGRATION_TEST_SCRIPT=""
readonly PLATFORM_PYTEST_WRAPPER="tests/plugin/backend/npu/run_pytest.py"
readonly PLATFORM_PATCH_MODULE="tests/plugin/backend/npu/npu_patch.py"
readonly PLATFORM_PYTEST_UNIT_MARKERS="not slow and not integration"
readonly PLATFORM_PYTEST_INTEGRATION_MARKERS="integration"
readonly PLATFORM_PYTEST_EXTRA_ARGS="--tb=short --verbose"
readonly PLATFORM_PYTEST_DESELECT=""
readonly PLATFORM_COVERAGE_ENABLED=true
