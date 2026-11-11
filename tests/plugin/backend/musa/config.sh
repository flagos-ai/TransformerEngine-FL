#!/usr/bin/env bash
# MUSA Backend Test Configuration

set -euo pipefail

readonly PLATFORM_NAME="musa"
readonly PLATFORM_DISPLAY_NAME="MooreThreads MUSA"
readonly PLATFORM_TYPE="shell_launcher"
readonly PLATFORM_NPROC_PER_NODE=8
readonly PLATFORM_DEVICE_ENV_VAR="MTHREADS_VISIBLE_DEVICES"
readonly PLATFORM_UNIT_TIMEOUT=7200
readonly PLATFORM_INTEGRATION_TIMEOUT=1800
readonly PLATFORM_UNIT_TEST_PATHS=(
    "tests/plugin/backend/musa"
    "tests/plugin/backend/reference"
    "tests/plugin/backend/flagos"
)
readonly PLATFORM_INTEGRATION_TEST_SCRIPT="tests/plugin/backend/musa/integration_test.sh"
readonly PLATFORM_PATCH_SCRIPT="tests/plugin/backend/musa/patch_megatron_mccl.py"
readonly PLATFORM_PYTEST_UNIT_MARKERS="not slow and not integration"
readonly PLATFORM_PYTEST_INTEGRATION_MARKERS="integration"
readonly PLATFORM_PYTEST_EXTRA_ARGS="--tb=short --verbose"
readonly PLATFORM_PYTEST_DESELECT=""
readonly PLATFORM_COVERAGE_ENABLED=true
