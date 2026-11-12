#!/usr/bin/env bash
# MetaX Backend Test Configuration

set -euo pipefail

readonly PLATFORM_NAME="metax"
readonly PLATFORM_DISPLAY_NAME="MetaX"
readonly PLATFORM_TYPE="shell_launcher"
readonly PLATFORM_NPROC_PER_NODE=4
readonly PLATFORM_DEVICE_ENV_VAR="METAX_VISIBLE_DEVICES"
readonly PLATFORM_UNIT_TIMEOUT=14400
readonly PLATFORM_INTEGRATION_TIMEOUT=1800
readonly PLATFORM_UNIT_TEST_PATHS=(
    "tests/plugin/backend/metax"
    "tests/plugin/backend/reference"
    "tests/plugin/backend/flagos"
)
readonly PLATFORM_INTEGRATION_TEST_SCRIPT="qa/L1_pytorch_mcore_integration/test.sh"
readonly PLATFORM_PYTEST_UNIT_MARKERS="not slow and not integration"
readonly PLATFORM_PYTEST_INTEGRATION_MARKERS="integration"
readonly PLATFORM_PYTEST_EXTRA_ARGS="--tb=short --verbose"
readonly PLATFORM_PYTEST_DESELECT=""
readonly PLATFORM_COVERAGE_ENABLED=true
