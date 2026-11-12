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

# Exact plugin-node quarantines. Node paths are relative to tests/plugin/.
# Literal entries only, one quoted node|reason per line; shared pytest collection
# reads this array for both the outer runner and fresh dispatch subprocesses.
readonly PLUGIN_IGNORE_TEST_NODES=(
    'plugin/triton_dispatch_numerics_cases.py::test_cross_entropy_forward_and_backward_match_native_exactly|cross_entropy num_warps=32 exceeds the MACA compiler limit; restore after launch adaptation'
)
