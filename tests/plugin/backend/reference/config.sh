#!/usr/bin/env bash
# Reference Backend Test Configuration

set -euo pipefail

readonly PLATFORM_NAME="reference"
readonly PLATFORM_DISPLAY_NAME="Reference"
readonly PLATFORM_TYPE="python_native"
readonly PLATFORM_NPROC_PER_NODE=1
readonly PLATFORM_DEVICE_ENV_VAR=""
readonly PLATFORM_UNIT_TIMEOUT=300
readonly PLATFORM_INTEGRATION_TIMEOUT=600
readonly PLATFORM_PYTEST_PATH="tests/plugin/backend/reference"
readonly PLATFORM_UNIT_TEST_PATHS=("$PLATFORM_PYTEST_PATH")
readonly PLATFORM_INTEGRATION_TEST_SCRIPT=""
readonly PLATFORM_PYTEST_UNIT_MARKERS="not slow and not integration"
readonly PLATFORM_PYTEST_INTEGRATION_MARKERS="integration"
readonly PLATFORM_PYTEST_EXTRA_ARGS="--tb=short --verbose"
readonly PLATFORM_PYTEST_DESELECT=""
readonly PLATFORM_COVERAGE_ENABLED=true
