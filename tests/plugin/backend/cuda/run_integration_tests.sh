#!/usr/bin/env bash
# CUDA Backend Integration Tests Entry Point

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"

echo "Running ${PLATFORM_DISPLAY_NAME} integration tests"
exec bash "$TE_PATH/$PLATFORM_INTEGRATION_TEST_SCRIPT"
