#!/usr/bin/env bash
# Ascend NPU Backend Integration Tests Entry Point

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"

echo "${PLATFORM_DISPLAY_NAME} has no independent integration test entry."
exit 0
