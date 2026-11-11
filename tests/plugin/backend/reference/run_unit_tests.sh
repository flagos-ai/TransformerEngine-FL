#!/usr/bin/env bash
# Reference Backend Unit Tests Entry Point

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"

echo "${PLATFORM_DISPLAY_NAME} is a python_native collection at ${PLATFORM_PYTEST_PATH}."
echo "Its tests are collected by the active hardware platform launcher."
exit 0
