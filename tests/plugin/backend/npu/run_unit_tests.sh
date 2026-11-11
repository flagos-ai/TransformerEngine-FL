#!/usr/bin/env bash
# Ascend NPU Backend Unit Tests Entry Point

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"

if [ "$#" -gt 0 ]; then
    targets=("$@")
else
    targets=("${PLATFORM_UNIT_TEST_PATHS[@]}")
fi

cd "$TE_PATH"
exec python3 "$TE_PATH/$PLATFORM_PYTEST_WRAPPER" "${targets[@]}"
