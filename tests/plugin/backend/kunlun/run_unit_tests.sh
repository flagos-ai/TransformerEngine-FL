#!/usr/bin/env bash
# KunlunXin backend test entrypoint.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"

TE_PATH="${TE_PATH:-/opt/transformerengine}"
XML_LOG_DIR="${XML_LOG_DIR:-/logs}"
mkdir -p "$XML_LOG_DIR"
XML_LOG_ROOT="$XML_LOG_DIR"

FAIL=0
OVERALL_FAIL=0

run_suite() {
    local script
    case "$1" in
        debug | pytorch_debug) script="qa/L0_pytorch_debug_unittest/test.sh" ;;
        unittest | pytorch_unittest) script="qa/L0_pytorch_unittest/test.sh" ;;
        distributed | pytorch_distributed_unittest) script="qa/L1_pytorch_distributed_unittest/test.sh" ;;
        onnx | pytorch_onnx_unittest) script="qa/L1_pytorch_onnx_unittest/test.sh" ;;
        -h | --help)
            echo "Usage: $0 [debug] [unittest] [distributed] [onnx]"
            echo "If no suite is specified, all suites are run serially."
            exit 0
            ;;
        *)
            echo "Error: unsupported Kunlun test suite: $1" >&2
            exit 2
            ;;
    esac

    echo "-------------------------------------------------------"
    echo "[RUN] Executing: $script"
    bash "$TE_PATH/$script" || FAIL=1
}

if [ "$#" -eq 0 ]; then
    set -- debug unittest distributed onnx
fi

for suite in "$@"; do
    FAIL=0
    export XML_LOG_DIR="$XML_LOG_ROOT/$suite"
    mkdir -p "$XML_LOG_DIR"
    echo "===== START $suite ====="
    run_suite "$suite"
    echo "===== END $suite rc=$FAIL ====="
    if [ "$FAIL" -ne 0 ]; then
        OVERALL_FAIL=1
    fi
done

if [ "$OVERALL_FAIL" -ne 0 ]; then
    exit 1
fi

exit 0
