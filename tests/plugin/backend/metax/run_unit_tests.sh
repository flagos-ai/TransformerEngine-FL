#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TE_PATH="${TE_PATH:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd)}"
XML_LOG_ROOT="${XML_LOG_DIR:-${RUNNER_TEMP:-$TE_PATH/logs}/metax}"

FAILED=0
FAILED_SUITES=()

usage() {
    echo "Usage: $0 [debug] [unittest] [distributed] [onnx]"
    echo "If no suite is specified, all suites are run serially."
}

run_suite() {
    local suite=$1
    local script
    local suite_status=0
    case "$suite" in
        debug) script="qa/L0_pytorch_debug_unittest/test.sh" ;;
        unittest) script="qa/L0_pytorch_unittest/test.sh" ;;
        distributed) script="qa/L1_pytorch_distributed_unittest/test.sh" ;;
        onnx) script="qa/L1_pytorch_onnx_unittest/test.sh" ;;
        -h | --help) usage; exit 0 ;;
        *)
            echo "Unsupported MetaX test suite: $suite" >&2
            usage >&2
            exit 2
            ;;
    esac

    local suite_log_dir="$XML_LOG_ROOT/$suite"
    mkdir -p "$suite_log_dir"
    echo "===== START $suite ====="
    if ! XML_LOG_DIR="$suite_log_dir" bash "$TE_PATH/$script"; then
        suite_status=1
        FAILED=1
        FAILED_SUITES+=("$suite")
    fi
    echo "===== END $suite rc=$suite_status ====="
}

if [ "$#" -eq 0 ]; then
    set -- debug unittest distributed onnx
fi

for suite in "$@"; do
    run_suite "$suite"
done

if [ "$FAILED" -ne 0 ]; then
    echo "Failed MetaX test suites: ${FAILED_SUITES[*]}" >&2
    exit 1
fi

echo "All selected MetaX test suites passed."
