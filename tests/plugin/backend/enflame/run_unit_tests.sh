#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"

XML_LOG_ROOT="$XML_LOG_DIR"
FAIL=0
OVERALL_FAIL=0
FAILED_CASES=()

usage() {
    cat <<'EOF'
Usage: tests/plugin/backend/enflame/run_unit_tests.sh [debug] [unittest] [distributed] [onnx]

Runs the selected Enflame/GCU test group.
If no suite is specified, all suites are run.
EOF
}

# The launcher prepares the Enflame environment and delegates test scope to QA.
run_suite() {
    local suite=$1 script
    case "$suite" in
        debug) script="qa/L0_pytorch_debug_unittest/test.sh" ;;
        unittest)
            script="qa/L0_pytorch_unittest/test.sh"
            local -x QA_PYTHON_BIN="$PYTHON_BIN"
            ;;
        distributed) script="qa/L1_pytorch_distributed_unittest/test.sh" ;;
        onnx) script="qa/L1_pytorch_onnx_unittest/test.sh" ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown suite: $suite" >&2
            usage >&2
            exit 2
            ;;
    esac

    echo "===== START $suite ====="
    echo "-------------------------------------------------------"
    echo "[RUN][$suite] $script"
    echo "-------------------------------------------------------"
    if ! bash "$TE_PATH/$script"; then
        FAIL=1
        FAILED_CASES+=("$suite:$script")
        echo "Error: sub-test failed: $suite:$script"
    fi
    echo "===== END $suite rc=$FAIL ====="
}

if [ "$#" -eq 0 ]; then
    set -- debug unittest distributed onnx
fi

for suite in "$@"; do
    FAIL=0
    export XML_LOG_DIR="$XML_LOG_ROOT/$suite"
    mkdir -p "$XML_LOG_DIR"
    run_suite "$suite"
    if [ "$FAIL" -ne 0 ]; then
        OVERALL_FAIL=1
    fi
done

if [ "$OVERALL_FAIL" -ne 0 ]; then
    echo "Error in the following test cases: ${FAILED_CASES[*]}"
    exit 1
fi

echo "All selected Enflame/GCU test suites passed."
