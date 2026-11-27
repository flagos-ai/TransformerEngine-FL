#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"
export TE_PATH="${TE_PATH:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd)}"
XML_LOG_ROOT="${XML_LOG_DIR:-${RUNNER_TEMP:-$TE_PATH/logs}/ascend}"

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
        # All four suites run shared QA; Ascend differences live in its backend
        # environment, NPU pytest adapter and platform YAML.
        debug) script="qa/L0_pytorch_debug_unittest/test.sh" ;;
        unittest) script="qa/L0_pytorch_unittest/test.sh" ;;
        distributed) script="qa/L1_pytorch_distributed_unittest/test.sh" ;;
        onnx) script="qa/L1_pytorch_onnx_unittest/test.sh" ;;
        -h | --help) usage; exit 0 ;;
        *)
            echo "Unsupported Ascend test suite: $suite" >&2
            usage >&2
            exit 2
            ;;
    esac

    local suite_log_dir="$XML_LOG_ROOT/$suite"
    mkdir -p "$suite_log_dir"
    echo "===== START $suite ====="
    # Apply the NPU adapter before collecting shared unit tests. Plugin tests
    # retain their own per-file environment through run_ci_test_group.py.
    local pytest_command="${TE_TEST_PYTEST_COMMAND:-python3 $TE_PATH/tests/plugin/backend/npu/run_pytest.py}"
    if ! env NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=1 \
        TE_TEST_PYTEST_COMMAND="$pytest_command" XML_LOG_DIR="$suite_log_dir" \
        bash "$TE_PATH/$script"; then
        suite_status=1
        FAILED=1
        FAILED_SUITES+=("$suite")
    fi
    if [[ "$suite" == unittest ]]; then
        # This NPU-only checkpoint round trip is a standalone program, not a
        # pytest target. Keep it alongside the common QA suite.
        if ! python3 "$TE_PATH/tests/pytorch/test_grouped_tensor_checkpoint_npu.py"; then
            suite_status=1
            FAILED=1
            FAILED_SUITES+=("$suite:npu-checkpoint")
        fi
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
    echo "Failed Ascend test suites: ${FAILED_SUITES[*]}" >&2
    exit 1
fi

echo "All selected Ascend test suites passed."
