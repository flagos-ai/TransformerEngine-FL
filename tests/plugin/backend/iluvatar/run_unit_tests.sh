#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
usage() {
    echo "Usage: $0 [debug] [unittest] [distributed] [onnx]"
    echo "If no suite is specified, all suites run serially."
}
for suite in "$@"; do
    case "$suite" in
        -h|--help) usage; exit 0 ;;
        debug|unittest|distributed|onnx) ;;
        *) echo "Unsupported Iluvatar test suite: $suite" >&2; usage >&2; exit 2 ;;
    esac
done
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"
XML_LOG_ROOT="$XML_LOG_DIR"
FAILED=0
FAILED_TESTS=()

# Test scope comes from shared QA and the platform YAML selection rules.
run_suite() {
    local suite=$1 script
    case "$suite" in
        debug) script="qa/L0_pytorch_debug_unittest/test.sh" ;;
        unittest)
            script="qa/L0_pytorch_unittest/test.sh"
            local -x QA_PYTHON_BIN="$PYTHON_BIN"
            ;;
        distributed)
            "$PYTHON_BIN" - <<'CHECK'
import os
import torch
required = int(os.environ["ILUVATAR_NPROC_PER_NODE"])
assert torch.cuda.device_count() >= required, f"Distributed QA needs {required} visible devices"
assert torch.distributed.is_nccl_available(), "Vendor NCCL-compatible backend is required"
CHECK
            script="qa/L1_pytorch_distributed_unittest/test.sh"
            ;;
        onnx) script="qa/L1_pytorch_onnx_unittest/test.sh" ;;
    esac

    echo "[RUN][$suite] $script"
    if ! bash "$TE_PATH/$script"; then
        FAILED=1
        FAILED_TESTS+=("$suite:$script")
    fi
}

if [ "$#" -eq 0 ]; then
    set -- debug unittest distributed onnx
fi
for suite in "$@"; do
    export XML_LOG_DIR="$XML_LOG_ROOT/$suite"
    mkdir -p "$XML_LOG_DIR"
    echo "===== START $suite ====="
    run_suite "$suite"
    echo "===== END $suite ====="
done
if [ "$FAILED" -ne 0 ]; then
    echo "Failed Iluvatar tests: ${FAILED_TESTS[*]}" >&2
    exit 1
fi
echo "All selected Iluvatar test suites passed."
