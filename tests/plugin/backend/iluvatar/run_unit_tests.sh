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

run_pytest() {
    local name=$1
    local target=$2
    shift 2
    if [ "$target" = tests/plugin ]; then
        local plugin_target plugin_name plugin_targets
        plugin_targets=$(TE_TEST_GROUP_PRESET=plugin-unit "$PYTHON_BIN" "$TE_PATH/tests/test_utils/run_ci_test_group.py" --list-targets) || { FAILED=1; return; }
        while IFS= read -r plugin_target; do
            [ -n "$plugin_target" ] || continue
            plugin_name="${plugin_target//\//_}"
            run_pytest "${name}_${plugin_name%.py}" "$plugin_target" "$@"
        done <<< "$plugin_targets"
        return
    fi
    echo "[RUN][$suite] $target"
    if ! TE_TEST_GROUP_PRESET=plugin-unit "$PYTHON_BIN" "$TE_PATH/tests/test_utils/run_ci_test_group.py" \
        --target-env "$target" "$PYTHON_BIN" -m pytest -v -s --tb=short \
        "$TE_PATH/$target" "$@" --junitxml="$XML_LOG_DIR/$name.xml"
    then
        FAILED=1
        FAILED_TESTS+=("$suite:$name")
    fi
}

# debug, distributed and onnx run the shared qa scripts like every other
# platform; Iluvatar differences live only in .github/configs/iluvatar.yml.
run_shared() {
    echo "[RUN][$suite] $1"
    if ! XML_LOG_DIR="$XML_LOG_DIR" bash "$TE_PATH/$1"; then
        FAILED=1
        FAILED_TESTS+=("$suite:$1")
    fi
}

run_suite() {
    local target
    local index=0
    case "$suite" in
        debug)
            run_shared qa/L0_pytorch_debug_unittest/test.sh
            ;;
        unittest)
            for target in "${ILUVATAR_UNIT_TEST_TARGETS[@]}"; do
                index=$((index + 1))
                run_pytest "unit_$index" "$target"
            done
            ;;
        distributed)
            "$PYTHON_BIN" - <<'CHECK'
import os
import torch
required = int(os.environ["ILUVATAR_NPROC_PER_NODE"])
assert torch.cuda.device_count() >= required, f"Distributed QA needs {required} visible devices"
assert torch.distributed.is_nccl_available(), "Vendor NCCL-compatible backend is required"
CHECK
            run_shared qa/L1_pytorch_distributed_unittest/test.sh
            ;;
        onnx)
            run_shared qa/L1_pytorch_onnx_unittest/test.sh
            ;;
    esac
}

if [ "$#" -eq 0 ]; then
    set -- debug unittest distributed onnx
fi
for suite in "$@"; do
    export XML_LOG_DIR="$XML_LOG_ROOT/$suite"
    mkdir -p "$XML_LOG_DIR"
    echo "===== START $suite ====="
    run_suite
    echo "===== END $suite ====="
done
if [ "$FAILED" -ne 0 ]; then
    echo "Failed Iluvatar tests: ${FAILED_TESTS[*]}" >&2
    exit 1
fi
echo "All selected Iluvatar test suites passed."
