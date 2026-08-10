#!/usr/bin/env bash

set -u

: "${TE_PATH:=${GITHUB_WORKSPACE:-$(pwd)}}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L1_pytorch_onnx_unittest-ascend}"
mkdir -p "$XML_LOG_DIR"

FAIL=0

test_fail() {
    FAIL=1
    echo "Error: sub-test failed: $1"
}

pytest_command() {
    local -n out=$1

    if [ -n "${TE_TEST_PYTEST_COMMAND:-}" ]; then
        # shellcheck disable=SC2206
        out=(${TE_TEST_PYTEST_COMMAND})
    else
        out=(python3 -m pytest)
    fi
}

require_modules() {
    python3 - "$@" <<'PY'
import importlib
import sys

missing = []
for module_name in sys.argv[1:]:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        missing.append(module_name)

if missing:
    print("missing modules: " + ", ".join(missing))
    raise SystemExit(1)
PY
}

run_pytest_step() {
    local label=$1
    local junit=$2
    shift 2

    local cmd=()
    pytest_command cmd
    cmd+=(-v -s --tb=auto "--junitxml=$XML_LOG_DIR/$junit")
    cmd+=("$@")

    echo "-------------------------------------------------------"
    echo "[RUN] Executing: $label"
    "${cmd[@]}" || test_fail "$label"
}

if ! require_modules onnxruntime onnxruntime_extensions; then
    test_fail "ONNX export tests"
elif [ -z "${TE_TEST_PYTEST_COMMAND:-}" ]; then
    NVTE_FLASH_ATTN=0 \
    NVTE_FUSED_ATTN=0 \
    NVTE_UNFUSED_ATTN=1 \
    NVTE_UnfusedDPA_Emulate_FP8=1 \
        run_pytest_step "ONNX export tests that do not require the NPU pytest runner" \
            "test_onnx_export.xml" \
            "$TE_PATH/tests/pytorch/test_onnx_export.py::test_export_ctx_manager" \
            "$TE_PATH/tests/pytorch/test_onnx_export.py::test_export_layernorm_zero_centered_gamma"
else
    NVTE_FLASH_ATTN=0 \
    NVTE_FUSED_ATTN=0 \
    NVTE_UNFUSED_ATTN=1 \
    NVTE_UnfusedDPA_Emulate_FP8=1 \
        run_pytest_step "ONNX export tests" "test_onnx_export.xml" \
            "$TE_PATH/tests/pytorch/test_onnx_export.py"
fi

if [ "$FAIL" -ne 0 ]; then
    echo "Some tests failed."
    exit 1
fi
