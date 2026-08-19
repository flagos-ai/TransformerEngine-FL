#!/usr/bin/env bash

set -u

: "${TE_PATH:=${GITHUB_WORKSPACE:-$(pwd)}}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L1_pytorch_distributed_unittest-ascend}"
mkdir -p "$XML_LOG_DIR"

FAIL=0

test_fail() {
    FAIL=1
    echo "Error: sub-test failed: $1"
}

pytest_command() {
    local use_platform_runner=$1
    local -n out=$2

    if [ "$use_platform_runner" = "true" ] && [ -n "${TE_TEST_PYTEST_COMMAND:-}" ]; then
        # shellcheck disable=SC2206
        out=(${TE_TEST_PYTEST_COMMAND})
    else
        out=(python3 -m pytest)
    fi
}

run_pytest_step() {
    local label=$1
    local junit=$2
    local use_platform_runner=$3
    shift 3

    local cmd=()
    pytest_command "$use_platform_runner" cmd
    cmd+=(-v -s --tb=short "--junitxml=$XML_LOG_DIR/$junit")
    cmd+=("$@")

    echo "-------------------------------------------------------"
    echo "[RUN] Executing: $label"
    "${cmd[@]}" || test_fail "$label"
}

if python3 - <<'PY'
import importlib.util

required = ("torch", "transformer_engine")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("Skipping context parallel utilities; missing modules: " + ", ".join(missing))
    raise SystemExit(1)
PY
then
    run_pytest_step "context parallel utilities" "pytest_test_cp_utils.xml" false \
        "$TE_PATH/tests/pytorch/attention/test_cp_utils.py"
fi

if [ -n "${TE_TEST_PYTEST_COMMAND:-}" ]; then
    NVTE_FLASH_ATTN=0 \
    NVTE_FUSED_ATTN=0 \
    NVTE_UNFUSED_ATTN=1 \
        run_pytest_step "distributed non-FP8 numerics" "pytest_distributed_numerics_none.xml" true \
            "$TE_PATH/tests/pytorch/distributed/test_numerics.py::test_ascend_distributed_smoke"
else
    echo "-------------------------------------------------------"
    echo "[SKIP] distributed non-FP8 numerics: Ascend shared PyTorch tests require the NPU pytest runner"
fi

echo "Skipping Ascend HCCL communication tests."

if [ "$FAIL" -ne 0 ]; then
    echo "Some tests failed."
    exit 1
fi
