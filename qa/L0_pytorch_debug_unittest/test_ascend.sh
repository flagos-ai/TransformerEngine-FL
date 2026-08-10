#!/usr/bin/env bash

set -u

: "${TE_PATH:=${GITHUB_WORKSPACE:-$(pwd)}}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L0_pytorch_debug_unittest-ascend}"
: "${NVTE_TEST_NVINSPECT_FEATURE_DIRS:=$TE_PATH/transformer_engine/debug/features}"
: "${NVTE_TEST_NVINSPECT_CONFIGS_DIR:=$TE_PATH/tests/pytorch/debug/test_configs/}"
mkdir -p "$XML_LOG_DIR"

export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

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

run_pytest_step() {
    local label=$1
    local junit=$2
    shift 2

    local cmd=()
    pytest_command cmd
    cmd+=(-v -s "--junitxml=$XML_LOG_DIR/$junit")
    cmd+=("$@")

    echo "-------------------------------------------------------"
    echo "[RUN] Executing: $label"
    "${cmd[@]}" || test_fail "$label"
}

if [ -z "${TE_TEST_PYTEST_COMMAND:-}" ]; then
    echo "Running Ascend PyTorch debug tests that do not require the NPU pytest runner."
    run_pytest_step "debug config" "test_config.xml" \
        "$TE_PATH/tests/pytorch/debug/test_config.py" \
        "--feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS"

    if [ "$FAIL" -ne 0 ]; then
        echo "Some tests failed."
        exit 1
    fi
    exit 0
fi

run_pytest_step "debug sanity" "test_sanity.xml" \
    "$TE_PATH/tests/pytorch/debug/test_sanity.py" \
    "--feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS"

run_pytest_step "debug config" "test_config.xml" \
    "$TE_PATH/tests/pytorch/debug/test_config.py" \
    "--feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS"

run_pytest_step "debug numerics" "test_numerics.xml" \
    "$TE_PATH/tests/pytorch/debug/test_numerics.py" \
    "--feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS"

run_pytest_step "debug log" "test_log.xml" \
    "$TE_PATH/tests/pytorch/debug/test_log.py" \
    "--feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS" \
    "--configs_dir=$NVTE_TEST_NVINSPECT_CONFIGS_DIR"

NVTE_TORCH_COMPILE=0 run_pytest_step "debug API features" "test_api_features.xml" \
    "$TE_PATH/tests/pytorch/debug/test_api_features.py" \
    --no-header \
    "--feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS" \
    "--configs_dir=$NVTE_TEST_NVINSPECT_CONFIGS_DIR"

run_pytest_step "debug performance" "test_perf.xml" \
    "$TE_PATH/tests/pytorch/debug/test_perf.py" \
    "--feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS" \
    "--configs_dir=$NVTE_TEST_NVINSPECT_CONFIGS_DIR"

if [ "$FAIL" -ne 0 ]; then
    echo "Some tests failed."
    exit 1
fi
