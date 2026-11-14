#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"

PYTHON="${PYTHON_BIN:-python3}"
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

run_cmd() {
    local suite=$1
    local label=$2
    shift 2
    echo "-------------------------------------------------------"
    echo "[RUN][$suite] $label"
    echo "-------------------------------------------------------"
    if ! "$@"; then
        FAIL=1
        FAILED_CASES+=("$suite:$label")
        echo "Error: sub-test failed: $suite:$label"
    fi
}

run_pytest_step() {
    local suite=$1
    local label=$2
    shift 2
    run_cmd "$suite" "$label" "$PYTHON" -m pytest "$@"
}

run_debug_suite() {
    echo "===== START debug ====="
    run_pytest_step debug test_config.py \
        -v -s --tb=auto --junitxml="$XML_LOG_DIR/test_config.xml" \
        "$TE_PATH/tests/pytorch/debug/test_config.py" \
        --feature_dirs="$TE_PATH/transformer_engine/debug/features"
    run_pytest_step debug test_log.py \
        -v -s --tb=auto --junitxml="$XML_LOG_DIR/test_log.xml" \
        "$TE_PATH/tests/pytorch/debug/test_log.py" \
        --feature_dirs="$TE_PATH/transformer_engine/debug/features" \
        -k "test_compute_max_blockwise_dynamic_range_direct"
    run_pytest_step debug test_api_features.py \
        -v -s --tb=auto --no-header --junitxml="$XML_LOG_DIR/test_api_features.xml" \
        "$TE_PATH/tests/pytorch/debug/test_api_features.py" \
        --feature_dirs="$TE_PATH/transformer_engine/debug/features" \
        --configs_dir="$TE_PATH/tests/pytorch/debug/test_configs" \
        -k "test_transformer_engine_no_config"
    echo "===== END debug rc=$FAIL ====="
}

run_unittest_suite() {
    echo "===== START unittest ====="
    run_pytest_step unittest test_deferred_init.py \
        -s -v --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_deferred_init.xml" \
        "$TE_PATH/tests/pytorch/test_deferred_init.py"
    run_pytest_step unittest test_jit.py \
        -s -v --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_jit.xml" \
        "$TE_PATH/tests/pytorch/test_jit.py"
    run_pytest_step unittest test_fused_optimizer.py \
        -s -v --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_fused_optimizer.xml" \
        "$TE_PATH/tests/pytorch/test_fused_optimizer.py"
    run_pytest_step unittest test_hf_integration.py \
        -s -v --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_hf_integration.xml" \
        "$TE_PATH/tests/pytorch/test_hf_integration.py"
    if ! TE_TEST_GROUP_PRESET=plugin-unit "$PYTHON" "$TE_PATH/tests/test_utils/run_ci_test_group.py"; then
        FAIL=1
        FAILED_CASES+=("unittest:plugin_core")
    fi
    echo "===== END unittest rc=$FAIL ====="
}

run_distributed_suite() {
    echo "===== START distributed ====="
    run_pytest_step distributed test_cast_master_weights_to_fp8.py \
        -v -s --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_cast_master_weights_to_fp8.xml" \
        "$TE_PATH/tests/pytorch/distributed/test_cast_master_weights_to_fp8.py"
    run_pytest_step distributed test_cp_utils.py \
        -v -s --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_cp_utils.xml" \
        "$TE_PATH/tests/pytorch/attention/test_cp_utils.py"
    run_pytest_step distributed test_numerics.py \
        -v -s --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_numerics.xml" \
        "$TE_PATH/tests/pytorch/distributed/test_numerics.py"
    run_pytest_step distributed test_numerics_exact.py \
        -v -s --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_numerics_exact.xml" \
        "$TE_PATH/tests/pytorch/distributed/test_numerics_exact.py"
    run_pytest_step distributed test_torch_fsdp2.py \
        -v -s --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_torch_fsdp2.xml" \
        "$TE_PATH/tests/pytorch/distributed/test_torch_fsdp2.py"
    echo "===== END distributed rc=$FAIL ====="
}

run_onnx_suite() {
    echo "===== START onnx ====="
    run_pytest_step onnx test_onnx_export.py \
        --tb=auto --junitxml="$XML_LOG_DIR/test_onnx_export.xml" \
        "$TE_PATH/tests/pytorch/test_onnx_export.py"
    echo "===== END onnx rc=$FAIL ====="
}

run_suite() {
    case "$1" in
        debug) run_debug_suite ;;
        unittest) run_unittest_suite ;;
        distributed) run_distributed_suite ;;
        onnx) run_onnx_suite ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown suite: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
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
