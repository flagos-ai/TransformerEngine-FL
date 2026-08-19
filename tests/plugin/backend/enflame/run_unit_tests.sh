#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env.sh"
source "$SCRIPT_DIR/config.sh"

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

join_with_or() {
    local result=""
    local item
    for item in "$@"; do
        if [ -z "$result" ]; then
            result="$item"
        else
            result="$result or $item"
        fi
    done
    printf '%s' "$result"
}

contains_item() {
    local needle=$1
    shift
    local item
    for item in "$@"; do
        if [ "$item" = "$needle" ]; then
            return 0
        fi
    done
    return 1
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
    local fused_optimizer_skip_expr
    fused_optimizer_skip_expr="$(join_with_or "${ENFLAME_UNITTEST_SKIP_FUSED_OPTIMIZER[@]}")"
    run_pytest_step unittest test_deferred_init.py \
        -s -v --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_deferred_init.xml" \
        "$TE_PATH/tests/pytorch/test_deferred_init.py"
    echo "[SKIP] Enflame: test_jit.py requires unsupported Torch JIT/TorchDynamo paths"
    run_pytest_step unittest test_fused_optimizer.py \
        -s -v --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_fused_optimizer.xml" \
        "$TE_PATH/tests/pytorch/test_fused_optimizer.py" \
        -k "not ($fused_optimizer_skip_expr)"
    local hf_integration_skip_expr
    hf_integration_skip_expr="$(join_with_or "${ENFLAME_UNITTEST_SKIP_HF_INTEGRATION[@]}")"
    run_pytest_step unittest test_hf_integration.py \
        -s -v --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_hf_integration.xml" \
        "$TE_PATH/tests/pytorch/test_hf_integration.py" \
        -k "not ($hf_integration_skip_expr)"
    run_pytest_step unittest plugin_policy.py \
        -s -v --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_plugin_policy.xml" \
        "$TE_PATH/tests/plugin/plugin/test_policy.py"
    run_pytest_step unittest plugin_manager.py \
        -s -v --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_plugin_manager.xml" \
        "$TE_PATH/tests/plugin/plugin/test_manager.py"
    run_pytest_step unittest backend_reference_lifecycle.py \
        -s -v --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_backend_reference.xml" \
        "$TE_PATH/tests/plugin/backend/reference/test_lifecycle.py"
    run_pytest_step unittest backend_reference_activation.py \
        -s -v --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_backend_reference_activation.xml" \
        "$TE_PATH/tests/plugin/backend/reference/test_activation.py"
    run_pytest_step unittest backend_reference_dropout.py \
        -s -v --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_backend_reference_dropout.xml" \
        "$TE_PATH/tests/plugin/backend/reference/test_dropout.py"
    run_pytest_step unittest backend_reference_gemm.py \
        -s -v --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_backend_reference_gemm.xml" \
        "$TE_PATH/tests/plugin/backend/reference/test_gemm.py"
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
    if contains_item "tests/pytorch/distributed/test_numerics.py" "${ENFLAME_DISTRIBUTED_SKIP_FILES[@]}"; then
        echo "[SKIP][distributed] test_numerics.py via ENFLAME_DISTRIBUTED_SKIP_FILES"
    else
        run_pytest_step distributed test_numerics.py \
            -v -s --tb=auto --junitxml="$XML_LOG_DIR/pytest_test_numerics.xml" \
            "$TE_PATH/tests/pytorch/distributed/test_numerics.py"
    fi
    echo "===== END distributed rc=$FAIL ====="
}

run_onnx_suite() {
    echo "===== START onnx ====="
    local skip_expr
    skip_expr="$(join_with_or "${ENFLAME_ONNX_SKIP_GROUPS[@]}")"
    skip_expr="not ($skip_expr)"
    run_pytest_step onnx test_onnx_export.py \
        --tb=auto --junitxml="$XML_LOG_DIR/test_onnx_export.xml" \
        "$TE_PATH/tests/pytorch/test_onnx_export.py" \
        -k "$skip_expr"
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

echo "Selected Enflame/GCU tests passed (some optional groups might have been skipped)."
