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
Usage: tests/plugin/backend/hygon/run_unit_tests.sh [debug] [unittest] [distributed] [onnx]

Runs the selected Hygon/DTK reference-baseline test group.
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

python_has_module() {
    "$PYTHON" - "$1" <<'PY'
import importlib
import sys

try:
    importlib.import_module(sys.argv[1])
except ModuleNotFoundError:
    raise SystemExit(1)
PY
}

install_python_package() {
    local module_name=$1
    local package_spec=$2

    if python_has_module "$module_name"; then
        return 0
    fi

    if [ "${HYGON_SKIP_DEP_INSTALL:-0}" = "1" ]; then
        echo "ERROR: Python module '$module_name' is missing and dependency installation is disabled" >&2
        return 1
    fi

    "$PYTHON" -m pip install "$package_spec"
}

skip_step() {
    local label=$1
    local reason=$2
    echo "-------------------------------------------------------"
    echo "[SKIP] Hygon/DTK: $label ($reason)"
    echo "-------------------------------------------------------"
}

run_cmd() {
    local suite=$1
    local label=$2
    local xml_name=$3
    shift 3

    echo "-------------------------------------------------------"
    echo "[RUN][$suite] $label"
    echo "-------------------------------------------------------"
    if ! "$@" --junitxml="$XML_LOG_DIR/$xml_name"; then
        FAIL=1
        FAILED_CASES+=("$suite:$label")
        echo "Error: sub-test failed: $suite:$label"
    fi
}

run_pytest_step() {
    local label=$1
    local xml_name=$2
    shift 2
    run_cmd "unittest" "$label" "$xml_name" "$@"
}

run_distributed_step() {
    local label=$1
    local xml_name=$2
    shift 2
    run_cmd "distributed" "$label" "$xml_name" "$@"
}

install_base_deps() {
    install_python_package pytest "${PYTEST_PACKAGE_SPEC:-pytest==8.2.1}"
}

install_l0_deps() {
    install_base_deps
    install_python_package expecttest "${EXPECTTEST_PACKAGE_SPEC:-expecttest}"
}

install_onnx_deps() {
    install_python_package onnxruntime "${ONNXRUNTIME_PACKAGE_SPEC:-onnxruntime}"
    install_python_package onnxruntime_extensions "${ONNXRUNTIME_EXTENSIONS_PACKAGE_SPEC:-onnxruntime_extensions}"
    install_base_deps
}

run_debug_suite() {
    echo "===== START debug $(date '+%F %T') ====="
    install_base_deps

    if ! "$PYTHON" -c "import nvdlfw_inspect.api" >/dev/null 2>&1; then
        skip_step "tests/pytorch/debug/*" "nvdlfw_inspect is unavailable"
        skip_step "tests/pytorch/test_sanity.py" "debug nvinspect path not required by current DTK plugin workflow"
        skip_step "tests/pytorch/test_numerics.py" "debug nvinspect path not required by current DTK plugin workflow"
        echo "===== END debug rc=0 $(date '+%F %T') ====="
        return 0
    fi

    local feature_dirs="${NVTE_TEST_NVINSPECT_FEATURE_DIRS:-$TE_PATH/transformer_engine/debug/features}"
    local configs_dir="${NVTE_TEST_NVINSPECT_CONFIGS_DIR:-$TE_PATH/tests/pytorch/debug/test_configs}"
    local dummy_config="${NVTE_TEST_NVINSPECT_DUMMY_CONFIG_FILE:-$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml}"

    run_cmd "debug" "debug/test_config.py" "test_config.xml" \
        "$PYTHON" -m pytest -v -s "$TE_PATH/tests/pytorch/debug/test_config.py" \
        --feature_dirs="$feature_dirs"
    run_cmd "debug" "debug/test_log.py" "test_log.xml" \
        "$PYTHON" -m pytest -v -s "$TE_PATH/tests/pytorch/debug/test_log.py" \
        --feature_dirs="$feature_dirs" --configs_dir="$configs_dir"
    run_cmd "debug" "debug/test_api_features.py" "test_api_features.xml" \
        env NVTE_TORCH_COMPILE=0 "$PYTHON" -m pytest -v -s "$TE_PATH/tests/pytorch/debug/test_api_features.py" \
        --no-header --feature_dirs="$feature_dirs" --configs_dir="$configs_dir"
    run_cmd "debug" "test_sanity.py (nvinspect)" "test_sanity_2.xml" \
        env NVTE_TEST_NVINSPECT_ENABLED=1 \
        NVTE_TEST_NVINSPECT_CONFIG_FILE="$dummy_config" \
        NVTE_TEST_NVINSPECT_FEATURE_DIRS="$feature_dirs" \
        PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
        "$PYTHON" -m pytest -v -s "$TE_PATH/tests/pytorch/test_sanity.py" --no-header

    echo "===== END debug rc=$FAIL $(date '+%F %T') ====="
}

run_unittest_suite() {
    echo "===== START unittest $(date '+%F %T') ====="
    install_l0_deps

    # Keep this group limited to upstream smoke tests that exercise the
    # reference path on a real Hygon device.
    run_pytest_step "test_deferred_init.py" "pytest_test_deferred_init.xml" \
        "$PYTHON" -m pytest -s -v --tb=auto "$TE_PATH/tests/pytorch/test_deferred_init.py"
    run_pytest_step "test_jit.py" "pytest_test_jit.xml" \
        "$PYTHON" -m pytest -s -v --tb=auto "$TE_PATH/tests/pytorch/test_jit.py" -k "not (test_torch_dynamo)"

    local plugin_root="$TE_PATH/tests/plugin"
    run_pytest_step "plugin/test_policy.py" "pytest_test_plugin_policy.xml" \
        "$PYTHON" -m pytest -s -v --tb=auto "$plugin_root/plugin/test_policy.py"
    run_pytest_step "plugin/test_manager.py" "pytest_test_plugin_manager.xml" \
        "$PYTHON" -m pytest -s -v --tb=auto "$plugin_root/plugin/test_manager.py"
    run_pytest_step "plugin/test_policy_selection.py" "pytest_test_plugin_policy_selection.xml" \
        "$PYTHON" -m pytest -s -v --tb=auto "$plugin_root/plugin/test_policy_selection.py"
    run_pytest_step "reference/test_lifecycle.py" "pytest_test_backend_reference.xml" \
        "$PYTHON" -m pytest -s -v --tb=auto "$plugin_root/backend/reference/test_lifecycle.py"
    run_pytest_step "reference/test_activation.py" "pytest_test_backend_reference_activation.xml" \
        "$PYTHON" -m pytest -s -v --tb=auto "$plugin_root/backend/reference/test_activation.py"
    run_pytest_step "reference/test_dropout.py" "pytest_test_backend_reference_dropout.xml" \
        "$PYTHON" -m pytest -s -v --tb=auto "$plugin_root/backend/reference/test_dropout.py"
    run_pytest_step "reference/test_gemm.py" "pytest_test_backend_reference_gemm.xml" \
        "$PYTHON" -m pytest -s -v --tb=auto "$plugin_root/backend/reference/test_gemm.py"

    echo "===== END unittest rc=$FAIL $(date '+%F %T') ====="
}

run_distributed_suite() {
    echo "===== START distributed $(date '+%F %T') ====="
    install_base_deps

    run_distributed_step "attention/test_cp_utils.py" "pytest_test_cp_utils.xml" \
        "$PYTHON" -m pytest -v -s "$TE_PATH/tests/pytorch/attention/test_cp_utils.py"

    echo "===== END distributed rc=$FAIL $(date '+%F %T') ====="
}

run_onnx_suite() {
    echo "===== START onnx $(date '+%F %T') ====="
    install_onnx_deps

    local skip_expr
    skip_expr="$(join_with_or "${HYGON_ONNX_SKIP_GROUPS[@]}")"
    skip_expr="not ($skip_expr)"
    echo "[SKIP] Hygon/DTK ONNX groups: $skip_expr"
    run_cmd "onnx" "test_onnx_export.py" "test_onnx_export.xml" \
        env NVTE_UnfusedDPA_Emulate_FP8=1 \
        "$PYTHON" -m pytest --tb=auto "$TE_PATH/tests/pytorch/test_onnx_export.py" -k "$skip_expr"

    echo "===== END onnx rc=$FAIL $(date '+%F %T') ====="
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

echo "Selected Hygon/DTK reference-baseline tests passed (some optional groups might have been skipped)."
