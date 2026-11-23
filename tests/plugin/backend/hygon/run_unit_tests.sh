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
Usage: tests/plugin/backend/hygon/run_unit_tests.sh [debug] [unittest] [distributed] [onnx]

Runs the selected Hygon/DTK reference-baseline test group.
If no suite is specified, all suites are run.
EOF
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

install_base_deps() {
    install_python_package pytest "${PYTEST_PACKAGE_SPEC:-pytest==8.2.1}"
}

install_l0_deps() {
    install_base_deps
    install_python_package expecttest "${EXPECTTEST_PACKAGE_SPEC:-expecttest}"
}

# debug, distributed and onnx run the shared qa scripts like every other
# platform; Hygon differences live only in .github/configs/hygon.yml.
run_shared_suite() {
    local suite=$1 script=$2
    echo "===== START $suite $(date '+%F %T') ====="
    echo "[RUN][$suite] $script"
    if ! XML_LOG_DIR="$XML_LOG_DIR" bash "$TE_PATH/$script"; then
        FAIL=1
        FAILED_CASES+=("$suite:$script")
    fi
    echo "===== END $suite rc=$FAIL $(date '+%F %T') ====="
}

run_debug_suite() { run_shared_suite debug qa/L0_pytorch_debug_unittest/test.sh; }

run_unittest_suite() {
    echo "===== START unittest $(date '+%F %T') ====="
    install_l0_deps

    # Keep this group limited to upstream smoke tests that exercise the
    # reference path on a real Hygon device.
    run_pytest_step "test_deferred_init.py" "pytest_test_deferred_init.xml" \
        "$PYTHON" -m pytest -s -v --tb=auto "$TE_PATH/tests/pytorch/test_deferred_init.py"
    run_pytest_step "test_jit.py" "pytest_test_jit.xml" \
        "$PYTHON" -m pytest -s -v --tb=auto "$TE_PATH/tests/pytorch/test_jit.py"

    # Attention targets verified passing on BW1000 without touching shared
    # source: GQA and HF integration. The fused-attention suite and the
    # attention-accuracy numerics remain out of scope; see
    # ignored_test_groups.attention.
    run_pytest_step "test_gqa.py" "pytest_test_gqa.xml" \
        "$PYTHON" -m pytest -s -v --tb=auto "$TE_PATH/tests/pytorch/test_gqa.py"
    run_pytest_step "test_hf_integration.py" "pytest_test_hf_integration.xml" \
        "$PYTHON" -m pytest -s -v --tb=auto "$TE_PATH/tests/pytorch/test_hf_integration.py"

    if ! TE_TEST_GROUP_PRESET=plugin-unit "$PYTHON" "$TE_PATH/tests/test_utils/run_ci_test_group.py"; then
        FAIL=1
        FAILED_CASES+=("unittest:plugin_core")
    fi
    echo "===== END unittest rc=$FAIL $(date '+%F %T') ====="
}

run_distributed_suite() { run_shared_suite distributed qa/L1_pytorch_distributed_unittest/test.sh; }

run_onnx_suite() { run_shared_suite onnx qa/L1_pytorch_onnx_unittest/test.sh; }

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

echo "All selected Hygon/DTK reference-baseline test suites passed."
