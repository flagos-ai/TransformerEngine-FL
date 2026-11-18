#!/usr/bin/env bash
# KunlunXin backend test entrypoint.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"

TE_PATH="${TE_PATH:-/opt/transformerengine}"
XML_LOG_DIR="${XML_LOG_DIR:-/logs}"
mkdir -p "$XML_LOG_DIR"
XML_LOG_ROOT="$XML_LOG_DIR"

FAIL=0
OVERALL_FAIL=0

run_test_step() {
    local label=$1
    shift
    echo "-------------------------------------------------------"
    echo "[RUN] Executing: $label"
    eval "$*" || FAIL=1
}

run_suite() {
    case "$1" in
        debug | pytorch_debug)
        run_test_step "test_sanity.xml" \
            "python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/test_sanity.xml \
            $TE_PATH/tests/pytorch/debug/test_sanity.py \
            --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS"

        run_test_step "test_config.xml" \
            "python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/test_config.xml \
            $TE_PATH/tests/pytorch/debug/test_config.py \
            --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS"

        run_test_step "test_numerics.xml" \
            "python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/test_numerics.xml \
            $TE_PATH/tests/pytorch/debug/test_numerics.py \
            --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS"

        run_test_step "test_log.xml" \
            "python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/test_log.xml \
            $TE_PATH/tests/pytorch/debug/test_log.py \
            --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS \
            --configs_dir=$NVTE_TEST_NVINSPECT_CONFIGS_DIR"

        run_test_step "test_api_features.xml" \
            "NVTE_TORCH_COMPILE=0 python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/test_api_features.xml \
            $TE_PATH/tests/pytorch/debug/test_api_features.py \
            --no-header \
            --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS \
            --configs_dir=$NVTE_TEST_NVINSPECT_CONFIGS_DIR"

        run_test_step "test_perf.xml" \
            "python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/test_perf.xml \
            $TE_PATH/tests/pytorch/debug/test_perf.py \
            --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS \
            --configs_dir=$NVTE_TEST_NVINSPECT_CONFIGS_DIR"
            ;;
        unittest | pytorch_unittest)
        run_test_step "pytest_kunlun_sanity.xml" \
            "NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=1 \
            python3 -m pytest -s -v --tb=auto \
            --junitxml=$XML_LOG_DIR/pytest_kunlun_sanity.xml \
            $TE_PATH/tests/pytorch/test_sanity.py \
            -k 'test_sanity_normalization_amp or test_sanity_linear or test_sanity_layernorm_linear or test_sanity_layernorm_mlp' \
            --no-header"

        run_test_step "pytest_test_recipe.xml" \
            "python3 -m pytest -s -v --tb=auto \
            --junitxml=$XML_LOG_DIR/pytest_test_recipe.xml \
            $TE_PATH/tests/pytorch/test_recipe.py"

        run_test_step "pytest_test_deferred_init.xml" \
            "python3 -m pytest -s -v --tb=auto \
            --junitxml=$XML_LOG_DIR/pytest_test_deferred_init.xml \
            $TE_PATH/tests/pytorch/test_deferred_init.py"

        run_test_step "plugin-unit" \
            "TE_TEST_GROUP_PRESET=plugin-unit python3 \
            $TE_PATH/tests/test_utils/run_ci_test_group.py"
            ;;
        # distributed and onnx run the shared qa scripts like every other platform;
        # Kunlun differences live only in .github/configs/kunlun.yml ignored_tests.
        distributed | pytorch_distributed_unittest)
        run_test_step "qa/L1_pytorch_distributed_unittest/test.sh" \
            "XML_LOG_DIR=$XML_LOG_DIR bash $TE_PATH/qa/L1_pytorch_distributed_unittest/test.sh"
            ;;
        onnx | pytorch_onnx_unittest)
        run_test_step "qa/L1_pytorch_onnx_unittest/test.sh" \
            "XML_LOG_DIR=$XML_LOG_DIR bash $TE_PATH/qa/L1_pytorch_onnx_unittest/test.sh"
            ;;
        -h | --help)
            echo "Usage: $0 [debug] [unittest] [distributed] [onnx]"
            echo "If no suite is specified, all suites are run serially."
            exit 0
            ;;
        *)
            echo "Error: unsupported Kunlun test suite: $1" >&2
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
    echo "===== START $suite ====="
    run_suite "$suite"
    echo "===== END $suite rc=$FAIL ====="
    if [ "$FAIL" -ne 0 ]; then
        OVERALL_FAIL=1
    fi
done

if [ "$OVERALL_FAIL" -ne 0 ]; then
    exit 1
fi

exit 0
