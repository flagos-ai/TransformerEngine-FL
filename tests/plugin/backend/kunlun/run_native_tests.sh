#!/usr/bin/env bash
# KunlunXin backend test entrypoint.
set -euo pipefail

TE_PATH="${TE_PATH:-/opt/transformerengine}"
XML_LOG_DIR="${XML_LOG_DIR:-/logs}"
mkdir -p "$XML_LOG_DIR"

export PLATFORM="${PLATFORM:-kunlunxin}"
export TE_FL_SKIP_CUDA="${TE_FL_SKIP_CUDA:-1}"
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-0}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export NVTE_UNFUSED_ATTN="${NVTE_UNFUSED_ATTN:-1}"
export NVTE_TEST_NVINSPECT_FEATURE_DIRS="${NVTE_TEST_NVINSPECT_FEATURE_DIRS:-$TE_PATH/transformer_engine/debug/features}"
export NVTE_TEST_NVINSPECT_CONFIGS_DIR="${NVTE_TEST_NVINSPECT_CONFIGS_DIR:-$TE_PATH/tests/pytorch/debug/test_configs/}"

if [ -n "${TE_TEST_GROUP_JSON:-}" ]; then
    GROUP_NAME=$(python3 - <<'PY'
import json
import os

group = json.loads(os.environ["TE_TEST_GROUP_JSON"])
print(group.get("name", ""))
PY
    )
else
    GROUP_NAME="${1:-}"
fi

if [ -z "$GROUP_NAME" ]; then
    echo "Error: TE_TEST_GROUP_JSON is required" >&2
    exit 1
fi

FAIL=0

run_test_step() {
    local label=$1
    shift
    echo "-------------------------------------------------------"
    echo "[RUN] Executing: $label"
    eval "$*" || FAIL=1
}

case "$GROUP_NAME" in
    pytorch_debug)
        run_test_step "test_sanity.xml" \
            "python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/test_sanity.xml \
            $TE_PATH/tests/pytorch/debug/test_sanity.py -k 'not fake_quant' \
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
            -k 'not (per_tensor_scaling or fake_quant or statistics)' \
            --no-header \
            --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS \
            --configs_dir=$NVTE_TEST_NVINSPECT_CONFIGS_DIR"

        run_test_step "test_perf.xml" \
            "python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/test_perf.xml \
            $TE_PATH/tests/pytorch/debug/test_perf.py \
            --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS \
            --configs_dir=$NVTE_TEST_NVINSPECT_CONFIGS_DIR"
        ;;
    pytorch_unittest)
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
        ;;
    pytorch_distributed_unittest)
        run_test_step "pytest_test_numerics.xml" \
            "python3 -m pytest -v -s \
            --junitxml=$XML_LOG_DIR/pytest_test_numerics.xml \
            $TE_PATH/tests/pytorch/distributed/test_numerics.py"
        run_test_step "pytest_test_numerics_exact.xml" \
            "python3 -m pytest -v -s \
            --junitxml=$XML_LOG_DIR/pytest_test_numerics_exact.xml \
            $TE_PATH/tests/pytorch/distributed/test_numerics_exact.py"
        run_test_step "pytest_test_torch_fsdp2.xml" \
            "PYTEST_ADDOPTS= COVERAGE_PROCESS_START= python3 -m pytest -v -s \
            --junitxml=$XML_LOG_DIR/pytest_test_torch_fsdp2.xml \
            $TE_PATH/tests/pytorch/distributed/test_torch_fsdp2.py \
            -k 'not test_distributed and not test_fsdp2_mem_leak_tests'"
        run_test_step "pytest_test_cp_utils.xml" \
            "python3 -m pytest -v -s \
            --junitxml=$XML_LOG_DIR/pytest_test_cp_utils.xml \
            $TE_PATH/tests/pytorch/attention/test_cp_utils.py"
        run_test_step "pytest_test_cast_master_weights_to_fp8.xml" \
            "python3 -m pytest -v -s \
            --junitxml=$XML_LOG_DIR/pytest_test_cast_master_weights_to_fp8.xml \
            $TE_PATH/tests/pytorch/distributed/test_cast_master_weights_to_fp8.py"
        ;;
    pytorch_onnx_unittest)
        export NVTE_UnfusedDPA_Emulate_FP8="${NVTE_UnfusedDPA_Emulate_FP8:-1}"

        run_test_step "test_onnx_export_linear.xml" \
            "NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=1 \
            NVTE_UnfusedDPA_Emulate_FP8=1 python3 -m pytest -v -s --tb=short \
            --junitxml=$XML_LOG_DIR/test_onnx_export_linear.xml \
            $TE_PATH/tests/pytorch/test_onnx_export.py \
            -k 'test_export_linear_recipe or test_export_linear_use_bias or test_export_linear_return_bias'"
        run_test_step "test_onnx_export_normalization.xml" \
            "NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=1 \
            NVTE_UnfusedDPA_Emulate_FP8=1 python3 -m pytest -v -s --tb=short \
            --junitxml=$XML_LOG_DIR/test_onnx_export_normalization.xml \
            $TE_PATH/tests/pytorch/test_onnx_export.py \
            -k 'test_export_layernorm_recipe or test_export_layernorm_zero_centered_gamma or test_export_layernorm_normalization'"
        run_test_step "test_onnx_export_layernorm_linear.xml" \
            "NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=1 \
            NVTE_UnfusedDPA_Emulate_FP8=1 python3 -m pytest -v -s --tb=short \
            --junitxml=$XML_LOG_DIR/test_onnx_export_layernorm_linear.xml \
            $TE_PATH/tests/pytorch/test_onnx_export.py \
            -k 'test_export_layernorm_linear_recipe or test_export_layernorm_linear_return_ln_out or test_export_layernorm_linear_zero_centered_gamma or test_export_layernorm_linear_normalization or test_export_layernorm_linear_no_bias or test_export_layernorm_linear_return_bias'"
        run_test_step "test_onnx_export_layernorm_mlp.xml" \
            "NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=1 \
            NVTE_UnfusedDPA_Emulate_FP8=1 python3 -m pytest -v -s --tb=short \
            --junitxml=$XML_LOG_DIR/test_onnx_export_layernorm_mlp.xml \
            $TE_PATH/tests/pytorch/test_onnx_export.py \
            -k 'test_export_layernorm_mlp or test_export_layernorm_mlp_return_layernorm_output or test_export_layernorm_mlp_return_bias or test_export_layernorm_mlp_no_bias or test_export_layernorm_mlp_zero_centered_gamma or test_export_layernorm_mlp_normalization or test_export_layernorm_mlp_activation'"
        run_test_step "test_onnx_export_context.xml" \
            "NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=1 \
            NVTE_UnfusedDPA_Emulate_FP8=1 python3 -m pytest -v -s --tb=short \
            --junitxml=$XML_LOG_DIR/test_onnx_export_context.xml \
            $TE_PATH/tests/pytorch/test_onnx_export.py \
            -k 'test_export_ctx_manager'"
        ;;
    *)
        echo "Error: unsupported Kunlun test group: ${GROUP_NAME}" >&2
        exit 1
        ;;
esac

if [ "$FAIL" -ne 0 ]; then
    exit 1
fi

exit 0
