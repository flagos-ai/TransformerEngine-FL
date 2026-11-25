#!/bin/bash


: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

# Retain the launcher's interpreter and optional import-time pytest adapter.
# Plugin tests use run_ci_test_group.py directly so their isolation is preserved.
if [[ -n "${QA_PYTHON_BIN:-}${TE_TEST_PYTEST_COMMAND:-}" ]]; then
    python3() {
        if [[ "${1:-}" == -m && "${2:-}" == pytest && -n "${TE_TEST_PYTEST_COMMAND:-}" ]]; then
            shift 2
            command "${QA_PYTHON_BIN:-python3}" -c '
import os
import shlex
import sys
command = shlex.split(os.environ["TE_TEST_PYTEST_COMMAND"].replace("{workspace}", os.environ.get("TE_PATH", "/opt/transformerengine")))
os.execvpe(command[0], command + sys.argv[1:], os.environ)
' "$@"
        else
            command "${QA_PYTHON_BIN:-python3}" "$@"
        fi
    }
fi

pip install pytest==8.2.1
# solve test_fused_optimizer import error
pip install expecttest
FAIL=0

test_fail() {
    FAIL=1
    echo "Error: sub-test failed: $1"
}


# Platform ignores live in .github/configs/<platform>.yml (ignored_tests); the
# CI runner applies them to every pytest call below. Do not filter here.
run_test_step() {
    local xml_file=$1
    local test_path=$2
    local cmd=$3
    local label=$4

    echo "-------------------------------------------------------"
    echo "[RUN] Executing: $label"

    eval "$cmd" || test_fail "$label"
}


# Step: Sanity
SANITY_CMD="python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_sanity.xml $TE_PATH/tests/pytorch/test_sanity.py --no-header"
run_test_step "pytest_test_sanity.xml" "$TE_PATH/tests/pytorch/test_sanity.py" "$SANITY_CMD" "test_sanity.py"


# Step: Recipe
run_test_step "pytest_test_recipe.xml" "$TE_PATH/tests/pytorch/test_recipe.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_recipe.xml $TE_PATH/tests/pytorch/test_recipe.py" "test_recipe.py"

# Step: Deferred Init
run_test_step "pytest_test_deferred_init.xml" "$TE_PATH/tests/pytorch/test_deferred_init.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_deferred_init.xml $TE_PATH/tests/pytorch/test_deferred_init.py" "test_deferred_init.py"

# Step: Numerics
NUMERICS_CMD="PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0 python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_numerics.xml $TE_PATH/tests/pytorch/test_numerics.py --no-header"
run_test_step "pytest_test_numerics.xml" "$TE_PATH/tests/pytorch/test_numerics.py" "$NUMERICS_CMD" "test_numerics.py"

# Step: CUDA Graphs
run_test_step "pytest_test_cuda_graphs.xml" "$TE_PATH/tests/pytorch/test_cuda_graphs.py" \
"PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0 python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cuda_graphs.xml $TE_PATH/tests/pytorch/test_cuda_graphs.py" "test_cuda_graphs.py"

# Step: JIT
JIT_CMD="python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_jit.xml $TE_PATH/tests/pytorch/test_jit.py --no-header"
run_test_step "pytest_test_jit.xml" "$TE_PATH/tests/pytorch/test_jit.py" "$JIT_CMD" "test_jit.py"


# Step: Fused Rope
run_test_step "pytest_test_fused_rope.xml" "$TE_PATH/tests/pytorch/test_fused_rope.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fused_rope.xml $TE_PATH/tests/pytorch/test_fused_rope.py" "test_fused_rope.py"

# Step: NVFP4 (Directory)
run_test_step "pytest_test_nvfp4.xml" "$TE_PATH/tests/pytorch/nvfp4" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_nvfp4.xml $TE_PATH/tests/pytorch/nvfp4" "test_nvfp4"

# Step: Quantized Tensors
run_test_step "pytest_test_quantized_tensor.xml" "$TE_PATH/tests/pytorch/test_quantized_tensor.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_quantized_tensor.xml $TE_PATH/tests/pytorch/test_quantized_tensor.py" "test_quantized_tensor.py"

# Step: Float8 Blockwise Tensor
run_test_step "pytest_test_float8blockwisetensor.xml" "$TE_PATH/tests/pytorch/test_float8blockwisetensor.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8blockwisetensor.xml $TE_PATH/tests/pytorch/test_float8blockwisetensor.py" "test_float8blockwisetensor.py"

# Step: Float8 Blockwise Scaling Exact
run_test_step "pytest_test_float8_blockwise_scaling_exact.xml" "$TE_PATH/tests/pytorch/test_float8_blockwise_scaling_exact.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8_blockwise_scaling_exact.xml $TE_PATH/tests/pytorch/test_float8_blockwise_scaling_exact.py" "test_float8_blockwise_scaling_exact.py"

# Step: Float8 Blockwise GEMM Exact
run_test_step "pytest_test_float8_blockwise_gemm_exact.xml" "$TE_PATH/tests/pytorch/test_float8_blockwise_gemm_exact.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8_blockwise_gemm_exact.xml $TE_PATH/tests/pytorch/test_float8_blockwise_gemm_exact.py" "test_float8_blockwise_gemm_exact.py"

# Step: GQA
run_test_step "pytest_test_gqa.xml" "$TE_PATH/tests/pytorch/test_gqa.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_gqa.xml $TE_PATH/tests/pytorch/test_gqa.py" "test_gqa.py"

# Step: Fused Optimizer
run_test_step "pytest_test_fused_optimizer.xml" "$TE_PATH/tests/pytorch/test_fused_optimizer.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fused_optimizer.xml $TE_PATH/tests/pytorch/test_fused_optimizer.py" "test_fused_optimizer.py"

# Step: Multi Tensor
run_test_step "pytest_test_multi_tensor.xml" "$TE_PATH/tests/pytorch/test_multi_tensor.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_multi_tensor.xml $TE_PATH/tests/pytorch/test_multi_tensor.py" "test_multi_tensor.py"

# Step: Fusible Ops
run_test_step "pytest_test_fusible_ops.xml" "$TE_PATH/tests/pytorch/test_fusible_ops.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fusible_ops.xml $TE_PATH/tests/pytorch/test_fusible_ops.py" "test_fusible_ops.py"

# Step: Permutation
run_test_step "pytest_test_permutation.xml" "$TE_PATH/tests/pytorch/test_permutation.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_permutation.xml $TE_PATH/tests/pytorch/test_permutation.py" "test_permutation.py"

# Step: Parallel Cross Entropy
run_test_step "pytest_test_parallel_cross_entropy.xml" "$TE_PATH/tests/pytorch/test_parallel_cross_entropy.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_parallel_cross_entropy.xml $TE_PATH/tests/pytorch/test_parallel_cross_entropy.py" "test_parallel_cross_entropy.py"

# Step: CPU Offloading
run_test_step "pytest_test_cpu_offloading.xml" "$TE_PATH/tests/pytorch/test_cpu_offloading.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cpu_offloading.xml $TE_PATH/tests/pytorch/test_cpu_offloading.py" "test_cpu_offloading.py"

# Step: CPU Offloading V1
run_test_step "pytest_test_cpu_offloading_v1.xml" "$TE_PATH/tests/pytorch/test_cpu_offloading_v1.py" \
"NVTE_FLASH_ATTN=0 NVTE_CPU_OFFLOAD_V1=1 python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cpu_offloading_v1.xml $TE_PATH/tests/pytorch/test_cpu_offloading_v1.py" "test_cpu_offloading_v1.py"

# Step: Attention
run_test_step "pytest_test_attention.xml" "$TE_PATH/tests/pytorch/attention/test_attention.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_attention.xml $TE_PATH/tests/pytorch/attention/test_attention.py" "test_attention.py"

# Step: KV Cache
run_test_step "pytest_test_kv_cache.xml" "$TE_PATH/tests/pytorch/attention/test_kv_cache.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_kv_cache.xml $TE_PATH/tests/pytorch/attention/test_kv_cache.py" "test_kv_cache.py"

# Step: HF Integration
run_test_step "pytest_test_hf_integration.xml" "$TE_PATH/tests/pytorch/test_hf_integration.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_hf_integration.xml $TE_PATH/tests/pytorch/test_hf_integration.py" "test_hf_integration.py"

# Step: Checkpoint
run_test_step "pytest_test_checkpoint.xml" "$TE_PATH/tests/pytorch/test_checkpoint.py" \
"NVTE_TEST_CHECKPOINT_ARTIFACT_PATH=$TE_PATH/artifacts/tests/pytorch/test_checkpoint python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_checkpoint.xml $TE_PATH/tests/pytorch/test_checkpoint.py" "test_checkpoint.py"

# Additional functional tests found by the QA scope audit.
run_test_step "pytest_test_backward_override.xml" "$TE_PATH/tests/pytorch/test_backward_override.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_backward_override.xml $TE_PATH/tests/pytorch/test_backward_override.py" "test_backward_override.py"

run_test_step "pytest_test_custom_recipe.xml" "$TE_PATH/tests/pytorch/test_custom_recipe.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_custom_recipe.xml $TE_PATH/tests/pytorch/test_custom_recipe.py" "test_custom_recipe.py"

run_test_step "pytest_test_float8_current_scaling_exact.xml" "$TE_PATH/tests/pytorch/test_float8_current_scaling_exact.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8_current_scaling_exact.xml $TE_PATH/tests/pytorch/test_float8_current_scaling_exact.py" "test_float8_current_scaling_exact.py"

run_test_step "pytest_test_fused_router.xml" "$TE_PATH/tests/pytorch/test_fused_router.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fused_router.xml $TE_PATH/tests/pytorch/test_fused_router.py" "test_fused_router.py"

run_test_step "pytest_test_grouped_linear.xml" "$TE_PATH/tests/pytorch/test_grouped_linear.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_grouped_linear.xml $TE_PATH/tests/pytorch/test_grouped_linear.py" "test_grouped_linear.py"

run_test_step "pytest_test_grouped_mlp.xml" "$TE_PATH/tests/pytorch/test_grouped_mlp.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_grouped_mlp.xml $TE_PATH/tests/pytorch/test_grouped_mlp.py" "test_grouped_mlp.py"

run_test_step "pytest_test_grouped_tensor.xml" "$TE_PATH/tests/pytorch/test_grouped_tensor.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_grouped_tensor.xml $TE_PATH/tests/pytorch/test_grouped_tensor.py" "test_grouped_tensor.py"

run_test_step "pytest_test_mhc.xml" "$TE_PATH/tests/pytorch/test_mhc.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_mhc.xml $TE_PATH/tests/pytorch/test_mhc.py" "test_mhc.py"

run_test_step "pytest_test_nvfp4_fsdp2_hooks.xml" "$TE_PATH/tests/pytorch/test_nvfp4_fsdp2_hooks.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_nvfp4_fsdp2_hooks.xml $TE_PATH/tests/pytorch/test_nvfp4_fsdp2_hooks.py" "test_nvfp4_fsdp2_hooks.py"

run_test_step "pytest_test_partial_cast.xml" "$TE_PATH/tests/pytorch/test_partial_cast.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_partial_cast.xml $TE_PATH/tests/pytorch/test_partial_cast.py" "test_partial_cast.py"

run_test_step "pytest_test_qk_norm.xml" "$TE_PATH/tests/pytorch/test_qk_norm.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_qk_norm.xml $TE_PATH/tests/pytorch/test_qk_norm.py" "test_qk_norm.py"

run_test_step "pytest_test_torch_compile.xml" "$TE_PATH/tests/pytorch/test_torch_compile.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_torch_compile.xml $TE_PATH/tests/pytorch/test_torch_compile.py" "test_torch_compile.py"

run_test_step "pytest_test_mxfp8.xml" "$TE_PATH/tests/pytorch/mxfp8" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_mxfp8.xml $TE_PATH/tests/pytorch/mxfp8" "mxfp8"

run_test_step "pytest_test_cu_seqlens_cache.xml" "$TE_PATH/tests/pytorch/attention/test_cu_seqlens_cache.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cu_seqlens_cache.xml $TE_PATH/tests/pytorch/attention/test_cu_seqlens_cache.py" "test_cu_seqlens_cache.py"

run_test_step "pytest_test_flex_attention.xml" "$TE_PATH/tests/pytorch/attention/test_flex_attention.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_flex_attention.xml $TE_PATH/tests/pytorch/attention/test_flex_attention.py" "test_flex_attention.py"

# Keep MXFP8 attention correctness in unit QA. Timed speedup assertions are
# outside this change; no performance QA or CI entry is added for now.
run_test_step "pytest_test_linear_mxfp8_attention.xml" "$TE_PATH/tests/pytorch/attention/test_linear_mxfp8_attention.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_linear_mxfp8_attention.xml $TE_PATH/tests/pytorch/attention/test_linear_mxfp8_attention.py::TestLinearMXFP8Attention::test_accuracy $TE_PATH/tests/pytorch/attention/test_linear_mxfp8_attention.py::TestLinearMXFP8Attention::test_backward" "MXFP8 attention correctness"

# ==============================================================================
# New Step: Plugin Core
# ==============================================================================
# The unified unit job owns core, backend and cross-backend accuracy tests.
TE_TEST_GROUP_PRESET=plugin-unit python3 "$TE_PATH/tests/test_utils/run_ci_test_group.py" || test_fail "plugin"

if [ "$FAIL" -ne 0 ]; then
    echo "Some tests failed."
    exit 1
fi
echo "All assigned tests passed (some might have been skipped)."
exit 0
