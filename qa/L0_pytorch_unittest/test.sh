#!/bin/bash


: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

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
