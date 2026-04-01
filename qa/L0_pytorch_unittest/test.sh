#!/bin/bash


: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

pip install pytest==8.2.1
FAIL=0

IS_CUDA_BACKEND=$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null)

test_fail() {
    FAIL=1
    echo "Error: sub-test failed: $1"
}


run_test_step() {
    local xml_file=$1
    local test_path=$2
    local cmd=$3
    local label=$4


    if [ "$PLATFORM" = "metax" ]; then
        case "$test_path" in
            *"test_numerics.py" | \
            *"test_sanity.py" | \
            *"test_parallel_cross_entropy.py" | \
            *"test_cuda_graphs.py" | \
            *"test_fused_rope.py" | \
            *"test_gqa.py" | \
            *"test_fused_optimizer.py" | \
            *"test_multi_tensor.py" | \
            *"test_cpu_offloading.py" | \
            *"test_attention.py" | \
            *"test_kv_cache.py" | \
            *"test_checkpoint.py" | \
            *"test_fused_router.py")
                echo "-------------------------------------------------------"
                echo "[SKIP] Platform MetaX: Ignoring $label"
                echo "-------------------------------------------------------"
                return 0
                ;;
        esac
    fi

    if [[ "$IS_CUDA_BACKEND" == *"cuda"* ]]; then
        if [[ "$test_path" == *"test_checkpoint.py" || "$test_path" == *"test_cpu_offloading.py" || "$test_path" == *"test_attention.py" ]]; then
            echo "-------------------------------------------------------"
            echo "[SKIP] CUDA Backend detected: Ignoring $label"
            echo "-------------------------------------------------------"
            return 0
        fi
    fi


    echo "-------------------------------------------------------"
    echo "[RUN] Executing: $label"

    eval "$cmd" || test_fail "$label"
}


# Step: Sanity
run_test_step "pytest_test_sanity.xml" "$TE_PATH/tests/pytorch/test_sanity.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_sanity.xml $TE_PATH/tests/pytorch/test_sanity.py -k \"not (test_sanity_layernorm_mlp or test_sanity_gpt or test_sanity_bert or test_sanity_T5 or test_sanity_amp_and_nvfuser or test_sanity_drop_path or test_sanity_fused_qkv_params or test_sanity_gradient_accumulation_fusion or test_inference_mode or test_sanity_normalization_amp or test_sanity_layernorm_linear or test_sanity_linear_with_zero_tokens or test_sanity_grouped_linear)\" --no-header" "test_sanity.py"

# Step: Recipe
run_test_step "pytest_test_recipe.xml" "$TE_PATH/tests/pytorch/test_recipe.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_recipe.xml $TE_PATH/tests/pytorch/test_recipe.py" "test_recipe.py"

# Step: Deferred Init
run_test_step "pytest_test_deferred_init.xml" "$TE_PATH/tests/pytorch/test_deferred_init.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_deferred_init.xml $TE_PATH/tests/pytorch/test_deferred_init.py" "test_deferred_init.py"

# Step: Numerics
run_test_step "pytest_test_numerics.xml" "$TE_PATH/tests/pytorch/test_numerics.py" \
"PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0 python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_numerics.xml $TE_PATH/tests/pytorch/test_numerics.py -k \"not (test_layernorm_mlp_accuracy or test_grouped_linear_accuracy or test_gpt_cuda_graph or test_transformer_layer_hidden_states_format or test_grouped_gemm or test_noncontiguous or test_gpt_checkpointing or test_gpt_accuracy or test_mha_accuracy or test_linear_accuracy or test_linear_accuracy_delay_wgrad_compute or test_rmsnorm_accuracy or test_layernorm_accuracy or test_layernorm_linear_accuracy)\" --no-header" "test_numerics.py"

# Step: CUDA Graphs
run_test_step "pytest_test_cuda_graphs.xml" "$TE_PATH/tests/pytorch/test_cuda_graphs.py" \
"PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0 python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cuda_graphs.xml $TE_PATH/tests/pytorch/test_cuda_graphs.py" "test_cuda_graphs.py"

# Step: JIT
run_test_step "pytest_test_jit.xml" "$TE_PATH/tests/pytorch/test_jit.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_jit.xml $TE_PATH/tests/pytorch/test_jit.py -k \"not (test_torch_dynamo)\"" "test_jit.py"

# Step: Fused Rope
run_test_step "pytest_test_fused_rope.xml" "$TE_PATH/tests/pytorch/test_fused_rope.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fused_rope.xml $TE_PATH/tests/pytorch/test_fused_rope.py" "test_fused_rope.py"

# Step: NVFP4 (Directory)
run_test_step "pytest_test_nvfp4.xml" "$TE_PATH/tests/pytorch/nvfp4" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_nvfp4.xml $TE_PATH/tests/pytorch/nvfp4" "test_nvfp4"

# Step: Float8 Tensors
run_test_step "pytest_test_float8tensor.xml" "$TE_PATH/tests/pytorch/test_float8tensor.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8tensor.xml $TE_PATH/tests/pytorch/test_float8tensor.py" "test_float8tensor.py"

# Step: GQA
run_test_step "pytest_test_gqa.xml" "$TE_PATH/tests/pytorch/test_gqa.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_gqa.xml $TE_PATH/tests/pytorch/test_gqa.py" "test_gqa.py"

# Step: Fused Optimizer
run_test_step "pytest_test_fused_optimizer.xml" "$TE_PATH/tests/pytorch/test_fused_optimizer.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fused_optimizer.xml $TE_PATH/tests/pytorch/test_fused_optimizer.py" "test_fused_optimizer.py"

# Step: Parallel Cross Entropy
run_test_step "pytest_test_parallel_cross_entropy.xml" "$TE_PATH/tests/pytorch/test_parallel_cross_entropy.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_parallel_cross_entropy.xml $TE_PATH/tests/pytorch/test_parallel_cross_entropy.py" "test_parallel_cross_entropy.py"

# Step: CPU Offloading
run_test_step "pytest_test_cpu_offloading.xml" "$TE_PATH/tests/pytorch/test_cpu_offloading.py" \
"NVTE_FLASH_ATTN=0 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cpu_offloading.xml $TE_PATH/tests/pytorch/test_cpu_offloading.py" "test_cpu_offloading.py"

# Step: Attention
run_test_step "pytest_test_attention.xml" "$TE_PATH/tests/pytorch/attention/test_attention.py" \
"python3 -m pytest -s -v --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_attention.xml $TE_PATH/tests/pytorch/attention/test_attention.py" "test_attention.py"

# Step: Checkpoint
run_test_step "pytest_test_checkpoint.xml" "$TE_PATH/tests/pytorch/test_checkpoint.py" \
"NVTE_TEST_CHECKPOINT_ARTIFACT_PATH=$TE_PATH/artifacts/tests/pytorch/test_checkpoint python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_checkpoint.xml $TE_PATH/tests/pytorch/test_checkpoint.py" "test_checkpoint.py"


if [ "$FAIL" -ne 0 ]; then
    echo "Some tests failed."
    exit 1
fi
echo "All assigned tests passed (some might have been skipped)."
exit 0
