#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${TE_PATH:=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)}"
: "${XML_LOG_DIR:=/logs}"

mkdir -p "${XML_LOG_DIR}"

FAILED=0

run_pytest() {
    local name=$1
    local target=$2
    shift 2

    echo "-------------------------------------------------------"
    echo "[RUN] ${name}: ${target}"
    if ! python3 -m pytest -s -v --tb=auto \
        --junitxml="${XML_LOG_DIR}/${name}.xml" \
        "${target}" "$@"; then
        echo "[FAIL] ${name}"
        FAILED=1
    fi
}

run_without_cuda_compat() {
    local name=$1
    local target=$2
    shift 2

    if python3 -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
        echo "[SKIP] MUSA compatibility environment exposes torch.cuda: ${name}"
        return 0
    fi
    run_pytest "${name}" "${target}" "$@"
}

run_debug() {
    local feature_dirs="${TE_PATH}/transformer_engine/debug/features"
    local configs_dir="${TE_PATH}/tests/pytorch/debug/test_configs/"

    NVTE_TORCH_COMPILE=0 \
    TORCHDYNAMO_DISABLE=1 \
    TORCH_COMPILE_DISABLE=1 \
        run_pytest test_debug_sanity \
        "${TE_PATH}/tests/pytorch/debug/test_sanity.py" \
        --feature_dirs="${feature_dirs}"

    run_pytest test_debug_config \
        "${TE_PATH}/tests/pytorch/debug/test_config.py" \
        --feature_dirs="${feature_dirs}"

    run_pytest test_debug_numerics \
        "${TE_PATH}/tests/pytorch/debug/test_numerics.py" \
        --feature_dirs="${feature_dirs}"

    run_pytest test_debug_log \
        "${TE_PATH}/tests/pytorch/debug/test_log.py" \
        --feature_dirs="${feature_dirs}" \
        --configs_dir="${configs_dir}"

    NVTE_TORCH_COMPILE=0 \
        run_pytest test_debug_api_features \
        "${TE_PATH}/tests/pytorch/debug/test_api_features.py" \
        --no-header \
        --feature_dirs="${feature_dirs}" \
        --configs_dir="${configs_dir}"

    run_pytest test_debug_perf \
        "${TE_PATH}/tests/pytorch/debug/test_perf.py" \
        --feature_dirs="${feature_dirs}" \
        --configs_dir="${configs_dir}"
}

run_pytorch() {
    local tests_root="${TE_PATH}/tests/pytorch"
    run_pytest test_sanity "${tests_root}/test_sanity.py" \
        -k "not (test_sanity_gpt or test_sanity_gpt_126m or test_sanity_bert or test_sanity_T5 or test_sanity_layernorm_mlp or test_sanity_amp_and_nvfuser or test_sanity_drop_path or test_sanity_fused_qkv_params or test_sanity_gradient_accumulation_fusion or test_inference_mode or test_sanity_normalization_amp or test_sanity_layernorm_linear or test_sanity_linear_with_zero_tokens or test_sanity_grouped_linear)" \
        --no-header

    run_pytest test_recipe "${tests_root}/test_recipe.py"
    run_pytest test_deferred_init "${tests_root}/test_deferred_init.py"

    PYTORCH_JIT=0 \
    NVTE_TORCH_COMPILE=0 \
    NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
    NVTE_FUSED_ATTN=0 \
        run_pytest test_numerics "${tests_root}/test_numerics.py" \
        -k "not (test_gpt_accuracy or test_mha_accuracy or test_dpa_accuracy or test_gpt_checkpointing or test_gpt_cuda_graph or test_grouped_linear_accuracy or test_grouped_gemm or test_noncontiguous or test_rmsnorm_accuracy or test_layernorm_accuracy or test_linear_accuracy or test_layernorm_linear_accuracy or test_layernorm_mlp_accuracy or test_transformer_layer_hidden_states_format)" \
        --no-header

    echo "[SKIP] MUSA: test_jit.py requires unsupported TorchDynamo/JIT fusion paths"
    run_pytest test_fused_rope "${tests_root}/test_fused_rope.py"
    run_pytest test_nvfp4 "${tests_root}/nvfp4"
    run_pytest test_quantized_tensor "${tests_root}/test_quantized_tensor.py"
    run_pytest test_float8blockwisetensor "${tests_root}/test_float8blockwisetensor.py"
    run_pytest test_float8_blockwise_scaling_exact \
        "${tests_root}/test_float8_blockwise_scaling_exact.py"
    run_pytest test_float8_blockwise_gemm_exact \
        "${tests_root}/test_float8_blockwise_gemm_exact.py"
    echo "[SKIP] MUSA: test_gqa.py requires unsupported TorchDynamo/Inductor paths"

    run_pytest test_fused_optimizer "${tests_root}/test_fused_optimizer.py" \
        -k "not test_bf16_exp_avg_and_exp_avg_sq"

    run_pytest test_multi_tensor \
        "${tests_root}/test_multi_tensor.py::test_multi_tensor_compute_scale_and_scale_inv" \
        --no-header

    run_pytest test_fusible_ops "${tests_root}/test_fusible_ops.py" \
        -k "not (test_layer_norm or test_rmsnorm or test_layernorm_mlp or test_grouped_mlp or test_custom or test_l2normalization)"

    run_pytest test_permutation "${tests_root}/test_permutation.py" \
        --deselect "tests/pytorch/test_permutation.py::test_permutation_mask_map[" \
        --deselect "tests/pytorch/test_permutation.py::test_permutation_and_padding_mask_map[" \
        --deselect "tests/pytorch/test_permutation.py::test_permutation_and_padding_with_merging_probs[" \
        --deselect "tests/pytorch/test_permutation.py::test_permutation_mask_map_alongside_probs[" \
        --deselect "tests/pytorch/test_permutation.py::test_permutation_mask_map_topk1_no_probs[" \
        --deselect "tests/pytorch/test_permutation.py::test_chunk_permutation["

    run_without_cuda_compat test_cpu_offloading "${tests_root}/test_cpu_offloading.py"
    NVTE_FLASH_ATTN=0 NVTE_CPU_OFFLOAD_V1=1 \
        run_without_cuda_compat test_cpu_offloading_v1 "${tests_root}/test_cpu_offloading_v1.py"
    run_without_cuda_compat test_attention "${tests_root}/attention/test_attention.py"
    run_without_cuda_compat test_kv_cache "${tests_root}/attention/test_kv_cache.py"
    run_without_cuda_compat test_hf_integration "${tests_root}/test_hf_integration.py"
    NVTE_TEST_CHECKPOINT_ARTIFACT_PATH="${TE_PATH}/artifacts/tests/pytorch/test_checkpoint" \
        run_without_cuda_compat test_checkpoint "${tests_root}/test_checkpoint.py"

}

run_distributed() {
    NVTE_FLASH_ATTN=0 \
    NVTE_FUSED_ATTN=0 \
    NVTE_UNFUSED_ATTN=1 \
        run_pytest test_cp_utils \
        "${TE_PATH}/tests/pytorch/attention/test_cp_utils.py"
}

run_onnx() {
    NVTE_UnfusedDPA_Emulate_FP8=1 \
        run_pytest test_onnx_export \
        "${TE_PATH}/tests/pytorch/test_onnx_export.py" \
        -k "test_export_layernorm_recipe or test_export_layernorm_zero_centered_gamma or test_export_layernorm_normalization or (test_export_core_attention and not arbitrary) or test_export_ctx_manager" \
        --no-header
}

GROUP=${1:-}
if [ -z "${GROUP}" ] && [ -n "${TE_TEST_GROUP_JSON:-}" ]; then
    GROUP=$(python3 -c \
        'import json, os; print(json.loads(os.environ["TE_TEST_GROUP_JSON"])["name"])')
fi

case "${GROUP}" in
    debug | pytorch_debug)
        run_debug
        ;;
    pytorch | pytorch_unittest)
        run_pytorch
        ;;
    distributed | pytorch_distributed_utils)
        run_distributed
        ;;
    onnx | pytorch_onnx_unittest)
        run_onnx
        ;;
    *)
        echo "Usage: $0 {debug|pytorch|distributed|onnx}" >&2
        exit 2
        ;;
esac

if [ "${FAILED}" -ne 0 ]; then
    echo "One or more MUSA test steps failed." >&2
    exit 1
fi

echo "All selected MUSA test steps passed."
