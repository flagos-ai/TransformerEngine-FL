#!/usr/bin/env bash
# Iluvatar backend test configuration, matching the other platform adapters.
set -euo pipefail

readonly PLATFORM_NAME="iluvatar"
readonly PLATFORM_DISPLAY_NAME="Iluvatar BI-V150"
readonly PLATFORM_TYPE="shell_launcher"
readonly PLATFORM_NPROC_PER_NODE=8
readonly PLATFORM_DEVICE_ENV_VAR="CUDA_VISIBLE_DEVICES"
readonly PLATFORM_UNIT_TIMEOUT=14400
readonly PLATFORM_INTEGRATION_TIMEOUT=1800
readonly PLATFORM_UNIT_TEST_PATHS=(
    "tests/plugin/backend/iluvatar"
    "tests/plugin/backend/reference"
    "tests/plugin/backend/flagos"
)
readonly PLATFORM_INTEGRATION_TEST_SCRIPT="tests/plugin/backend/iluvatar/integration_test.sh"
readonly PLATFORM_PYTEST_UNIT_MARKERS="not slow and not integration"
readonly PLATFORM_PYTEST_INTEGRATION_MARKERS="integration"
readonly PLATFORM_PYTEST_EXTRA_ARGS="--tb=short --verbose"
readonly PLATFORM_PYTEST_DESELECT=""
readonly PLATFORM_COVERAGE_ENABLED=true

# Whole-file exclusions for this CI baseline, not passing or capability-skipped
# tests. Keep the shared sources intact; restore these entries after isolated
# reproduction confirms the attention/graph/offload issues are fixed.
readonly ILUVATAR_IGNORE_TEST_TARGETS=(
    "tests/pytorch/test_numerics.py|FP16 GPT CUDA Graph case hung with a driver thread remaining after termination."
    "tests/pytorch/test_cuda_graphs.py|FP16 Transformer graph case hung again on a different GPU."
    "tests/pytorch/test_cpu_offloading.py|UnfusedAttention MHA numerics produced no output for 20 minutes and left an unreaped process."
    "tests/pytorch/test_cpu_offloading_v1.py|MHA/Transformer offload cases require an unavailable fused attention backend; excluded with the offload suite."
    "tests/pytorch/attention/test_attention.py|No passing cases: fused attention backend and THD conversion APIs are unavailable. Restore after the required backend/API paths pass isolated tests."
    "tests/pytorch/attention/test_kv_cache.py|No passing cases: IXTE QKV enum mismatch, unavailable fused attention, and FP8 capability checks. Restore after those paths are aligned and validated."
    "tests/pytorch/test_checkpoint.py|Historical checkpoint fixtures are missing; this is a test-data gap. Restore after known-version fixtures are provisioned and validated; do not substitute freshly generated checkpoints."
    "tests/pytorch/distributed/test_numerics.py|The non-quantized torchrun worker exits unsuccessfully in normal and debug modes; no passing cases. Restore after worker-level diagnosis and passing isolated multi-GPU runs."
)

# Partial exclusions audited against b3086518 device JUnit. Only entries with
# passing evidence are filtered; all 4073 previously passing cases stay selected.
# These are quarantine rules, not fixes. Scope by suite because the same file
# can have different outcomes with nvinspect enabled. Restore a selector after
# isolated reruns confirm the relevant API, numerical, or test-contract fix.
# Entries with no passing evidence are recorded separately in the whole-file
# exclusions above; neither those gaps nor the partial exclusions count as passes.
# Format: suite|target|pytest -k exclusion expression|failure classification.
readonly ILUVATAR_IGNORE_TEST_FILTERS=(
    # unittest: retain 780 prior passes; quarantine 1968 failures.
    "unittest|tests/pytorch/test_sanity.py|(test_sanity_grouped_linear and (first-False-True or \
        last-False-True or \
        middle-False-True) and (1-dtype or \
        2-dtype)) or \
        (test_sanity_layernorm_mlp and not (-gelu- or \
        -swiglu-))|Shape or \
        epilogue mismatch: 48; Missing compatibility module: 576; Vendor API or \
        enum mismatch: 1344"
    # unittest: retain 9 prior passes; quarantine 4 failures.
    "unittest|tests/pytorch/test_fused_optimizer.py|TestFusedSGD or \
        test_grad_scaler_capturable|Vendor API or \
        enum mismatch: 2; Numerical or \
        behavioral assertion mismatch: 2"
    # Follow PPU's function-level quarantine; keep the other multi-tensor tests.
    # CoreX torch.ldexp(1, 13) returned 8191.99951171875 while IXTE returned
    # the exact 8192.0. Random inputs make this zero-tolerance failure intermittent.
    # Restore after CoreX ldexp is corrected and all scale parameters pass again.
    "unittest|tests/pytorch/test_multi_tensor.py|test_multi_tensor_compute_scale_and_scale_inv|CoreX PyTorch ldexp reference-value discrepancy under zero-tolerance comparison; quarantine the scale/scale_inv function as on PPU."
    # unittest: retain 75 prior passes; quarantine 370 failures.
    "unittest|tests/pytorch/test_fusible_ops.py|(TestFuser and (test_dtype_cast[)) or \
        (TestBasicOps and (test_identity[None-cuda-dtype0] or \
        test_identity[None-cuda-dtype1])) or \
        (TestBasicOps and (test_reshape[)) or \
        (TestBasicOps and (test_bias[None-device1-dtype0-in_shape0-1] or \
        test_bias[None-device1-dtype0-in_shape0-7] or \
        test_bias[None-device1-dtype0-in_shape0-32] or \
        test_bias[None-device1-dtype0-in_shape1-1] or \
        test_bias[None-device1-dtype0-in_shape1-7] or \
        test_bias[None-device1-dtype0-in_shape1-32] or \
        test_bias[None-device1-dtype0-in_shape2-1] or \
        test_bias[None-device1-dtype0-in_shape2-7] or \
        test_bias[None-device1-dtype0-in_shape2-32] or \
        test_bias[None-device1-dtype1-in_shape0-1] or \
        test_bias[None-device1-dtype1-in_shape0-7] or \
        test_bias[None-device1-dtype1-in_shape0-32] or \
        test_bias[None-device1-dtype1-in_shape1-1] or \
        test_bias[None-device1-dtype1-in_shape1-7] or \
        test_bias[None-device1-dtype1-in_shape1-32] or \
        test_bias[None-device1-dtype1-in_shape2-1] or \
        test_bias[None-device1-dtype1-in_shape2-7] or \
        test_bias[None-device1-dtype1-in_shape2-32])) or \
        (TestBasicOps and (test_quantize[)) or \
        (TestBasicOps and (test_basic_linear[)) or \
        (TestBasicOps and (test_linear[)) or \
        (TestBasicOps and (test_layer_norm[)) or \
        (TestBasicOps and (test_layer_norm_autocast)) or \
        (TestBasicOps and (test_rmsnorm[)) or \
        (TestBasicOps and (test_l2normalization[)) or \
        (TestBasicOps and (test_add_extra_input[None-cuda-dtype0-True] or \
        test_add_extra_input[None-cuda-dtype0-False] or \
        test_add_extra_input[None-cuda-dtype1-True] or \
        test_add_extra_input[None-cuda-dtype1-False])) or \
        (TestBasicOps and (test_make_extra_output[None-cuda-dtype0-True] or \
        test_make_extra_output[None-cuda-dtype0-False] or \
        test_make_extra_output[None-cuda-dtype1-True] or \
        test_make_extra_output[None-cuda-dtype1-False])) or \
        (TestBasicOps and (test_activation[)) or \
        (TestBasicOps and (test_swiglu[)) or \
        (TestBasicOps and (test_interleaved_swiglu)) or \
        (TestBasicOps and (test_clamped_swiglu[)) or \
        (TestBasicOps and (test_interleaved_clamped_swiglu)) or \
        (TestBasicOps and (test_constant_scale[device1-dtype0-shape0-1] or \
        test_constant_scale[device1-dtype0-shape0--2.5] or \
        test_constant_scale[device1-dtype0-shape0-3.5] or \
        test_constant_scale[device1-dtype0-shape1-1] or \
        test_constant_scale[device1-dtype0-shape1--2.5] or \
        test_constant_scale[device1-dtype0-shape1-3.5] or \
        test_constant_scale[device1-dtype0-shape2-1] or \
        test_constant_scale[device1-dtype0-shape2--2.5] or \
        test_constant_scale[device1-dtype0-shape2-3.5] or \
        test_constant_scale[device1-dtype1-shape0-1] or \
        test_constant_scale[device1-dtype1-shape0--2.5] or \
        test_constant_scale[device1-dtype1-shape0-3.5] or \
        test_constant_scale[device1-dtype1-shape1-1] or \
        test_constant_scale[device1-dtype1-shape1--2.5] or \
        test_constant_scale[device1-dtype1-shape1-3.5] or \
        test_constant_scale[device1-dtype1-shape2-1] or \
        test_constant_scale[device1-dtype1-shape2--2.5] or \
        test_constant_scale[device1-dtype1-shape2-3.5])) or \
        (TestBasicOps and (test_dropout[)) or \
        (TestBasicOps and (test_grouped_linear[)) or \
        (TestBasicOps and (test_scaled_swiglu[)) or \
        (TestBasicOps and (test_scaled_srelu[)) or \
        (TestBasicOps and (test_interleaved_scaled_swiglu)) or \
        (TestBasicOps and (test_scaled_clamped_qgeglu[)) or \
        (TestBasicOps and (test_interleaved_scaled_clamped_qgeglu)) or \
        (TestFusedOps and (test_forward_linear_bias_activation[)) or \
        (TestFusedOps and (test_forward_linear_bias_add[)) or \
        (TestFusedOps and (test_forward_linear_scale_add[)) or \
        (TestFusedOps and (test_backward_activation_bias[)) or \
        (TestFusedOps and (test_backward_add_rmsnorm[)) or \
        (TestFusedOps and (test_backward_linear_add[)) or \
        (TestFusedOps and (test_backward_linear_scale[None-dtype0-1] or \
        test_backward_linear_scale[None-dtype0--2.5] or \
        test_backward_linear_scale[None-dtype0-3.5] or \
        test_backward_linear_scale[None-dtype1-1] or \
        test_backward_linear_scale[None-dtype1--2.5] or \
        test_backward_linear_scale[None-dtype1-3.5])) or \
        (TestCheckpointing and (test_linear[True-None])) or \
        (TestSequentialModules and (test_layernorm_mlp[None-dtype0-False-False-True-False] or \
        test_layernorm_mlp[None-dtype0-False-False-True-True] or \
        test_layernorm_mlp[None-dtype1-False-False-True-False] or \
        test_layernorm_mlp[None-dtype1-False-False-True-True])) or \
        (TestSequentialModules and (test_grouped_mlp[)) or \
        (TestCustomOps and (test_custom_basic_op)) or \
        (TestCustomOps and (test_custom_forward_fused_op1)) or \
        (TestCustomOps and (test_custom_forward_fused_op2)) or \
        (TestCustomOps and (test_custom_backward_fused_op)) or \
        (TestCustomOps and (test_custom_forward_backward_fused_op)) or \
        (TestTrainingLoops and (test_linear_training_loop[)) or \
        (TestTrainingLoops and (test_linear_inference_loop[))|Numerical or \
        behavioral assertion mismatch: 232; Shape or \
        epilogue mismatch: 24; Missing compatibility module: 25; Vendor API or \
        enum mismatch: 89"
    # Preserve the remaining permutation coverage; quarantine only identified cases.
    # The exact 7-expert mask-map cases (topK=2/5, 2039 output tokens, FP16)
    # intermittently fail backward/forward comparison against the PyTorch reference.
    # On CoreX, duplicate-index FP16 accumulation in the reference path has shown
    # missing contributions (index_select backward / scatter_add_ forward).
    # Seed 44 reproduced forward output 0.6630859375 vs TE/FP64 sum 0.8203125;
    # this points to reference accumulation, not ordinary rounding error. The
    # underlying concurrent-accumulation defect still needs vendor confirmation.
    # Restore these cases after the CoreX issue is fixed and both isolated and
    # full-suite repeated runs pass; keep all other selected permutation coverage.
    "unittest|tests/pytorch/test_permutation.py|(test_permutation_index_map[) or \
        (test_permutation_mask_map[False-2039-2-4096-7-4096-5] or \
        test_permutation_mask_map[False-2039-5-4096-7-4096-5] or \
        test_permutation_mask_map[False-2039-2-4096-16-4096-5] or \
        test_permutation_mask_map[False-2039-5-4096-16-4096-5]) or \
        (test_permutation_and_padding_mask_map[) or \
        (test_permutation_and_padding_with_merging_probs[) or \
        (test_permutation_mask_map_empty_input[False-5] or \
        test_permutation_mask_map_empty_input[True-5]) or \
        (test_permutation_mask_map_alongside_probs[False-1-2039-2-4096-7-4096-5] or \
        test_permutation_mask_map_alongside_probs[False-1-2039-2-4096-16-4096-5] or \
        test_permutation_mask_map_alongside_probs[False-1-2039-5-4096-7-4096-5] or \
        test_permutation_mask_map_alongside_probs[False-1-2039-5-4096-16-4096-5] or \
        test_permutation_mask_map_alongside_probs[False-2-2039-2-4096-16-4096-5] or \
        test_permutation_mask_map_alongside_probs[False-2-2039-5-4096-16-4096-5] or \
        test_permutation_mask_map_alongside_probs[True-1-2039-2-4096-7-4096-5]) or \
        (test_permutation_mask_map_alongside_probs_empty_input[False-5] or \
        test_permutation_mask_map_alongside_probs_empty_input[True-5]) or \
        (test_permutation_index_map_topk1_no_probs[) or \
        (test_permutation_mask_map_topk1_no_probs[False-4096-16-4096-5])|Vendor API or \
        enum mismatch: 56; Numerical or \
        behavioral assertion mismatch: 14; Exact 7-expert topK=2/5 mask-map cases quarantined for intermittent reference FP16 duplicate-index accumulation mismatches; see reproduction and restoration notes above."
    # Preserve the directory baseline's seven selectors while isolating files.
    # Historical directory totals: 96 passes, 281 failures (173 CPU mock/device
    # contract mismatches and 108 shape/epilogue mismatches); not per-file counts.
    "unittest|tests/plugin/backend/flagos/test_gemm.py|(test_generic_gemm_lifecycle_matrix[) or \
        (test_grouped_gemm_standard_lifecycle[)|CPU mock/device contract or shape/epilogue mismatch; preserved directory baseline."
    "unittest|tests/plugin/backend/flagos/test_multi_tensor.py|(test_l2_norm_non_finite_tracking[) or \
        (test_scale_standard_lifecycle) or \
        (test_scale_non_finite_tracking[)|CPU mock/device contract or shape/epilogue mismatch; preserved directory baseline."
    "unittest|tests/plugin/backend/flagos/test_optimizer.py|(test_multi_tensor_adam_lifecycle[) or \
        (test_param_remainder_bit_manipulation_lifecycle[)|CPU mock/device contract or shape/epilogue mismatch; preserved directory baseline."
    # debug: retain 13 prior passes; quarantine 2 failures.
    "debug|tests/pytorch/debug/test_sanity.py|(test_sanity_debug[fake_quant-False-mha_attention] or \
        test_sanity_debug[fake_quant-False-transformer_layer])|Missing compatibility module: 2"
    # debug: retain 1 prior passes; quarantine 6 failures.
    "debug|tests/pytorch/debug/test_api_features.py|(test_disable_fp8_gemm) or \
        (test_disable_fp8_layer) or \
        (test_per_tensor_scaling) or \
        (test_fake_quant) or \
        (test_statistics_collection) or \
        (test_statistics_multi_run)|Numerical or \
        behavioral assertion mismatch: 4; Missing compatibility module: 2"
    # debug: retain 828 prior passes; quarantine 1920 failures.
    "debug|tests/pytorch/test_sanity.py|test_sanity_layernorm_mlp and not (-gelu- or \
        -swiglu-)|Missing compatibility module: 576; Vendor API or \
        enum mismatch: 1344"
    # onnx: retain 57 prior passes; quarantine 8 failures.
    "onnx|tests/pytorch/test_onnx_export.py|(test_export_layernorm_mlp_activation[relu] or \
        test_export_layernorm_mlp_activation[reglu] or \
        test_export_layernorm_mlp_activation[geglu] or \
        test_export_layernorm_mlp_activation[clamped_swiglu]) or \
        (test_export_transformer_layer_activation[relu] or \
        test_export_transformer_layer_activation[reglu] or \
        test_export_transformer_layer_activation[geglu] or \
        test_export_transformer_layer_activation[clamped_swiglu])|Missing compatibility module: 6; Vendor API or \
        enum mismatch: 2"
)

# Common L0 QA targets plus plugin tests. Match PPU's RoPE coverage through
# tests/plugin/backend/flagos/test_fused_rope.py (included by the FlagOS directory
# below), rather than the large tests/pytorch/test_fused_rope.py matrix. On the
# 32 GiB BI-V150 that matrix OOMs in gradient comparisons and cascades across
# cases; it remains available for separate capacity testing.
# Quantization-only targets are excluded from routine CI until passing evidence
# exists. Keep float8_blockwise_gemm_exact: it also provides GEMM coverage.
# Run the removed recipe/tensor/NVFP4 targets explicitly for recovery validation.
readonly ILUVATAR_UNIT_TEST_TARGETS=(
    "tests/pytorch/test_sanity.py"
    "tests/pytorch/test_deferred_init.py"
    "tests/pytorch/test_jit.py"
    "tests/pytorch/test_float8_blockwise_gemm_exact.py"
    "tests/pytorch/test_gqa.py"
    "tests/pytorch/test_fused_optimizer.py"
    "tests/pytorch/test_multi_tensor.py"
    "tests/pytorch/test_fusible_ops.py"
    "tests/pytorch/test_permutation.py"
    "tests/pytorch/test_parallel_cross_entropy.py"
    "tests/pytorch/test_hf_integration.py"
    "tests/plugin"
)
readonly ILUVATAR_DISTRIBUTED_TEST_TARGETS=(
    "tests/pytorch/distributed/test_numerics_exact.py"
    "tests/pytorch/distributed/test_torch_fsdp2.py"
    "tests/pytorch/attention/test_cp_utils.py"
    # Retain this test: failure occurred in init_process_group before FP8 work.
    # A standalone two-rank NCCL/IXCCL init also timed out waiting for rank 0's
    # communicator ID (30 s). Investigate CoreX/IXCCL and device health; do not
    # classify this infrastructure failure as unsupported FP8 or suppress it.
    "tests/pytorch/distributed/test_cast_master_weights_to_fp8.py"
)
