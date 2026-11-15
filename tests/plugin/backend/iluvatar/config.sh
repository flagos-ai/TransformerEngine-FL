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
readonly PLATFORM_COVERAGE_ENABLED=true

# Test scope is selected by the platform YAML. Per-test quarantines belong in
# .github/configs/iluvatar.yml so the planner and every pytest child share one
# source of truth.
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
    "tests/pytorch/distributed/test_cast_master_weights_to_fp8.py"
)

# Parameter-matrix exclusions. `ignored_tests` in iluvatar.yml only supports
# whole functions or exact parameter ids, so the partial `-k` filters from the
# pre-centralization launcher live here, keyed by "<suite>:<target>" and applied
# by the launcher as `-k "not (...)"`. The suite prefix keeps the original
# scoping: unittest quarantined part of test_sanity_grouped_linear, while the
# debug suite never carried that rule.
declare -A ILUVATAR_TARGET_EXCLUDE_K=(
  [unittest:tests/pytorch/test_sanity.py]='(test_sanity_grouped_linear and (first-False-True or last-False-True or middle-False-True) and (1-dtype or 2-dtype)) or (test_sanity_layernorm_mlp and not (-gelu- or -swiglu-))'
  [debug:tests/pytorch/test_sanity.py]='test_sanity_layernorm_mlp and not (-gelu- or -swiglu-)'
  [unittest:tests/pytorch/test_fusible_ops.py]='(TestFuser and (test_dtype_cast[)) or (TestBasicOps and (test_identity[None-cuda-dtype0] or test_identity[None-cuda-dtype1])) or (TestBasicOps and (test_reshape[)) or (TestBasicOps and (test_bias[None-device1-dtype0-in_shape0-1] or test_bias[None-device1-dtype0-in_shape0-7] or test_bias[None-device1-dtype0-in_shape0-32] or test_bias[None-device1-dtype0-in_shape1-1] or test_bias[None-device1-dtype0-in_shape1-7] or test_bias[None-device1-dtype0-in_shape1-32] or test_bias[None-device1-dtype0-in_shape2-1] or test_bias[None-device1-dtype0-in_shape2-7] or test_bias[None-device1-dtype0-in_shape2-32] or test_bias[None-device1-dtype1-in_shape0-1] or test_bias[None-device1-dtype1-in_shape0-7] or test_bias[None-device1-dtype1-in_shape0-32] or test_bias[None-device1-dtype1-in_shape1-1] or test_bias[None-device1-dtype1-in_shape1-7] or test_bias[None-device1-dtype1-in_shape1-32] or test_bias[None-device1-dtype1-in_shape2-1] or test_bias[None-device1-dtype1-in_shape2-7] or test_bias[None-device1-dtype1-in_shape2-32])) or (TestBasicOps and (test_quantize[)) or (TestBasicOps and (test_basic_linear[)) or (TestBasicOps and (test_linear[)) or (TestBasicOps and (test_layer_norm[)) or (TestBasicOps and (test_layer_norm_autocast)) or (TestBasicOps and (test_rmsnorm[)) or (TestBasicOps and (test_l2normalization[)) or (TestBasicOps and (test_add_extra_input[None-cuda-dtype0-True] or test_add_extra_input[None-cuda-dtype0-False] or test_add_extra_input[None-cuda-dtype1-True] or test_add_extra_input[None-cuda-dtype1-False])) or (TestBasicOps and (test_make_extra_output[None-cuda-dtype0-True] or test_make_extra_output[None-cuda-dtype0-False] or test_make_extra_output[None-cuda-dtype1-True] or test_make_extra_output[None-cuda-dtype1-False])) or (TestBasicOps and (test_activation[)) or (TestBasicOps and (test_swiglu[)) or (TestBasicOps and (test_interleaved_swiglu)) or (TestBasicOps and (test_clamped_swiglu[)) or (TestBasicOps and (test_interleaved_clamped_swiglu)) or (TestBasicOps and (test_constant_scale[device1-dtype0-shape0-1] or test_constant_scale[device1-dtype0-shape0--2.5] or test_constant_scale[device1-dtype0-shape0-3.5] or test_constant_scale[device1-dtype0-shape1-1] or test_constant_scale[device1-dtype0-shape1--2.5] or test_constant_scale[device1-dtype0-shape1-3.5] or test_constant_scale[device1-dtype0-shape2-1] or test_constant_scale[device1-dtype0-shape2--2.5] or test_constant_scale[device1-dtype0-shape2-3.5] or test_constant_scale[device1-dtype1-shape0-1] or test_constant_scale[device1-dtype1-shape0--2.5] or test_constant_scale[device1-dtype1-shape0-3.5] or test_constant_scale[device1-dtype1-shape1-1] or test_constant_scale[device1-dtype1-shape1--2.5] or test_constant_scale[device1-dtype1-shape1-3.5] or test_constant_scale[device1-dtype1-shape2-1] or test_constant_scale[device1-dtype1-shape2--2.5] or test_constant_scale[device1-dtype1-shape2-3.5])) or (TestBasicOps and (test_dropout[)) or (TestBasicOps and (test_grouped_linear[)) or (TestBasicOps and (test_scaled_swiglu[)) or (TestBasicOps and (test_scaled_srelu[)) or (TestBasicOps and (test_interleaved_scaled_swiglu)) or (TestBasicOps and (test_scaled_clamped_qgeglu[)) or (TestBasicOps and (test_interleaved_scaled_clamped_qgeglu)) or (TestFusedOps and (test_forward_linear_bias_activation[)) or (TestFusedOps and (test_forward_linear_bias_add[)) or (TestFusedOps and (test_forward_linear_scale_add[)) or (TestFusedOps and (test_backward_activation_bias[)) or (TestFusedOps and (test_backward_add_rmsnorm[)) or (TestFusedOps and (test_backward_linear_add[)) or (TestFusedOps and (test_backward_linear_scale[None-dtype0-1] or test_backward_linear_scale[None-dtype0--2.5] or test_backward_linear_scale[None-dtype0-3.5] or test_backward_linear_scale[None-dtype1-1] or test_backward_linear_scale[None-dtype1--2.5] or test_backward_linear_scale[None-dtype1-3.5])) or (TestCheckpointing and (test_linear[True-None])) or (TestSequentialModules and (test_layernorm_mlp[None-dtype0-False-False-True-False] or test_layernorm_mlp[None-dtype0-False-False-True-True] or test_layernorm_mlp[None-dtype1-False-False-True-False] or test_layernorm_mlp[None-dtype1-False-False-True-True])) or (TestSequentialModules and (test_grouped_mlp[)) or (TestCustomOps and (test_custom_basic_op)) or (TestCustomOps and (test_custom_forward_fused_op1)) or (TestCustomOps and (test_custom_forward_fused_op2)) or (TestCustomOps and (test_custom_backward_fused_op)) or (TestCustomOps and (test_custom_forward_backward_fused_op)) or (TestTrainingLoops and (test_linear_training_loop[)) or (TestTrainingLoops and (test_linear_inference_loop[))'
  [unittest:tests/pytorch/test_permutation.py]='(test_permutation_index_map[) or (test_permutation_mask_map[False-2039-2-4096-7-4096-5] or test_permutation_mask_map[False-2039-5-4096-7-4096-5] or test_permutation_mask_map[False-2039-2-4096-16-4096-5] or test_permutation_mask_map[False-2039-5-4096-16-4096-5] or test_permutation_mask_map[True-2039-2-4096-7-4096-5]) or (test_permutation_and_padding_mask_map[) or (test_permutation_and_padding_with_merging_probs[) or (test_permutation_mask_map_empty_input[False-5] or test_permutation_mask_map_empty_input[True-5]) or (test_permutation_mask_map_alongside_probs[False-1-2039-2-4096-7-4096-5] or test_permutation_mask_map_alongside_probs[False-1-2039-2-4096-16-4096-5] or test_permutation_mask_map_alongside_probs[False-1-2039-5-4096-7-4096-5] or test_permutation_mask_map_alongside_probs[False-1-2039-5-4096-16-4096-5] or test_permutation_mask_map_alongside_probs[False-2-2039-2-4096-7-4096-5] or test_permutation_mask_map_alongside_probs[False-2-2039-5-4096-7-4096-5] or test_permutation_mask_map_alongside_probs[False-2-2039-2-4096-16-4096-5] or test_permutation_mask_map_alongside_probs[False-2-2039-5-4096-16-4096-5] or test_permutation_mask_map_alongside_probs[True-1-2039-2-4096-7-4096-5]) or (test_permutation_mask_map_alongside_probs_empty_input[False-5] or test_permutation_mask_map_alongside_probs_empty_input[True-5]) or (test_permutation_index_map_topk1_no_probs[) or (test_permutation_mask_map_topk1_no_probs[False-4096-16-4096-5])'
  [onnx:tests/pytorch/test_onnx_export.py]='(test_export_layernorm_mlp_activation[relu] or test_export_layernorm_mlp_activation[reglu] or test_export_layernorm_mlp_activation[geglu] or test_export_layernorm_mlp_activation[clamped_swiglu]) or (test_export_transformer_layer_activation[relu] or test_export_transformer_layer_activation[reglu] or test_export_transformer_layer_activation[geglu] or test_export_transformer_layer_activation[clamped_swiglu])'
)
