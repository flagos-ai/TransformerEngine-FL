#!/usr/bin/env bash
# PPU Backend Test Configuration

set -euo pipefail

readonly PLATFORM_NAME="ppu"
readonly PLATFORM_DISPLAY_NAME="PPU ZW810E"
readonly PLATFORM_TYPE="shell_launcher"
readonly PLATFORM_NPROC_PER_NODE=2
readonly PLATFORM_DEVICE_ENV_VAR="CUDA_VISIBLE_DEVICES"
readonly PLATFORM_UNIT_TIMEOUT=14400
readonly PLATFORM_INTEGRATION_TIMEOUT=1800
readonly PLATFORM_UNIT_TEST_PATHS=(
    "tests/plugin/backend/ppu"
    "tests/plugin/backend/reference"
    "tests/plugin/backend/flagos"
)
readonly PLATFORM_INTEGRATION_TEST_SCRIPT="qa/L1_pytorch_mcore_integration/test.sh"
readonly PLATFORM_PYTEST_UNIT_MARKERS="not slow and not integration"
readonly PLATFORM_PYTEST_INTEGRATION_MARKERS="integration"
readonly PLATFORM_PYTEST_EXTRA_ARGS="--tb=short --verbose"
readonly PLATFORM_PYTEST_DESELECT=""
readonly PLATFORM_COVERAGE_ENABLED=true

# PPU-only scope. Exclusions are known backend gaps and never count as passes.
PPU_DEBUG_TARGETS=(tests/pytorch/debug/test_sanity.py tests/pytorch/debug/test_config.py tests/pytorch/debug/test_numerics.py tests/pytorch/debug/test_log.py tests/pytorch/debug/test_api_features.py tests/pytorch/debug/test_perf.py)
PPU_UNITTEST_TARGETS=(tests/pytorch/test_sanity.py tests/pytorch/test_recipe.py tests/pytorch/test_deferred_init.py tests/pytorch/test_numerics.py tests/pytorch/test_cuda_graphs.py tests/pytorch/test_jit.py tests/pytorch/nvfp4 tests/pytorch/test_quantized_tensor.py tests/pytorch/test_float8blockwisetensor.py tests/pytorch/test_float8_blockwise_scaling_exact.py tests/pytorch/test_float8_blockwise_gemm_exact.py tests/pytorch/test_gqa.py tests/pytorch/test_fused_optimizer.py tests/pytorch/test_multi_tensor.py tests/pytorch/test_fusible_ops.py tests/pytorch/test_permutation.py tests/pytorch/test_parallel_cross_entropy.py tests/pytorch/test_cpu_offloading.py tests/pytorch/test_cpu_offloading_v1.py tests/pytorch/attention/test_attention.py tests/pytorch/attention/test_kv_cache.py tests/pytorch/test_hf_integration.py tests/plugin)
PPU_DISTRIBUTED_TARGETS=(tests/pytorch/distributed/test_numerics.py tests/pytorch/distributed/test_numerics_exact.py tests/pytorch/distributed/test_torch_fsdp2.py tests/pytorch/distributed/test_cast_master_weights_to_fp8.py tests/pytorch/attention/test_cp_utils.py)
PPU_ONNX_TARGETS=(tests/pytorch/test_onnx_export.py)

# Entire-file exclusions are recorded separately from pytest results. Restore
# each target after the PPU backend supports the corresponding operation.
PPU_IGNORE_TARGETS=(
  tests/pytorch/test_permutation.py
  tests/pytorch/attention/test_attention.py
  tests/pytorch/attention/test_kv_cache.py
)
declare -A PPU_IGNORE_REASON=(
  [tests/pytorch/test_permutation.py]='Legacy PPU filter deselects the entire file; underlying failures require triage'
  [tests/pytorch/attention/test_attention.py]='Legacy PPU filter deselects the entire file; underlying failures require triage'
  [tests/pytorch/attention/test_kv_cache.py]='Legacy PPU filter deselects the entire file; underlying failures require triage'
)
declare -A PPU_IGNORE_OWNER=(
  [tests/pytorch/test_permutation.py]='PPU/plugin team'
  [tests/pytorch/attention/test_attention.py]='PPU/plugin team'
  [tests/pytorch/attention/test_kv_cache.py]='PPU/plugin team'
)
declare -A PPU_IGNORE_RESTORE=(
  [tests/pytorch/test_permutation.py]='Capture failure evidence, assign the defect, and restore after the target passes'
  [tests/pytorch/attention/test_attention.py]='Capture failure evidence, assign the defect, and restore after the target passes'
  [tests/pytorch/attention/test_kv_cache.py]='Capture failure evidence, assign the defect, and restore after the target passes'
)

# Partial known failures, owned by the PPU/plugin team; restore when fixed upstream.
declare -A PPU_IGNORE_K=(
  [tests/pytorch/debug/test_sanity.py]='test_sanity_debug and fake_quant and False and (mha_attention or transformer_layer)'
  [tests/pytorch/debug/test_api_features.py]='test_per_tensor_scaling or test_fake_quant or test_statistics_collection or test_statistics_multi_run'
  [tests/pytorch/test_sanity.py]='test_sanity_grouped_linear and (1-dtype or 2-dtype)'
  [tests/pytorch/test_numerics.py]='(test_gpt_cuda_graph and (dtype1 or dtype2)) or (test_layernorm_accuracy and (dtype1 or dtype2)) or (test_transformer_layer_hidden_states_format and 126m-2-dtype)'
  [tests/pytorch/test_cuda_graphs.py]='test_make_graphed_callables_with_kwargs or (test_make_graphed_callables and (transformer or mha) and (dtype1 or dtype2)) or (test_make_graphed_callables_with_dot_product_attention and (dtype1 or dtype2))'
  [tests/pytorch/test_fused_optimizer.py]='TestFusedSGD or test_bf16_exp_avg_and_exp_avg_sq'
  [tests/pytorch/test_multi_tensor.py]='test_multi_tensor_compute_scale_and_scale_inv'
  [tests/pytorch/test_fusible_ops.py]='test_grouped_linear or test_backward_add_rmsnorm or test_grouped_mlp or (test_basic_linear and not test_basic_linear_quantized and (in_shape0 or in_shape1 or in_shape2)) or (test_layer_norm and not test_layer_norm_autocast and (dtype1 or dtype2)) or (test_rmsnorm and True and (dtype1 or dtype2)) or (test_activation and dtype1 and (qgelu or qgeglu or (glu and not (geglu or reglu or sreglu or swiglu)))) or (test_clamped_swiglu and dtype1) or (test_dropout and (dtype1 or dtype2) and shape2 and True and 0.5) or (test_forward_linear_bias_activation and (dtype1 or dtype2) and (in_shape0 or in_shape2)) or (test_forward_linear_bias_add and dtype1 and True) or (test_forward_linear_scale_add and dtype1 and (2.5 or 3.5)) or (TestCheckpointing and test_linear and True)'
  [tests/pytorch/test_cpu_offloading.py]='(test_memory and (multihead_attention or transformer_layer)) or (test_numerics and transformer_layer) or (test_numerics and UnfusedAttention and True-multihead_attention)'
  [tests/pytorch/test_cpu_offloading_v1.py]='test_cpu_offload and (multihead_attention or transformer_layer)'
)

# Exact plugin-node quarantines. Node paths are relative to tests/plugin/.
# Literal entries only, one quoted node|reason per line; shared pytest collection
# reads this array for both the outer runner and fresh dispatch subprocesses.
readonly PLUGIN_IGNORE_TEST_NODES=(
    'plugin/triton_dispatch_numerics_cases.py::test_dispatch_entry_does_not_shrink_the_compiled_graph|torch.compile/FlagTree autotune grid compatibility failure; restore after compiled tracing passes'
    'backend/flagos/test_optimizer_numerical.py::test_multi_tensor_adam_matches_formula[False-0.0-0-0]|repeated Adam numerical mismatch; first-call cause unconfirmed; restore after fresh-process numerics pass'
)
