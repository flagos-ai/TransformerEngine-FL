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
readonly PLATFORM_COVERAGE_ENABLED=true

# Test scope is selected here; per-test quarantines belong in
# .github/configs/ppu.yml so the planner and every pytest child share one
# source of truth.
PPU_DEBUG_TARGETS=(tests/pytorch/debug/test_sanity.py tests/pytorch/debug/test_config.py tests/pytorch/debug/test_numerics.py tests/pytorch/debug/test_log.py tests/pytorch/debug/test_api_features.py tests/pytorch/debug/test_perf.py)
PPU_UNITTEST_TARGETS=(tests/pytorch/test_sanity.py tests/pytorch/test_recipe.py tests/pytorch/test_deferred_init.py tests/pytorch/test_numerics.py tests/pytorch/test_cuda_graphs.py tests/pytorch/test_jit.py tests/pytorch/nvfp4 tests/pytorch/test_quantized_tensor.py tests/pytorch/test_float8blockwisetensor.py tests/pytorch/test_float8_blockwise_scaling_exact.py tests/pytorch/test_float8_blockwise_gemm_exact.py tests/pytorch/test_gqa.py tests/pytorch/test_fused_optimizer.py tests/pytorch/test_multi_tensor.py tests/pytorch/test_fusible_ops.py tests/pytorch/test_permutation.py tests/pytorch/test_parallel_cross_entropy.py tests/pytorch/test_cpu_offloading.py tests/pytorch/test_cpu_offloading_v1.py tests/pytorch/attention/test_attention.py tests/pytorch/attention/test_kv_cache.py tests/pytorch/test_hf_integration.py tests/plugin)
PPU_DISTRIBUTED_TARGETS=(tests/pytorch/distributed/test_numerics.py tests/pytorch/distributed/test_numerics_exact.py tests/pytorch/distributed/test_torch_fsdp2.py tests/pytorch/distributed/test_cast_master_weights_to_fp8.py tests/pytorch/attention/test_cp_utils.py)
PPU_ONNX_TARGETS=(tests/pytorch/test_onnx_export.py)
