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
