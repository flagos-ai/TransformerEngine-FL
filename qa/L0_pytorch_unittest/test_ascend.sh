#!/usr/bin/env bash

set -u

: "${TE_PATH:=${GITHUB_WORKSPACE:-$(pwd)}}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L0_pytorch_unittest-ascend}"
mkdir -p "$XML_LOG_DIR"

FAIL=0

test_fail() {
    FAIL=1
    echo "Error: sub-test failed: $1"
}

pytest_command() {
    local use_platform_runner=$1
    local -n out=$2

    if [ "$use_platform_runner" = "true" ] && [ -n "${TE_TEST_PYTEST_COMMAND:-}" ]; then
        # shellcheck disable=SC2206
        out=(${TE_TEST_PYTEST_COMMAND})
    else
        out=(python3 -m pytest)
    fi
}

run_pytest_step() {
    local label=$1
    local junit=$2
    local use_platform_runner=$3
    shift 3

    local cmd=()
    pytest_command "$use_platform_runner" cmd
    cmd+=(-v -s --tb=short "--junitxml=$XML_LOG_DIR/$junit")
    cmd+=("$@")

    echo "-------------------------------------------------------"
    echo "[RUN] Executing: $label"
    "${cmd[@]}" || test_fail "$label"
}

run_pytest_step_with_unfused_attention() {
    local label=$1
    local junit=$2
    shift 2

    if [ -z "${TE_TEST_PYTEST_COMMAND:-}" ]; then
        echo "-------------------------------------------------------"
        echo "[IGNORE] $label: Ascend shared PyTorch tests require the NPU pytest runner"
        return
    fi

    NVTE_FLASH_ATTN=0 \
    NVTE_FUSED_ATTN=0 \
    NVTE_UNFUSED_ATTN=1 \
        run_pytest_step "$label" "$junit" true "$@"
}

# Include both NPU test files and every shared plugin test in the unit job.
TE_TEST_GROUP_JSON='{
  "name": "plugin-unit", "runner": "pytest", "max_retries": 0,
  "pytest_args": ["-v", "-s", "--tb=short"],
  "steps": [{
    "name": "plugin", "targets": ["tests/plugin/"],
    "isolate_files": true, "use_platform_runner": false,
    "ignore": [],
    "unset_env_by_path": {"tests/plugin/plugin/": [
      "TE_FL_ALLOW_VENDORS", "TE_FL_PREFER", "TE_FL_PER_OP", "TE_FL_STRICT"
    ]}
  }]
}' python3 "$TE_PATH/tests/test_utils/run_ci_test_group.py" || test_fail "plugin"

run_pytest_step_with_unfused_attention "shared portable sanity tests" "pytest_shared_sanity_portable.xml" \
    "$TE_PATH/tests/pytorch/test_sanity.py::test_sanity_normalization_amp[LayerNorm-False-False-small-dtype0]" \
    "$TE_PATH/tests/pytorch/test_sanity.py::test_sanity_normalization_amp[RMSNorm-False-False-small-dtype0]" \
    "$TE_PATH/tests/pytorch/test_sanity.py::test_sanity_linear[False-False-False-small-None-dtype0]" \
    "$TE_PATH/tests/pytorch/test_sanity.py::test_sanity_layernorm_linear[False-LayerNorm-False-False-False-small-None-dtype0]" \
    "$TE_PATH/tests/pytorch/test_sanity.py::test_sanity_layernorm_linear[False-RMSNorm-False-False-False-small-None-dtype0]" \
    "$TE_PATH/tests/pytorch/test_sanity.py::test_sanity_layernorm_mlp[False-False-LayerNorm-gelu-False-False-False-small-None-dtype0]" \
    "$TE_PATH/tests/pytorch/test_sanity.py::test_sanity_layernorm_mlp[False-False-RMSNorm-silu-False-False-False-small-None-dtype0]"

run_pytest_step_with_unfused_attention "shared non-FP8 numerics and unfused attention tests" "pytest_shared_numerics_portable.xml" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_linear_accuracy[False-False-small-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_linear_accuracy[False-False-small-1-dtype1]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_layernorm_accuracy[False-1e-05-126m-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_rmsnorm_accuracy[False-1e-05-126m-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_layernorm_linear_accuracy[False-False-False-LayerNorm-small-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_layernorm_linear_accuracy[False-False-False-RMSNorm-small-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_layernorm_mlp_accuracy[False-False-LayerNorm-gelu-small-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_layernorm_mlp_accuracy[False-False-RMSNorm-silu-small-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_dpa_accuracy[126m-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_mha_accuracy[causal-small-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_mha_accuracy[no_mask-small-1-dtype0]"

if [ "$FAIL" -ne 0 ]; then
    echo "Some tests failed."
    exit 1
fi
