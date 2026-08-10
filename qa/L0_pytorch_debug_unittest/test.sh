# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.



: ${TE_PATH:=/opt/transformerengine}
: ${NVTE_TEST_NVINSPECT_FEATURE_DIRS:=$TE_PATH/transformer_engine/debug/features}
: ${NVTE_TEST_NVINSPECT_CONFIGS_DIR:=$TE_PATH/tests/pytorch/debug/test_configs/}

: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

# Config with the dummy feature which prevents nvinspect from being disabled.
# Nvinspect will be disabled if no feature is active.
: ${NVTE_TEST_NVINSPECT_DUMMY_CONFIG_FILE:=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml}

FAIL=0

# It is not installed as a requirement,
# because it is not available on PyPI.
pip install pytest==8.2.1

METAX_IGNORED_TESTS=(
    "$TE_PATH/tests/pytorch/test_numerics.py"
    "$TE_PATH/tests/pytorch/test_sanity.py"
    "$TE_PATH/tests/pytorch/debug/test_sanity.py"
)

should_skip_on_metax() {
    local test_path=$1

    [ "$PLATFORM" = "metax" ] || return 1

    local ignored_test
    for ignored_test in "${METAX_IGNORED_TESTS[@]}"; do
        if [ "$test_path" = "$ignored_test" ]; then
            echo "[SKIP] Platform MetaX: Ignoring $test_path"
            return 0
        fi
    done

    return 1
}


run_test_step() {
    local xml_file=$1
    local test_path=$2
    local cmd=$3

    if should_skip_on_metax "$test_path"; then
        return 0
    fi

    echo "-------------------------------------------------------"
    echo "[RUN] Executing: $test_path"
    eval "$cmd" || FAIL=1
}



# Step 1: Sanity
run_test_step "test_sanity.xml" "$TE_PATH/tests/pytorch/debug/test_sanity.py" \
"pytest -v -s --junitxml=$XML_LOG_DIR/test_sanity.xml $TE_PATH/tests/pytorch/debug/test_sanity.py --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS"

# Step 2: Config
run_test_step "test_config.xml" "$TE_PATH/tests/pytorch/debug/test_config.py" \
"pytest -v -s --junitxml=$XML_LOG_DIR/test_config.xml $TE_PATH/tests/pytorch/debug/test_config.py --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS"

# Step 3: Numerics
run_test_step "test_numerics.xml" "$TE_PATH/tests/pytorch/debug/test_numerics.py" \
"pytest -v -s --junitxml=$XML_LOG_DIR/test_numerics.xml $TE_PATH/tests/pytorch/debug/test_numerics.py --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS"

# Step 4: Log
run_test_step "test_log.xml" "$TE_PATH/tests/pytorch/debug/test_log.py" \
"pytest -v -s --junitxml=$XML_LOG_DIR/test_log.xml $TE_PATH/tests/pytorch/debug/test_log.py --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS --configs_dir=$NVTE_TEST_NVINSPECT_CONFIGS_DIR"

# Step 5: API Features
run_test_step "test_api_features.xml" "$TE_PATH/tests/pytorch/debug/test_api_features.py" \
"NVTE_TORCH_COMPILE=0 pytest -v -s --junitxml=$XML_LOG_DIR/test_api_features.xml $TE_PATH/tests/pytorch/debug/test_api_features.py --no-header --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS --configs_dir=$NVTE_TEST_NVINSPECT_CONFIGS_DIR"

# Step 6: Performance
run_test_step "test_perf.xml" "$TE_PATH/tests/pytorch/debug/test_perf.py" \
"pytest -v -s --junitxml=$XML_LOG_DIR/test_perf.xml $TE_PATH/tests/pytorch/debug/test_perf.py --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS --configs_dir=$NVTE_TEST_NVINSPECT_CONFIGS_DIR"


# Step 7: Sanity 2
run_test_step "test_sanity_2.xml" "$TE_PATH/tests/pytorch/test_sanity.py" \
"NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$NVTE_TEST_NVINSPECT_DUMMY_CONFIG_FILE NVTE_TEST_NVINSPECT_FEATURE_DIRS=$NVTE_TEST_NVINSPECT_FEATURE_DIRS PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
pytest -v -s --junitxml=$XML_LOG_DIR/test_sanity_2.xml $TE_PATH/tests/pytorch/test_sanity.py --no-header"

# Step 8: Numerics 2
run_test_step "test_numerics_2.xml" "$TE_PATH/tests/pytorch/test_numerics.py" \
"NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$NVTE_TEST_NVINSPECT_DUMMY_CONFIG_FILE NVTE_TEST_NVINSPECT_FEATURE_DIRS=$NVTE_TEST_NVINSPECT_FEATURE_DIRS PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0 \
pytest -v -s --junitxml=$XML_LOG_DIR/test_numerics_2.xml $TE_PATH/tests/pytorch/test_numerics.py -k \"not (test_linear_accuracy or test_layernorm_linear_accuracy or test_layernorm_mlp_accuracy or test_transformer_layer_hidden_states_format)\" --no-header"

exit $FAIL
