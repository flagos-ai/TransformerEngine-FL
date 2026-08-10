# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

function error_exit() {
    echo "Error: $1"
    exit 1
}

function test_fail() {
    RET=1
    FAILED_CASES="$FAILED_CASES $1"
    echo "Error: sub-test failed: $1"
}

RET=0
FAILED_CASES=""
DEBUG_TESTS_READY=0

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

# The current CUDA 12.8 test container hits a fused-attention runtime loader
# issue, so keep the distributed numerics suite on the unfused attention path.
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-0}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export NVTE_UNFUSED_ATTN="${NVTE_UNFUSED_ATTN:-1}"

# Make CUDA runtime libraries discoverable for fused attention kernels.
if [ -z "${CUDA_HOME:-}" ]; then
    if [ -d /usr/local/cuda ]; then
        export CUDA_HOME=/usr/local/cuda
    elif [ -d /usr/local/cuda-12.8 ]; then
        export CUDA_HOME=/usr/local/cuda-12.8
    fi
fi
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME:-}}"

CUDA_LIB_DIRS=()
for path in \
    "${CUDA_HOME:-}/lib64" \
    "${CUDA_HOME:-}/targets/x86_64-linux/lib" \
    "$(python3 - <<'PY'
import site
from pathlib import Path

for root in site.getsitepackages():
    candidate = Path(root) / "torch" / "lib"
    if candidate.exists():
        print(candidate)
        break
PY
)" \
    "$(python3 - <<'PY'
import site
from pathlib import Path

for root in site.getsitepackages():
    candidate = Path(root) / "nvidia" / "cuda_runtime" / "lib"
    if candidate.exists():
        print(candidate)
        break
PY
)"; do
    if [ -n "$path" ] && [ -d "$path" ]; then
        CUDA_LIB_DIRS+=("$path")
    fi
done

if [ "${#CUDA_LIB_DIRS[@]}" -gt 0 ]; then
    CUDA_LIB_PATH="$(IFS=:; echo "${CUDA_LIB_DIRS[*]}")"
    export LD_LIBRARY_PATH="${CUDA_LIB_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

python3 - <<'PY'
import ctypes

for name in ("libcudart.so", "libcudart.so.12"):
    try:
        ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        print(f"[CUDA] Preloaded {name}")
        break
    except OSError as exc:
        print(f"[CUDA] Failed to preload {name}: {exc}")
PY


# It is not installed as a requirement,
# because it is not available on PyPI.
pip uninstall -y nvdlfw-inspect
if pip install git+https://github.com/NVIDIA/nvidia-dlfw-inspect.git && \
   python3 -c "import nvdlfw_inspect.api" >/dev/null 2>&1; then
    DEBUG_TESTS_READY=1
else
    echo "Warning: nvdlfw_inspect is unavailable; debug numerics test will be skipped"
fi

pip3 install pytest==8.2.1 || error_exit "Failed to install pytest"

run_test_step() {
    local xml_file=$1
    local test_path=$2
    local cmd=$3
    local label=$4

    if [ "$PLATFORM" = "metax" ]; then
        case "$test_path" in
            *"test_numerics.py" | \
            *"test_numerics_exact.py" | \
            *"test_torch_fsdp2.py" | \
            *"test_cast_master_weights_to_fp8.py")
                echo "-------------------------------------------------------"
                echo "[SKIP] Platform MetaX: Ignoring $label"
                echo "-------------------------------------------------------"
                return 0
                ;;
        esac
    fi

    echo "-------------------------------------------------------"
    echo "[RUN] Executing: $label"
    eval "$cmd" || test_fail "$label"
}

# python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_sanity.xml $TE_PATH/tests/pytorch/distributed/test_sanity.py || test_fail "test_sanity.py"
run_test_step "pytest_test_numerics.xml" "$TE_PATH/tests/pytorch/distributed/test_numerics.py" \
"python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_numerics.xml $TE_PATH/tests/pytorch/distributed/test_numerics.py" \
"test_numerics.py"
run_test_step "pytest_test_numerics_exact.xml" "$TE_PATH/tests/pytorch/distributed/test_numerics_exact.py" \
"python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_numerics_exact.xml $TE_PATH/tests/pytorch/distributed/test_numerics_exact.py" \
"test_numerics_exact.py"
# python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_fusible_ops.xml $TE_PATH/tests/pytorch/distributed/test_fusible_ops.py || test_fail "test_fusible_ops.py"
run_test_step "pytest_test_torch_fsdp2.xml" "$TE_PATH/tests/pytorch/distributed/test_torch_fsdp2.py" \
"python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_torch_fsdp2.xml $TE_PATH/tests/pytorch/distributed/test_torch_fsdp2.py -k 'not (test_distributed)'" \
"test_torch_fsdp2.py"
# python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_comm_gemm_overlap.xml $TE_PATH/tests/pytorch/distributed/test_comm_gemm_overlap.py || test_fail "test_comm_gemm_overlap.py"
# python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_fusible_ops_with_userbuffers.xml $TE_PATH/tests/pytorch/distributed/test_fusible_ops_with_userbuffers.py || test_fail "test_fusible_ops_with_userbuffers.py"
# python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_attention_with_cp.xml $TE_PATH/tests/pytorch/attention/test_attention_with_cp.py || test_fail "test_attention_with_cp.py"
run_test_step "pytest_test_cp_utils.xml" "$TE_PATH/tests/pytorch/attention/test_cp_utils.py" \
"python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_cp_utils.xml $TE_PATH/tests/pytorch/attention/test_cp_utils.py" \
"test_cp_utils.py"
run_test_step "pytest_test_cast_master_weights_to_fp8.xml" "$TE_PATH/tests/pytorch/distributed/test_cast_master_weights_to_fp8.py" \
"python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_cast_master_weights_to_fp8.xml $TE_PATH/tests/pytorch/distributed/test_cast_master_weights_to_fp8.py" \
"test_cast_master_weights_to_fp8.py"


# debug tests


# Config with the dummy feature which prevents nvinspect from being disabled.
# Nvinspect will be disabled if no feature is active.
: ${NVTE_TEST_NVINSPECT_DUMMY_CONFIG_FILE:=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml}
: ${NVTE_TEST_NVINSPECT_FEATURE_DIRS:=$TE_PATH/transformer_engine/debug/features}

# pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_distributed.xml $TE_PATH/tests/pytorch/debug/test_distributed.py --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS || test_fail "debug test_distributed.py"
# standard numerics tests with initialized debug
if [ "$DEBUG_TESTS_READY" -eq 1 ]; then
    run_test_step "pytest_test_numerics_2.xml" "$TE_PATH/tests/pytorch/distributed/test_numerics.py" \
    "NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$NVTE_TEST_NVINSPECT_DUMMY_CONFIG_FILE NVTE_TEST_NVINSPECT_FEATURE_DIRS=$NVTE_TEST_NVINSPECT_FEATURE_DIRS pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_numerics_2.xml $TE_PATH/tests/pytorch/distributed/test_numerics.py" \
    "test_numerics.py (debug)"
else
    echo "Skipping debug test_numerics.py because nvdlfw_inspect is unavailable"
fi

if [ "$RET" -ne 0 ]; then
    echo "Error in the following test cases:$FAILED_CASES"
    exit 1
fi
echo "All tests passed"
exit 0
