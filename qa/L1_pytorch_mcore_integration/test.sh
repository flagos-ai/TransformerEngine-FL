# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

retry_command() {
    local attempts=$1
    local delay_seconds=$2
    shift 2

    local attempt
    for attempt in $(seq 1 "${attempts}"); do
        if "$@"; then
            return 0
        fi
        if [ "${attempt}" -lt "${attempts}" ]; then
            echo "Command failed (attempt ${attempt}/${attempts}): $*"
            echo "Retrying in ${delay_seconds}s..."
            sleep "${delay_seconds}"
        fi
    done

    echo "Command failed after ${attempts} attempts: $*"
    return 1
}

detect_platform() {
    if command -v nvidia-smi &>/dev/null; then
        echo cuda
    elif command -v mx-smi &>/dev/null || [ -d /opt/maca ]; then
        echo metax
    elif command -v npu-smi &>/dev/null || [ -d /usr/local/Ascend ]; then
        echo ascend
    else
        echo unknown
    fi
}

# Paths
: "${TE_PATH:=$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
: "${MCORE_PATH:=/workspace/Megatron-LM-FL}"
: "${MCORE_REPO_URL:=https://github.com/flagos-ai/Megatron-LM-FL.git}"
: "${MCORE_REF:=175ae90ec92a9e6fea2d74ccd24d6a1835d3ae82}"
: "${OUTPUT_DIR:=${TE_PATH}/qa/L1_pytorch_mcore_integration/output}"
: "${DATA_CACHE_PATH:=/tmp/data_cache}"
: "${PLATFORM:=$(detect_platform)}"
: "${TE_FL_PREFER:=vendor}"

: "${DISTRIBUTED_BACKEND:=nccl}"
if [ "${PLATFORM}" = "ascend" ]; then
    : "${NUM_LAYERS:=2}"
    : "${HIDDEN_SIZE:=128}"
    : "${NUM_ATTENTION_HEADS:=4}"
    : "${SEQ_LENGTH:=128}"
    : "${MICRO_BATCH_SIZE:=1}"
    : "${GLOBAL_BATCH_SIZE:=1}"
    : "${ENABLE_DIAGNOSTICS:=0}"
else
    : "${NUM_LAYERS:=12}"
    : "${HIDDEN_SIZE:=512}"
    : "${NUM_ATTENTION_HEADS:=8}"
    : "${SEQ_LENGTH:=1024}"
    : "${MICRO_BATCH_SIZE:=4}"
    : "${GLOBAL_BATCH_SIZE:=32}"
    : "${ENABLE_DIAGNOSTICS:=1}"
    : "${CUDA_DEVICE_MAX_CONNECTIONS:=1}"
    : "${CUBLAS_WORKSPACE_CONFIG:=:4096:8}"
fi

export PLATFORM TE_FL_PREFER MCORE_REPO_URL MCORE_REF DISTRIBUTED_BACKEND
export NUM_LAYERS HIDDEN_SIZE NUM_ATTENTION_HEADS SEQ_LENGTH
export MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE ENABLE_DIAGNOSTICS
if [ -n "${CUDA_DEVICE_MAX_CONNECTIONS:-}" ]; then
    export CUDA_DEVICE_MAX_CONNECTIONS
fi
if [ -n "${CUBLAS_WORKSPACE_CONFIG:-}" ]; then
    export CUBLAS_WORKSPACE_CONFIG
fi

# Check whether FP8 is supported
WITH_FP8=
if command -v nvidia-smi &>/dev/null; then
    DEVICE_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/[^0-9]//g')
    if [[ ${DEVICE_ARCH} -ge 89 ]]; then
        WITH_FP8=1
    fi
elif command -v mx-smi &>/dev/null; then
    # Metax hardware does not support FP8; leave WITH_FP8 unset
    :
fi

# Download or sync Megatron-LM-FL to the requested repo/ref.
if [ ! -d "${MCORE_PATH}" ]; then
    mkdir -p "$(dirname "${MCORE_PATH}")"
    git config --global --unset-all credential.helper 2>/dev/null || true
    git config --system --unset-all credential.helper 2>/dev/null || true
    retry_command 3 5 git clone --filter=blob:none --no-checkout \
        "${MCORE_REPO_URL}" "${MCORE_PATH}"
fi

if [ -d "${MCORE_PATH}/.git" ]; then
    git -C "${MCORE_PATH}" remote set-url origin "${MCORE_REPO_URL}"
    retry_command 3 5 git -C "${MCORE_PATH}" fetch --depth 1 origin "${MCORE_REF}"
    git -C "${MCORE_PATH}" checkout --detach --force "FETCH_HEAD"
else
    echo "Megatron-LM-FL checkout is not a Git repository: ${MCORE_PATH}" >&2
    exit 1
fi

if [ "${DISTRIBUTED_BACKEND}" = "mccl" ]; then
    python3 "${TE_PATH}/tests/integration/musa/patch_megatron_mccl.py" "${MCORE_PATH}"
fi

# Megatron-LM-FL tokenizer imports happen at module import time, so direct
# source execution needs these Python deps available before pretrain_gpt.py
# starts.
python3 - <<'PY' || python3 -m pip install --disable-pip-version-check six regex
import regex
import six
print(f"six available: {six.__version__}")
print(f"regex available: {regex.__version__}")
PY

# Megatron's mock dataset requires its pybind11 helper extension. Source-only
# checkouts do not provide the compiled module.
if ! PYTHONPATH="${MCORE_PATH}:${PYTHONPATH:-}" python3 -c \
    "import megatron.core.datasets.helpers_cpp" 2>/dev/null; then
    (cd "${MCORE_PATH}" && python3 setup.py build_ext --inplace)
fi

CHECKPOINT_DIR=${OUTPUT_DIR}/checkpoints
TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard
mkdir -p "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" "${DATA_CACHE_PATH}" /tmp/checkpoints

echo "Using Megatron-LM-FL repo: ${MCORE_REPO_URL}"
echo "Using Megatron-LM-FL ref: ${MCORE_REF}"
git -C "${MCORE_PATH}" rev-parse --short HEAD
echo "Platform: ${PLATFORM}"
echo "Distributed backend: ${DISTRIBUTED_BACKEND}"
if [ -n "${WITH_FP8}" ]; then
    echo "FP8 enabled: yes"
else
    echo "FP8 enabled: no"
fi

# Megatron-LM-FL invocation. Keep the argument shape aligned with the
# previously validated tp1/pp1 mock-data GPT functional case while letting CI
# exit after a few steps.
DEVICE_ENV="NCCL_ALGO=${NCCL_ALGO:-Ring}"
if [ -n "${CUDA_DEVICE_MAX_CONNECTIONS:-}" ]; then
    DEVICE_ENV="${DEVICE_ENV}
CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}"
fi
if [ -n "${CUBLAS_WORKSPACE_CONFIG:-}" ]; then
    DEVICE_ENV="${DEVICE_ENV}
CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"
fi

DIAGNOSTIC_ARGS=""
if [ "${ENABLE_DIAGNOSTICS}" = "1" ]; then
    DIAGNOSTIC_ARGS="
--log-params-norm
--log-num-zeros-in-grad
--log-memory-to-tensorboard"
fi

COMMAND="
NVTE_TORCH_COMPILE=0
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
TORCHDYNAMO_DISABLE=1
TORCH_COMPILE_DISABLE=1
${DEVICE_ENV}

torchrun
--nnodes=1
--nproc_per_node=1

${MCORE_PATH}/pretrain_gpt.py
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1
--num-layers ${NUM_LAYERS}
--hidden-size ${HIDDEN_SIZE}
--num-attention-heads ${NUM_ATTENTION_HEADS}
${DIAGNOSTIC_ARGS}
--log-validation-ppl-to-tensorboard
--log-timers-to-tensorboard
--seq-length ${SEQ_LENGTH}
--max-position-embeddings ${SEQ_LENGTH}
--micro-batch-size ${MICRO_BATCH_SIZE}
--global-batch-size ${GLOBAL_BATCH_SIZE}
--train-iters 50
--eval-iters 10
--timing-log-level 0
--lr-decay-iters 320000
--save ${CHECKPOINT_DIR}
--split 949,50,1
--tokenizer-type NullTokenizer
--vocab-size 8192
--mock-data
--distributed-backend ${DISTRIBUTED_BACKEND}
--lr 0.00015
--lr-decay-style cosine
--min-lr 1.0e-5
--weight-decay 1e-2
--clip-grad 1.0
--lr-warmup-fraction .01
--log-interval 1
--save-interval 10000
--eval-interval 1000
--transformer-impl transformer_engine
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers 1
--deterministic-mode
--no-gradient-accumulation-fusion
--attention-softmax-in-fp32
--use-mcore-models
--ckpt-format torch_dist
--dist-ckpt-optim-fully-reshardable
--dist-ckpt-strictness log_all
--data-cache-path ${DATA_CACHE_PATH}
--bf16
--attention-backend unfused
--tensorboard-dir ${TENSORBOARD_DIR}
--exit-interval 4
${WITH_FP8:+--fp8-format hybrid}
"
COMMAND=$(echo "${COMMAND}" | tr '\n' ' ')

# Launch Megatron-LM-FL
bash -c "${COMMAND}"
