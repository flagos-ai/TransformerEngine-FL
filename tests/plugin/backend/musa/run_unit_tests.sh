#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"

mkdir -p "${XML_LOG_DIR}"
XML_LOG_ROOT="${XML_LOG_DIR}"

FAILED=0
OVERALL_FAILED=0

usage() {
    echo "Usage: $0 [debug] [unittest] [distributed] [onnx]"
    echo "If no suite is specified, all suites are run serially."
}

# Keep device/runtime adaptation in set_env.sh and exclusions in musa.yml.
run_suite() {
    local script
    case "$1" in
        debug | pytorch_debug)
            script="qa/L0_pytorch_debug_unittest/test.sh"
            local -x NVTE_TORCH_COMPILE=0 TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
            ;;
        unittest | pytorch | pytorch_unittest) script="qa/L0_pytorch_unittest/test.sh" ;;
        distributed | pytorch_distributed_utils | pytorch_distributed_unittest)
            script="qa/L1_pytorch_distributed_unittest/test.sh"
            ;;
        onnx | pytorch_onnx_unittest) script="qa/L1_pytorch_onnx_unittest/test.sh" ;;
        -h | --help) usage; exit 0 ;;
        *)
            echo "Unsupported MUSA test suite: $1" >&2
            usage >&2
            exit 2
            ;;
    esac

    echo "[RUN] $script"
    if ! bash "$TE_PATH/$script"; then
        echo "[FAIL] $script"
        FAILED=1
    fi
}

if [ "$#" -eq 0 ]; then
    set -- debug unittest distributed onnx
fi

for suite in "$@"; do
    FAILED=0
    export XML_LOG_DIR="${XML_LOG_ROOT}/${suite}"
    mkdir -p "${XML_LOG_DIR}"
    echo "===== START ${suite} ====="
    run_suite "${suite}"
    echo "===== END ${suite} rc=${FAILED} ====="
    if [ "${FAILED}" -ne 0 ]; then
        OVERALL_FAILED=1
    fi
done

if [ "${OVERALL_FAILED}" -ne 0 ]; then
    echo "One or more MUSA test steps failed." >&2
    exit 1
fi

echo "All selected MUSA test steps passed."
