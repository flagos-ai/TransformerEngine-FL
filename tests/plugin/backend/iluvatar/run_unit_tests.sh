#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
usage() {
    echo "Usage: $0 [debug] [unittest] [distributed] [onnx]"
    echo "If no suite is specified, all suites run serially."
}
for suite in "$@"; do
    case "$suite" in
        -h|--help) usage; exit 0 ;;
        debug|unittest|distributed|onnx) ;;
        *) echo "Unsupported Iluvatar test suite: $suite" >&2; usage >&2; exit 2 ;;
    esac
done
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"
XML_LOG_ROOT="$XML_LOG_DIR"
FAILED=0
FAILED_TESTS=()

run_pytest() {
    local name=$1
    local target=$2
    shift 2
    if [ "$target" = tests/plugin ]; then
        local plugin_file plugin_target plugin_name flagos_done=0
        while IFS= read -r plugin_file; do
            plugin_target="${plugin_file#"$TE_PATH/"}"
            if [[ "$plugin_target" == tests/plugin/backend/npu/* ]]; then
                echo "[IGNORE] $plugin_target: NPU tests require the Ascend platform"
                continue
            fi
            if [[ "$plugin_target" == tests/plugin/backend/flagos/* ]]; then
                # Keep directory-scoped ignore expressions and their membership
                # audit together; splitting them across files changes the audit.
                if [ "$flagos_done" -eq 0 ]; then
                    run_pytest "${name}_flagos" tests/plugin/backend/flagos "$@"
                    flagos_done=1
                fi
            else
                plugin_name="${plugin_target//\//_}"
                run_pytest "${name}_${plugin_name%.py}" "$plugin_target" "$@"
            fi
        done < <(find "$TE_PATH/tests/plugin" -type f -name 'test_*.py' | sort)
        return
    fi
    local ignore_rule
    for ignore_rule in "${ILUVATAR_IGNORE_TEST_TARGETS[@]}"; do
        if [ "$target" = "${ignore_rule%%|*}" ]; then
            echo "[IGNORE][$suite] $target: ${ignore_rule#*|}"
            return 0
        fi
    done
    local rule scope details expression reason
    local combined_filter=""
    local expressions=()
    for rule in "${ILUVATAR_IGNORE_TEST_FILTERS[@]}"; do
        scope=${rule%%|*}
        details=${rule#*|}
        if [ "$suite" = "$scope" ] && [ "$target" = "${details%%|*}" ]; then
            details=${details#*|}
            expression=${details%%|*}
            reason=${details#*|}
            echo "[IGNORE][$suite] $target: $reason"
            echo "[IGNORE][$suite] -k not ($expression)"
            expressions+=("$expression")
            combined_filter="${combined_filter:+$combined_filter and }not ($expression)"
        fi
    done
    if [ -n "$combined_filter" ]; then
        set -- "$@" -k "$combined_filter"
    fi
    echo "[RUN][$suite] $target"
    local python_command=("$PYTHON_BIN")
    if [[ "$target" == tests/plugin/plugin/* ]]; then
        # Policy/dispatch contract tests create mock vendors and need the default
        # policy, independent of the device QA's backend preference.
        python_command=(env -u TE_FL_ALLOW_VENDORS -u TE_FL_PREFER -u TE_FL_PER_OP -u TE_FL_STRICT "$PYTHON_BIN")
    fi
    # Use an inline plugin so each rule is checked before pytest applies -k.
    if ! "${python_command[@]}" - "${#expressions[@]}" "${expressions[@]}" \
        -v -s --tb=short "$TE_PATH/$target" "$@" --junitxml="$XML_LOG_DIR/$name.xml" <<'CHECK'
import sys
import pytest
from _pytest.mark import KeywordMatcher
from _pytest.mark.expression import Expression

count = int(sys.argv[1])
rules = sys.argv[2:2 + count]

class IgnoreAudit:
    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_modifyitems(self, items):
        for rule in rules:
            expression = Expression.compile(rule)
            matched = [item.nodeid for item in items if expression.evaluate(KeywordMatcher.from_item(item))]
            print(f"[IGNORE] {len(matched)}/{len(items)} matched: {rule}")
            for nodeid in matched:
                print(f"[IGNORE-NODE] {nodeid}")
            if not matched or len(matched) == len(items):
                raise pytest.UsageError(f"Ignore selector drift (zero or all tests matched): {rule}")

raise SystemExit(pytest.main(sys.argv[2 + count:], plugins=[IgnoreAudit()]))
CHECK
    then
        FAILED=1
        FAILED_TESTS+=("$suite:$name")
    fi
}

run_suite() {
    local target
    local index=0
    local feature_dirs="$TE_PATH/transformer_engine/debug/features"
    local configs_dir="$TE_PATH/tests/pytorch/debug/test_configs"
    case "$suite" in
        debug)
            for name in sanity config numerics log api_features perf; do
                run_pytest "debug_$name" "tests/pytorch/debug/test_$name.py" \
                    --feature_dirs="$feature_dirs" --configs_dir="$configs_dir"
            done
            NVTE_TEST_NVINSPECT_ENABLED=1 \
            NVTE_TEST_NVINSPECT_CONFIG_FILE="$configs_dir/dummy_feature.yaml" \
            NVTE_TEST_NVINSPECT_FEATURE_DIRS="$feature_dirs" \
                run_pytest debug_sanity_regular "tests/pytorch/test_sanity.py"
            NVTE_TEST_NVINSPECT_ENABLED=1 \
            NVTE_TEST_NVINSPECT_CONFIG_FILE="$configs_dir/dummy_feature.yaml" \
            NVTE_TEST_NVINSPECT_FEATURE_DIRS="$feature_dirs" \
                run_pytest debug_numerics_regular "tests/pytorch/test_numerics.py"
            ;;
        unittest)
            local ignore_rule
            for ignore_rule in "${ILUVATAR_IGNORE_TEST_TARGETS[@]}"; do
                echo "[IGNORE][$suite] ${ignore_rule%%|*}: ${ignore_rule#*|}"
            done
            for target in "${ILUVATAR_UNIT_TEST_TARGETS[@]}"; do
                index=$((index + 1))
                run_pytest "unit_$index" "$target"
            done
            ;;
        distributed)
            # The ordinary numerics target is omitted from the list; the debug
            # invocation below is guarded by the same whole-file ignore_rule.
            local ignore_rule
            for ignore_rule in "${ILUVATAR_IGNORE_TEST_TARGETS[@]}"; do
                case "${ignore_rule%%|*}" in
                    tests/pytorch/distributed/*)
                        echo "[IGNORE][$suite] ${ignore_rule%%|*}: ${ignore_rule#*|}"
                        ;;
                esac
            done
            "$PYTHON_BIN" - <<'CHECK'
import os
import torch
required = int(os.environ["ILUVATAR_NPROC_PER_NODE"])
assert torch.cuda.device_count() >= required, f"Distributed QA needs {required} visible devices"
assert torch.distributed.is_nccl_available(), "Vendor NCCL-compatible backend is required"
CHECK
            for target in "${ILUVATAR_DISTRIBUTED_TEST_TARGETS[@]}"; do
                index=$((index + 1))
                run_pytest "distributed_$index" "$target"
            done
            NVTE_TEST_NVINSPECT_ENABLED=1 \
            NVTE_TEST_NVINSPECT_CONFIG_FILE="$configs_dir/dummy_feature.yaml" \
            NVTE_TEST_NVINSPECT_FEATURE_DIRS="$feature_dirs" \
                run_pytest distributed_debug_numerics "tests/pytorch/distributed/test_numerics.py"
            ;;
        onnx)
            NVTE_UnfusedDPA_Emulate_FP8=1 \
                run_pytest onnx_export "tests/pytorch/test_onnx_export.py"
            ;;
    esac
}

if [ "$#" -eq 0 ]; then
    set -- debug unittest distributed onnx
fi
for suite in "$@"; do
    export XML_LOG_DIR="$XML_LOG_ROOT/$suite"
    mkdir -p "$XML_LOG_DIR"
    echo "===== START $suite ====="
    run_suite
    echo "===== END $suite ====="
done
if [ "$FAILED" -ne 0 ]; then
    echo "Failed Iluvatar tests: ${FAILED_TESTS[*]}" >&2
    exit 1
fi
echo "All selected Iluvatar test suites passed."
