#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env.sh"
source "$SCRIPT_DIR/config.sh"
# Individual pytest failures are collected and propagated after all targets.
set +e
cd "$TE_PATH"
PYTHON="${PYTHON_BIN:-python3}"
ROOT_LOG_DIR="$XML_LOG_DIR"
mkdir -p "$ROOT_LOG_DIR"
SUMMARY="$ROOT_LOG_DIR/summary.tsv"
EXCLUSIONS="$ROOT_LOG_DIR/exclusions.tsv"
printf 'suite\ttarget\treturncode\tstatus\treason\towner\trestore_condition\n' > "$SUMMARY"
printf 'suite\ttarget\treason\towner\trestore_condition\n' > "$EXCLUSIONS"
OVERALL_FAIL=0
is_excluded_target() {
  local candidate="$1" excluded
  for excluded in "${PPU_EXCLUDED_TARGETS[@]}"; do
    [ "$excluded" = "$candidate" ] && return 0
  done
  return 1
}
run_target() {
  local suite="$1" target="$2"; shift 2
  local name="${target//\//_}"; name="${name%.py}"
  local log="$XML_LOG_DIR/${name}.log" xml="$XML_LOG_DIR/${name}.xml"
  local timeout_seconds="${PPU_STEP_TIMEOUT:-7200}"
  if ! [[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "PPU_STEP_TIMEOUT must be a positive integer, got: $timeout_seconds" >&2
    OVERALL_FAIL=1
    printf '%s\t%s\t2\tfailed\t\t\t\n' "$suite" "$target" >> "$SUMMARY"
    return
  fi
  local -a cmd=("$PYTHON" -m pytest "$TE_PATH/$target" -v -s --tb=short -ra -o faulthandler_timeout=120 --junitxml="$xml")
  cmd+=("$@")
  echo "[RUN][$suite] $target"
  rm -f "$xml"
  local rc status
  timeout --signal=TERM --kill-after=10s "${timeout_seconds}s" "${cmd[@]}" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  status=passed
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    status=timed_out
    OVERALL_FAIL=1
    "$PYTHON" - "$xml" "$target" "$timeout_seconds" <<'PY'
import pathlib, sys, xml.etree.ElementTree as ET

xml = pathlib.Path(sys.argv[1])
target = sys.argv[2]
timeout_seconds = sys.argv[3]
suite = ET.Element('testsuite', name=target, tests='1', failures='0', errors='1', skipped='0')
case = ET.SubElement(suite, 'testcase', name='runner_timeout')
ET.SubElement(case, 'error', message=f'Target exceeded {timeout_seconds} seconds')
ET.ElementTree(suite).write(xml, encoding='unicode', xml_declaration=True)
PY
    if [ "$?" -ne 0 ]; then
      echo "Failed to write timeout JUnit report: $xml" >&2
    fi
  elif [ "$rc" -ne 0 ]; then
    status=failed
    OVERALL_FAIL=1
  fi
  printf '%s\t%s\t%s\t%s\t\t\t\n' "$suite" "$target" "$rc" "$status" >> "$SUMMARY"
  echo "[EXIT $rc][$suite] $target (log: $log)"
}
record_exclusion() {
  local suite="$1" target="$2"
  local reason="${PPU_EXCLUDED_REASON[$target]}"
  local owner="${PPU_EXCLUDED_OWNER[$target]}"
  local restore="${PPU_EXCLUDED_RESTORE[$target]}"
  echo "[EXCLUDED][$suite] $target: $reason; owner=$owner; restore=$restore"
  printf '%s\t%s\t\texcluded\t%s\t%s\t%s\n' "$suite" "$target" "$reason" "$owner" "$restore" >> "$SUMMARY"
  printf '%s\t%s\t%s\t%s\t%s\n' "$suite" "$target" "$reason" "$owner" "$restore" >> "$EXCLUSIONS"
}
run_suite() {
  local suite="$1" target
  local -n targets="PPU_${suite^^}_TARGETS"
  local previous_visible="${CUDA_VISIBLE_DEVICES:-}"
  local previous_visible_set=0
  [ -n "${CUDA_VISIBLE_DEVICES+x}" ] && previous_visible_set=1
  if [ "$suite" = distributed ]; then
    local nproc
    nproc="$($PYTHON -c 'import yaml; print(yaml.safe_load(open(".github/configs/ppu.yml"))["nproc_per_node"])')"
    if ! [[ "$nproc" =~ ^[0-9]+$ ]] || [ "$nproc" -lt 2 ]; then
      echo "ppu.yml: nproc_per_node must be an integer >= 2" >&2
      OVERALL_FAIL=1
      printf '%s\t<device-check>\t2\tfailed\t\t\t\n' "$suite" >> "$SUMMARY"
      return
    fi
    local hardware_count
    hardware_count="$($PYTHON -c 'import torch; print(torch.cuda.device_count())')"
    local visible="${CUDA_VISIBLE_DEVICES:-}"
    local -a devices=()
    if [ -n "$visible" ]; then
      IFS=',' read -r -a devices <<< "$visible"
    else
      local index
      for ((index=0; index<hardware_count; index++)); do devices+=("$index"); done
    fi
    if [ "$hardware_count" -lt "$nproc" ] || [ "${#devices[@]}" -lt "$nproc" ]; then
      echo "PPU distributed requires $nproc visible devices, found ${#devices[@]} (runtime count: $hardware_count)" >&2
      OVERALL_FAIL=1
      printf '%s\t<device-check>\t1\tfailed\t\t\t\n' "$suite" >> "$SUMMARY"
      return
    fi
    local selected="${devices[0]}"
    local index
    for ((index=1; index<nproc; index++)); do selected+=",${devices[$index]}"; done
    export CUDA_VISIBLE_DEVICES="$selected"
    echo "[distributed] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (nproc_per_node=$nproc)"
  fi
  mkdir -p "$ROOT_LOG_DIR/$suite"
  local old_log="$XML_LOG_DIR"; XML_LOG_DIR="$ROOT_LOG_DIR/$suite"
  for target in "${targets[@]}"; do
    if is_excluded_target "$target"; then
      record_exclusion "$suite" "$target"
      continue
    fi
    local -a extra=()
    [ "$suite" = debug ] && extra+=(--feature_dirs=transformer_engine/debug/features --configs_dir=tests/pytorch/debug/test_configs/)
    [ "$target" = tests/pytorch/test_cpu_offloading_v1.py ] && export NVTE_CPU_OFFLOAD_V1=1
    [ "$target" = tests/pytorch/test_onnx_export.py ] && export NVTE_UnfusedDPA_Emulate_FP8=1
    [ -n "${PPU_SKIP_K[$target]:-}" ] && extra+=(-k "not (${PPU_SKIP_K[$target]})")
    run_target "$suite" "$target" "${extra[@]}"
  done
  XML_LOG_DIR="$old_log"
  if [ "$suite" = distributed ]; then
    if [ "$previous_visible_set" -eq 1 ]; then
      export CUDA_VISIBLE_DEVICES="$previous_visible"
    else
      unset CUDA_VISIBLE_DEVICES
    fi
  fi
}
if [ "$#" -eq 0 ]; then set -- debug unittest distributed onnx; fi
for suite in "$@"; do
  case "$suite" in
    debug|unittest|distributed|onnx) run_suite "$suite" ;;
    -h|--help) echo "Usage: $0 [debug] [unittest] [distributed] [onnx]"; exit 0 ;;
    *) echo "Unknown suite: $suite" >&2; exit 2 ;;
  esac
done
summary_rc=0
"$PYTHON" - "$SUMMARY" "$ROOT_LOG_DIR/summary.json" <<'PY' || summary_rc=$?
import json, os, pathlib, re, sys, tempfile, xml.etree.ElementTree as ET

summary = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
rows = []
with summary.open() as f:
    next(f)
    for line in f:
        suite, target, rc, status, reason, owner, restore = line.rstrip('\n').split('\t')
        xml = summary.parent / suite / (target.replace('/', '_').removesuffix('.py') + '.xml')
        log = summary.parent / suite / (target.replace('/', '_').removesuffix('.py') + '.log')
        counts = {'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0}
        if status != 'excluded':
            if not xml.is_file():
                raise FileNotFoundError(f'Missing JUnit report: {xml}')
            root = ET.parse(xml)
            for key in counts: counts[key] = sum(int(x.get(key, 0)) for x in root.iter('testsuite'))
            if status == 'passed' and counts['tests'] == 0:
                raise ValueError(f'Passing target collected zero tests: {target}')
        counts['deselected'] = 0
        if log.exists():
            matches = re.findall(r'(\d+) deselected', log.read_text(errors='replace'))
            if matches: counts['deselected'] = int(matches[-1])
        counts.update(
            suite=suite,
            target=target,
            returncode=int(rc) if rc else None,
            status=status,
            reason=reason or None,
            owner=owner or None,
            restore_condition=restore or None,
        )
        rows.append(counts)
with tempfile.NamedTemporaryFile('w', dir=destination.parent, delete=False) as f:
    json.dump(rows, f, indent=2)
    f.write('\n')
    temporary = f.name
os.replace(temporary, destination)
PY
if [ "$summary_rc" -ne 0 ]; then
  echo "Failed to generate test summary from JUnit reports" >&2
  OVERALL_FAIL=1
fi
exit "$OVERALL_FAIL"
