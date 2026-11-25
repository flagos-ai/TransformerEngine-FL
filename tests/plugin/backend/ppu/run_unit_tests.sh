#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"
# Individual pytest failures are collected and propagated after all targets.
set +e
cd "$TE_PATH"
PYTHON="${PYTHON_BIN:-python3}"
ROOT_LOG_DIR="$XML_LOG_DIR"
mkdir -p "$ROOT_LOG_DIR"
SUMMARY="$ROOT_LOG_DIR/summary.tsv"
printf 'suite\ttarget\treturncode\tstatus\treason\towner\trestore_condition\n' > "$SUMMARY"
OVERALL_FAIL=0

# The platform launcher handles the device environment and delegates test scope to QA.
run_shared_suite() {
  local suite="$1" script rc status=passed
  local timeout_seconds="${PPU_STEP_TIMEOUT:-7200}"
  case "$suite" in
    debug) script="qa/L0_pytorch_debug_unittest/test.sh" ;;
    unittest)
      script="qa/L0_pytorch_unittest/test.sh"
      timeout_seconds="$PLATFORM_UNIT_TIMEOUT"
      ;;
    distributed) script="qa/L1_pytorch_distributed_unittest/test.sh" ;;
    onnx) script="qa/L1_pytorch_onnx_unittest/test.sh" ;;
  esac
  echo "[RUN][$suite] $script"
  XML_LOG_DIR="$XML_LOG_DIR" timeout --signal=TERM --kill-after=10s "${timeout_seconds}s" \
    bash "$TE_PATH/$script" 2>&1 | tee "$XML_LOG_DIR/shared-suite.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then status=timed_out; OVERALL_FAIL=1
  elif [ "$rc" -ne 0 ]; then status=failed; OVERALL_FAIL=1; fi
  printf '%s\t%s\t%s\t%s\t\t\t\n' "$suite" "$script" "$rc" "$status" >> "$SUMMARY"
  echo "[EXIT $rc][$suite] $script"
}
run_suite() {
  local suite="$1"
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
  run_shared_suite "$suite"
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
        log = summary.parent / suite / 'shared-suite.log'
        counts = {'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0}
        if target.startswith('qa/'):
            # A shared qa script writes one JUnit file per step into the suite dir.
            for report in sorted((summary.parent / suite).glob('*.xml')):
                root = ET.parse(report)
                for key in counts: counts[key] += sum(int(x.get(key, 0)) for x in root.iter('testsuite'))
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
