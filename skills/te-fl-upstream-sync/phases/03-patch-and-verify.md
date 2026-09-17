### Stage 5: Patch CUDA Hardcoding in Upstream Python Changes (`/stage5-patch-cuda-hardcoding`)

Run the Repo Detection Preamble to ensure you are in the TransformerEngine-FL directory.

The main branch carries a "patch" feature that abstracts CUDA-specific string references in
TransformerEngine's Python layer. The key mechanism:

- `TE_DEVICE_TYPE` (string, default `"cuda"`) — defined in `transformer_engine/__init__.py`,
  overridden by vendor `patches.py` at import time (e.g., MUSA sets it to `"musa"`)

Only `TE_DEVICE_TYPE` is needed in this stage. The `torch.cuda.*` API calls (streams, events,
synchronize, device queries, etc.) are handled separately by vendor `patches.py` files via
runtime monkey-patching (e.g., `torch.cuda.synchronize → torch.musa.synchronize`). Those
calls stay as `torch.cuda.*` in source code — no replacement needed.

When upstream merges introduce new or modified Python files, they may bring fresh `"cuda"`
string hardcoding. This stage detects and patches those instances.

#### What to patch vs what to leave alone

**Patch** — hardcoded `"cuda"` strings used as device identifiers:

| Pattern | Replacement |
|---------|-------------|
| `device="cuda"` | `device=TE_DEVICE_TYPE` |
| `torch.device("cuda")` | `torch.device(TE_DEVICE_TYPE)` |
| `torch.get_autocast_dtype("cuda")` | `torch.get_autocast_dtype(TE_DEVICE_TYPE)` |
| `.device.type == "cuda"` | `.device.type == TE_DEVICE_TYPE` |
| `"cuda"` in other device-selection contexts | `TE_DEVICE_TYPE` |

**Leave as-is** — everything else:

- All `torch.cuda.*` API calls — handled by vendor patches.py at runtime
- `torch.cuda.CUDAGraph` — CUDA-specific, no vendor equivalent
- `torch.cuda.nvtx.*` — profiling, handled by patches.py or no-op
- `torch.version.cuda` — build-time version query
- `.cuda_stream` — low-level C pointer
- `"cuda"` in comments, docstrings, and string messages
- `"cuda"` in device-type checks that guard CUDA-specific code blocks (these are intentional
  gates, not device selection)

#### Step 1: Scan the upstream Python diff for new `"cuda"` string hardcoding

```bash
git diff base..dev -- transformer_engine/pytorch/ ':(exclude)transformer_engine/pytorch/csrc/' \
  > /tmp/python_layer_diff.diff
```

Extract newly added lines containing `"cuda"` string patterns:

```bash
grep '^+' /tmp/python_layer_diff.diff | grep -v '^+++' \
  | grep -E 'device.*"cuda"|torch\.device\("cuda"\)|get_autocast_dtype.*"cuda"|\.device\.type.*==.*"cuda"' \
  > /tmp/cuda_string_candidates.txt
```

This is the candidate list. Each line needs manual triage in Step 2.

#### Step 2: Triage candidates

For each candidate line, decide patch or skip:

1. **Patch** if the `"cuda"` string is used for device selection in general-purpose code
   (modules, ops, distributed, quantization, attention, etc.)
2. **Skip** if the `"cuda"` string is inside a CUDA-specific guard (e.g.,
   `if device.type == "cuda": <do cuda-only thing>`) — these are intentional gates
3. **Skip** if it's in a docstring, comment, or log message
4. **Skip** if the file is inherently CUDA-only (e.g., `cuda_graphs.py`)

The key distinction: device *selection* (`device="cuda"`) should use `TE_DEVICE_TYPE` so
non-CUDA vendors get their device. Device *detection* (`if x == "cuda"`) that gates
CUDA-specific behavior should stay as `"cuda"`.

#### Step 3: Apply patches

For each file with lines to patch:

1. Read the current file content (post-merge, on the working branch)
2. Replace `"cuda"` string patterns per the table above
3. Add the import if not already present:
   ```python
   from transformer_engine import TE_DEVICE_TYPE
   ```
4. Syntax check: `python3 -c "import ast; ast.parse(open('<file>').read())"`

**Important**: Only patch lines introduced or modified by the upstream merge (the `^+` lines
from the diff). Do not retroactively patch pre-existing `"cuda"` references that the main
branch has already chosen to leave as-is.

**Do not modify** any files under `plugin/core/backends/vendor/` — those are vendor-customized.

#### Step 4: Verify

1. Syntax check all modified files
2. Grep to confirm no un-patched `"cuda"` device strings remain in newly added lines:
   ```bash
   git diff base..HEAD -- transformer_engine/pytorch/ ':(exclude)transformer_engine/pytorch/csrc/' \
     | grep '^+' | grep -v '^+++' \
     | grep -E 'device.*=.*"cuda"|torch\.device\("cuda"\)|get_autocast_dtype\("cuda"\)' \
     > /tmp/remaining_cuda_strings.txt
   ```
   Review — each remaining line should be a deliberate skip (guard, docstring, CUDA-only file).
3. Count patched vs skipped for the commit log.

#### Step 5: Commit

```bash
git add -A
pre-commit run --all-files
git add -A   # re-stage any formatting fixes
git commit -m "patch: normalize new upstream 'cuda' string hardcoding to TE_DEVICE_TYPE

Scanned Python-layer diff (base..dev, excluding csrc) for newly introduced
hardcoded 'cuda' device strings. Replaced <N> instances across <M> files:
- device='cuda' → device=TE_DEVICE_TYPE: <count>
- torch.device('cuda') → torch.device(TE_DEVICE_TYPE): <count>
- get_autocast_dtype('cuda') → get_autocast_dtype(TE_DEVICE_TYPE): <count>
- .device.type == 'cuda' → .device.type == TE_DEVICE_TYPE: <count>
Skipped <K> intentional guards and CUDA-specific blocks.
torch.cuda.* API calls left as-is (handled by vendor patches.py at runtime)."
```


### Stage 6: Detect & Fix Stale References in Fork-Specific Code (`/stage6-detect-stale-refs`)

Run the Repo Detection Preamble to ensure you are in the TransformerEngine-FL directory.

After Stages 7 and 8, the merge branch contains both upstream updates (from dev) and fork-specific
additions (from main). Because main was originally based on the base branch, fork-specific code may
reference functions, classes, or file paths that upstream has since renamed or relocated between base
and dev. These stale references won't cause merge conflicts (the fork code is new relative to dev),
but they will cause runtime errors — calling a function that no longer exists or importing from a
module that has moved.

**The core problem:** The merge branch = main + dev. Code that is new in the merge branch compared
to dev (i.e., fork-specific code) was written against the base branch's API surface. If dev renamed
`_load_cudnn` to `_load_cudnn_v2`, or moved `quantized_tensor.py` to `float8/quantized_tensor.py`,
the fork code still references the old names.

#### Step 1: Identify what upstream renamed or moved between base and dev

Build two inventories of upstream changes:

**1a. Renamed/removed Python symbols (functions, classes, constants):**

```bash
# Get all Python files that changed between base and dev (upstream evolution)
git diff --name-only base..dev -- '*.py' > /tmp/upstream_changed_py_files.txt

# Extract removed/renamed function and class definitions
# Lines starting with '-' that define functions or classes (but not '---' diff headers)
git diff base..dev -- '*.py' | \
  grep -E '^\-[^-]' | \
  grep -E '^\-(def |class |    def )' | \
  sed 's/^-//' | \
  sed 's/(.*//' | \
  sed 's/def //' | sed 's/class //' | \
  sed 's/://' | \
  tr -d ' ' | \
  sort -u > /tmp/upstream_removed_symbols.txt

# Extract added/renamed function and class definitions
git diff base..dev -- '*.py' | \
  grep -E '^\+[^+]' | \
  grep -E '^\+(def |class |    def )' | \
  sed 's/^+//' | \
  sed 's/(.*//' | \
  sed 's/def //' | sed 's/class //' | \
  sed 's/://' | \
  tr -d ' ' | \
  sort -u > /tmp/upstream_added_symbols.txt

# Symbols that were removed but NOT re-added are truly gone
# Symbols removed AND re-added with a different name are renames
comm -23 /tmp/upstream_removed_symbols.txt /tmp/upstream_added_symbols.txt \
  > /tmp/upstream_gone_symbols.txt

echo "=== Symbols removed/renamed in upstream (base→dev) ==="
cat /tmp/upstream_gone_symbols.txt
echo "Count: $(wc -l < /tmp/upstream_gone_symbols.txt)"
```

**1b. Relocated files (moved or renamed):**

```bash
# Detect file renames/moves between base and dev
git diff --diff-filter=R --name-status -M base..dev > /tmp/upstream_renamed_files.txt

# Also detect deleted files (might have been moved without git detecting the rename)
git diff --diff-filter=D --name-only base..dev > /tmp/upstream_deleted_files.txt

echo "=== Files renamed/moved in upstream ==="
cat /tmp/upstream_renamed_files.txt
echo ""
echo "=== Files deleted in upstream ==="
cat /tmp/upstream_deleted_files.txt
```

If both lists are empty, this stage is a no-op — skip to the commit step.

#### Step 2: Identify fork-specific code (new in merge branch vs dev)

This is the code at risk — it was written against the base branch API and has never been reconciled
with upstream's renames.

```bash
# Fork-specific content = lines present in current branch (merge result) but not in dev
# Focus on Python files in plugin/ and any fork-specific directories
git diff dev..HEAD -- '*.py' | \
  grep -E '^\+[^+]' | \
  grep -v '^+++' \
  > /tmp/fork_new_lines.txt

# Also get the list of fork-specific files (files that exist in HEAD but not in dev)
git diff --name-only dev..HEAD -- '*.py' > /tmp/fork_changed_files.txt

echo "=== Fork-specific changed files ==="
cat /tmp/fork_changed_files.txt
echo "Count: $(wc -l < /tmp/fork_changed_files.txt)"
```

#### Step 3: Cross-reference — find stale references

For each gone/renamed symbol from Step 1, check if fork-specific code references it:

```bash
echo "=== Scanning fork-specific code for stale references ==="
STALE_FOUND=0

while IFS= read -r symbol; do
  [ -z "$symbol" ] && continue
  # Search fork-changed files for references to this symbol
  MATCHES=$(grep -rn "$symbol" --include='*.py' \
    $(cat /tmp/fork_changed_files.txt) 2>/dev/null | \
    grep -v "^Binary" || true)
  if [ -n "$MATCHES" ]; then
    echo ""
    echo "⚠️  STALE REFERENCE: '$symbol' (removed/renamed in upstream)"
    echo "$MATCHES"
    STALE_FOUND=$((STALE_FOUND + 1))
  fi
done < /tmp/upstream_gone_symbols.txt

# Check for imports from relocated/deleted files
while IFS= read -r old_file; do
  [ -z "$old_file" ] && continue
  # Convert file path to module path for import matching
  MODULE=$(echo "$old_file" | sed 's/\.py$//' | sed 's/\//./g')
  BASENAME=$(basename "$old_file" .py)
  MATCHES=$(grep -rn "import.*$BASENAME\|from.*$MODULE" --include='*.py' \
    $(cat /tmp/fork_changed_files.txt) 2>/dev/null | \
    grep -v "^Binary" || true)
  if [ -n "$MATCHES" ]; then
    echo ""
    echo "⚠️  STALE IMPORT: references deleted/moved file '$old_file'"
    echo "$MATCHES"
    STALE_FOUND=$((STALE_FOUND + 1))
  fi
done < /tmp/upstream_deleted_files.txt

echo ""
echo "=== Summary: $STALE_FOUND stale reference(s) found ==="
```

#### Step 4: Resolve stale references

For each stale reference found, determine the correct replacement:

1. **Renamed symbol:** Find the new name in dev by searching for the function's context:
   ```bash
   # Example: if _load_cudnn was renamed, find what replaced it
   git log --all --oneline --diff-filter=M base..dev -- <file_containing_old_symbol>
   git diff base..dev -- <file_containing_old_symbol> | grep -A5 -B5 "old_symbol_name"
   ```
   The diff context usually shows the old name removed and the new name added nearby.

2. **Relocated file:** Use the rename detection from Step 1b, or search dev for the file:
   ```bash
   # Find where the file moved to
   git ls-tree -r --name-only dev | grep "<basename>"
   ```

3. **Truly removed (no replacement):** The upstream removed the functionality entirely. The fork
   code needs to be refactored to use the replacement API, or the old implementation needs to be
   kept as a fork-specific utility. Flag these for manual review.

Apply each fix, keeping fork-specific logic intact while updating references to match the current
upstream API surface.

#### Step 5: Verify fixes

```bash
# Re-run the stale reference scan — should find 0 issues
# (repeat Step 3 commands)

# Syntax-check all modified files
for f in $(git diff --name-only HEAD -- '*.py'); do
  python3 -c "import ast; ast.parse(open('$f').read())" 2>&1 && \
    echo "  ✅ $f" || echo "  ❌ $f"
done
```

#### Step 6: Commit

```bash
git add -A
pre-commit run --all-files
git add -A   # re-stage any formatting fixes
git commit -m "fix: update stale references in fork code to match upstream renames

Scanned fork-specific code (new in merge vs dev) for references to
functions, classes, and file paths that upstream renamed or relocated
between base and dev. Fixed <N> stale reference(s):
- <list of old_name → new_name replacements>
- <list of old_path → new_path import updates>"
```


### Stage 7: Build & Import Verification (`/stage7-basic-test`)

This stage validates that the merged code compiles and the core import chain works. It is the
first gate after all code changes (Stages 3–6). If this fails, nothing else matters — fix it
before moving on.

Run the Repo Detection Preamble to ensure you are in the TransformerEngine-FL directory.

#### Step 1: Environment Setup & Build

```bash
# Ensure we are in the TransformerEngine-FL directory (Repo Detection Preamble)
if [ -d "TransformerEngine-FL" ]; then
  cd TransformerEngine-FL
elif [ "$(basename $(pwd))" = "TransformerEngine-FL" ]; then
  echo "Already in TransformerEngine-FL"
elif [ -d "../TransformerEngine-FL" ]; then
  cd ../TransformerEngine-FL
else
  echo "ERROR: TransformerEngine-FL directory not found. Run /stage1-setup first."
  exit 1
fi
echo "Working directory: $(pwd)"

# Check and display current conda environment
conda info --envs
echo "Active conda env: $CONDA_DEFAULT_ENV"
conda info
python --version
which python
which pip

# Update third-party dependencies (upstream may bump submodule versions)
git submodule update --init --recursive

# NOTE: --no-build-isolation is REQUIRED. Without it, pip creates an isolated venv
# that won't have access to the current conda env's PyTorch/CUDA dependencies.
# NOTE: Do NOT remove build/ directories or .so files before installing.
# The editable install handles incremental builds correctly.
pip install --no-build-isolation -e . 2>&1 | tee build.log

# Verify shared objects were generated
find . -name "*.so" -newer build.log | head -10
```

#### Step 2: Import Verification

```bash
python -c "from transformer_engine import pytorch"
# Expected output:
# [CUDA] Successfully loaded CUDA libs
# [TE-FL manager.py INFO] OpManager initialized: 110 ops with 179 implementations
# [TE-FL manager.py INFO] Registered impl_ids: ['default.flagos', 'reference.torch', 'vendor.cuda']
```

If the import succeeds, the basic test passes — report success and proceed to Stage 8.

#### Step 3: Failure Diagnosis & Iterative Fix

If the import test fails (`from transformer_engine import pytorch`), do NOT proceed. Instead,
diagnose which of the previous stages (3/4/5/6) caused the issue and fix it.

```bash
# 3a: Trace the failing import chain
python -c "
import traceback
try:
    from transformer_engine import pytorch
except Exception:
    traceback.print_exc()
"

# 3b: Common failure patterns — analyze the traceback to identify the root cause:
#
#   - "transformer_engine_torch.xxx" AttributeError
#     → Upstream added new C++ APIs that the fork's plugin backend doesn't expose yet.
#     → Fix: Re-run Stage 4 (Plugin API Sync) to add the missing APIs.
#
#   - ImportError / ModuleNotFoundError for a moved/renamed module
#     → Upstream renamed or moved files between releases.
#     → Detect: git diff --name-status base..dev | grep '^R'  (renamed)
#              git diff --name-status base..dev | grep '^D'  (deleted/moved)
#     → Fix: Update fork imports to the new upstream path. This is typically
#       a Stage 3 (conflict resolution) or Stage 5 (CUDA patching) issue.
#
#   - SyntaxError or merge conflict markers in source
#     → A Stage 3 conflict resolution left bad content.
#     → Fix: Go back to Stage 3 and re-resolve the affected file.

# 3c: Cross-branch investigation to pinpoint the source
#   git show dev:<failing_file>    > /tmp/file_dev.py
#   git show main:<failing_file>   > /tmp/file_main.py
#   git show base:<failing_file>   > /tmp/file_base.py
#   diff /tmp/file_base.py /tmp/file_dev.py    # what upstream changed
#   diff /tmp/file_base.py /tmp/file_main.py   # what fork changed
#   diff /tmp/file_dev.py /tmp/file_main.py    # divergence between fork and upstream
# This reveals whether the bug is from a bad merge resolution, an upstream
# rename the fork still references, or a fork addition conflicting with upstream.

# 3d: Decision — fix directly or rollback
# If the problem can be directly fixed (e.g., missing import, typo, small patch):
#   → Fix it, commit with descriptive message, re-run Step 2.
# If the problem is systemic (e.g., entire stage needs re-execution):
#   → Rollback to the commit before that stage, re-execute the stage with
#     updated steps, then re-run from Step 1.

# 3e: After fixing, re-verify
python -c "from transformer_engine import pytorch; print('OK')"
python -c "from transformer_engine import te_device_type; print(te_device_type())"
```

Iterate until the import succeeds. Each fix should be committed separately with a descriptive message
(e.g., `fix: resolve import error for X after upstream merge`).

