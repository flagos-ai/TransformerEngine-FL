### Stage 8: Unit & Integration Tests (`/stage8-run-unit-tests`, `/stage8-run-full-cicd`)

After Stage 7 confirms the build and import work, run the full test suite. Three levels of testing,
each building on the previous. If any level fails, stop and diagnose before proceeding.

Run the Repo Detection Preamble to ensure you are in the TransformerEngine-FL directory.

#### Pre-check: CI Script Validation

Before running any tests, verify that the CI scripts in `qa/` reference test files that actually
exist. Upstream renames test files between releases (e.g. `test_float8tensor.py` →
`test_quantized_tensor.py`), and the TE-FL CI scripts may not have been updated to match.

Run this check:

```bash
# Extract all .py test file references from qa/L0_pytorch_unittest/test.sh and verify they exist
grep -oP '(?<=\$TE_PATH/)tests/pytorch/[^\s"]+\.py' qa/L0_pytorch_unittest/test.sh | sort -u | while read f; do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f"
  fi
done
# Also check directory references (e.g. nvfp4/)
grep -oP '(?<=\$TE_PATH/)tests/pytorch/[^\s"]+(?<!\.py)(?=")' qa/L0_pytorch_unittest/test.sh | sort -u | while read d; do
  if [ ! -e "$d" ]; then
    echo "MISSING DIR: $d"
  fi
done
```

If any files are reported as MISSING:

1. Check if the file was renamed upstream:
   ```bash
   git log --oneline --all --diff-filter=R --summary -- <missing-file> | head -5
   ```
2. Update the CI script to use the new filename.
3. Also check whether new test files exist in `tests/pytorch/` that are not yet referenced in the
   CI script — compare against the upstream version of `qa/L0_pytorch_unittest/test.sh`:
   ```bash
   git show upstream/main:qa/L0_pytorch_unittest/test.sh | grep -oP 'tests/pytorch/[^\s"]+\.py' | sort > /tmp/upstream_tests.txt
   grep -oP '(?<=\$TE_PATH/)tests/pytorch/[^\s"]+\.py' qa/L0_pytorch_unittest/test.sh | sort > /tmp/local_tests.txt
   diff /tmp/upstream_tests.txt /tmp/local_tests.txt
   ```
   Lines prefixed with `<` are in upstream but missing locally — add them if the files exist.

4. Fix any issues, run `pre-commit run --files qa/L0_pytorch_unittest/test.sh`, and commit before
   proceeding to the test levels below.

**Also check for tests that are in the skip list but never actually run** — a test appearing only
in the MetaX/CUDA skip block but not in any `run_test_step` call is a sign it was added to the
skip list when the test was introduced upstream, but the `run_test_step` call was never added.

**Known patterns to watch for after each upstream sync:**
- Test file renames (check `git log --diff-filter=R` on the upstream merge commit)
- New test files added under `tests/pytorch/` or `tests/pytorch/attention/`
- Tests moved into subdirectories (e.g. `test_attention.py` → `attention/test_attention.py`)

#### Per-Level Results Table

After each level finishes, parse the test output and display a results table to the console.
This gives immediate visibility into what passed/failed before moving to the next level.

The per-level table format (one row per test function):

| # | Test Name | Result | Duration | Details |
|---|-----------|--------|----------|---------|
| 1 | test_foo  | ✅ PASSED | 0.3s | |
| 2 | test_bar  | ❌ FAILED | 1.2s | AssertionError: expected X |
| 3 | test_baz  | ⚠️ SKIPPED | — | reason: no GPU |
|   | **Total** | **2/3 passed** | **1.5s** | **1 failed, 1 skipped** |

How to compile the table:
- Parse the pytest `-v` output: each line like `test_file.py::test_name PASSED/FAILED/SKIPPED` is a row
- Extract duration from the pytest summary line (e.g., `=== 42 passed in 5.23s ===`)
- For FAILED tests, include the first line of the failure reason from the `--tb=short` traceback
- For SKIPPED tests, include the skip reason if available
- Add a summary row at the bottom with totals

#### Level 1 — Plugin-Specific Tests
```bash
pytest transformer_engine/plugin/tests/ -k "plugin" -v --tb=short 2>&1 | tee plugin-test.log
```

**→ Display Level 1 results table, then continue.**

#### Level 2 — Integration Tests
```bash
pytest tests/pytorch/ -v --tb=short 2>&1 | tee integration-test.log
```

**→ Display Level 2 results table, then continue.**

#### Level 2.5 — CI Test Suites (OP API Validation)

This level runs three CI test suites in sequence: debug → unit → distributed. These validate the
plugin OP API interfaces across all vendor backends. This is critical after upstream merges because
upstream may change function signatures in ways that break the plugin dispatch layer.

```bash
cd "$TE_FL_DIR"
# L0 Debug tests
TE_PATH=$(pwd) bash qa/L0_pytorch_debug_unittest/test.sh 2>&1 | tee l0-debug.log
# L0 Unit tests
TE_PATH=$(pwd) bash qa/L0_pytorch_unittest/test.sh 2>&1 | tee l0-unittest.log
# L1 Distributed tests
TE_PATH=$(pwd) bash qa/L1_pytorch_distributed_unittest/test.sh 2>&1 | tee l1-distributed.log
```

**Common failure pattern: OP API signature mismatch**

If tests fail with errors like:
- `CUDABackend.fused_topk_with_score_function_bwd() takes 10 positional arguments but 11 were given`
- `CUDABackend.fused_score_for_moe_aux_loss_bwd() got an unexpected keyword argument 'grad_logits'`

This indicates that the upstream pytorch layer changed the call signature, but the plugin OP API
definition and backend implementations were not updated to match.

**Fix procedure:**

1. **Identify the failing OP** from the error message (e.g., `fused_topk_with_score_function_bwd`)

2. **Check the upstream caller** to see the expected signature:
   ```bash
   grep -n "tex.fused_topk_with_score_function_bwd" transformer_engine/pytorch/router.py
   ```
   Note all parameters being passed, including output tensors like `grad_logits`.

3. **Check the C++ extension signature** to confirm the expected interface:
   ```bash
   grep -A 10 "void fused_topk_with_score_function_bwd" transformer_engine/pytorch/csrc/extensions/router.cpp
   ```

4. **Update the OP API abstract method** in `transformer_engine/plugin/core/ops.py`:
   - Add any missing parameters (e.g., `grad_logits: torch.Tensor`)
   - Ensure parameter order matches the upstream caller

5. **Update ALL backend implementations** (not just CUDA):
   ```bash
   # Find all backends that implement this OP
   find transformer_engine/plugin/core/backends -name "*.py" | xargs grep -l "def fused_topk_with_score_function_bwd"
   ```
   
   For each backend file found:
   - Add the missing parameter to the function signature
   - Pass it through to the underlying `tex.*` call
   
   **Backends to check:**
   - `vendor/cuda/cuda.py`
   - `vendor/hygon/hygon.py`
   - `vendor/metax/metax.py`
   - `vendor/enflame/enflame.py`
   - `vendor/iluvatar/iluvatar.py`
   - `vendor/musa/musa.py`
   - `flagos/flagos.py` (if the OP is implemented)
   - `reference/reference.py` (if the OP is implemented)

6. **Run pre-commit** on all modified files:
   ```bash
   pre-commit run --files transformer_engine/plugin/core/ops.py \
     transformer_engine/plugin/core/backends/vendor/cuda/cuda.py \
     transformer_engine/plugin/core/backends/vendor/hygon/hygon.py \
     transformer_engine/plugin/core/backends/vendor/metax/metax.py \
     transformer_engine/plugin/core/backends/vendor/enflame/enflame.py \
     transformer_engine/plugin/core/backends/vendor/iluvatar/iluvatar.py \
     transformer_engine/plugin/core/backends/vendor/musa/musa.py
   ```

7. **Commit the fix**:
   ```bash
   git add transformer_engine/plugin/core/ops.py transformer_engine/plugin/core/backends/vendor/*/
   git commit -m "fix(plugin): update OP API signatures for <op_name>

   Upstream changed the call signature to include <new_parameter>.
   Updated abstract method in ops.py and all vendor backend implementations."
   ```

8. **Re-run all three CI test suites** to verify the fix:
   ```bash
   TE_PATH=$(pwd) bash qa/L0_pytorch_debug_unittest/test.sh 2>&1 | tee l0-debug-rerun.log
   TE_PATH=$(pwd) bash qa/L0_pytorch_unittest/test.sh 2>&1 | tee l0-unittest-rerun.log
   TE_PATH=$(pwd) bash qa/L1_pytorch_distributed_unittest/test.sh 2>&1 | tee l1-distributed-rerun.log
   ```

**→ Display Level 2.5 results table, then continue.**

#### Level 2.6 — Plugin Unit Tests (`run_all_tests.py`)

Run the plugin's own test suite, which validates backend registration, op dispatch, and
plugin-layer correctness independently of the upstream pytorch tests.

```bash
cd "$TE_FL_DIR"
python transformer_engine/plugin/tests/run_all_tests.py 2>&1 | tee plugin-run-all.log
```

If tests fail, check for:
- Missing op registrations in `transformer_engine/plugin/core/register_ops.py`
- Backend method not implemented (raises `NotImplementedError`)
- Import errors caused by renamed or removed upstream symbols

**→ Display Level 2.6 results table, then continue.**

#### Level 3 — End-to-End Tests
```bash
python tests/pytorch/test_sanity.py 2>&1 | tee e2e-test.log
```
For non-pytest scripts, parse stdout for pass/fail indicators and display a simplified table.

**→ Display Level 3 results table.**

#### Final Summary

After all levels complete (or on first failure), display the cumulative summary table:

| Level | Test Suite | Status | Passed | Failed | Skipped | Duration |
|-------|-----------|--------|--------|--------|---------|----------|
| L1 | Plugin Tests | ✅/❌ | N | N | N | Xs |
| L2 | Integration | ✅/❌ | N | N | N | Xs |
| L2.5a | L0 Debug Unit Tests | ✅/❌ | N | N | N | Xs |
| L2.5b | L0 PyTorch Unit Tests | ✅/❌ | N | N | N | Xs |
| L2.5c | L1 Distributed Tests | ✅/❌ | N | N | N | Xs |
| L2.6 | Plugin Unit Tests | ✅/❌ | N | N | N | Xs |
| L3 | End-to-End | ✅/❌ | N | N | N | Xs |
| | **Total** | | **N** | **N** | **N** | **Xs** |

If any level failed, list the specific failing tests below the summary table for quick reference.

Document the failure in the sync report so the next attempt can address it.

#### Bug Fix Commit

If any test failures are caused by plugin-layer issues (e.g., missing parameters, signature mismatches
between upstream callers and plugin wrappers), fix them immediately. After fixing:

1. **Run pre-commit** on all changed files:
   ```bash
   pre-commit run --files <changed-file-1> <changed-file-2> ...
   ```
   If pre-commit modifies files (e.g., black reformatting), re-run to confirm clean.

2. **Commit the fix** with a descriptive message:
   ```bash
   git add <changed-files>
   git commit -m "fix: <description of the bug fix>

   <details of what was wrong and what was fixed>"
   ```

3. **Re-run the failing test** to verify the fix before proceeding to the next level.

Remember: fixes should cover ALL vendor backends (cuda, enflame, hygon, iluvatar, metax, musa), not just the
one being tested. Check ops.py (abstract method) + all 6 vendor backend files.

---

### Stage 9: Merge to main (Tree Replacement Strategy) (`/stage9-merge-to-main`)

When the dev/merge branch (e.g. `merge/dev-to-main-20260410`) is a **superset** of main — meaning
all main's features have been incorporated during the preceding stages — use the tree replacement
strategy to create a clean merge commit suitable for a PR to main.

**Prerequisite:** The merge branch must be complete and verified (Stage 1–8 passed). If the merge
branch is missing features that exist on main, go back and fix it first — do NOT patch during
this stage.

#### Why tree replacement instead of `-X theirs`?

`-X theirs` resolves **conflicts** by taking theirs, but **non-conflicting changes from both
sides are still merged**. When both branches independently added the same patch (e.g. CUDA
patches, plugin additions), git sees them as non-conflicting additions and keeps both copies —
resulting in duplicate imports, duplicate code blocks, and broken code. Tree replacement avoids
this entirely.

#### Step 1: Identify the source branch

Ask the user to confirm the merge branch name. Default is `merge/dev-to-main-20260410`:

```
Which branch should be merged into main?
Default: merge/dev-to-main-20260410
```

Store the branch name:
```bash
MERGE_BRANCH="merge/dev-to-main-20260410"  # or user-provided value
```

#### Step 2: Create merge branch with tree replacement

```bash
# Ensure we're in the TransformerEngine-FL directory
cd "$TE_FL_DIR"

# Start from main
git checkout main
git pull origin main

# Create a new branch for the PR
git checkout -b merge-to-main-$(date +%Y%m%d)

# Tree replacement merge:
# "merge -s ours" records both parents but keeps main's tree,
# then "read-tree" replaces the tree with the merge branch's content.
git merge -s ours ${MERGE_BRANCH} --no-edit
git read-tree -m -u ${MERGE_BRANCH}

# Run pre-commit before finalizing
pre-commit run --all-files
# If pre-commit modified files, stage them
git add -A
git commit --amend --no-edit
```

After this, the working tree is **identical** to `${MERGE_BRANCH}`, but the commit has both
`main` and `${MERGE_BRANCH}` as parents (preserving full history).

#### Step 3: Remove intermediate sync records

The merge branch may contain files created during the sync process that should not land on main
(e.g., `SYNC_POINT.md`). Remove them:

```bash
# Remove intermediate sync/version record files
for f in SYNC_POINT.md MERGE_RECORD.md UPSTREAM_SYNC.md; do
    if [ -f "$f" ]; then
        git rm "$f"
    fi
done

# Commit if anything was removed
if ! git diff --cached --quiet; then
    git commit -m "chore: remove intermediate sync record files (not needed on main)"
fi
```

#### Step 4: Verify tree equality

```bash
# Should produce no output (or only the removed sync files)
git diff ${MERGE_BRANCH} HEAD --stat
```

If there is unexpected diff output beyond the removed sync files, something went wrong.
Investigate before proceeding.

#### Step 5: Incorporate new main commits (if any)

If `origin/main` received new commits after the merge branch was created:

```bash
git checkout main
git pull origin main
git checkout merge-to-main-$(date +%Y%m%d)
git merge main --no-edit
# Resolve any conflicts (typically few, since merge branch is a superset)
pre-commit run --all-files
git add -A
git commit  # if conflicts were resolved or pre-commit modified files
```

#### Step 6: Final verification

```bash
# No conflict markers
grep -rn "<<<<<<" transformer_engine/ tests/ .github/ 2>/dev/null | head

# History is correct — both parents present
git log --oneline --graph -5

# Build still works
pip install -e . 2>&1 | tail -5
python -c "import transformer_engine; print('OK')"
```

**Commit discipline:** Run `pre-commit run --all-files` before every commit in this stage.
If pre-commit modifies files, re-stage and re-run until clean.

**Checkpoint:**
1. Verify `git diff ${MERGE_BRANCH} HEAD` produces no output
2. Verify `git log --oneline --graph -3` shows both parents
3. Verify `pip install -e .` and import succeed
4. Verify `pre-commit run --all-files` passes clean

**Success criteria:** Merge branch tree equals `${MERGE_BRANCH}`, no duplicate code blocks, both
parents in history, pip-installable, pre-commit clean. PR is submitted manually by the user.

---

### Stage 10: FlagScale End-to-End Training Validation (`/stage10-flagscale-training`)

After the merge to main is prepared (Stage 9), validate the merged TransformerEngine-FL in a real
training scenario using FlagScale. This catches runtime integration issues that unit tests miss —
wrong tensor shapes, device mismatches, plugin dispatch failures under real workloads, etc.

**This stage delegates to the `e2e-stage-manager` skill.** Use it to:

1. Create a new unified stage config (e.g., stage11) with the correct model × implementation × backend matrix
2. Run batch training tests across all combinations
3. Compare results with previous stages

Refer to [`e2e-stage-manager/SKILL.md`](../e2e-stage-manager/SKILL.md) for full instructions on
config generation, batch execution, and result comparison.

#### Sync-specific prerequisites before invoking e2e-stage-manager

1. **Install the latest TE-FL** into the conda environment from the merge branch:
   ```bash
   cd /path/to/TransformerEngine-FL
   pip install -e . --no-build-isolation
   python -c "import transformer_engine; print(transformer_engine.__version__)"
   ```

2. **Run pylint** to catch lint errors before training (CI runs pylint with `set -e`):
   ```bash
   python3 -m pylint --recursive=y transformer_engine/common transformer_engine/pytorch transformer_engine/debug
   ```
   Fix any issues, run `pre-commit run --files <changed-files>`, and commit before proceeding.

3. **Use a separate FlagScale installation** — do NOT use the FlagScale in the same workspace as
   TransformerEngine-FL. The user must provide an absolute path to a compatible FlagScale repo.

4. **Run on the `merge-to-main-YYYYMMDD` branch** produced by Stage 9, not the intermediate merge branch.

#### Success criteria

At least one parameter combination completes 20 training steps without errors, and loss values are
decreasing. The batch comparison table (generated by e2e-stage-manager) makes it easy to spot which
combinations work and which need further investigation.

#### Fix-and-retry during training

If a combination fails, diagnose using these common patterns:

| Error Type | Root Cause | Fix |
|-----------|-----------|-----|
| TypeError / missing positional argument | Plugin API signature mismatch | Fix in ops.py + all vendor backends |
| AttributeError | Stale reference to renamed symbol | Find new name via `git diff base..dev` |
| RuntimeError: device mismatch | Un-patched `"cuda"` hardcoding | Replace with `TE_DEVICE_TYPE` |
| Plugin dispatch error | Missing op registration | Add to register_ops.py for all vendors |

After each fix: pre-commit, commit, rebuild (`pip install -e . --no-build-isolation`), then rerun
the same combination to verify. Apply fixes to ALL vendor backends (cuda, enflame, musa, iluvatar, hygon, metax).
