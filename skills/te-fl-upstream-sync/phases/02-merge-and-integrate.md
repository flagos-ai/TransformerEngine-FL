### Stage 3: Merge & Conflict Resolution (`/stage3-merge`, `/stage3-analyze-conflicts`, `/stage3-resolve-p0-conflict`, `/stage3-resolve-build-conflict`)

This is where upstream changes meet the fork's plugin system. Conflicts are expected and normal.
This stage covers both the merge itself and resolving any conflicts that arise.

Run the Repo Detection Preamble to ensure you are in the TransformerEngine-FL directory.

#### Step 1: Execute the merge (`/stage3-merge`)

1. Ensure working tree is clean:
   ```bash
   git status
   ```
   If dirty, ask the user to stash or commit first.

2. Checkout main:
   ```bash
   git checkout main
   ```

3. Create a dated merge branch (recommended for safety):
   ```bash
   git checkout -b merge/dev-to-main-$(date +%Y%m%d)
   ```

4. Execute the merge with no-fast-forward:
   ```bash
   git merge dev --no-ff -m "merge(dev): integrate upstream release_v2.14"
   ```

5. If the merge completes cleanly (rare but possible), skip to Stage 4.

6. If conflicts occur, list them:
   ```bash
   git diff --name-only --diff-filter=U
   ```

**Important:** Do NOT use `--strategy-option theirs` or `--strategy-option ours` globally. Each conflict
needs individual analysis based on the priority matrix.

#### Step 2: Auto-resolve unmodified files

Some conflicted files may not have been modified by the fork at all (i.e., `git diff base..main -- <file>`
is empty). For these files, it is safe to accept the upstream (dev) version directly.

```bash
for file in $(git diff --name-only --diff-filter=U); do
  if git diff base..main -- "$file" | grep -q '^'; then
    echo "FORK-MODIFIED: $file"
  else
    echo "AUTO-RESOLVE (take upstream): $file"
    git checkout --theirs "$file"
    git add "$file"
  fi
done
```

Only files printed as `FORK-MODIFIED` require priority-based resolution below.

#### Step 3: Analyze and categorize conflicts (`/stage3-analyze-conflicts`)

1. List all remaining conflicted files:
   ```bash
   git diff --name-only --diff-filter=U
   ```

2. Categorize each file into P0/P1/P2 based on path matching:
   - P0: paths containing `transformer_engine/pytorch/`
   - P1: `setup.py`, `CMakeLists.txt`, `pyproject.toml`, or files in core algorithm dirs
   - P2: everything else, such as `.github` cicd related

3. Present a table to the user showing file, priority, and recommended resolution strategy.

4. Suggest resolution order: all P0 first, then P1, then P2.

#### Step 4: Resolve P0 conflicts (`/stage3-resolve-p0-conflict <filepath>`)

For P0 files (plugin interfaces, CUDA patches):

1. Show the conflict diff for the specific file
2. Read the main branch version entirely. If newly add contents in main, copy to current branch.
3. Read the main branch version entirely. If `cuda` -> `te-device-type` related modification or other modification related to plugin system or patch system, apply to current branch.
3. Check security or critical bug, and fix it.

#### Step 5: Resolve P1 build conflicts (`/stage3-resolve-build-conflict <filepath>`)

For P1 build files (setup.py, CMakeLists.txt, pyproject.toml):

1. Show the three-way diff (base, ours, theirs)
2. Identify plugin-specific build sections in the fork version (look for comments like
   `# Plugin build targets`, or targets referencing `plugin/`)
3. Identify upstream improvements (new dependencies, version bumps, new build targets)
4. Merge manually: keep all plugin build sections from main, integrate upstream changes around them
5. After resolution, verify plugin build targets still exist:
   ```bash
   grep -n "plugin" setup.py CMakeLists.txt
   ```

For P2 files, use standard merge resolution — accept both sides where possible, prefer upstream
for upstream-specific content.

#### Step 6: Finalize merge

After all conflicts are resolved:
```bash
git add -A
pre-commit run --all-files
git add -A   # re-stage any formatting fixes
git commit --no-edit
```

### Stage 4: Plugin API Sync (`/stage4-add-apis`, `/stage4-verify-pybind-coverage`)

**Why this stage matters:** The main branch's plugin OP APIs were built to match the C++ bindings in
`pytorch/csrc/` as they existed in the base branch. After merging dev into main, `pytorch/csrc/` now
reflects the upstream release — which may have added new APIs, changed existing function signatures, or
removed deprecated ones. The plugin layer must be updated to stay consistent, otherwise you get
`AttributeError: module 'transformer_engine_torch' has no attribute 'xxx'` at runtime.

**Core principle:** Plugin OP APIs must mirror `pytorch/csrc/` pybind bindings 1:1. The diff between
base and dev in `pytorch/csrc/` tells you exactly what changed; the plugin layer must reflect those
same changes.

Run the Repo Detection Preamble to ensure you are in the TransformerEngine-FL directory.

**Key files to update:**
- `transformer_engine/plugin/core/ops.py` — base class `TEFLBackendBase` with abstract method stubs for every op
- Vendor backend implementations (one per vendor, each delegating to its own `tex` module):
  - `transformer_engine/plugin/core/backends/vendor/cuda/cuda.py` — `CUDABackend`, tex = `transformer_engine_torch_nv`
  - `transformer_engine/plugin/core/backends/vendor/enflame/enflame.py` — `EnflameBackend`, tex = `migration.patches.transformer_engine.v2_9_0`
  - `transformer_engine/plugin/core/backends/vendor/iluvatar/iluvatar.py` — `IluvatarBackend`, tex = `transformer_engine_iluvatar.pytorch.ixte_torch`
  - `transformer_engine/plugin/core/backends/vendor/metax/metax.py` — `MetaxBackend`, tex = `transformer_engine_torch_metax`
  - `transformer_engine/plugin/core/backends/vendor/musa/musa.py` — `MUSABackend`, tex = `transformer_engine_musa_torch`
  - `transformer_engine/plugin/core/backends/vendor/hygon/hygon.py` — `HygonBackend`, tex = `transformer_engine_torch_hygon`
- Vendor `register_ops.py` files (one per vendor, containing `OpImpl` registrations):
  - `transformer_engine/plugin/core/backends/vendor/cuda/register_ops.py`
  - `transformer_engine/plugin/core/backends/vendor/enflame/register_ops.py`
  - `transformer_engine/plugin/core/backends/vendor/iluvatar/register_ops.py`
  - `transformer_engine/plugin/core/backends/vendor/metax/register_ops.py`
  - `transformer_engine/plugin/core/backends/vendor/musa/register_ops.py`
  - `transformer_engine/plugin/core/backends/vendor/hygon/register_ops.py`
- Scan-only (check for changed interfaces, usually no changes needed):
  - `transformer_engine/plugin/core/backends/flagos/` — partial implementation, subset of ops
  - `transformer_engine/plugin/core/backends/reference/` — partial implementation, subset of ops

#### Step 1: Diff csrc between base and dev to identify all API changes

The base branch represents the upstream version that main's plugin APIs were originally built against.
The dev branch represents the new upstream release. Diffing base..dev shows exactly what changed.

```bash
# Full diff of C++ bindings between the old upstream (base) and new upstream (dev)
git diff base..dev -- transformer_engine/pytorch/csrc/ > /tmp/csrc_diff.diff

echo "=== csrc diff summary ==="
diffstat /tmp/csrc_diff.diff 2>/dev/null || echo "(diffstat not available, check /tmp/csrc_diff.diff manually)"
```

#### Step 2: Extract and categorize API changes (ADD / MODIFY / REMOVE)

Parse the diff to build a clear picture of what needs to change in the plugin layer.

```bash
# --- 2a: Newly ADDED pybind APIs (lines added with .def() in the diff) ---
echo "=== ADDED APIs ==="
grep -E '^\+.*\.def\(' /tmp/csrc_diff.diff | grep -v '^\+\+\+' | sed 's/^+//' | sed 's/^[[:space:]]*//'

# --- 2b: REMOVED pybind APIs (lines removed with .def() in the diff) ---
echo ""
echo "=== REMOVED APIs ==="
grep -E '^\-.*\.def\(' /tmp/csrc_diff.diff | grep -v '^\-\-\-' | sed 's/^-//' | sed 's/^[[:space:]]*//'

# --- 2c: MODIFIED APIs — appear in both added and removed with the same function name ---
echo ""
echo "=== MODIFIED APIs (name appears in both added and removed lines) ==="
ADDED_NAMES=$(grep -E '^\+.*\.def\("' /tmp/csrc_diff.diff | grep -v '^\+\+\+' | \
  sed 's/.*\.def("\([^"]*\)".*/\1/' | sort -u)
REMOVED_NAMES=$(grep -E '^\-.*\.def\("' /tmp/csrc_diff.diff | grep -v '^\-\-\-' | \
  sed 's/.*\.def("\([^"]*\)".*/\1/' | sort -u)
comm -12 <(echo "$ADDED_NAMES") <(echo "$REMOVED_NAMES")

# --- 2d: Purely new APIs (added but not in removed — truly new) ---
echo ""
echo "=== PURELY NEW APIs ==="
comm -23 <(echo "$ADDED_NAMES") <(echo "$REMOVED_NAMES")

# --- 2e: Purely removed APIs (removed but not re-added — truly deleted) ---
echo ""
echo "=== PURELY REMOVED APIs ==="
comm -13 <(echo "$ADDED_NAMES") <(echo "$REMOVED_NAMES")

# --- 2f: Also check Python-side references to transformer_engine_torch for new call sites ---
echo ""
echo "=== New Python-side transformer_engine_torch references ==="
git diff base..dev -- transformer_engine/pytorch/ ':(exclude)transformer_engine/pytorch/csrc/' | \
  grep -E '^\+.*transformer_engine_torch\.' | grep -v '^\+\+\+' | \
  sed 's/.*transformer_engine_torch\.//' | sed 's/[^a-zA-Z0-9_].*//' | sort -u
```

Review the output carefully. Save the categorized list — you will use it in the next steps.

#### Step 2g: Detect class-object parameter changes (indirect API changes)

Some plugin ops accept class objects (dataclasses, named tuples, or custom classes) as parameters
rather than primitive types. When the class definition changes (fields added, removed, or renamed),
the op's effective interface has changed even though its function signature is identical. The pybind
diff from Steps 2a–2f will NOT catch these — you must check explicitly.

**Why this matters:** Consider `get_attention_backend(attention_params: AttentionParams)`. If upstream
adds new fields to `AttentionParams` (e.g., `bottom_right_diagonal`, `cuda_graph`, `num_splits`),
the function signature is unchanged, so Steps 2a–2f report no change. But vendor backends that
construct or inspect `AttentionParams` fields will break or produce wrong results if they don't
account for the new fields. This is an indirect API change that must be detected and handled.

**How to detect:**

1. Identify all plugin ops whose parameters include class/dataclass types. The most common cases:
   - `get_attention_backend(attention_params: AttentionParams)` — `AttentionParams` is a dataclass
     in `transformer_engine/pytorch/attention/dot_product_attention/utils.py`
   - Any op parameter annotated with a class type defined in the TE codebase (not stdlib types)

2. For each such class, diff its definition between base and dev:

```bash
# --- 2g: Class-object parameter change detection ---
echo "=== Detecting class-object parameter changes ==="

# AttentionParams — consumed by get_attention_backend
echo ""
echo "--- AttentionParams field diff (base vs dev) ---"
echo "Base fields:"
git show base:transformer_engine/pytorch/attention/dot_product_attention/utils.py 2>/dev/null | \
  sed -n '/^class AttentionParams/,/^class \|^def /p' | grep -E "^\s+\w+\s*:" | sort
echo ""
echo "Dev fields:"
git show dev:transformer_engine/pytorch/attention/dot_product_attention/utils.py 2>/dev/null | \
  sed -n '/^class AttentionParams/,/^class \|^def /p' | grep -E "^\s+\w+\s*:" | sort

echo ""
echo "--- Fields ADDED in dev ---"
diff <(git show base:transformer_engine/pytorch/attention/dot_product_attention/utils.py 2>/dev/null | \
  sed -n '/^class AttentionParams/,/^class \|^def /p' | grep -E "^\s+\w+\s*:" | sed 's/[[:space:]]//g' | sort) \
  <(git show dev:transformer_engine/pytorch/attention/dot_product_attention/utils.py 2>/dev/null | \
  sed -n '/^class AttentionParams/,/^class \|^def /p' | grep -E "^\s+\w+\s*:" | sed 's/[[:space:]]//g' | sort) \
  | grep '^>' | sed 's/^> /  NEW: /'

echo ""
echo "--- Fields REMOVED in dev ---"
diff <(git show base:transformer_engine/pytorch/attention/dot_product_attention/utils.py 2>/dev/null | \
  sed -n '/^class AttentionParams/,/^class \|^def /p' | grep -E "^\s+\w+\s*:" | sed 's/[[:space:]]//g' | sort) \
  <(git show dev:transformer_engine/pytorch/attention/dot_product_attention/utils.py 2>/dev/null | \
  sed -n '/^class AttentionParams/,/^class \|^def /p' | grep -E "^\s+\w+\s*:" | sed 's/[[:space:]]//g' | sort) \
  | grep '^<' | sed 's/^< /  REMOVED: /'

# Repeat for any other class-object parameters found in plugin ops.
# To discover them, scan ops.py for parameter type annotations that reference TE-defined classes:
echo ""
echo "--- Other class-typed parameters in TEFLBackendBase ops ---"
grep -E "def \w+\(self.*:\s*[A-Z]\w+Params|def \w+\(self.*:\s*Optional\[[A-Z]\w+Params\]" \
  transformer_engine/plugin/core/ops.py | grep -v "FlashAttentionBase"
```

3. If new fields are found, treat the consuming op as MODIFIED — even though its signature is
   unchanged. Add it to the MODIFIED list from Step 2c. Then in Steps 4–5, update the plugin's
   handling of that op:
   - If the vendor backend passes the class object through transparently (e.g., CUDA/MetaX/MUSA
     just forward `attention_params` to the upstream implementation), no code change is needed —
     the new fields flow through automatically. Document this.
   - If the vendor backend constructs the class object, inspects its fields, or has custom logic
     that ignores the parameter (e.g., Hygon/Iluvatar/Reference use env-var logic and ignore
     `attention_params`), verify that the new fields don't break the custom logic. If the vendor
     ignores the parameter entirely, no change is needed — but document the finding.
   - If the class field change affects the op's return value or behavior in a way that downstream
     code depends on, the vendor implementation may need updating.

#### Step 3: Cross-reference with current plugin ops

Before making changes, check what already exists in the plugin layer to understand the gap.

```bash
# List all ops currently defined in the base class
echo "=== Current plugin op definitions (ops.py base class) ==="
grep -n "def " transformer_engine/plugin/core/ops.py | head -80

# List all backend implementations in the CUDA reference backend
echo ""
echo "=== Current CUDA backend implementations ==="
grep -n "def " transformer_engine/plugin/core/backends/vendor/cuda/cuda.py | head -80

# List all registered ops in the CUDA register_ops.py
echo ""
echo "=== Current CUDA registered ops ==="
grep "op_name" transformer_engine/plugin/core/backends/vendor/cuda/register_ops.py | head -80

# Cross-reference: find APIs from Step 2 that are NOT yet in the plugin
echo ""
echo "=== Missing from plugin (need to ADD) ==="
for api_name in $ADDED_NAMES; do
    if ! grep -q "$api_name" transformer_engine/plugin/core/ops.py 2>/dev/null; then
        echo "  MISSING: $api_name"
    fi
done

echo ""
echo "=== In plugin but signature may need UPDATE ==="
for api_name in $(comm -12 <(echo "$ADDED_NAMES") <(echo "$REMOVED_NAMES")); do
    if grep -q "$api_name" transformer_engine/plugin/core/ops.py 2>/dev/null; then
        echo "  CHECK: $api_name (exists in plugin, signature may have changed)"
    fi
done
```

This gives you a clear action list: which APIs to add, which to update, and which to remove.

Use the CUDA backend as the reference implementation — it is always the most complete and should be
updated first. All other vendor backends mirror CUDA's method signatures.

#### Step 4: Update base class op definitions (ops.py)

For each API change identified in Steps 2-3, update `transformer_engine/plugin/core/ops.py` (the
`TEFLBackendBase` class):

- **For ADDED APIs:** Add a new abstract method stub mirroring the C++ function's Python-visible
  signature. Read the full `.def(...)` block in the new csrc code to get the exact parameter names
  and types. Follow the pattern of existing methods in the class — each method is a thin stub that
  raises `NotImplementedError` or returns a default value.

- **For MODIFIED APIs:** Update the existing method signature to match the new parameters. Compare
  the old and new `.def(...)` blocks side by side. Common changes: added/removed parameters, changed
  default values, renamed arguments.

- **For REMOVED APIs:** If an API was removed from csrc, decide whether to:
  - Remove it from the plugin (if nothing in the fork depends on it)
  - Keep it with a deprecation warning (if fork-specific code still uses it)
  - Mark it clearly with a comment for manual review

- **CommOverlapP2P special case:** If new APIs relate to `CommOverlapP2P`, check whether they should
  be class methods on the `CommOverlapP2P` wrapper in ops.py (which routes to backend
  `create_comm_overlap_p2p`) or standalone methods on `TEFLBackendBase`.

#### Step 5: Update CUDA reference backend first (cuda.py + register_ops.py)

The CUDA backend is the reference implementation. Update it first, then replicate to other vendors.

**5a: Update `backends/vendor/cuda/cuda.py`**

Mirror every change from Step 4 in the `CUDABackend` class:

- **For ADDED ops:** Add a method that delegates to `tex.xxx(...)` where `tex` is the vendor's
  C extension module (`transformer_engine_torch_nv` for CUDA). Match the parameter list exactly
  with the ops.py base class definition. Key patterns to follow:
  - DType conversion: use `tex.DType(int(dtype))` when passing TE dtype enums to the C extension
  - Tensor parameters: pass through directly
  - Return values: return whatever `tex.xxx()` returns

- **For MODIFIED ops:** Update the method to pass the new parameters correctly. If parameters were
  added/removed, the `tex.xxx()` call must reflect that.

- **For REMOVED ops:** Remove or deprecate the corresponding method.

**5b: Update `backends/vendor/cuda/register_ops.py`**

For each ADDED op, add an `OpImpl` registration entry. Follow the existing pattern:

```python
OpImpl(
    op_name="new_api_name",
    impl_id="vendor.cuda",
    vendor="NVIDIA",
    priority=100,
    impl=CUDABackend.new_api_name,
    is_available=_bind_is_available(CUDABackend.new_api_name),
),
```

The `_bind_is_available` helper wraps the backend method for `OpImpl.is_available()` checks.
Place new entries in the same logical grouping as related existing ops.

#### Step 5c: Replicate to all other vendor backends

After CUDA is complete and verified, replicate the same changes to all other vendor backends.
Each vendor follows the identical pattern — only the class name, `tex` module, `impl_id`, and
`vendor` string differ:

| Vendor   | Class            | tex module                                      | impl_id           | vendor string |
|----------|------------------|------------------------------------------------|--------------------|---------------|
| CUDA     | `CUDABackend`    | `transformer_engine_torch_nv`                  | `vendor.cuda`      | `NVIDIA`      |
| Enflame  | `EnflameBackend` | `migration.patches.transformer_engine.v2_9_0`  | `vendor.enflame`   | `ENFLAME`     |
| Iluvatar | `IluvatarBackend`| `transformer_engine_iluvatar.pytorch.ixte_torch`| `vendor.iluvatar`  | `Iluvatar`    |
| MetaX    | `MetaxBackend`   | `transformer_engine_torch_metax`               | `vendor.metax`     | `METAX`       |
| MUSA     | `MUSABackend`    | `transformer_engine_musa_torch`                | `vendor.musa`      | `MUSA`        |
| Hygon    | `HygonBackend`   | `transformer_engine_torch_hygon`               | `vendor.hygon`     | `HYGON`       |

For each vendor, update both files:
1. `{vendor}/{vendor}.py` — add/modify/remove the same methods as CUDA, delegating to that
   vendor's `tex` module
2. `{vendor}/register_ops.py` — add/modify/remove the same `OpImpl` entries with the vendor's
   `impl_id` and `vendor` string

**Efficiency tip:** Since all vendors have identical method bodies (only `tex` module differs),
you can write a batch script to apply the same changes across all vendors simultaneously rather
than editing each file manually. Diff the CUDA register_ops op_names against each vendor's to
identify exactly which ops are missing.

**Note on Enflame:** Enflame's `tex` module is `migration.patches.transformer_engine.v2_9_0`
(loaded lazily via `_get_tex()`). It also has a custom `flash_attention.py` and
`get_attention_backend` implementation that routes through its own migration layer. When adding
new ops, follow the same delegation pattern as other vendors.

**Note on Hygon:** Hygon may have a pre-existing gap in OpImpl count compared to other vendors
(e.g., 8 fewer ops). This is expected — only add the new ops from this sync, don't try to
backfill the pre-existing gap.

**Critical: Explicit parameter signatures required.** All vendor backend methods MUST use
explicit parameter lists matching the CUDA backend exactly. Do NOT use `*args, **kwargs` as a
shortcut — this hides interface mismatches and causes silent failures when upstream adds new
parameters. The only exceptions are methods where CUDA itself uses `*args, **kwargs` (e.g.,
`te_general_grouped_gemm_for_*`, `nvfp4_compute_per_block_scale`, `nvfp4_expand_scale_to_fp8`,
`nvfp4_fused_scale`, `nvfp4_multi_tensor_2d_partial_cast`).

Example — correct:
```python
def mxfp8_scaling_compute_partial_amax(
    self,
    tensor: torch.Tensor,
    amax: torch.Tensor,
    h: int,
    w: int,
    start_offset: int,
    block_len: int,
) -> None:
    tex = self._get_tex()
    return tex.mxfp8_scaling_compute_partial_amax(tensor, amax, h, w, start_offset, block_len)
```

Example — wrong (do not do this):
```python
def mxfp8_scaling_compute_partial_amax(self, *args, **kwargs):
    tex = self._get_tex()
    return tex.mxfp8_scaling_compute_partial_amax(*args, **kwargs)
```

After adding/modifying methods in all vendors, verify consistency:
```bash
# Verify no new *args/**kwargs were introduced (excluding known exceptions)
for vendor in enflame hygon iluvatar metax musa; do
  f="transformer_engine/plugin/core/backends/vendor/$vendor/$vendor.py"
  echo "=== $vendor ==="
  grep -n "def.*\*args.*\*\*kwargs" "$f" | \
    grep -v "te_general_grouped_gemm_for_\|nvfp4_compute_per_block_scale\|nvfp4_expand_scale_to_fp8\|nvfp4_fused_scale\|nvfp4_multi_tensor_2d_partial_cast"
done
# If any output appears, those methods need explicit parameter lists from CUDA.
```

#### Step 5d: Scan flagos and reference backends for changed interfaces

The `flagos/` and `reference/` backends are partial implementations that only cover a subset of
ops. They do NOT need new method stubs for newly added APIs. However, if any MODIFIED APIs
(signature changes) affect methods that these backends implement, those signatures must be updated.

```bash
# Check if any of the modified API names exist in flagos/reference backends
echo "=== Checking flagos backend for modified APIs ==="
for api_name in <MODIFIED_API_NAMES>; do
    grep -rn "def $api_name" transformer_engine/plugin/core/backends/flagos/ 2>/dev/null
done

echo ""
echo "=== Checking reference backend for modified APIs ==="
for api_name in <MODIFIED_API_NAMES>; do
    grep -rn "def $api_name" transformer_engine/plugin/core/backends/reference/ 2>/dev/null
done
```

If any matches are found, update those method signatures to match the new parameters from Step 4.
If no matches are found (the common case), no changes are needed — document this in the commit log.

#### Step 5e: Add plugin-to-tex type conversions for new methods

The plugin layer (ops.py) defines its own Python enum types (`DType`, `NVTE_QKV_Layout`,
`NVTE_Bias_Type`, `NVTE_Mask_Type`, `NVTE_Softmax_Type`, `NVTE_QKV_Format`, `CommOverlapType`,
etc.) that mirror the C++ enums but are distinct Python objects. Each vendor's `tex` module has
its own versions of these enums (e.g., `tex.DType`, `tex.NVTE_QKV_Layout`). Since they share
the same integer values but are different types, conversion is required at the boundary when
calling `tex.*()`.

After adding all new methods to vendor backends, audit each one for parameters that carry
plugin-layer enum types or objects containing such types. Two conversion patterns apply:

**Pattern 1 — Standalone enum parameters:**
When a method receives a bare enum parameter (e.g., `dtype: DType`, `otype: DType`,
`qkv_layout`, `bias_type`, `attn_mask_type`), convert it before passing to `tex`:

```python
dtype = tex.DType(int(dtype)) if dtype is not None else None
qkv_layout = tex.NVTE_QKV_Layout(int(qkv_layout)) if qkv_layout is not None else None
bias_type = tex.NVTE_Bias_Type(int(bias_type)) if bias_type is not None else None
comm_type = tex.CommOverlapType(int(comm_type)) if comm_type is not None else None
```

The general form is `tex.<EnumType>(int(param))` with a `None` guard when the parameter is
optional. Look at the pybind `.def()` signature to determine which parameters are enums — they
will have C++ types like `DType`, `NVTE_QKV_Layout`, etc.

**Pattern 2 — Object attribute conversion (quantizer pattern):**
When a method receives a complex object (like a `quantizer`) that internally holds enum-typed
attributes, those attributes must be normalized before the object is passed to `tex`. The
quantizer's `.dtype` attribute is the most common case:

```python
# Normalize quantizer.dtype to this backend's `tex.DType`.
try:
    if quantizer is not None and hasattr(quantizer, "dtype") and hasattr(tex, "DType"):
        qdtype = quantizer.dtype
        if qdtype is not None:
            quantizer.dtype = tex.DType(int(qdtype))
except Exception:
    pass
```

This pattern is defensive (`try/except`, `hasattr` checks) because the quantizer object comes
from the Python layer and its structure may vary. Currently applies to `quantize` and
`bgrad_quantize`, but any new API that accepts a quantizer or similar wrapper object needs the
same treatment.

**How to identify which new methods need conversion:**
1. Read each new method's pybind `.def()` signature in csrc
2. Any parameter typed as a TE enum (`DType`, `NVTE_*`, `CommOverlapType`, etc.) → Pattern 1
3. Any parameter that is a complex object containing TE enums (e.g., quantizer) → Pattern 2
4. Plain tensors, ints, floats, bools, strings → no conversion needed

This step applies to all vendor backends equally — the conversion logic is identical, only the
`tex` module differs.

#### Step 6: Verify API consistency across all backends

Before committing, verify that the plugin layer is fully consistent with csrc across all vendors.

```bash
echo "=== Verification: Plugin API Consistency Check ==="

# 6a: Extract all pybind API names from the current (merged) csrc
ALL_CSRC_APIS=$(grep -rh '\.def("' transformer_engine/pytorch/csrc/ 2>/dev/null | \
  sed 's/.*\.def("\([^"]*\)".*/\1/' | sort -u)

# 6b: Extract all method names from the base class (ops.py)
ALL_OPS_METHODS=$(grep -E "^\s+def " transformer_engine/plugin/core/ops.py 2>/dev/null | \
  sed 's/.*def \([a-zA-Z_][a-zA-Z0-9_]*\).*/\1/' | grep -v "^__" | sort -u)

# 6c: Find csrc APIs missing from ops.py base class
echo "--- csrc APIs missing from ops.py (should be empty) ---"
comm -23 <(echo "$ALL_CSRC_APIS") <(echo "$ALL_OPS_METHODS")

# 6d: Verify all vendor backends have the same method count as CUDA reference
echo ""
echo "=== Vendor backend method counts (should match CUDA) ==="
for vendor in cuda enflame iluvatar metax musa hygon; do
    VENDOR_FILE="transformer_engine/plugin/core/backends/vendor/$vendor/$vendor.py"
    if [ -f "$VENDOR_FILE" ]; then
        COUNT=$(grep -c "^\s*def " "$VENDOR_FILE" 2>/dev/null)
        echo "  $vendor: $COUNT methods"
    else
        echo "  $vendor: FILE NOT FOUND"
    fi
done

# 6e: Verify all vendor register_ops have consistent OpImpl counts
echo ""
echo "=== Vendor register_ops OpImpl counts ==="
for vendor in cuda enflame iluvatar metax musa hygon; do
    REG_FILE="transformer_engine/plugin/core/backends/vendor/$vendor/register_ops.py"
    if [ -f "$REG_FILE" ]; then
        COUNT=$(grep -c "op_name" "$REG_FILE" 2>/dev/null)
        echo "  $vendor: $COUNT op_name entries"
    else
        echo "  $vendor: FILE NOT FOUND"
    fi
done

# 6f: Quick sanity check
CSRC_COUNT=$(echo "$ALL_CSRC_APIS" | wc -l | tr -d ' ')
OPS_COUNT=$(echo "$ALL_OPS_METHODS" | wc -l | tr -d ' ')
echo ""
echo "csrc API count: $CSRC_COUNT"
echo "ops.py method count: $OPS_COUNT"

if [ "$CSRC_COUNT" -le "$OPS_COUNT" ]; then
    echo "✅ Plugin covers all csrc APIs"
else
    echo "⚠️  Plugin may be missing $(($CSRC_COUNT - $OPS_COUNT)) API(s) — review the list above"
fi

# 6g: Syntax-check all modified Python files
echo ""
echo "=== Syntax validation ==="
for f in transformer_engine/plugin/core/ops.py \
         transformer_engine/plugin/core/backends/vendor/cuda/cuda.py \
         transformer_engine/plugin/core/backends/vendor/cuda/register_ops.py \
         transformer_engine/plugin/core/backends/vendor/enflame/enflame.py \
         transformer_engine/plugin/core/backends/vendor/enflame/register_ops.py \
         transformer_engine/plugin/core/backends/vendor/iluvatar/iluvatar.py \
         transformer_engine/plugin/core/backends/vendor/iluvatar/register_ops.py \
         transformer_engine/plugin/core/backends/vendor/metax/metax.py \
         transformer_engine/plugin/core/backends/vendor/metax/register_ops.py \
         transformer_engine/plugin/core/backends/vendor/musa/musa.py \
         transformer_engine/plugin/core/backends/vendor/musa/register_ops.py \
         transformer_engine/plugin/core/backends/vendor/hygon/hygon.py \
         transformer_engine/plugin/core/backends/vendor/hygon/register_ops.py; do
    python3 -c "import ast; ast.parse(open('$f').read())" 2>&1 && echo "  ✅ $f" || echo "  ❌ $f"
done
```

If the verification shows missing APIs, go back to Steps 4-5 and fix them before proceeding.

#### Step 7: Sync FlashAttention Class Hierarchy with Upstream

Steps 1–6 above handle `TEFLBackendBase` ops (including ordinary methods like `get_attention_backend`
that accept class-object parameters — see Step 2g for how those are detected). This step handles a
separate concern: the **FlashAttention class hierarchy**, which is independent of `TEFLBackendBase`
and uses its own inheritance-based plugin pattern.

**Architecture:** The plugin defines `FlashAttentionBase` in `ops.py` (inherits `torch.nn.Module` +
`ABC`). Each vendor provides a subclass that either delegates to a vendor-specific flash attention
library or implements its own attention logic:

- **Delegation pattern** (CUDA, MetaX, MUSA, Hygon, Iluvatar): The vendor subclass's `_forward_impl`
  has the same signature as `FlashAttentionBase._forward_impl` and delegates the actual computation
  to a vendor library. When upstream adds a new parameter, these subclasses need the parameter added
  to their signature and passed through to the delegation call.
- **Custom implementation pattern** (KunlunXin, Reference, FlagOS): The vendor subclass implements
  its own attention logic (e.g., using `torch.nn.functional.scaled_dot_product_attention` or
  `flag_gems`). When upstream adds a new parameter, these subclasses need the parameter in their
  signature but may choose to ignore it or implement support for it depending on their backend's
  capabilities.

**Note on `get_attention_backend` and `FusedAttention`:** `get_attention_backend` is an ordinary
method inside `TEFLBackendBase` — it is handled by Steps 1–6 (with class-object parameter changes
like `AttentionParams` detected by Step 2g). `FusedAttention` is an upstream class that the plugin
does NOT wrap (no `FusedAttentionBase` exists) — upstream's class is used directly, so signature
changes to `FusedAttention.forward()` do not require plugin changes.

**Files to update:**
- `transformer_engine/plugin/core/ops.py` — `FlashAttentionBase._forward_impl` + `forward` signatures
- `transformer_engine/plugin/core/backends/vendor/cuda/flash_attention.py`
- `transformer_engine/plugin/core/backends/vendor/enflame/flash_attention.py`
- `transformer_engine/plugin/core/backends/vendor/hygon/flash_attention.py`
- `transformer_engine/plugin/core/backends/vendor/metax/flash_attention.py`
- `transformer_engine/plugin/core/backends/vendor/musa/flash_attention.py`
- `transformer_engine/plugin/core/backends/vendor/iluvatar/flash_attention.py` (if exists)
- `transformer_engine/plugin/core/backends/vendor/kunlunxin/flash_attention.py` (if exists)
- `transformer_engine/plugin/core/backends/reference/flash_attention.py`
- `transformer_engine/plugin/core/backends/flagos/attention/dot_product_attention/backends.py`

**7a: Diff upstream FlashAttention.forward() signature changes**

```bash
# --- FlashAttention.forward() ---
echo "=== Base branch FlashAttention.forward signature ==="
git show base:transformer_engine/pytorch/attention/dot_product_attention/backends.py 2>/dev/null | \
  sed -n '/class FlashAttention/,/"""flash-attn fprop"""/p' | grep -E "^\s+\w+.*[:=]"

echo ""
echo "=== Dev branch (current) FlashAttention.forward signature ==="
sed -n '/class FlashAttention(torch.nn.Module)/,/"""flash-attn fprop"""/p' \
  transformer_engine/pytorch/attention/dot_product_attention/backends.py | grep -E "^\s+\w+.*[:=]"
```

For each interface, identify parameters that were added, removed, or had their defaults changed.

For `get_attention_backend`: the function signature may be unchanged while `AttentionParams` gains
new fields — check both. Also check whether the function body reads new fields from
`AttentionParams` (compare the field extraction block at the top of the function).

For `FusedAttention`: if the plugin does not wrap it (no `FusedAttentionBase` in `ops.py`), then
signature changes do not require plugin updates — document this finding and move on.

Report the diff results. If no parameters were added/removed/changed, FlashAttention needs no plugin
updates — document this and skip to Step 8. If changes exist, proceed with 7b–7e below.

**7b: Compare with plugin FlashAttentionBase**

```bash
echo "=== Plugin FlashAttentionBase._forward_impl signature ==="
sed -n '/class FlashAttentionBase/,/raise NotImplementedError/p' \
  transformer_engine/plugin/core/ops.py | grep -E "^\s+\w+.*[:=]"

echo ""
echo "=== Plugin FlashAttentionBase.forward signature ==="
sed -n '/class FlashAttentionBase/,/def backend_name/p' \
  transformer_engine/plugin/core/ops.py | grep -E "^\s+\w+.*[:=]"
```

Cross-reference each parameter with the upstream `FlashAttention.forward()` signature from 7a.
Any parameter present in upstream but missing from the plugin is a gap that must be filled.

**7c: Update FlashAttentionBase in ops.py**

For each new/changed parameter identified in 7a–7b:

1. Add the parameter to `_forward_impl()` abstract method signature (with the same default value as upstream)
2. Add the parameter to `forward()` method signature (same default)
3. Add the parameter to the `_forward_impl()` call inside `forward()` (the direct call path)
4. Add the parameter to the `call_impl_fn` lambda/closure inside `forward()` (the fallback dispatch path)

Preserve the parameter ordering from upstream. New parameters typically go at the end, before
`**kwargs` if present.

**7d: Update all vendor FlashAttention subclasses**

For each vendor's `flash_attention.py`:

1. Add the new parameter(s) to `_forward_impl()` with the same default value
2. For **delegation-pattern** vendors (CUDA, MetaX, MUSA, Hygon, Iluvatar): pass the new parameter
   through to the delegation call (e.g., the `tex.fused_attn_*` call or the upstream function call)
3. For **custom-implementation** vendors (KunlunXin, Reference, FlagOS): add the parameter to the
   signature. Whether to implement support depends on the backend's capabilities:
   - If the parameter controls an optimization the backend doesn't support (e.g., `num_splits` for
     a torch SDPA backend), accept the parameter but don't use it — the default value should be
     safe to ignore
   - If the parameter changes semantics (e.g., a new attention mask type), the implementation may
     need updating

```bash
# Check all vendor flash_attention files for current _forward_impl signatures
echo "=== Vendor FlashAttention _forward_impl signatures ==="
for vendor in cuda enflame hygon metax musa iluvatar kunlunxin; do
    FILE="transformer_engine/plugin/core/backends/vendor/$vendor/flash_attention.py"
    if [ -f "$FILE" ]; then
        echo "--- $vendor ---"
        grep -A 30 "def _forward_impl" "$FILE" | head -35 | grep -E "^\s+\w+.*[:=]"
        echo ""
    fi
done

echo "--- reference ---"
grep -A 30 "def _forward_impl" \
  transformer_engine/plugin/core/backends/reference/flash_attention.py | head -35 | grep -E "^\s+\w+.*[:=]"

echo ""
echo "--- flagos ---"
grep -A 30 "def _forward_impl" \
  transformer_engine/plugin/core/backends/flagos/attention/dot_product_attention/backends.py | \
  head -35 | grep -E "^\s+\w+.*[:=]"
```

Update each file to include the new parameter(s). For delegation-pattern vendors, also update the
delegation call to pass the new parameter through.

**7e: Update flagos and reference backends (custom implementations)**

The flagos (`FlashAttentionFL`) and reference (`FlashAttentionTorch`) backends have their own
attention implementations rather than delegating to upstream. When new parameters are added:

1. Add the parameter to `_forward_impl()` signature (same default as upstream)
2. Decide whether to implement support:
   - Parameters that affect the core attention computation (e.g., new mask types, new dropout
     modes) should be implemented if the backend supports them, or raise `NotImplementedError`
     with a clear message if not
   - Parameters that are optimization hints (e.g., `num_splits` for controlling parallelism)
     can safely be accepted and ignored — the default value produces correct results
3. For KunlunXin's `FlashAttentionTorch` (uses `torch.nn.functional.scaled_dot_product_attention`):
   check if PyTorch's SDPA supports the new parameter natively

```bash
# Verify upstream call sites pass the new parameter(s)
echo "=== Upstream flash_attention call sites ==="
grep -n "self\.flash_attention(" transformer_engine/pytorch/ -r --include="*.py" -A 30 | \
  grep -E "self\.flash_attention\(|^\s+\w+\s*=" | head -10

echo ""
echo "=== Plugin FlashAttentionBase params (after update) ==="
grep -A 30 "def _forward_impl" transformer_engine/plugin/core/ops.py | \
  head -35 | grep -E "^\s+\w+.*[:=]"
```

Confirm every argument passed at each upstream call site is accepted by the plugin's
`FlashAttentionBase._forward_impl` and all vendor subclass implementations.

#### Step 8: Commit and log all changes

```bash
# Build a detailed change log
LOG_FILE="/tmp/plugin_api_changes.log"
echo "=== Plugin API changes for upstream sync ===" > "$LOG_FILE"
echo "Date: $(date -Iseconds)" >> "$LOG_FILE"
echo "Diff base: base..dev" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

echo "--- Files changed ---" >> "$LOG_FILE"
git diff --name-only HEAD >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

echo "--- Op definition changes (ops.py base class) ---" >> "$LOG_FILE"
git diff HEAD -- transformer_engine/plugin/core/ops.py >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

echo "--- Vendor backend changes ---" >> "$LOG_FILE"
for vendor in cuda enflame iluvatar metax musa hygon; do
    echo "=== $vendor ===" >> "$LOG_FILE"
    git diff HEAD -- "transformer_engine/plugin/core/backends/vendor/$vendor/" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
done

cat "$LOG_FILE"

git add -A
pre-commit run --all-files
git add -A   # re-stage any formatting fixes
git commit -m "plugin: sync plugin APIs with upstream csrc changes

Updated plugin OP API layer to match pytorch/csrc/ pybind changes
between base and dev branches. Changes applied to:
- ops.py base class (TEFLBackendBase)
- ops.py FlashAttentionBase (synced forward/\_forward\_impl signatures with upstream FlashAttention)
- All vendor FlashAttention subclasses (cuda, enflame, hygon, metax, musa, iluvatar, kunlunxin)
- All 6 vendor backends (cuda, enflame, iluvatar, metax, musa, hygon)
- All 6 vendor register_ops.py files
- Scanned flagos/reference backends for changed interfaces
See /tmp/plugin_api_changes.log for details."
```


#### Step 9: Full-Surface Pybind Coverage Audit (`/stage4-verify-pybind-coverage`)

Steps 1–8 above handle new/modified/removed APIs found by diffing `pytorch/csrc/` between base and dev.
But omissions can also come from:
- Pybind exports in `common/util/pybind_helper.h` (enums, utility functions via `NVTE_DECLARE_COMMON_PYBIND11_HANDLES`)
- Functions added in earlier upstream versions that were never wrapped in the plugin layer
- Exports that the diff-based steps missed (e.g., unchanged but unwrapped)

This step does a full-surface audit: extract every pybind-exported symbol, compare against the plugin
layer, and flag any gaps.

**Key source files:**
- `transformer_engine/pytorch/csrc/extensions/pybind.cpp` — main `PYBIND11_MODULE` with `m.def()` calls
- `transformer_engine/common/util/pybind_helper.h` — `NVTE_DECLARE_COMMON_PYBIND11_HANDLES` macro (enums + utility functions)
- `transformer_engine/plugin/core/ops.py` — `TEFLBackendBase` class (abstract method stubs for every op)
- `transformer_engine/plugin/core/backends/vendor/cuda/register_ops.py` — `OpImpl` registrations

#### Step 1: Extract all pybind-exported function names

```bash
# 1a: Extract m.def() function names from pybind.cpp
grep -oP 'm\.def\("\K[^"]+' transformer_engine/pytorch/csrc/extensions/pybind.cpp | sort -u > /tmp/pybind_exports.txt

# 1b: Extract utility function names from pybind_helper.h
# These are inside NVTE_DECLARE_COMMON_PYBIND11_HANDLES macro
grep -oP 'm\.def\("\K[^"]+' transformer_engine/common/util/pybind_helper.h >> /tmp/pybind_exports.txt

# 1c: Check for any other pybind modules in common/ or pytorch/csrc/
grep -rl 'PYBIND11_MODULE\|m\.def(' transformer_engine/common/ transformer_engine/pytorch/csrc/ 2>/dev/null | \
  grep -v pybind.cpp | grep -v pybind_helper.h

sort -u -o /tmp/pybind_exports.txt /tmp/pybind_exports.txt
echo "Total pybind exports: $(wc -l < /tmp/pybind_exports.txt)"
```

#### Step 2: Extract all plugin-layer op names

```bash
# 2a: Method names from TEFLBackendBase in ops.py
grep -oP 'def \K\w+' transformer_engine/plugin/core/ops.py | grep -v '^_' | sort -u > /tmp/plugin_ops.txt

# 2b: Registered op_names from CUDA register_ops.py (most complete vendor)
grep -oP 'op_name="\K[^"]+' transformer_engine/plugin/core/backends/vendor/cuda/register_ops.py | sort -u > /tmp/registered_ops.txt

echo "TEFLBackendBase methods: $(wc -l < /tmp/plugin_ops.txt)"
echo "CUDA registered ops: $(wc -l < /tmp/registered_ops.txt)"
```

#### Step 3: Find omissions

```bash
# 3a: Pybind exports missing from TEFLBackendBase
echo "=== Pybind exports NOT in TEFLBackendBase ==="
comm -23 /tmp/pybind_exports.txt /tmp/plugin_ops.txt

# 3b: Pybind exports missing from CUDA register_ops.py
echo ""
echo "=== Pybind exports NOT in CUDA register_ops.py ==="
comm -23 /tmp/pybind_exports.txt /tmp/registered_ops.txt

# 3c: In TEFLBackendBase but not registered (abstract method with no implementation)
echo ""
echo "=== In TEFLBackendBase but NOT registered in CUDA ==="
comm -23 /tmp/plugin_ops.txt /tmp/registered_ops.txt
```

#### Step 4: Categorize and triage omissions

For each omission found in Step 3, categorize it:

| Category | Action |
|----------|--------|
| Compute ops (kernels, gemm, attention, norm, activation) | Must add: ops.py stub + vendor impl + register_ops.py |
| Utility/query functions (device_supports_X, get_version, etc.) | May skip if only used internally by the C++ layer, or add if Python code calls `tex.xxx()` |
| Enum/class exports (DType, FP8TensorMeta, etc.) | Handled separately by TEFLModule constructor — verify they're in the enum/class setup |

For each compute op omission, check if Python code actually calls it:
```bash
# For each missing op, check if it's called via tex.xxx
for op in $(comm -23 /tmp/pybind_exports.txt /tmp/registered_ops.txt); do
    count=$(grep -r "tex\.$op" transformer_engine/pytorch/ --include="*.py" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "CRITICAL: tex.$op called $count times but not in plugin layer"
    else
        echo "LOW: $op not called via tex in Python code"
    fi
done
```

#### Step 5: Fix omissions (same pattern as Steps 4–5 above)

For each CRITICAL omission (pybind export that Python code calls via `tex.xxx` but plugin doesn't wrap):

1. Add abstract method stub to `TEFLBackendBase` in `ops.py`, matching the pybind signature
2. Add implementation to `CUDABackend` in `cuda.py` that delegates to `self._get_tex().xxx(...)`
3. Add `OpImpl` registration in `cuda/register_ops.py`
4. Repeat for ALL other vendor backends (iluvatar, metax, musa, hygon) — each vendor must get:
   - A method in its backend class (e.g., `iluvatar.py`, `metax.py`, `musa.py`, `hygon.py`)
     that delegates to its own native tex module via `self._get_tex().xxx(...)`
   - An `OpImpl` registration in its `register_ops.py`
   - If the vendor's native module doesn't support the function, implement a safe fallback
     (e.g., return `False` for query functions, raise `NotImplementedError` for compute ops)
5. Check if flagos/reference backends need an implementation

For LOW-priority omissions (not called via `tex` in Python), document them but don't add unless needed.

#### Step 6: Verify completeness

```bash
# Re-run the comparison to confirm zero critical omissions remain
echo "=== Remaining omissions ==="
comm -23 /tmp/pybind_exports.txt /tmp/registered_ops.txt | while read op; do
    count=$(grep -r "tex\.$op" transformer_engine/pytorch/ --include="*.py" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "STILL MISSING: tex.$op ($count call sites)"
    fi
done
echo "If no output above, all critical omissions are resolved."
```

#### Pre-commit and commit

Run pre-commit hooks before committing to ensure code formatting is correct:

```bash
pre-commit run --files $(git diff --name-only --cached) || true
# If pre-commit modified files, stage them again
git add -u
```

```
git commit -m "fix: add missing pybind exports to plugin layer

Audited all pybind-exported symbols against plugin ops.py / register_ops.py.
Found N omissions, M critical (called via tex.xxx in Python code).
Added ops.py stubs, vendor implementations, and registrations for:
- <list of added ops>
Skipped N low-priority utility functions not called from Python."
```

