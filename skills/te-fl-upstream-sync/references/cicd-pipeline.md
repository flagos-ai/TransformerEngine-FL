# CI/CD Pipeline Reference

Detailed instructions for each verification level after an upstream sync merge.

## Level 1 — Compile Test (`/run-level1-compile`)

The most basic check: does the code compile with plugin support?

```bash
# Clean any previous build artifacts
rm -rf build/ dist/ *.egg-info

# Install in editable mode with verbose output to catch warnings
pip install -e . -v 2>&1 | tee build.log

# Verify shared libraries were generated
echo "=== Checking .so files ==="
find . -name "*.so" -newer build.log | head -20

# Specifically check for plugin-related .so
echo "=== Plugin .so files ==="
find . -name "*plugin*" -name "*.so"

# Check for compilation warnings related to plugin
echo "=== Plugin-related warnings ==="
grep -i "plugin\|cuda_patch" build.log | grep -i "warn\|error" || echo "No plugin warnings found"
```

**Pass criteria:**
- Exit code 0 from pip install
- At least one `.so` file generated
- No errors mentioning plugin or cuda_patch in build log

**Common failures after upstream sync:**
- Missing include paths for plugin headers → check CMakeLists.txt
- Undefined symbols → check if upstream renamed/removed functions the plugin depends on
- CUDA version mismatch → check CUDA toolkit version vs upstream requirements

---

## Level 2 — Unit Tests

```bash
# Run upstream unit tests
pytest tests/unit/ -v --tb=short 2>&1 | tee unit-test.log

# Summary
echo "=== Unit Test Summary ==="
tail -5 unit-test.log
```

**Pass criteria:**
- All pre-existing tests pass
- No new test failures compared to upstream's test results for the same release

**If tests fail:**
- Check if failures are in plugin-related tests → likely a merge issue
- Check if failures are in upstream tests → might be environment issue, compare with upstream CI

---

## Level 3 — Integration Tests (PyTorch)

```bash
# Run PyTorch integration tests
pytest tests/pytorch/ -v --tb=short -x 2>&1 | tee integration-test.log

# Summary
echo "=== Integration Test Summary ==="
tail -10 integration-test.log
```

**Pass criteria:**
- All PyTorch integration tests pass
- No regressions from previous sync

**Note:** These tests may require a GPU. If running in a CPU-only environment, skip and note
in the sync report that Level 3 was not verified.

---

## Level 4 — Plugin-Specific Tests (`/run-level4-plugin`)

These are the fork-specific validations. They verify that the plugin system survived the merge intact.

### 4.1 Plugin Directory Integrity

```bash
echo "=== Plugin Directory Check ==="

# Check directory exists
if [ -d "transformer_engine/common/plugin" ]; then
    echo "✅ Plugin directory exists"
    ls -la transformer_engine/common/plugin/
else
    echo "❌ Plugin directory MISSING"
    exit 1
fi

# Check CUDA patches directory
if [ -d "transformer_engine/common/cuda_patches" ]; then
    echo "✅ CUDA patches directory exists"
    ls -la transformer_engine/common/cuda_patches/
else
    echo "❌ CUDA patches directory MISSING"
    exit 1
fi
```

### 4.2 OP API Interface Verification

```bash
echo "=== OP API Interface Check ==="

# Check register_plugin signature
grep -n "register_plugin" transformer_engine/common/plugin/*.h
if [ $? -eq 0 ]; then
    echo "✅ register_plugin() found"
else
    echo "❌ register_plugin() MISSING"
    exit 1
fi

# Check unregister_plugin signature
grep -n "unregister_plugin" transformer_engine/common/plugin/*.h
if [ $? -eq 0 ]; then
    echo "✅ unregister_plugin() found"
else
    echo "❌ unregister_plugin() MISSING"
    exit 1
fi

# Check for ABI-breaking changes (compare signatures with known-good)
echo "=== Signature Diff ==="
grep -A2 "register_plugin\|unregister_plugin" transformer_engine/common/plugin/*.h
```

### 4.3 CUDA Patches Applicability

```bash
echo "=== CUDA Patches Check ==="

PATCH_DIR="transformer_engine/common/cuda_patches"
FAIL=0

for patch in "$PATCH_DIR"/*.patch; do
    [ -f "$patch" ] || continue
    echo "Testing: $patch"
    if git apply --check "$patch" 2>/dev/null; then
        echo "  ✅ Applies cleanly"
    else
        echo "  ❌ FAILS to apply"
        FAIL=1
    fi
done

if [ $FAIL -eq 1 ]; then
    echo "⚠️  Some patches need rebasing"
else
    echo "✅ All patches apply cleanly"
fi
```

### 4.4 Build Configuration Check

```bash
echo "=== Build Config Check ==="

# Check setup.py for plugin references
if grep -q "plugin" setup.py 2>/dev/null; then
    echo "✅ setup.py contains plugin build targets"
else
    echo "⚠️  setup.py may be missing plugin targets"
fi

# Check CMakeLists.txt for plugin targets
if grep -q "plugin" CMakeLists.txt 2>/dev/null; then
    echo "✅ CMakeLists.txt contains plugin targets"
else
    echo "⚠️  CMakeLists.txt may be missing plugin targets"
fi
```

### 4.5 Python Bindings Check

```bash
echo "=== Python Bindings Check ==="

python -c "
try:
    import transformer_engine
    print('✅ transformer_engine imports successfully')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)

# Check for plugin module
try:
    from transformer_engine.pytorch import plugin
    print('✅ Plugin module accessible')
except (ImportError, AttributeError) as e:
    print(f'⚠️  Plugin module check: {e}')
    print('   (This may be expected if plugin is loaded differently)')
"
```

---

## Level 5 — End-to-End Tests

Only run if a GPU is available and the environment supports training.

```bash
echo "=== Environment Check ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "=== Running E2E Test ==="
    # Small-scale model training test
    python -c "
import torch
import transformer_engine.pytorch as te

# Simple forward pass with TE layers
model = te.Linear(256, 256)
x = torch.randn(2, 256, device='cuda')
with te.fp8_autocast():
    y = model(x)
    loss = y.sum()
    loss.backward()
print('✅ E2E forward/backward pass successful')
print(f'   Output shape: {y.shape}')
"
else
    echo "⚠️  No GPU available — skipping Level 5"
    echo "   Record this in the sync report"
fi
```

---

## Rollback Strategy

If any level fails and cannot be fixed:

```bash
# Find the merge commit
git log --oneline --merges -1

# Revert the merge (keeping the main branch's history)
git revert -m 1 <merge-commit-sha>

# Push the revert
git push origin main
```

The `-m 1` tells git to revert to the first parent (main), effectively undoing the merge while
preserving history.
