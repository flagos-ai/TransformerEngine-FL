#!/usr/bin/env bash
# validate_plugin.sh — Plugin system integrity checks for TransformerEngine-FL
# Usage: bash scripts/validate_plugin.sh [repo_root]
# Exit code: 0 = all checks pass, 1 = critical failure, 2 = warnings only

set -euo pipefail

REPO_ROOT="${1:-.}"
FAIL=0
WARN=0

echo "============================================"
echo "  TransformerEngine-FL Plugin Validation"
echo "============================================"
echo ""

# --- Check 1: Plugin directory exists ---
echo "[1/6] Plugin directory integrity"
PLUGIN_DIR="$REPO_ROOT/transformer_engine/common/plugin"
if [ -d "$PLUGIN_DIR" ]; then
    FILE_COUNT=$(find "$PLUGIN_DIR" -type f | wc -l | tr -d ' ')
    echo "  ✅ Plugin directory exists ($FILE_COUNT files)"
    find "$PLUGIN_DIR" -type f -name "*.h" | while read -r f; do
        echo "     - $(basename "$f")"
    done
else
    echo "  ❌ CRITICAL: Plugin directory missing: $PLUGIN_DIR"
    FAIL=1
fi

# --- Check 2: CUDA patches directory ---
echo ""
echo "[2/6] CUDA patches directory"
CUDA_DIR="$REPO_ROOT/transformer_engine/common/cuda_patches"
if [ -d "$CUDA_DIR" ]; then
    PATCH_COUNT=$(find "$CUDA_DIR" -type f | wc -l | tr -d ' ')
    echo "  ✅ CUDA patches directory exists ($PATCH_COUNT files)"
else
    echo "  ❌ CRITICAL: CUDA patches directory missing: $CUDA_DIR"
    FAIL=1
fi

# --- Check 3: OP API signatures ---
echo ""
echo "[3/6] OP API interface signatures"
if [ -d "$PLUGIN_DIR" ]; then
    REG=$(grep -rl "register_plugin" "$PLUGIN_DIR" 2>/dev/null | wc -l | tr -d ' ')
    UNREG=$(grep -rl "unregister_plugin" "$PLUGIN_DIR" 2>/dev/null | wc -l | tr -d ' ')

    if [ "$REG" -gt 0 ]; then
        echo "  ✅ register_plugin() found in $REG file(s)"
        grep -n "register_plugin" "$PLUGIN_DIR"/*.h 2>/dev/null | head -5 | sed 's/^/     /'
    else
        echo "  ❌ CRITICAL: register_plugin() not found"
        FAIL=1
    fi

    if [ "$UNREG" -gt 0 ]; then
        echo "  ✅ unregister_plugin() found in $UNREG file(s)"
        grep -n "unregister_plugin" "$PLUGIN_DIR"/*.h 2>/dev/null | head -5 | sed 's/^/     /'
    else
        echo "  ❌ CRITICAL: unregister_plugin() not found"
        FAIL=1
    fi
else
    echo "  ⏭️  Skipped (plugin directory missing)"
fi

# --- Check 4: CUDA patches applicability ---
echo ""
echo "[4/6] CUDA patches applicability"
if [ -d "$CUDA_DIR" ]; then
    PATCH_FAIL=0
    for patch in "$CUDA_DIR"/*.patch; do
        [ -f "$patch" ] || continue
        BASENAME=$(basename "$patch")
        if git apply --check "$patch" 2>/dev/null; then
            echo "  ✅ $BASENAME applies cleanly"
        else
            echo "  ⚠️  $BASENAME fails to apply (may need rebasing)"
            WARN=1
            PATCH_FAIL=1
        fi
    done
    if [ $PATCH_FAIL -eq 0 ] && [ "$(find "$CUDA_DIR" -name '*.patch' | wc -l | tr -d ' ')" -eq 0 ]; then
        echo "  ℹ️  No .patch files found (patches may use a different format)"
    fi
else
    echo "  ⏭️  Skipped (CUDA patches directory missing)"
fi

# --- Check 5: Build configuration ---
echo ""
echo "[5/6] Build configuration"
for BUILD_FILE in setup.py CMakeLists.txt pyproject.toml; do
    FPATH="$REPO_ROOT/$BUILD_FILE"
    if [ -f "$FPATH" ]; then
        if grep -qi "plugin" "$FPATH"; then
            echo "  ✅ $BUILD_FILE references plugin targets"
        else
            echo "  ⚠️  $BUILD_FILE exists but no plugin references found"
            WARN=1
        fi
    else
        echo "  ℹ️  $BUILD_FILE not found (may not be applicable)"
    fi
done

# --- Check 6: Python bindings ---
echo ""
echo "[6/6] Python bindings"
if python -c "import transformer_engine" 2>/dev/null; then
    echo "  ✅ transformer_engine imports successfully"
    python -c "
try:
    from transformer_engine.pytorch import plugin
    print('  ✅ Plugin module accessible')
except (ImportError, AttributeError, ModuleNotFoundError) as e:
    print(f'  ⚠️  Plugin module: {e}')
    print('     (May need pip install -e . first)')
" 2>/dev/null || echo "  ⚠️  Could not check plugin module"
else
    echo "  ⚠️  transformer_engine not installed (run pip install -e . first)"
    WARN=1
fi

# --- Summary ---
echo ""
echo "============================================"
if [ $FAIL -gt 0 ]; then
    echo "  ❌ VALIDATION FAILED — critical issues found"
    echo "============================================"
    exit 1
elif [ $WARN -gt 0 ]; then
    echo "  ⚠️  VALIDATION PASSED WITH WARNINGS"
    echo "============================================"
    exit 2
else
    echo "  ✅ ALL CHECKS PASSED"
    echo "============================================"
    exit 0
fi
