# Stage 4: csrc API Recovery Report

## Executive Summary

**Problem**: Fork-specific csrc APIs were lost during Stage 3 merge due to incorrect priority classification.

**Impact**: 8 MoE/allocate APIs referenced by all 6 vendor backends are missing from csrc layer, causing plugin layer to fail at runtime.

**Root Cause**: csrc directory is outside `plugin/` tree, so it was treated as P2 (accept upstream) instead of P0 (preserve fork).

---

## Detailed Analysis

### Lost APIs (8 total)

#### Router APIs (6)
1. `fused_topk_with_score_function_fwd`
2. `fused_topk_with_score_function_bwd`
3. `fused_score_for_moe_aux_loss_fwd`
4. `fused_score_for_moe_aux_loss_bwd`
5. `fused_moe_aux_loss_fwd`
6. `fused_moe_aux_loss_bwd`

#### Allocate APIs (2)
7. `convert_host_pointers_to_tensor`
8. `get_device_pointer_for_data_and_scales`

### Affected Files

| File | Status | main | HEAD | Action Required |
|------|--------|------|------|-----------------|
| `router.cpp` | Different | 184 lines | 260 lines | ✓ Restore fork functions |
| `utils.cpp` | **Identical** | 165 lines | 165 lines | ✗ No action needed |
| `pybind.cpp` | Different | 638 lines | 760 lines | ✓ Restore 8 .def() calls |

**Key Finding**: `utils.cpp` is identical between main and HEAD, meaning the 2 allocate APIs are already present!

### Impact Scope

All 6 vendor backends reference these APIs:
- cuda: 32 references
- enflame: 32 references
- hygon: 32 references
- iluvatar: 32 references
- metax: 32 references
- musa: 32 references

**Total**: 192 plugin-layer call sites will fail if csrc APIs are not restored.

---

## Recovery Plan

### Option A: Complete Recovery (Recommended for Production)

1. **router.cpp**
   - Extract fork-added functions from `main:transformer_engine/pytorch/csrc/extensions/router.cpp`
   - Merge into HEAD version (preserving upstream v2.17 additions)
   - Estimated LOC: ~150 lines of fork code

2. **pybind.cpp**
   - Extract 8 `.def()` calls from `main:transformer_engine/pytorch/csrc/extensions/pybind.cpp`
   - Insert into HEAD version in router/utils sections
   - Estimated LOC: ~30 lines

3. **extensions.h** (if needed)
   - Check if function declarations need updates
   - Add missing declarations for 8 APIs

4. **Verification**
   - Build C++ extension: `cd transformer_engine/pytorch/csrc && make`
   - Import test: `python -c "import transformer_engine_torch; print(dir(transformer_engine_torch))" | grep fused_moe`
   - Runtime test: Run one vendor backend's router test

### Option B: Defer to Next Sync (For Testing Only)

Skip csrc recovery, continue with Stage 5-7 to test other aspects. Mark as known issue for manual fix.

**Tradeoff**: Plugin layer will not work, but we can validate:
- Stage 5: CUDA hardcoding detection still works
- Stage 6: Obsolete reference detection still works
- Stage 7: Build system (non-router parts) still works

---

## Recommendation

**Choose Option A** if goal is production-ready v2.17 sync.

**Choose Option B** if goal is workflow validation only.

---

## Lessons Learned

1. **P0 scope too narrow**: csrc should be P0 when it contains fork-specific extensions, not just `plugin/`
2. **API dependency check missing**: Should have validated plugin → csrc call graph before accepting P2 files
3. **Skill workflow gap**: Phase 02 doc doesn't cover "fork csrc extensions" case explicitly

### Proposed Skill Update

Add to Phase 02, Stage 3:

```markdown
### P0.5: Fork-specific csrc extensions

Before accepting P2 files, check if csrc contains fork-added functions:

\`\`\`bash
git diff base..main -- transformer_engine/*/csrc/ | grep "^+.*def("
\`\`\`

If fork csrc APIs exist, treat them as P0 (manual merge).
```

---

## Next Steps

1. User decides: Option A (recover) or Option B (defer)
2. If Option A: Execute 2-file recovery plan
3. If Option B: Continue to Stage 5, mark Stage 4 as "deferred"
4. Update `te-fl-upstream-sync` skill with lesson learned
