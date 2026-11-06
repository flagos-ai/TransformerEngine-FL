---
name: te-preserve-runtime-patches
description: Audit and preserve invasive TransformerEngine-FL runtime changes across an upstream merge. Use when upstream modifies transformer_engine/pytorch, transformer_engine/__init__.py, common, debug, device selection, plugin dispatch, tensor/attention wrappers, or vendor patches and the fork must retain behavior across CUDA and non-CUDA backends.
---

# Preserve TransformerEngine-FL Runtime Patches

Treat fork changes to upstream-owned runtime files as semantic patches, not ordinary additions. Audit before editing and require a focused acceptance test for every invariant.

## Workflow

1. Require matching fork-delta, conflict, and API inventories.
2. Identify fork-only and both-changed runtime files from the inventories.
3. Extract fork invariants: plugin dispatch calls, TE_DEVICE_TYPE/device substitutions, non-CUDA guards, vendor selection, tensor/attention argument forwarding, and import/initialization ordering.
4. Compare base, fork, and target call sites. Search target for new hard-coded CUDA behavior, renamed imports, changed signatures, and altered defaults. For every fork-added symbol, verify import, definition, and call/reference coupling; a call such as te_device_type() is invalid if its import was removed.
5. Create a patch ledger with path, invariant, upstream change, preservation method, test, owner, and status.
6. Apply changes only in an isolated branch. Keep upstream functional changes while reintroducing fork dispatch at the narrowest boundary.
7. Run static scans, import tests, plugin dispatch tests, and at least one representative non-CUDA path when hardware is available.

## Outputs

Write runtime-patch-ledger.json, runtime-patch-ledger.tsv, runtime-patch-audit.md, and raw scan logs under /share/project/zhaoyingli/flagos/temp/<run>.

## Acceptance Criteria

- Every fork-modified runtime file appears exactly once in the ledger.
- Every both-changed runtime file has an explicit invariant and preservation decision.
- Every plugin dispatch and device abstraction marker is accounted for. Module-level dispatch blocks are retained atomically: native fallback alias, plugin replacement, original callback save, replacement callback, and required imports.
- No unexplained new hard-coded CUDA path remains in target-derived runtime code.
- Import order and optional vendor dependencies are tested.
- Python signatures and forwarded keyword arguments match target callers. Every te_device_type() call has a valid import and a focused assertion/device test.
- CUDA and at least one non-CUDA backend have evidence, or a blocked reason with owner.
- No conflict markers or unrelated files enter the patch.
- Main remains unchanged and all artifacts are under /share/project/zhaoyingli/flagos/temp, never /tmp.

Never mass-replace cuda, keep fork patches only by line count, or use blanket ours/theirs resolution.

## Cross-module patch chains

Treat runtime changes as chains of definition, import, alias registration, and call site. Include `sys.modules` substitutions and compatibility modules in the ledger. A patch is not preserved until its downstream import-time consumers load successfully; record the chain owner and staged import evidence.
