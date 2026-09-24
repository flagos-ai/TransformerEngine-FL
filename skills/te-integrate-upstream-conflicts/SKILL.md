---
name: te-integrate-upstream-conflicts
description: Analyze and safely resolve files changed by both TransformerEngine-FL and NVIDIA upstream during an upgrade. Use for three-way merge planning, conflict classification, semantic preservation of plugin dispatch and invasive runtime patches, dry-run merge validation, or reviewing whether every conflict has an owner and acceptance test.
---

# Integrate Upstream Conflicts

Operate on an isolated worktree or branch. Never modify main, push, or resolve conflicts by blindly choosing ours/theirs.

## Workflow

1. Require the classifier inventory and exact base, fork, and target SHAs.
2. Run scripts/conflict_inventory.py and write outputs under /share/project/zhaoyingli/flagos/temp.
3. For every both-changed path, inspect base, fork, and target versions. Record fork invariant, upstream change, resolution strategy, owner, and acceptance test.
4. Classify each path P0 (plugin/runtime/device), P1 (build/package/submodule/tests), or P2 (CI/QA/docs/metadata). Treat fork-only additions as owned content: they are not conflicts, but they must be preserved unless an explicit decision removes them.
5. Simulate the merge before editing. Record textual conflicts separately from clean-but-semantic conflicts.
6. Require explicit approval for each P0 strategy. Use manual edits and narrow commits; never use blanket ours/theirs.
7. After resolution, run conflict-marker, diff-scope, and invariant checks.

## Acceptance Criteria

- Every both-changed path appears exactly once.
- Every path has priority, owner, strategy, acceptance test, and status.
- P0 paths cannot be resolved without explicit user approval.
- Textual conflicts and clean-but-semantic risks are both reported.
- No blanket ours/theirs operation is used, and no whole-file theirs/tree-replacement operation is used on fork-owned plugin, runtime, build, or CI files.
- No conflict markers remain in resolved files.
- Resolved diff contains no unexpected path outside the approved manifest.
- Each P0 invariant has a passing focused test or recorded blocked reason. Module-level plugin dispatch blocks must retain their imports, saved native fallback, original callback, replacement callback, and call sites as one unit.
- Worktree is isolated and main is unchanged.
- All artifacts are under /share/project/zhaoyingli/flagos/temp, never /tmp.

Stop on missing inventory, ambiguous ownership, unapproved P0 decisions, unexpected files, or unavailable test evidence.
