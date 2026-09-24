---
name: te-classify-fork-delta
description: Inventory and classify all fork-owned changes before upgrading a TransformerEngine fork. Use when establishing exact upstream base and target refs, producing a complete fork delta manifest, finding files changed by both fork and upstream, discovering plugin backends and CI surfaces, ranking merge risks, or checking that an upgrade plan covers every modified file.
---

# Classify TransformerEngine Fork Delta

Create the authoritative input for later upgrade skills. Do not merge, edit source, create branches, or update submodules while using this skill.

## Inputs

Require repository path and exact base, fork, and target refs. Resolve them to commits. Stop if a ref is missing. Record whether base is an ancestor of target. If release histories diverge, require an explicit `--allow-divergent-upstream` decision and record their merge-base; never silently treat a divergent history as linear.

## Workflow

1. Record worktree status, branch, remotes, tags, and submodule status without modification.
2. Run `scripts/classify_fork_delta.py` with explicit refs and an untracked output directory.
3. Use `references/classification.md` to review category and priority assignments.
4. Resolve every `unclassified` entry with a narrow, justified rule; never hide it in `other`.
5. Review every `both_changed` path and record its invariant, observing test, and downstream owner.
6. Confirm all discovered backends, CI, QA, tests, build files, packaging files, and submodules have a downstream owner.
7. Present the manifest and risk register for approval. Do not begin integration.

```bash
python scripts/classify_fork_delta.py --repo /path/to/repo   --base <base-ref> --fork <fork-ref> --target <target-ref> --allow-divergent-upstream --output /share/project/zhaoyingli/flagos/temp/te-upgrade-inventory
```

Outputs are `inventory.json`, `inventory.md`, `fork-changes.tsv`, and `both-changed.tsv`.

## Priority

1. P0: plugin contracts, invasive runtime/device patches, or files changed on both sides.
2. P1: build, packaging, submodules, discovery, and tests gating P0 behavior.
3. P2: CI/CD, QA, executable examples, docs, and repository metadata.
4. P3: additive material with no runtime, build, test, or release effect.

Never lower priority merely because Git predicts a clean merge.

## Acceptance Criteria

- Record all refs as full SHAs, ancestry result, and merge-base. Require explicit authorization for divergent upstream tags.
- Record dirty worktree state before analysis.
- Include every `git diff --name-status base..fork` path exactly once.
- Leave zero unclassified paths.
- Include every both-side path in `both-changed.tsv`.
- Discover plugin backends dynamically; use no fixed vendor list.
- List CI, QA, tests, build, packaging, and submodule surfaces even when empty.
- Record base/fork/target gitlink SHAs or explicit absence for every submodule.
- Make JSON, Markdown, and TSV totals agree.
- Obtain user approval before merge or source modification.

Stop on unresolved refs, uncertain base, failed Git commands, inconsistent totals, or unclassified paths.
## FL design baseline

When the fork contains a plugin, hardware matrix, or invasive runtime layer, also produce a design baseline from `base..fork` (not only a path manifest). Group paths into plugin contract/core, reference and vendor backends, runtime patches, native build/submodules, CI/CD, QA/tests, and docs/observability. For each group record background, public entry points, call-chain owner, invariants, and an explicit `preserve`, `adapt`, or `drop` decision placeholder.

The baseline must include dynamic module aliases (especially `sys.modules["transformer_engine_torch"]`) and public enum/callable compatibility. Compare the plugin-provided module against the upstream Python/native public contract before integration. Store the report under the approved artifact directory (never `/tmp`), for example `te-fl-design-baseline-<base>-<fork>.md`.

Acceptance: every fork-owned path is assigned to a design group and owner; every group has a documented purpose and acceptance evidence; dynamic aliases and enum members are listed; CI/build/submodule surfaces are represented; the report is referenced by the upgrade decision log.
