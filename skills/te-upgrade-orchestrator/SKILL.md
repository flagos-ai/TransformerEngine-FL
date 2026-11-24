---
name: te-upgrade-orchestrator
description: Orchestrate the gated TransformerEngine-FL upstream upgrade workflow across fork inventory, conflict, plugin API, runtime patch, build/submodule, CI/CD, test matrix, and finalization skills. Use when starting or resuming a complete upstream upgrade and when every phase needs explicit inputs, evidence, approvals, and stop conditions.
---

# Orchestrate TransformerEngine-FL Upgrade

Use the phase skills in dependency order. The orchestrator coordinates; specialized skills perform analysis and tests.

## Phase Order

1. Classify fork delta and produce the FL design baseline.
2. Integrate upstream conflicts.
3. Audit plugin API.
4. Preserve runtime patches.
5. Integrate build and submodules.
6. Audit CI/CD.
7. Run the test matrix.
8. Finalize evidence and rollback.
9. Request separate approval for merge-to-main, push, PR, tag, or release.

## Gates

Before each phase verify:

- exact base, fork, and target refs;
- prior phase artifact exists and refs agree;
- prior acceptance criteria pass;
- no unresolved P0 decision;
- worktree and branch isolation;
- output directory is under /share/project/zhaoyingli/flagos/temp.

Pause after analysis, before source edits, before GPU runs, before merge, and before any external publication. A blocked hardware test pauses only the affected capability, but must remain visible in the final report.

## Safety

Do not use blanket ours/theirs, reset, clean, force push, or tree replacement. Do not count blocked non-NVIDIA tests as pass. Do not mutate main. Keep each phase in a focused commit or uncommitted review state until approved.

## Acceptance Criteria

- All eight specialized phases have an artifact and status.
- Ref identities match across artifacts.
- Every P0 item has an approved decision or remains blocked.
- Every blocked test and missing backend has owner and reason.
- Final report includes evidence index and rollback plan.
- No push, PR, tag, merge-to-main, or release occurs without explicit user approval.
- No artifact or instruction references /tmp.


## Design-baseline gate

Conflict integration cannot begin until the design baseline for the selected `base..fork` delta exists and every design group has an owner plus a preserve/adapt/drop decision placeholder. Carry those decisions into finalization.
