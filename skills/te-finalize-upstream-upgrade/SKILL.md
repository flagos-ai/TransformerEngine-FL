---
name: te-finalize-upstream-upgrade
description: Finalize and hand off a TransformerEngine-FL upstream upgrade after conflict, plugin API, runtime patch, build, CI, and test audits. Use for evidence completeness, unresolved blocker review, commit/branch hygiene, rollback planning, release notes, or preparing a PR without pushing automatically.
---

# Finalize TransformerEngine-FL Upgrade

Treat finalization as an evidence gate, not a formatting step.

## Workflow

1. Require all phase inventories and their exact commit refs.
2. Verify each acceptance criterion and classify it pass, fail, blocked, or not-applicable.
3. Check no unresolved P0 decisions, missing API dispositions, unowned CI gaps, or unexplained blocked tests remain.
4. Review worktree, branch ancestry, diff scope, generated artifacts, submodule gitlinks, and commit messages.
5. Produce a rollback plan naming the merge/upgrade commits and a recoverable command.
6. Produce a handoff report with changed surfaces, test evidence, known limitations, and explicit user decisions required.
7. Stop before push, PR creation, tag, or main mutation; request separate authorization.

## Outputs

Write final-report.md, evidence-index.json, blockers.tsv, and rollback-plan.md under /share/project/zhaoyingli/flagos/temp/<run>.

## Acceptance Criteria

- All phase artifacts exist and refs agree.
- Every phase has a status and evidence path.
- No unresolved P0 item is marked pass.
- Every blocked non-NVIDIA test has owner and reason.
- Worktree and main status are recorded.
- Diff contains no unapproved generated artifact.
- Rollback command is explicit and non-destructive.
- Push/PR/tag actions are not performed without separate approval.
- All output is under /share/project/zhaoyingli/flagos/temp, never /tmp.

## Design decision ledger

The handoff must reference the FL design baseline and include a `preserve/adapt/drop` decision for each design group: plugin core, each backend family, runtime patch chain, native build/submodules, CI matrix, QA/tests, and observability/docs. No group may be omitted because Git merged it cleanly.
