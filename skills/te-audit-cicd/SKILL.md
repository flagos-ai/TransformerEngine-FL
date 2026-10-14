---
name: te-audit-cicd
description: Audit TransformerEngine-FL GitHub Actions, CI configs, QA entrypoints, reusable workflows, local actions, runner labels, and vendor test coverage after an upstream upgrade. Use when .github, qa, workflow matrices, hardware backends, or CI scripts may be stale or silently incomplete.
---

# Audit TransformerEngine-FL CI/CD

Perform a static audit before relying on remote CI. Do not edit workflows or trigger jobs in this skill.

## Workflow

1. Read the fork-delta inventory and identify changed workflow, config, script, and QA paths.
2. Parse every YAML workflow/config; report syntax and duplicate-key failures.
3. Resolve local action, reusable-workflow, script, Dockerfile, config, and test-entrypoint references. Compare fork-only workflows against the target tree and require an explicit preserve/remove decision; never assume upstream absence means deletion.
4. Extract vendor/backend and test-group matrices dynamically. Compare supported plugin backends with CI coverage and mark intentional exclusions.
5. Check runner labels, images, permissions, secrets, event triggers, path filters, and failure masking.
6. Run shell syntax checks for referenced scripts and static QA entrypoint checks.
7. Write a coverage matrix and require owner/reason for every missing or blocked path.

## Outputs

Write cicd-audit.json, workflow-matrix.tsv, missing-references.tsv, cicd-audit.md, and raw parser logs under /share/project/zhaoyingli/flagos/temp/<run>.

## Acceptance Criteria

- Every workflow/config parses or has an owned blocker.
- Every local reference resolves. Fork-only workflows and CI configs remain in the merged tree unless an explicit removal decision has evidence.
- Every discovered plugin backend has CI coverage or an explicit exclusion.
- Vendor matrices and test groups have no unexplained gaps.
- Permissions, triggers, runner labels, and secrets are recorded.
- Referenced shell scripts pass syntax checks or have an owned blocker.
- No workflow silently suppresses test failures. Do not use tree replacement or whole-file theirs resolution on fork-owned workflows.
- Audit is deterministic, main unchanged, and all artifacts are under /share/project/zhaoyingli/flagos/temp, never /tmp.

Do not infer CI success from YAML syntax alone.
