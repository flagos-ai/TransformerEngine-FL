---
name: te-fl-upstream-sync
description: >
  Manages the upstream sync workflow for the TransformerEngine-FL fork (flagos-ai/TransformerEngine-FL
  forked from Nvidia/TransformerEngine). Handles creating dev branches aligned with upstream releases,
  merging into main with conflict resolution, validating the custom plugin system (OP API interfaces,
  CUDA patches, Python bindings), and running multi-level CI/CD verification. Use this skill whenever
  the user mentions syncing upstream, updating from Nvidia/TransformerEngine, merging upstream releases
  (e.g. release_v2.14), fork sync, repo update, pulling upstream changes, plugin validation, checking
  plugin API, verifying CUDA patches, or any upstream integration workflow for TransformerEngine-FL.
  Also trigger when the user references conflict resolution for plugin files, build system merges
  involving setup.py/CMakeLists.txt with plugin targets, or running CI/CD pipelines after an upstream
  merge.
---

# TransformerEngine-FL Upstream Sync Workflow

You are guiding a developer through syncing their fork `flagos-ai/TransformerEngine-FL` with upstream
`Nvidia/TransformerEngine`. The fork's core value is a custom plugin system — every decision you make
must protect plugin functionality above all else.

## Fork Architecture

The fork adds these components on top of upstream:

- **Plugin OP API interfaces**: `transformer_engine/plugin/` - Plugin systems
- **CUDA patches**: `transformer_engine/__init__.py` — Patches to upstream torch.cuda apis
- **Build integration**: `setup.py`, `CMakeLists.txt`, `pyproject.toml` — plugin compilation targets
  woven into the upstream build system.
- **Other github workflow related**

## Repo Detection Preamble

Every stage below assumes you are inside the `TransformerEngine-FL` directory. Run this detection
snippet before any stage if the working directory is uncertain:

```bash
if [ -d "TransformerEngine-FL" ]; then
  cd TransformerEngine-FL
elif [ "$(basename $(pwd))" = "TransformerEngine-FL" ]; then
  echo "Already in TransformerEngine-FL"
elif [ -d "../TransformerEngine-FL" ]; then
  cd ../TransformerEngine-FL
else
  echo "ERROR: TransformerEngine-FL directory not found in current or parent directory"
  echo "Please clone the repo first, or cd to the correct location."
  exit 1
fi
echo "Working directory: $(pwd)"
```

---

## The Multi-Stage Sync Workflow

This workflow is sequential. Each stage has a command the user can invoke, but you should also guide
them through the full flow when they ask to "sync upstream" or similar.

### Phase Index

| Phase | File | Stages | Description |
|-------|------|--------|-------------|
| 1 | [phases/01-setup-and-analysis.md](phases/01-setup-and-analysis.md) | Stage 1-2 | Repo Setup & Branch Preparation + Identify Plugin Changes |
| 2 | [phases/02-merge-and-integrate.md](phases/02-merge-and-integrate.md) | Stage 3-4 | Merge & Conflict Resolution + Plugin API Sync |
| 3 | [phases/03-patch-and-verify.md](phases/03-patch-and-verify.md) | Stage 5-7 | CUDA Patching + Stale Refs + Build Verify |
| 4 | [phases/04-test-and-finalize.md](phases/04-test-and-finalize.md) | Stage 8-10 | Tests + Merge to Main + FlagScale Training |

Read the relevant phase file when the user enters that stage. Each phase is self-contained with
full instructions for its stages.

---

## Sync Report (`/generate-sync-report`)

After the workflow completes (success or failure), generate a report. Use the script at
`scripts/generate_sync_report.sh` to collect the data, or assemble manually:

```markdown
# TransformerEngine-FL Upstream Sync Report

## Summary
- **Date**: <date>
- **Upstream Release**: release_v2.14
- **Upstream Commit SHA**: <sha>
- **Status**: ✅ Complete / ❌ Failed at Stage N

## Stage Results
| Stage | Status | Notes |
|-------|--------|-------|
| 1. Repo setup & branch preparation | ✅/❌ | |
| 2. Identify plugin changes | ✅/❌ | |
| 3. Merge & conflict resolution | ✅/❌ | N conflicts, P0: N, P1: N, P2: N |
| 4. Plugin API sync | ✅/❌ | N APIs added/modified/removed, N omissions found/fixed |
| 5. Patch CUDA hardcoding | ✅/❌ | |
| 6. Detect & fix stale references | ✅/❌ | N stale refs found/fixed |
| 7. Build & import verification | ✅/❌ | |
| 8. Unit & integration tests | ✅/❌ | CI script validation: N missing refs fixed, N tests added |
| 9. Merge to main | ✅/❌ | tree replacement merge, PR opened |
| 10. FlagScale training validation | ✅/❌ | N/M combinations passed (see batch comparison table) |

## Conflicts Resolved
<list of files and resolution strategy used>

## Plugin System Status
- Plugin directory: ✅ intact
- OP API signatures: ✅ unchanged
- CUDA patches: ✅ present and applicable
- Build targets: ✅ present
- Python bindings: ✅ functional

## CI/CD Results
<level-by-level results table>

## Rollback Info
- Merge commit: <sha>
- Rollback command: `git revert -m 1 <sha>`
```

---

## Critical Rules

These are non-negotiable because the fork's entire value proposition is the plugin system:

1. **Plugin files are sacred.** Carefully auto-resolve a P0 conflict toward upstream. Always keep the
   fork version and manually review upstream changes.
2. **No silent failures.** Every validation step must produce visible output. If a check can't run
   (e.g., no GPU for Level 5), say so explicitly rather than skipping silently.
3. **Rollback is always an option.** If things go sideways, the user should never feel stuck. Always
   have `git revert -m 1 <merge-commit>` ready.
