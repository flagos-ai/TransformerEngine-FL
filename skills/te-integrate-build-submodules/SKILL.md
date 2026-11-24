---
name: te-integrate-build-submodules
description: Audit and integrate TransformerEngine-FL build, packaging, wheel, CMake, setup, and third-party submodule changes across an upstream upgrade. Use when setup.py, pyproject.toml, build_tools, MANIFEST.in, .gitmodules, 3rdparty gitlinks, native extensions, or install/import behavior changes.
---

# Integrate Build and Submodules

Audit the build graph before source integration. Do not update gitlinks or build artifacts without an explicit three-way decision.

## Workflow

1. Require matching fork-delta and conflict inventories.
2. Compare base, fork, and target versions of setup.py, pyproject.toml, MANIFEST.in, build_tools, CMake files, package data, and extension source lists.
3. Extract plugin compilation targets, include paths, libraries, device guards, version metadata, and wheel package inclusion. Mark fork-only build and workflow files as preserved content; do not replace them with upstream trees.
4. Compare .gitmodules and every 3rdparty gitlink at base, fork, and target. Record absent, added, removed, and changed submodules.
5. Build a decision matrix: path/gitlink, fork requirement, upstream change, selected value, compatibility risk, owner, and validation command.
6. In an isolated environment run static packaging checks, editable build/import, and wheel content inspection. Use the approved GPU machine/environment only when a CUDA extension build is required.
7. Record blocked dependency, compiler, CUDA, or network conditions explicitly.

## Outputs

Write build-audit.json, build-matrix.tsv, submodule-matrix.tsv, build-audit.md, and logs under /share/project/zhaoyingli/flagos/temp/<run>.

## Acceptance Criteria

- Every fork-modified build/package file has one matrix row.
- Every submodule path in any of base/fork/target has base/fork/target gitlink or explicit absent value.
- Plugin source files, extension targets, include paths, package data, and version metadata are all accounted for.
- .gitmodules entries and checkout URLs are valid.
- No unexpected generated files enter the source diff. Fork-only build/CI files remain present after merge unless an explicit removal decision is recorded.
- Editable install/import and wheel manifest checks pass, or have an owned blocked reason.
- Selected submodule commits are reachable and reproducible.
- Build artifacts are not committed.
- Main is unchanged and all artifacts are under /share/project/zhaoyingli/flagos/temp, never /tmp.

Never assume a clean Python import proves native extensions are correctly built.
