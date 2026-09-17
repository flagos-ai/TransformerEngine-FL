---
name: te-audit-plugin-api
description: Audit TransformerEngine-FL plugin API compatibility against an upstream release. Use when upstream pybind bindings, Python call sites, enums, dataclasses, or attention signatures change; when adding/removing plugin ops; or when verifying every implementation and registration across dynamically discovered backends.
---

# Audit TransformerEngine-FL Plugin API

Use an isolated tree and exact base, fork, and target refs. This skill audits before edits; it does not invent fallbacks or silently suppress missing APIs.

## Workflow

1. Read the fork-delta and conflict inventories. Require their refs to match.
2. Extract target pybind exports, Python-side tex call sites, plugin base methods, registered op names, and backend methods.
3. Compare base and target signatures, including multiline definitions, constructors, enum types, dataclasses, and AttentionParams fields.
4. Build a matrix: symbol, upstream status, plugin base, CUDA/reference/FlagOS/vendor implementations, registration, signature risk, disposition, owner, test.
5. Dynamically discover backends from transformer_engine/plugin/core/backends. Never use a hard-coded vendor list.
6. Classify each gap as required, intentionally unsupported, fallback, or obsolete. Every non-required disposition needs a written reason.
7. Update code only after the matrix is approved. Re-run the audit after each API change.

## Outputs

Write under /share/project/zhaoyingli/flagos/temp/<run>:

- api-inventory.json
- api-matrix.tsv
- api-audit.md
- raw upstream/plugin symbol lists
- decisions.tsv

## Acceptance Criteria

- Base/fork/target SHAs match the upstream and conflict inventories.
- Binding extraction includes all binding files and macro-generated exports considered.
- Every target export and Python call site has a matrix row.
- Every matrix row records base/target signature status and enum/dataclass risk.
- Every required symbol has a base stub, implementation, registration, and focused test.
- Every backend is dynamically discovered and has native/fallback/unsupported status per required symbol.
- No unexplained missing, stale, duplicate, or signature-mismatch symbols remain.
- Allowlisted unsupported symbols have owner, reason, and test or blocked evidence.
- Re-running the audit is deterministic and produces identical symbol sets.
- All artifacts are under /share/project/zhaoyingli/flagos/temp; never /tmp.

Stop on macro-generated exports not accounted for, mismatched refs, ambiguous capability, or unapproved API decisions.

## Dynamic public-module contract

When the fork registers `transformer_engine_torch` dynamically, audit that alias as a first-class public surface. Compare Python import-time consumers against both the native binding and the plugin-provided module, including enum member sets, sentinel values, signatures, and required callables. Run the audit in staged imports so the first exception does not hide later incompatibilities.

Acceptance additionally requires: `transformer_engine.pytorch.constants`, `pytorch.cpp_extensions`, and full `transformer_engine.pytorch` imports pass; all public enum sets used at import time match; every deliberate plugin-only difference has an owner, rationale, and test.
