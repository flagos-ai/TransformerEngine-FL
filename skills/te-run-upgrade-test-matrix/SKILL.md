---
name: te-run-upgrade-test-matrix
description: Plan, execute, and report TransformerEngine-FL upstream-upgrade tests by capability and hardware. Use after merge/plugin/build audits for plugin tests, Python/C++ tests, NVIDIA GPU validation, vendor backend checks, CI/QA entrypoints, and FlagScale E2E; explicitly record blocked non-NVIDIA tests instead of treating them as passed.
---

# Run TransformerEngine-FL Upgrade Test Matrix

Separate test selection from execution and never hide unavailable hardware.

## Workflow

1. Require completed conflict, API, runtime, build, and CI inventories.
2. Generate a matrix of static, CPU/import, NVIDIA CUDA, plugin, and vendor-specific tests.
3. Mark each row runnable, blocked, skipped-by-policy, or not-applicable with reason and owner.
4. Run cheap static checks first, then install/import, plugin tests, focused CUDA tests, broader CUDA tests, and finally FlagScale E2E.
5. Use the approved NVIDIA machine and conda environment only for GPU-required rows. Do not claim non-NVIDIA rows passed on that host.
6. Capture command, commit, environment, start/end time, exit code, log path, and artifact path for every row.
7. Re-run failed rows only after recording diagnosis and a scoped change.

## Hardware Policy

The available GPU host is NVIDIA-only. CUDA rows may be runnable there. Hygon, MUSA, NPU, Ascend, Kunlunxin, Enflame, MetaX, Iluvatar, Tsingmicro, and other non-NVIDIA native rows must be blocked or static-only unless a matching host is later authorized.

## Outputs

Write test-matrix.json, test-matrix.tsv, test-report.md, environment.txt, and per-row logs under /share/project/zhaoyingli/flagos/temp/<run>.

## Acceptance Criteria

- Every discovered test group has one matrix row and hardware classification.
- Static and NVIDIA-runnable rows have command, result, and log evidence.
- Every blocked non-NVIDIA row has explicit reason and owner.
- No skipped or blocked row is counted as pass.
- Tests run against the recorded commit and environment.
- Failed tests retain first failure logs and diagnosis.
- Focused plugin/API tests run before broad E2E.
- Main remains unchanged and all artifacts are under /share/project/zhaoyingli/flagos/temp, never /tmp.

Never run destructive cleanup or alter test fixtures to make a row pass.

## Import-contract smoke tier

Before GPU kernels, run a staged public-contract smoke tier: native/core import, dynamic `transformer_engine_torch` registration, enum/callable comparison, `pytorch.constants`, `pytorch.cpp_extensions`, and full PyTorch import. Record the exact module path and enum sets.
