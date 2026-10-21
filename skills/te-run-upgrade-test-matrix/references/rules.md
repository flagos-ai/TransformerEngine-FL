# Test Matrix Rules

Recommended order:

1. Python syntax, YAML/workflow and stale-reference scans.
2. Editable install/import and package manifest.
3. Plugin lifecycle, policy, registry, and backend unit tests.
4. NVIDIA CUDA focused operator/API tests.
5. NVIDIA broader QA and integration tests.
6. FlagScale single-config smoke test, then batch E2E.

Use statuses pass, fail, blocked, skipped, and not-applicable. A blocked hardware test is evidence of missing coverage, not success.
