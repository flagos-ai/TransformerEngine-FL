# TransformerEngine-FL Plugin Tests

This directory owns tests added for the TransformerEngine-FL plugin layer.
Upstream Transformer Engine tests remain in `tests/cpp`, `tests/jax`, and
`tests/pytorch`.

Platform CI launchers live with their backend support files under
`backend/<platform>/`. Use `run_unit_tests.sh` and
`run_integration_tests.sh` for the two standard entry points. See
[`../CI_TESTING_GUIDE.md`](../CI_TESTING_GUIDE.md) for the complete convention.

The test layout follows the implementation boundary:

- `plugin/`: plugin manager, policy, registry, and discovery behavior.
- `backend/`: shared backend contracts and operation suites.
- `backend/reference/`: reference backend tests.
- `backend/flagos/`: FlagOS backend tests that do not require a specific device.
- `backend/npu/`: Ascend NPU tests, runtime compatibility patches, and the
  backend-local pytest entry point used to run selected upstream tests.

Ascend tests that need runtime compatibility setup are launched through
`backend/npu/run_pytest.py`. The launcher applies the NPU runtime patch before
pytest collects tests. Platform-specific behavior stays in `backend/npu/` and
is not added to the common CI workflow.

Platforms that do not need an import-time adapter continue to use the normal
`python -m pytest` path.
