# TransformerEngine-FL Plugin Tests

This directory owns tests added for the TransformerEngine-FL plugin layer.
Upstream Transformer Engine tests remain in `tests/cpp`, `tests/jax`, and
`tests/pytorch`.

Platform CI launchers live with their backend support files under
`backend/<platform>/`. Use `run_unit_tests.sh` and
`run_integration_tests.sh` for the two standard entry points. See
[`../CI_TESTING_GUIDE.md`](../CI_TESTING_GUIDE.md) for the complete convention.

## Backend Test Organization

Every backend platform directory follows the same four-file contract:

```text
backend/<platform>/
  config.sh                    # platform constants only
  set_env.sh                   # runtime environment setup
  run_unit_tests.sh            # unit-test entry point
  run_integration_tests.sh     # integration-test entry point
```

Both CI matrices invoke the platform-owned entry points. A launcher may
delegate to a shared `qa/` implementation, but `.github/configs/` must point
to `tests/plugin/backend/<platform>/run_integration_tests.sh`, never directly
to `qa/`.

`config.sh` declares platform identity, device metadata, timeouts, test paths,
coverage settings, and any patch or wrapper paths. `set_env.sh` exports
`TE_PATH`, `PYTHONPATH`, device visibility, vendor library paths, and other
runtime variables. Platform-specific suite selection remains inside the
launcher when the vendor runtime requires it.

The supported platform types are:

- `shell_launcher`: a platform-specific shell launcher runs the suites.
- `python_wrapper`: a Python pytest wrapper applies import-time compatibility
  patches before collection, as used by NPU.
- `python_native`: tests are collected by another platform launcher, as used
  by FlagOS and Reference; their four scripts are declarations and no-op
  entry points rather than independent CI jobs.

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
