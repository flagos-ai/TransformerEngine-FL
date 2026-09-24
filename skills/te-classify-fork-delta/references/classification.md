# Classification Reference

| Category | Scope | Downstream owner |
|---|---|---|
| plugin-core | Plugin framework and registry | Plugin API audit |
| plugin-backend | Backend implementations and registrations | Capability audit |
| invasive-runtime | Modified upstream runtime files | Semantic integration |
| device-abstraction | Device constants and patches | Device audit |
| build-packaging | Setup, manifests, build tools | Build/package audit |
| submodule | Gitmodules and gitlinks | Submodule integration |
| cicd | GitHub automation | CI/CD audit |
| qa | QA entrypoints | QA audit |
| tests | Tests and test utilities | Test matrix audit |
| docs-examples-benchmarks | Support and executable examples | Compatibility audit |
| repository-metadata | Lint, license, ignore, contribution files | Finalization audit |

P0 affects numerical behavior, dispatch, API/ABI, device placement, backend selection, or is changed on both sides. P1 affects build, install, import, tests, or dependency checkout. P2 affects automation and developer workflows. P3 has no executable or release effect.

For every both-changed file record: fork behavior, upstream change, invariant, observing test, and downstream owner.
