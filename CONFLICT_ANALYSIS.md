# Merge Conflict Analysis (dev → main)

**Date**: 2026-08-10
**Merge**: upstream v2.17 → fork main
**Total Conflicts**: 229 files

## Priority Classification

### P0 (Plugin System - Critical) - 0 conflicts ✅
**No conflicts in transformer_engine/plugin/** - Plugin system is entirely new to the fork, no upstream overlap.

### P1 (Build System - Manual Merge Required) - 1 conflict
1. `setup.py` - Fork adds plugin compilation targets, upstream has build improvements

### P2 (CI/CD, Tests, Docs - Accept Upstream with Fork Patches) - 228 conflicts

**CI/CD (18 files)**:
- `.github/workflows/build.yml`
- `.github/workflows/lint.yml`
- `.github/actions/build-pytorch-wheel/Dockerfile`
- Other workflow files

**Tests (130+ files)**:
- `tests/cpp/operator/` - Multiple test files
- `tests/jax/` - JAX test suite
- `tests/pytorch/` - PyTorch test suite
- `qa/L0_*/test.sh` - QA test scripts
- `qa/L1_*/test.sh` - Integration test scripts

**Docs (20+ files)**:
- `docs/api/pytorch.rst`
- `docs/conf.py`
- `docs/index.rst`
- Various feature documentation

**Build Tools (10 files)**:
- `build_tools/VERSION.txt`
- `build_tools/wheel_utils/`

**Other**:
- `.gitignore`
- `README.rst`
- `3rdparty/cudnn-frontend` (submodule)

## Resolution Strategy

