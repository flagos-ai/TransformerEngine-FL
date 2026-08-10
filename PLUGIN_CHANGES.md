# Plugin Changes (base → main)

**Summary**: Fork added comprehensive plugin system with 82 new files in `transformer_engine/plugin/` and extensive integration changes.

## Statistics
- **Total diff lines**: 48,150 lines
- **Files changed**: 236 files
- **Insertions**: 44,027 lines
- **Deletions**: 639 lines

## New Files (P0 - Plugin Directory)

### Core Plugin System (82 files added)
All files in `transformer_engine/plugin/` are NEW (P0 priority):

**Plugin Core Infrastructure:**
- `transformer_engine/plugin/__init__.py`
- `transformer_engine/plugin/core/__init__.py`
- `transformer_engine/plugin/core/_build_config.py.template`
- `transformer_engine/plugin/core/_module_setup.py`
- `transformer_engine/plugin/core/backends/__init__.py`
- `transformer_engine/plugin/core/backends/fa_utils.py`

**Vendor Backends (7 backends):**
1. **CUDA**: `vendor/cuda/*.py` (cuda.py, flash_attention.py, register_ops.py)
2. **Enflame**: `vendor/enflame/*.py`
3. **Hygon**: `vendor/hygon/*.py`
4. **Iluvatar**: `vendor/iluvatar/*.py`
5. **Kunlunxin**: `vendor/kunlunxin/*.py`
6. **Metax**: `vendor/metax/*.py`
7. **MUSA**: `vendor/musa/*.py`

**FlagOS Backend:**
- `backends/flagos/flagos.py`
- `backends/flagos/register_ops.py`
- `backends/flagos/impl/`: fused_adam.py, gemm.py, multi_tensor.py, normalization.py, rmsnorm.py, softmax.py
- `backends/flagos/impl/trition/fused_rope.py`
- `backends/flagos/attention/dot_product_attention/backends.py`

**Reference Backend:**
- `backends/reference/reference.py`
- `backends/reference/register_ops.py`
- `backends/reference/impl/`: activation.py, dropout.py, gemm.py, normalization.py, optimizer.py, rmsnorm.py, softmax.py
- `backends/reference/flash_attention.py`

**Benchmarks:**
- `plugin/benchmarks/benchmark_all_backends.py`

## Modified Files (P1/P2)

### P1 - Build System (must be carefully merged)
- **setup.py**: Plugin compilation targets added
- **transformer_engine/__init__.py**: CUDA patches and plugin initialization

### P2 - PyTorch Integration (44 files modified)
Modified files in `transformer_engine/pytorch/` to integrate plugin API calls.

### CI/CD Changes
- `.github/workflows/`: 18 new workflow files for multi-backend testing
- `.github/configs/`: 6 new config files (ascend.yml, cuda.yml, hygon.yml, metax.yml, musa.yml)
- `.github/scripts/`: 5 new setup scripts for different backends

### Test Suite (Plugin Tests)
- `tests/plugin/`: Complete test suite for plugin system (30+ test files)
  - Backend-specific tests: flagos, hygon, musa, npu, reference
  - Integration tests for all 7 vendor backends

## Critical P0 Files (Plugin System Core)

The following directories MUST be preserved during merge:

1. **`transformer_engine/plugin/`** - Entire directory (82 files)
   - All vendor backends
   - All FlagOS implementations
   - All reference implementations
   - Plugin core infrastructure

2. **`tests/plugin/`** - Entire test suite (30+ files)

## CUDA Patches

Modified: `transformer_engine/__init__.py`
- Adds runtime patches to torch.cuda APIs
- Enables device-agnostic code via TE_DEVICE_TYPE environment variable

## Build System Modifications

Modified: `setup.py`
- Adds plugin compilation targets
- Integrates vendor backend build configuration

## API Changes (Python)

44 files modified in `transformer_engine/pytorch/`:
- OP calls updated to route through plugin system
- Device type abstraction added (torch.cuda → torch.device(TE_DEVICE_TYPE))

## Merge Strategy

**P0 (Plugin files)**: In case of conflict, ALWAYS keep fork version
**P1 (setup.py, __init__.py)**: Manual merge required, must preserve plugin integration
**P2 (pytorch/*.py)**: Accept upstream changes, then re-apply device abstraction patches

