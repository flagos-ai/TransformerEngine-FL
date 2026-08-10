# Stage 5: CUDA Hardcoding Analysis

**Date**: 2025-01-10  
**Upgrade**: v2.14 → v2.17  
**Branch**: merge/dev-to-main-20260810  

## Summary

Scanned Python layer diff (base..dev, excluding csrc) for newly introduced hardcoded `"cuda"` device strings.

**Total candidates**: 26 lines across 9 files

## Triage Results

按照 Phase 03 文档规则，逐行分析每个候选：

### Files to Patch

**utils.py** (4 patches):
- Line 687: `device = torch.device("cuda", ...)` → PATCH (device selection)
- Line 691: `device = torch.device("cuda", ...)` → PATCH (device selection)
- Line 852: `torch.get_autocast_dtype("cuda")` → PATCH (autocast device type)
- Line 857: `device_type="cuda"` → PATCH (autocast context device type)

**distributed.py** (3 patches):
- Line 109: `device = torch.device("cuda", device)` → PATCH (device selection)
- Line 131/135: `device = torch.device("cuda", ...)` → PATCH (device selection)
- Line 295: `torch.get_autocast_dtype("cuda")` → PATCH (autocast)

**quantization.py** (6 patches):
- Lines 1425, 1474, 1517, 1559, 1642, 1873: `device = torch.device("cuda")` → PATCH (device selection)

**quantized_tensor.py** (1 patch):
- Line 331: `device = torch.device("cuda")` → PATCH (device selection)

**cpu_offload.py** (1 patch):
- Line 350: `device=torch.device("cuda")` → PATCH (device selection)

**jit.py** (3 patches):
- Lines 298-300, 335, 339, 373: `device="cuda"` → PATCH (test fixtures)

### Files to Skip (No Patching)

**utils.py** (2 skips):
- Line 686: `if device.type != "cuda":` → SKIP (CUDA detection guard)
- Line 690: `if device.type == "cuda" and device.index is None:` → SKIP (CUDA detection guard)
- Line 713: `if device1.type == "cuda":` → SKIP (CUDA-specific index check)

**distributed.py** (4 skips):
- Line 99: `device: Union[int, str, torch.device] = "cuda"` → SKIP (default parameter, will be converted by torch.device())
- Line 1029: `device = "cuda"` → SKIP (string literal for backward compat)
- Line 1853: `device = torch.device("cuda", ...)` → CONTEXT NEEDED (may be in CUDA-only block)

**ep.py** (7 skips):
- Line 123: `ep_group._get_backend(torch.device("cuda"))` → SKIP (low-level backend API, must specify exact backend)
- Line 239: `device = torch.device("cuda", ...)` → SKIP (EP-specific CUDA path)
- Lines 271, 291, 312, 330, 350: `device_types="cuda"` → SKIP (PyTorch registration parameter)

**newton_schulz.py** (1 skip):
- Line 155: `group._get_backend(torch.device("cuda"))` → SKIP (low-level backend API)

**transformer.py** (2 skips):
- Lines 195, 347: `device = "cuda"` → SKIP (docstring default value)

## Patch Plan

**Total patches**: 18 lines across 6 files

### Replacement patterns:

1. `device="cuda"` → `device=TE_DEVICE_TYPE`
2. `torch.device("cuda")` → `torch.device(TE_DEVICE_TYPE)`
3. `torch.device("cuda", index)` → `torch.device(TE_DEVICE_TYPE, index)`
4. `torch.get_autocast_dtype("cuda")` → `torch.get_autocast_dtype(TE_DEVICE_TYPE)`
5. `device_type="cuda"` → `device_type=TE_DEVICE_TYPE`

### Import addition:

All patched files need:
```python
from transformer_engine import TE_DEVICE_TYPE
```

## Next Steps

1. Apply patches to 6 files
2. Syntax check all modified files
3. Verify no un-patched device strings remain
4. Commit with detailed message
