"""Python-side compatibility patches for the NPU vendor backend."""

from __future__ import annotations

from collections.abc import Callable

import torch

try:
    import torch_npu
except ImportError:
    pass

from types import SimpleNamespace


def _noop(*args, **kwargs):
    return None


def get_npu_device_properties(device=None):
    return SimpleNamespace(
        name="Fake NPU",
        total_memory=16 * 1024**3,
        major=9,
        minor=0,
        multi_processor_count=80,
        uuid="fake-uuid-12345",
    )


_PATCH_CALLS: list[tuple[object, str, Callable[..., object]]] = [
    # We do not recommend replace is_available, due to its device-related behavior.
    (torch.cuda, "get_device_properties", get_npu_device_properties),
    (torch.cuda, "device", torch_npu.npu.device),
    (torch.cuda, "current_device", torch_npu.npu.current_device),
    (torch.cuda, "synchronize", torch_npu.npu.synchronize),
    (torch.cuda, "is_current_stream_capturing", torch_npu.npu.is_current_stream_capturing),
    # TODO: Add NVTX patches for NPU.
    # NVTX is CUDA-specific; make it a no-op on NPU.
    (torch.cuda.nvtx, "range_push", _noop),
    (torch.cuda.nvtx, "range_pop", _noop),
    # TODO: Add other patches for NPU.
]


def apply_patch() -> None:
    """Apply NPU Python-side patches (idempotent, best-effort)."""
    try:
        import torch_npu

        if not torch_npu.npu.is_available():
            return

    except Exception as e:
        print(f"[TE-FL] NPU backend not available: {e}")
        # If backend availability can't be determined, don't patch.
        return

    # Mark TE global device type for Python-side callers.
    # IMPORTANT: do not import `transformer_engine` here, because TE's `__init__.py`
    # imports this module to run patches and that would cause a circular import.
    try:
        import transformer_engine

        transformer_engine.TE_DEVICE_TYPE = "npu"
        transformer_engine.TE_PLATFORM = torch_npu.npu
    except Exception as e:
        print(f"[TE-FL NPU Patches] Error setting TE device type or platform: {e}")
        # Best-effort: don't fail patching if we can't set the global.
        pass

    # Only patch when torch_npu.npu exists and is usable.
    if not hasattr(torch_npu, "npu"):
        return
    try:
        if not torch_npu.npu.is_available():
            return
    except Exception:
        return

    for parent, attr, replacement in _PATCH_CALLS:
        if not hasattr(parent, attr):
            continue
        try:
            setattr(parent, attr, replacement)
        except Exception:
            # Best-effort: patching should never crash import/initialization.
            continue
    print(f"[TE-FL] NPU backend patches applied")
