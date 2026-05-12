"""Python-side compatibility patches for the Tsingmicro(TXDA) vendor backend."""

from __future__ import annotations

from collections.abc import Callable

import torch


def _noop(*args, **kwargs):
    return None


# Patches: (parent_object, attribute_name, replacement_callable)
_PATCH_CALLS: list[tuple[object, str, Callable[..., object]]] = [
    (torch.cuda, "is_available", torch.txda.is_available),   
    (torch.cuda, "get_device_properties", torch.txda.get_device_properties),
    (torch.cuda, "device", torch.txda.device),
    (torch.cuda, "current_device", torch.txda.current_device),
    (torch.cuda, "synchronize", torch.txda.synchronize),
    (torch.cuda, "is_current_stream_capturing", torch.txda.is_current_stream_capturing),
    # NVTX is CUDA-specific; make it a no-op on TXDA.
    (torch.cuda.nvtx, "range_push", _noop),
    (torch.cuda.nvtx, "range_pop", _noop),
]


def apply_patch() -> None:
    """Apply TXDA Python-side patches (idempotent, best-effort)."""
    try:
        import torch_txda
        import flag_gems
        from torch_txda import transfer_to_txda
    except Exception as e:
        return

    # Only patch when torch.txda exists and is usable.
    if not hasattr(torch, "txda"):
        return
    try:
        if not torch.txda.is_available():
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
    print(f"[TE-FL] Tsingmicro(TXDA) backend patches applied")
