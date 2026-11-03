"""Python-side compatibility patches for the Biren/SUPA vendor backend."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch


def _noop(*args, **kwargs):
    return None


_PATCH_CALLS: list[tuple[object, str, Callable[..., object]]] = [
    (torch.cuda.nvtx, "range_push", _noop),
    (torch.cuda.nvtx, "range_pop", _noop),
]


def apply_patch() -> None:
    """Apply Biren/SUPA Python-side patches (idempotent, best-effort)."""
    if not bool(int(os.getenv("TE_FL_USE_SUPA", "1"))):
        return

    # Set TE device type to "supa" so that TE's device checks match the real
    # tensor device type reported by torch_supa (avoids torch.compile conflicts
    # that arise from device_type(True) faking "cuda").
    try:
        import torch_supa  # noqa: F401
        import transformer_engine

        transformer_engine.TE_DEVICE_TYPE = "supa"
        transformer_engine.TE_PLATFORM = torch.supa
    except Exception:
        pass

    for parent, attr, replacement in _PATCH_CALLS:
        if not hasattr(parent, attr):
            continue
        try:
            setattr(parent, attr, replacement)
        except Exception:
            continue
