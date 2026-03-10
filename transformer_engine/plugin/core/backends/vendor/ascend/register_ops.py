# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
ASCEND vendor backend operator registrations.

This module registers all VENDOR (ASCEND) implementations.
"""

from __future__ import annotations

import functools

from ....types import OpImpl, BackendImplKind


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """
    Register all ASCEND (VENDOR) operator implementations.

    Args:
        registry: Registry to register into
    """
    # Import ASCEND backend to get all the wrapped tex functions
    from .ascend import AscendBackend

    # Create a backend instance to access the methods
    backend = AscendBackend()

    # Check if ASCEND is available before registering
    if not backend.is_available():
        return

    # Bind is_available to all methods
    is_avail = backend.is_available

    impls = [
        # Activations - Forward
        OpImpl(op_name="gelu", impl_id="vendor.ascend", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.gelu, is_avail), vendor="Ascend", priority=100),

        # FlashAttention class getter
        OpImpl(op_name="get_flash_attention_class", impl_id="vendor.ascend", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.get_flash_attention_class, is_avail), vendor="Ascend", priority=100),
    ]

    registry.register_many(impls)
