# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
NPU backend operator registrations.

This module registers all Ascend NPU PyTorch implementations into the
TE-FL plugin registry.
"""

from __future__ import annotations

import functools

from transformer_engine.plugin.core.types import OpImpl, BackendImplKind


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """
    Register all NPU operator implementations.

    Args:
        registry: Registry to register into
    """
    from .npu import NPUBackend

    backend = NPUBackend()

    if not backend.is_available():
        return

    is_avail = backend.is_available

    # Keep this list limited to kernels used by the profiled Qwen3.5 MoE
    # training path and backed by a meaningful Ascend-specific implementation.
    op_names = (
        "get_flash_attention_class",
        "get_attention_backend",
        "get_permutation_class",
        "rmsnorm_fwd",
        "rmsnorm_bwd",
        "te_general_grouped_gemm",
    )
    impls = [
        OpImpl(
            op_name=op_name,
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(getattr(backend, op_name), is_avail),
            vendor="NPU",
            priority=100,
        )
        for op_name in op_names
    ]

    registry.register_many(impls)
