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

    impls = [
        # FlashAttention class getter
        OpImpl(
            op_name="get_flash_attention_class",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.get_flash_attention_class, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # RMSNorm forward
        OpImpl(
            op_name="rmsnorm_fwd",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rmsnorm_fwd, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # RMSNorm backward
        OpImpl(
            op_name="rmsnorm_bwd",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rmsnorm_bwd, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Multi-tensor scale
        OpImpl(
            op_name="multi_tensor_scale",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.multi_tensor_scale, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Multi-tensor L2 norm
        OpImpl(
            op_name="multi_tensor_l2norm",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.multi_tensor_l2norm, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Multi-tensor compute scale and scale_inv
        OpImpl(
            op_name="multi_tensor_compute_scale_and_scale_inv",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.multi_tensor_compute_scale_and_scale_inv, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Multi-tensor compute scale_inv E8M0
        OpImpl(
            op_name="multi_tensor_compute_scale_inv_e8m0",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.multi_tensor_compute_scale_inv_e8m0, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Attention backend selector
        OpImpl(
            op_name="get_attention_backend",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.get_attention_backend, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Multi-tensor: unscale + L2 norm
        OpImpl(
            op_name="multi_tensor_unscale_l2norm",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.multi_tensor_unscale_l2norm, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Generic GEMM
        OpImpl(
            op_name="generic_gemm",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.generic_gemm, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Grouped GEMM
        OpImpl(
            op_name="te_general_grouped_gemm",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.te_general_grouped_gemm, is_avail),
            vendor="NPU",
            priority=100,
        ),
    ]

    registry.register_many(impls)
