# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
KunLunXin backend operator registrations.

This module registers all KunLunXin PyTorch implementations.
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
    Register all KunLunXin PyTorch operator implementations.

    Args:
        registry: Registry to register into
    """
    from .kunlunxin import KunLunXinBackend

    # Create a backend instance to access the methods
    backend = KunLunXinBackend()

    if not backend.is_available():
        return
    # Bind is_available to all methods
    is_avail = backend.is_available

    impls = [
        # FlashAttention class getter
        OpImpl(
            op_name="get_flash_attention_class",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.get_flash_attention_class, is_avail),
            vendor="KUNLUNXIN",
            priority=100,
        ),
        OpImpl(
            op_name="rmsnorm_bwd",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rmsnorm_bwd, is_avail),
            vendor="KUNLUNXIN",
            priority=100,
        ),
        OpImpl(
            op_name="multi_tensor_adam",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.multi_tensor_adam, is_avail),
            vendor="KUNLUNXIN",
            priority=100,
        ),
        OpImpl(
            op_name="multi_tensor_scale",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.multi_tensor_scale, is_avail),
            vendor="KUNLUNXIN",
            priority=100,
        ),
        OpImpl(
            op_name="rmsnorm_fwd",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rmsnorm_fwd, is_avail),
            vendor="KUNLUNXIN",
            priority=100,
        ),
        OpImpl(
            op_name="multi_tensor_adam_fp8",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.multi_tensor_adam_fp8, is_avail),
            vendor="KUNLUNXIN",
            priority=100,
        ),
        OpImpl(
            op_name="multi_tensor_adam_capturable",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.multi_tensor_adam_capturable, is_avail),
            vendor="KUNLUNXIN",
            priority=100,
        ),
        OpImpl(
            op_name="multi_tensor_adam_capturable_master",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.multi_tensor_adam_capturable_master, is_avail),
            vendor="KUNLUNXIN",
            priority=100,
        ),
        OpImpl(
            op_name="cast_to_fp8",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.cast_to_fp8, is_avail),
            vendor="KUNLUNXIN",
            priority=100,
        ),
        OpImpl(
            op_name="bulk_overlap_ag_with_external_gemm",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.bulk_overlap_ag_with_external_gemm, is_avail),
            vendor="KUNLUNXIN",
            priority=100,
        ),
        OpImpl(
            op_name="multi_tensor_l2norm",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.multi_tensor_l2norm, is_avail),
            vendor="KUNLUNXIN",
            priority=100,
        ),
        OpImpl(
            op_name="get_cudnn_version",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.get_cudnn_version, is_avail),
            vendor="KUNLUNXIN",
            priority=100,
        ),
        OpImpl(
            op_name="get_attention_backend",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.get_attention_backend, is_avail),
            vendor="KUNLUNXIN",
            priority=100,
        ),
        OpImpl(
            op_name="scaled_masked_softmax_forward",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.scaled_masked_softmax_forward, is_avail),
            vendor="KUNLUNXIN",
            priority=100,
        ),
        OpImpl(
            op_name="scaled_masked_softmax_backward",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.scaled_masked_softmax_backward, is_avail),
            vendor="KUNLUNXIN",
            priority=100,
        ),
        OpImpl(
            op_name="multi_tensor_compute_scale_and_scale_inv",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.multi_tensor_compute_scale_and_scale_inv, is_avail),
            vendor="KUNLUNXIN",
            priority=100,
        ),
        OpImpl(
            op_name="multi_tensor_compute_scale_inv_e8m0",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.multi_tensor_compute_scale_inv_e8m0, is_avail),
            vendor="KUNLUNXIN",
            priority=100,
        ),
    ]

    registry.register_many(impls)
