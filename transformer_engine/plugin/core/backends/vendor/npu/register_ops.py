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
        # cuDNN version (returns 0 for NPU)
        OpImpl(
            op_name="get_cudnn_version",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.get_cudnn_version, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Quantize
        OpImpl(
            op_name="quantize",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.quantize, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation: gelu
        OpImpl(
            op_name="gelu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.gelu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation: relu
        OpImpl(
            op_name="relu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.relu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation: silu
        OpImpl(
            op_name="silu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.silu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation: swiglu
        OpImpl(
            op_name="swiglu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.swiglu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation: geglu
        OpImpl(
            op_name="geglu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.geglu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation: reglu
        OpImpl(
            op_name="reglu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.reglu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation: srelu
        OpImpl(
            op_name="srelu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.srelu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation: glu
        OpImpl(
            op_name="glu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.glu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation: qgelu
        OpImpl(
            op_name="qgelu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.qgelu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation: qgeglu
        OpImpl(
            op_name="qgeglu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.qgeglu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation: sreglu
        OpImpl(
            op_name="sreglu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.sreglu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation backward: dgelu
        OpImpl(
            op_name="dgelu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.dgelu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation backward: dsilu
        OpImpl(
            op_name="dsilu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.dsilu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation backward: drelu
        OpImpl(
            op_name="drelu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.drelu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation backward: dsrelu
        OpImpl(
            op_name="dsrelu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.dsrelu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation backward: dqgelu
        OpImpl(
            op_name="dqgelu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.dqgelu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation backward: dgeglu
        OpImpl(
            op_name="dgeglu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.dgeglu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation backward: dswiglu
        OpImpl(
            op_name="dswiglu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.dswiglu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation backward: dreglu
        OpImpl(
            op_name="dreglu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.dreglu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation backward: dsreglu
        OpImpl(
            op_name="dsreglu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.dsreglu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation backward: dqgeglu
        OpImpl(
            op_name="dqgeglu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.dqgeglu, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # Activation backward: dglu
        OpImpl(
            op_name="dglu",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.dglu, is_avail),
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
        # Multi-tensor: scale tensor
        OpImpl(
            op_name="multi_tensor_scale_tensor",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.multi_tensor_scale_tensor, is_avail),
            vendor="NPU",
            priority=100,
        ),

        OpImpl(
            op_name="generic_gemm",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.generic_gemm, is_avail),
            vendor="NPU",
            priority=100,
        )
    ]

    registry.register_many(impls)
