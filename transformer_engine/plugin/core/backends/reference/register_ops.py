# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Reference backend operator registrations.

This module registers all REFERENCE (PyTorch) implementations.
"""

from __future__ import annotations

import functools

from ...types import OpImpl, BackendImplKind


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """
    Register all PyTorch (REFERENCE) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .reference import ReferenceBackend

    # Create a backend instance to access the methods
    backend = ReferenceBackend()

    # Bind is_available to all methods
    is_avail = backend.is_available

    impls = [
        # Normalization
        OpImpl(op_name="rmsnorm_fwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.rmsnorm_fwd, is_avail), vendor=None, priority=50),
        OpImpl(op_name="rmsnorm_bwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.rmsnorm_bwd, is_avail), vendor=None, priority=50),
        OpImpl(op_name="layernorm_fwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.layernorm_fwd, is_avail), vendor=None, priority=50),
        OpImpl(op_name="layernorm_bwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.layernorm_bwd, is_avail), vendor=None, priority=50),

        # GEMM
        OpImpl(op_name="generic_gemm", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.generic_gemm, is_avail), vendor=None, priority=50),

        # Activations - Forward
        OpImpl(op_name="gelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.gelu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="geglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.geglu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="qgelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.qgelu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="qgeglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.qgeglu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="relu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.relu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="reglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.reglu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="srelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.srelu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="sreglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.sreglu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="silu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.silu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="swiglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.swiglu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="clamped_swiglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.clamped_swiglu, is_avail), vendor=None, priority=50),

        # Activations - Backward
        OpImpl(op_name="dgelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.dgelu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="dgeglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.dgeglu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="dqgelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.dqgelu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="dqgeglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.dqgeglu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="drelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.drelu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="dreglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.dreglu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="dsrelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.dsrelu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="dsreglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.dsreglu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="dsilu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.dsilu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="dswiglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.dswiglu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="clamped_dswiglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.clamped_dswiglu, is_avail), vendor=None, priority=50),

        # Activations - Bias + Backward
        OpImpl(op_name="dbias_dgelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.dbias_dgelu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="dbias_dsilu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.dbias_dsilu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="dbias_drelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.dbias_drelu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="dbias_dqgelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.dbias_dqgelu, is_avail), vendor=None, priority=50),
        OpImpl(op_name="dbias_dsrelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.dbias_dsrelu, is_avail), vendor=None, priority=50),

        # Softmax
        OpImpl(op_name="scaled_softmax_forward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.scaled_softmax_forward, is_avail), vendor=None, priority=50),
        OpImpl(op_name="scaled_softmax_backward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.scaled_softmax_backward, is_avail), vendor=None, priority=50),
        OpImpl(op_name="scaled_masked_softmax_forward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.scaled_masked_softmax_forward, is_avail), vendor=None, priority=50),
        OpImpl(op_name="scaled_masked_softmax_backward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.scaled_masked_softmax_backward, is_avail), vendor=None, priority=50),
        OpImpl(op_name="scaled_upper_triang_masked_softmax_forward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.scaled_upper_triang_masked_softmax_forward, is_avail), vendor=None, priority=50),
        OpImpl(op_name="scaled_upper_triang_masked_softmax_backward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.scaled_upper_triang_masked_softmax_backward, is_avail), vendor=None, priority=50),
        OpImpl(op_name="scaled_aligned_causal_masked_softmax_forward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.scaled_aligned_causal_masked_softmax_forward, is_avail), vendor=None, priority=50),
        OpImpl(op_name="scaled_aligned_causal_masked_softmax_backward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.scaled_aligned_causal_masked_softmax_backward, is_avail), vendor=None, priority=50),

        # Fused attention backend getter
        OpImpl(op_name="get_fused_attn_backend", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.get_fused_attn_backend, is_avail), vendor=None, priority=50),

        # Dropout
        OpImpl(op_name="dropout_fwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.dropout_fwd, is_avail), vendor=None, priority=50),
        OpImpl(op_name="dropout_bwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.dropout_bwd, is_avail), vendor=None, priority=50),

        # Library version getters
        OpImpl(op_name="get_cublasLt_version", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.get_cublasLt_version, is_avail), vendor=None, priority=50),
        OpImpl(op_name="get_cudnn_version", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.get_cudnn_version, is_avail), vendor=None, priority=50),
        OpImpl(op_name="get_num_cublas_streams", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.get_num_cublas_streams, is_avail), vendor=None, priority=50),

        # Multi-tensor optimizer operations
        OpImpl(op_name="multi_tensor_scale", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.multi_tensor_scale, is_avail), vendor=None, priority=50),
        OpImpl(op_name="multi_tensor_l2norm", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.multi_tensor_l2norm, is_avail), vendor=None, priority=50),
        OpImpl(op_name="multi_tensor_unscale_l2norm", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.multi_tensor_unscale_l2norm, is_avail), vendor=None, priority=50),
        OpImpl(op_name="multi_tensor_adam", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.multi_tensor_adam, is_avail), vendor=None, priority=50),
        OpImpl(op_name="multi_tensor_adam_param_remainder", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.multi_tensor_adam_param_remainder, is_avail), vendor=None, priority=50),
        OpImpl(op_name="multi_tensor_sgd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.multi_tensor_sgd, is_avail), vendor=None, priority=50),

        # FlashAttention class getter
        OpImpl(op_name="get_flash_attention_class", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.get_flash_attention_class, is_avail), vendor=None, priority=50),

        # Attention backend selection
        OpImpl(op_name="get_attention_backend", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=_bind_is_available(backend.get_attention_backend, is_avail), vendor=None, priority=50),
    ]

    registry.register_many(impls)
