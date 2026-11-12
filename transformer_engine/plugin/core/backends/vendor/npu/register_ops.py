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
        # cuDNN is not available on Ascend NPU.
        OpImpl(
            op_name="get_cudnn_version",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.get_cudnn_version, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # FlashAttention class getter
        OpImpl(
            op_name="get_flash_attention_class",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.get_flash_attention_class, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # LayerNorm forward
        OpImpl(
            op_name="layernorm_fwd",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.layernorm_fwd, is_avail),
            vendor="NPU",
            priority=100,
        ),
        # LayerNorm backward
        OpImpl(
            op_name="layernorm_bwd",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.layernorm_bwd, is_avail),
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
        # Multi-tensor Adam/AdamW
        OpImpl(
            op_name="multi_tensor_adam",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.multi_tensor_adam, is_avail),
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

    impls.extend(_triton_permutation_impls(is_avail))

    registry.register_many(impls)


# Triton APIs claimed by NPU, named after the TE function each replaces.
# Dispatch selects per operator, so this list grows one entry at a time;
# anything left out stays on te.native.
#
# Empty on purpose. TE's own Triton kernels do run on Ascend -- slower than a
# native kernel, but correct -- so every API here resolves to te.native until
# a faster Ascend implementation exists. Uncomment an entry once
# triton_permutation.py actually implements it; a listed-but-unimplemented API
# raises NotImplementedError on NPU.
#
# One caveat when editing: entries are independent only while they stay
# equivalent to native. An implementation that encodes row_id_map its own way
# has to cover the APIs on both ends of that map (see triton_permutation.py).
_TRITON_PERMUTATION_APIS = (
    # "make_row_id_map",
    # "permute_with_mask_map",
    # "unpermute_with_mask_map",
    # "unpermute_with_mask_map_bwd_with_merging_probs",
    # "make_chunk_sort_map",
    # "sort_chunks_by_map",
)


def _triton_permutation_impls(is_available_fn) -> list:
    """Build OpImpls for the Triton permutation APIs claimed by NPU."""
    from . import triton_permutation

    return [
        OpImpl(
            op_name=f"triton_{api_name}",
            impl_id="vendor.npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(getattr(triton_permutation, api_name), is_available_fn),
            vendor="NPU",
            priority=100,
        )
        for api_name in _TRITON_PERMUTATION_APIS
    ]
