# Copyright (c) 2025, BAAI. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Gated Delta Net implementation backed by local GDN implementation."""

from __future__ import annotations

import logging
from types import NotImplementedType
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def is_gated_delta_net_available() -> bool:
    """Check required Python operators and NPU availability, without launching kernels."""
    try:
        from .gdn_operators import validate_runtime

        validate_runtime()
        return hasattr(torch, "npu") and torch.npu.is_available()
    except (ImportError, RuntimeError, OSError) as e:
        logger.warning(f"[GDN] Runtime validation failed: {e}")
        return False


def gated_delta_net_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | NotImplementedType:
    """Run GDN while preserving the TE-FL BSHD contract.

    TE-FL receives ``query``, ``key`` and ``value`` in BSHD layout. The GDN
    implementation expects BHSD layout and returns output in BSHD layout.
    ``g`` and ``beta`` remain BSH tensors.

    Return NotImplemented before kernel execution for unsupported inputs or
    missing optional dependencies. The caller owns native fallback; a cached
    callable rechecks this contract on every call. Invalid inputs and kernel
    execution errors raise normally, never becoming fallback results.
    """
    _validate_inputs(query, key, value, g, beta, initial_state, chunk_size)
    if (
        query.device.type != "npu"
        or query.dtype not in (torch.float16, torch.bfloat16)
        or any(size == 0 for size in query.shape[:3])
        or query.shape[-1] != 128
        or value.shape[-1] != 128
        or beta.dtype != query.dtype
        or g.dtype != torch.float32
        or chunk_size != 64
        or initial_state is not None
        or output_final_state
    ):
        return NotImplemented
    if not is_gated_delta_net_available():
        return NotImplemented

    from .gdn_impl import flash_gated_delta_rule

    # Call implementation with layout conversion BSHD -> BHSD
    output, final_state = flash_gated_delta_rule(
        q=query.transpose(1, 2).contiguous(),
        k=key.transpose(1, 2).contiguous(),
        v=value.transpose(1, 2).contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=None,  # Will be computed inside
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        chunk_size=chunk_size,
    )

    return output.contiguous(), final_state


def _validate_inputs(query, key, value, g, beta, initial_state, chunk_size):
    if type(chunk_size) is not int or chunk_size <= 0 or chunk_size & (chunk_size - 1):
        raise ValueError("chunk_size must be a positive power of two.")
    if query.ndim != 4 or key.shape != query.shape or value.ndim != 4:
        raise ValueError("query/key must match and query/key/value must be BSHD tensors.")
    if value.shape[:3] != query.shape[:3] or g.shape != query.shape[:3] or beta.shape != g.shape:
        raise ValueError("query/key/value/g/beta must have matching BSH dimensions.")
    if query.shape[-1] == 0 or value.shape[-1] == 0:
        raise ValueError("Head dimensions must be positive.")
    tensors = (query, key, value, g, beta)
    if any(t.device != query.device or not t.is_floating_point() for t in tensors):
        raise ValueError("All inputs must be floating point tensors on the same device.")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("query/key/value must have the same dtype.")
    if initial_state is not None:
        expected = (query.shape[0], query.shape[2], query.shape[3], value.shape[3])
        if initial_state.shape != expected or initial_state.device != query.device:
            raise ValueError("initial_state must be BHKV on the input device.")
        if not initial_state.is_floating_point():
            raise ValueError("initial_state must be floating point.")
