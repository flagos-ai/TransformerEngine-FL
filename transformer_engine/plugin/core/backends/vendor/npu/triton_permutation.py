# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""NPU implementations of TE's Triton permutation APIs.

Each function mirrors one public function in
``transformer_engine/pytorch/triton/permutation.py``: same name, same
parameter names and order, same return arity. Dispatch forwards ``*args,
**kwargs`` untouched and never adapts a signature, and the middle layer does
pass some arguments by keyword, so a renamed or reordered parameter breaks the
call rather than being adjusted.

Shapes and dtypes are documented on the TE functions and not restated here.

An implementation must be semantically equivalent to the function it replaces,
**including the intermediates it produces or consumes**. ``make_row_id_map``
builds the ``row_id_map`` the permute and unpermute APIs read. Such a value is
opaque to TE, so a vendor may encode it its own way -- but then every API
touching it must come from the same vendor, or a native consumer misreads it
and returns a wrong result with no error. Given an equivalent encoding, any
single API can be replaced on its own.

TE pins the mask-map layout from outside this file: ``register_fake`` in
``pytorch/permutation.py`` hardcodes the shape and ``moe_unpermute`` recovers
``num_experts`` from it. A wrong output shape or dtype is not validated in
eager; it surfaces as ``AssertionError: ... incorrect fake`` only under
``torch.compile``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


_NPU_MAX_GRID = 65535


def _token_grid(num_tokens, hidden_tiles):
    return (min(num_tokens * hidden_tiles, _NPU_MAX_GRID),)


def _not_implemented(api_name: str) -> NotImplementedError:
    """Error raised while an NPU kernel is missing.

    Dispatch never retries, so a placeholder return value would be consumed
    as a real result.
    """
    return NotImplementedError(
        f"NPU implementation for Triton API '{api_name}' is not available yet. "
        "Unregister it in register_ops.py to fall back to te.native."
    )


def make_row_id_map(
    routing_map: torch.Tensor,
    num_tokens: int,
    num_experts: int,
) -> torch.Tensor:
    """Build the mask-map token-to-expert map.

    Building the map is a separate step from permuting it.
    """
    raise _not_implemented("make_row_id_map")


def permute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: Optional[torch.Tensor],
    scale: Optional[torch.Tensor],
    pad_offsets: Optional[torch.Tensor],
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
    scale_hidden_dim: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Permute tokens according to a mask-map ``row_id_map``.

    ``permuted_probs`` keeps ``probs.dtype``; it does not adopt ``inp.dtype``.
    """
    import triton

    from .triton_permutation_kernels import _permute_kernel_npu

    # Use torch.zeros when pad_offsets is provided to ensure padding regions are zeroed.
    # The kernel writes only to valid positions, leaving padding positions at zero.
    alloc = torch.zeros if pad_offsets is not None else torch.empty
    output = alloc((num_out_tokens, hidden_size), dtype=inp.dtype, device=inp.device)
    permuted_probs = (
        alloc((num_out_tokens,), dtype=probs.dtype, device=inp.device)
        if probs is not None
        else None
    )
    permuted_scale = (
        alloc((num_out_tokens, scale_hidden_dim), dtype=scale.dtype, device=inp.device)
        if scale is not None
        else None
    )
    # pylint: disable=unnecessary-lambda-assignment
    grid = lambda meta: _token_grid(num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))
    _permute_kernel_npu[grid](
        inp,
        row_id_map,
        probs,
        scale,
        permuted_scale,
        pad_offsets,
        # Pass output buffers as input parameters (for JAX input_output_aliases compatibility).
        # In PyTorch, these point to the same memory as the output pointers below.
        output,
        permuted_probs,
        scale_hidden_dim,
        num_tokens,
        num_out_tokens,
        row_id_map.stride(0),
        row_id_map.stride(1),
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        probs.stride(0) if probs is not None else None,
        probs.stride(1) if probs is not None else None,
        scale.stride(0) if scale is not None else None,
        scale.stride(1) if scale is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        permuted_scale.stride(0) if permuted_scale is not None else None,
        permuted_scale.stride(1) if permuted_scale is not None else None,
        output,
        permuted_probs,
        num_experts,
        hidden_size,
        PERMUTE_PROBS=probs is not None,
        PERMUTE_SCALE=scale is not None,
        FUSION_PAD=pad_offsets is not None,
        npu_token_count=num_tokens,
    )
    return output, permuted_scale, permuted_probs


def unpermute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Optional[torch.Tensor],
    permuted_probs: Optional[torch.Tensor],
    pad_offsets: Optional[torch.Tensor],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Scatter permuted tokens back, optionally weighting by merging probs."""
    import triton

    from .triton_permutation_kernels import _unpermute_kernel_npu

    output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device=inp.device)
    if permuted_probs is not None:
        unpermuted_probs = torch.empty(
            (num_tokens, num_experts), dtype=permuted_probs.dtype, device=inp.device
        )
    else:
        unpermuted_probs = None
    # pylint: disable=unnecessary-lambda-assignment
    grid = lambda meta: _token_grid(num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))
    _unpermute_kernel_npu[grid](
        inp,
        row_id_map,
        merging_probs,
        permuted_probs,
        pad_offsets,
        # Dummy buffer parameters for kernel signature consistency with _permute_kernel.
        # These are unused in unpermute but maintain consistent interface.
        output,  # output_buf_ptr (unused, passed for signature consistency)
        unpermuted_probs,  # unpermuted_probs_buf_ptr (unused, passed for signature consistency)
        row_id_map.stride(0),
        row_id_map.stride(1),
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        merging_probs.stride(0) if merging_probs is not None else None,
        merging_probs.stride(1) if merging_probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        unpermuted_probs.stride(0) if unpermuted_probs is not None else None,
        unpermuted_probs.stride(1) if unpermuted_probs is not None else None,
        output,
        unpermuted_probs,
        num_experts,
        hidden_size,
        PROBS_LOAD_WIDTH=triton.next_power_of_2(num_experts),
        WITH_MERGING_PROBS=merging_probs is not None,
        PERMUTE_PROBS=permuted_probs is not None,
        FUSION_UNPAD=pad_offsets is not None,
        npu_token_count=num_tokens,
    )
    return output, unpermuted_probs


def unpermute_with_mask_map_bwd_with_merging_probs(
    fwd_output_grad: torch.Tensor,
    row_id_map: torch.Tensor,
    fwd_input: torch.Tensor,
    merging_probs: torch.Tensor,
    pad_offsets: Optional[torch.Tensor],
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward of the merging-probs unpermute.

    Reads the same ``row_id_map`` as :func:`unpermute_with_mask_map`.
    """
    raise _not_implemented("unpermute_with_mask_map_bwd_with_merging_probs")


def make_chunk_sort_map(
    split_sizes: torch.Tensor,
    sorted_indices: torch.Tensor,
    num_tokens: int,
    num_splits: int,
) -> torch.Tensor:
    """Build the chunk-sort row map."""
    import triton

    from .triton_permutation_kernels import _make_chunk_sort_map_kernel_npu

    row_id_map = torch.empty((num_tokens,), dtype=torch.int32, device=split_sizes.device)
    grid = _token_grid(num_tokens, 1)
    _make_chunk_sort_map_kernel_npu[grid](
        split_sizes,
        sorted_indices,
        row_id_map,
        num_splits,
        IDX_LOAD_WIDTH=triton.next_power_of_2(num_splits),
        npu_token_count=num_tokens,
    )
    return row_id_map


def sort_chunks_by_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: Optional[torch.Tensor],
    num_tokens: int,
    hidden_size: int,
    is_forward: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Reorder token chunks using a chunk-sort ``row_id_map``.

    Both directions come through this one API via ``is_forward``, so
    registering it covers the backward pass too.
    """
    import triton

    from .triton_permutation_kernels import _sort_chunks_by_map_kernel_npu

    output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device=inp.device)
    if probs is not None:
        permuted_probs = torch.empty((num_tokens,), dtype=probs.dtype, device=inp.device)
    else:
        permuted_probs = None
    # pylint: disable=unnecessary-lambda-assignment
    grid = lambda meta: _token_grid(num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))
    _sort_chunks_by_map_kernel_npu[grid](
        inp,
        row_id_map,
        probs,
        output,  # no use in Pytorch side, serves as WAR for JAX side
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        probs.stride(0) if probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        output,
        permuted_probs,
        hidden_size,
        PERMUTE_PROBS=probs is not None,
        FORWARD=is_forward,
        npu_token_count=num_tokens,
    )
    return output, permuted_probs
