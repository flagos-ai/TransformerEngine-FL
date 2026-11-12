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


def _supports_row_tiles(inp, num_rows, hidden_size):
    return (
        num_rows > 0
        and hidden_size > 0
        and inp.stride(1) == 1
        and inp.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def _row_tile_shape(triton, hidden_size):
    cols = min(1024, max(32, triton.next_power_of_2(hidden_size)))
    return (4 if cols <= 512 else 2), cols


def _launch_row_gather(
    inp, source_rows, expert_rows, probs, output, output_probs, num_rows, hidden_size, mask_map
):
    import triton

    from .triton_permutation_kernels import _gather_permutation_rows

    rows, cols = _row_tile_shape(triton, hidden_size)
    if hidden_size == 2048 and num_rows >= 4096 and inp.dtype in (torch.float16, torch.bfloat16):
        # Larger row tiles and a bounded persistent grid amortize scheduling
        # overhead for the large 16-bit gathers used by MoE chunk sorting.
        rows = 8 if mask_map else 16
        tasks = triton.cdiv(num_rows, rows) * triton.cdiv(hidden_size, cols)
        grid = (min(tasks, 768),)
    else:
        grid = _token_grid(triton.cdiv(num_rows, rows), triton.cdiv(hidden_size, cols))
    _gather_permutation_rows[grid](
        inp,
        source_rows,
        expert_rows,
        probs,
        output,
        output_probs,
        num_rows,
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        probs.stride(0) if probs is not None else 0,
        probs.stride(1) if probs is not None and mask_map else 0,
        output_probs.stride(0) if output_probs is not None else 0,
        HIDDEN=hidden_size,
        WITH_PROBS=probs is not None,
        MASK_MAP=mask_map,
        ROWS=rows,
        COLS=cols,
    )


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
    num_out_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Build TE's row map; the optional output-row hint is not required.

    The scan derives every route from routing_map, including when the caller
    supplies an output allocation size or leaves it unknown for padded paths.
    """
    import triton

    from .triton_permutation_kernels import (
        _row_map_scan_blocks,
        _row_map_block_offsets,
        _row_map_write,
    )

    row_map = torch.zeros(
        (num_tokens, num_experts * 2 + 1), dtype=torch.int32, device=routing_map.device
    )
    if num_tokens == 0 or num_experts == 0:
        return row_map

    block_tokens = 128
    block_experts = 16
    num_blocks = triton.cdiv(num_tokens, block_tokens)
    local_prefix = torch.empty(
        (num_tokens, num_experts), dtype=torch.int32, device=routing_map.device
    )
    block_counts = torch.empty(
        (num_blocks, num_experts), dtype=torch.int32, device=routing_map.device
    )
    block_offsets = torch.empty_like(block_counts)
    expert_counts = torch.empty((num_experts,), dtype=torch.int32, device=routing_map.device)
    _row_map_scan_blocks[(num_blocks, triton.cdiv(num_experts, block_experts))](
        routing_map,
        local_prefix,
        block_counts,
        num_tokens,
        routing_map.stride(0),
        routing_map.stride(1),
        EXPERTS=num_experts,
        TOKENS=block_tokens,
        COLUMNS=block_experts,
    )
    _row_map_block_offsets[(num_experts,)](
        block_counts,
        block_offsets,
        expert_counts,
        BLOCKS=num_blocks,
        EXPERTS=num_experts,
        WIDTH=triton.next_power_of_2(num_blocks),
    )
    tokens_per_tile = 16
    _row_map_write[(min(triton.cdiv(num_tokens, tokens_per_tile), 256),)](
        routing_map,
        local_prefix,
        block_offsets,
        expert_counts,
        row_map,
        num_tokens,
        routing_map.stride(0),
        routing_map.stride(1),
        EXPERTS=num_experts,
        WIDTH=triton.next_power_of_2(num_experts),
        TOKENS=tokens_per_tile,
        BLOCK_TOKENS=block_tokens,
    )
    return row_map


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
    if num_tokens == 0:
        return output, permuted_scale, permuted_probs
    if (
        pad_offsets is None
        and scale is None
        and _supports_row_tiles(inp, num_out_tokens, hidden_size)
    ):
        from .triton_permutation_kernels import _pack_mask_map_routes

        # -1 leaves untouched output rows untouched, as in the scatter path.
        source_rows = torch.full((num_out_tokens,), -1, dtype=torch.int32, device=inp.device)
        expert_rows = torch.empty_like(source_rows) if probs is not None else None
        tokens_per_tile = 64
        _pack_mask_map_routes[_token_grid(triton.cdiv(num_tokens, tokens_per_tile), 1)](
            row_id_map,
            source_rows,
            expert_rows,
            num_tokens,
            num_out_tokens,
            row_id_map.stride(0),
            row_id_map.stride(1),
            EXPERTS=num_experts,
            WITH_PROBS=probs is not None,
            TOKENS=tokens_per_tile,
        )
        _launch_row_gather(
            inp,
            source_rows,
            expert_rows,
            probs,
            output,
            permuted_probs,
            num_out_tokens,
            hidden_size,
            mask_map=True,
        )
        return output, permuted_scale, permuted_probs

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
    if num_tokens == 0:
        return output, unpermuted_probs
    if (
        pad_offsets is None
        and permuted_probs is None
        and _supports_row_tiles(inp, num_tokens, hidden_size)
    ):
        from .triton_permutation_kernels import _combine_mask_map_rows

        rows, cols = _row_tile_shape(triton, hidden_size)
        grid = _token_grid(triton.cdiv(num_tokens, rows), triton.cdiv(hidden_size, cols))
        _combine_mask_map_rows[grid](
            inp,
            row_id_map,
            merging_probs,
            output,
            num_tokens,
            row_id_map.stride(0),
            row_id_map.stride(1),
            inp.stride(0),
            inp.stride(1),
            output.stride(0),
            output.stride(1),
            merging_probs.stride(0) if merging_probs is not None else 0,
            merging_probs.stride(1) if merging_probs is not None else 0,
            EXPERTS=num_experts,
            HIDDEN=hidden_size,
            WITH_WEIGHTS=merging_probs is not None,
            ROWS=rows,
            COLS=cols,
        )
        return output, unpermuted_probs

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

    from .triton_permutation_kernels import _chunk_sort_expand, _chunk_sort_offsets

    if num_splits == 0:
        return torch.arange(num_tokens, dtype=torch.int32, device=split_sizes.device)
    row_id_map = torch.empty((num_tokens,), dtype=torch.int32, device=split_sizes.device)
    if num_tokens == 0:
        return row_id_map

    ends = torch.empty((num_splits,), dtype=torch.int32, device=split_sizes.device)
    output_starts = torch.empty_like(ends)
    _chunk_sort_offsets[(1,)](
        split_sizes,
        sorted_indices,
        ends,
        output_starts,
        num_splits,
        triton.next_power_of_2(num_splits),
    )
    block = 256
    _chunk_sort_expand[(min(triton.cdiv(num_tokens, block), 64),)](
        ends, output_starts, row_id_map, num_tokens, num_splits, num_splits.bit_length(), block
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
    if num_tokens == 0:
        return output, permuted_probs
    if row_id_map.is_contiguous() and _supports_row_tiles(inp, num_tokens, hidden_size):
        source_rows = row_id_map
        if is_forward:
            from .triton_permutation_kernels import _invert_chunk_row_map

            source_rows = torch.empty_like(row_id_map)
            index_block = 256
            _invert_chunk_row_map[_token_grid(triton.cdiv(num_tokens, index_block), 1)](
                row_id_map,
                source_rows,
                num_tokens,
                ROWS=index_block,
            )
        _launch_row_gather(
            inp,
            source_rows,
            None,
            probs,
            output,
            permuted_probs,
            num_tokens,
            hidden_size,
            mask_map=False,
        )
        return output, permuted_probs

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
