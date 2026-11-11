# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.


"""Ascend Triton kernels for the TE permutation plugin."""

import triton
import triton.language as tl

from transformer_engine.common.triton.permutation import _permutation_autotune_configs


# The launcher caps the physical grid. Each program processes every remaining
# logical tile at its grid stride.
@triton.autotune(configs=_permutation_autotune_configs(), key=["hidden_size"])
@triton.jit(do_not_specialize=["npu_token_count"])
def _permute_kernel_npu(
    # input pointers
    input_ptr,
    row_id_map_ptr,
    probs_ptr,
    scale_ptr,
    permuted_scale_ptr,
    pad_offsets_ptr,
    # Pre-allocated output buffers for JAX input_output_aliases.
    # These are aliased to output_ptr/permuted_probs_ptr in JAX, so they point to the same memory.
    # In PyTorch, pass the same tensors as output_ptr/permuted_probs_ptr.
    output_buf_ptr,  # pylint: disable=unused-argument
    permuted_probs_buf_ptr,  # pylint: disable=unused-argument
    # sizes
    scale_hidden_dim,
    num_tokens,  # pylint: disable=unused-argument
    num_out_tokens,  # pylint: disable=unused-argument
    # strides
    stride_row_id_map_token,
    stride_row_id_map_expert,
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_probs_token,
    stride_probs_expert,
    stride_scale_token,
    stride_scale_hidden,
    stride_permuted_probs_token,
    stride_permuted_scale_token,
    stride_permuted_scale_hidden,
    # output pointers
    output_ptr,
    permuted_probs_ptr,
    # metas
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    PERMUTE_PROBS: tl.constexpr,
    PERMUTE_SCALE: tl.constexpr,
    FUSION_PAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    npu_token_count,
):
    # Note: When FUSION_PAD=True, output buffers should be pre-zeroed by the caller
    # to ensure padding positions contain zeros.
    # PyTorch: Use torch.zeros() for output buffer allocation
    # JAX: Pre-zeroed buffers should be passed (when input_output_aliases works)

    hidden_tiles = tl.cdiv(hidden_size, BLOCK_SIZE)
    task_count = npu_token_count * hidden_tiles
    for task in range(tl.program_id(0), task_count, tl.num_programs(0)):
        pid_t = task % npu_token_count
        pid_h = task // npu_token_count
        expert_idx = 0
        cur_off = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cur_off < hidden_size

        src_row = pid_t.to(tl.int64)
        input_off = src_row * stride_input_token + cur_off * stride_input_hidden
        inp = tl.load(input_ptr + input_off, mask=mask)
        if PERMUTE_SCALE:
            mask_scale = cur_off < scale_hidden_dim
            scale_off = pid_t * stride_scale_token + cur_off * stride_scale_hidden
            scale = tl.load(scale_ptr + scale_off, mask=mask_scale)
        n_routed = tl.load(
            row_id_map_ptr
            + pid_t * stride_row_id_map_token
            + num_experts * 2 * stride_row_id_map_expert
        )
        for idx in tl.range(n_routed):
            dst_row = tl.load(
                row_id_map_ptr + pid_t * stride_row_id_map_token + idx * stride_row_id_map_expert
            ).to(tl.int64)
            if FUSION_PAD or PERMUTE_PROBS:
                expert_idx = tl.load(
                    row_id_map_ptr
                    + pid_t * stride_row_id_map_token
                    + (num_experts + idx) * stride_row_id_map_expert
                )
            if FUSION_PAD:
                pad_off = tl.load(pad_offsets_ptr + expert_idx)
                dst_row = dst_row + pad_off
            output_off = dst_row * stride_output_token + cur_off * stride_output_hidden
            if PERMUTE_SCALE:
                permuted_scale_off = (
                    dst_row * stride_permuted_scale_token + cur_off * stride_permuted_scale_hidden
                )
                tl.store(permuted_scale_ptr + permuted_scale_off, scale, mask=mask_scale)
            if PERMUTE_PROBS:
                prob_off = pid_t * stride_probs_token + expert_idx * stride_probs_expert
                prob = tl.load(probs_ptr + prob_off)
                if pid_h == 0:
                    permuted_prob_off = dst_row * stride_permuted_probs_token
                    tl.store(permuted_probs_ptr + permuted_prob_off, prob)
                if prob == 0.0:
                    # for routing_map padding
                    # dst_row != -1 and prob == 0.0 means that this slot is padded
                    tl.store(output_ptr + output_off, 0.0, mask=mask)
                else:
                    tl.store(output_ptr + output_off, inp, mask=mask)
            else:
                tl.store(output_ptr + output_off, inp, mask=mask)


@triton.autotune(configs=_permutation_autotune_configs(), key=["hidden_size"])
@triton.jit(do_not_specialize=["npu_token_count"])
def _unpermute_kernel_npu(
    # input pointers
    input_ptr,
    row_id_map_ptr,
    merging_probs_ptr,
    permuted_probs_ptr,
    pad_offsets_ptr,
    # Dummy parameters for JAX input_output_aliases compatibility (matches _permute_kernel signature pattern)
    # These are unused in the unpermute kernel but maintain consistency with the permute kernel.
    output_buf_ptr,  # pylint: disable=unused-argument
    unpermuted_probs_buf_ptr,  # pylint: disable=unused-argument
    # strides
    stride_row_id_map_token,
    stride_row_id_map_expert,
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_merging_probs_token,
    stride_merging_probs_expert,
    stride_permuted_probs_token,
    stride_unpermuted_probs_token,
    stride_unpermuted_probs_expert,
    # output pointers
    output_ptr,
    unpermuted_probs_ptr,
    # metas
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    PROBS_LOAD_WIDTH: tl.constexpr,
    WITH_MERGING_PROBS: tl.constexpr,
    PERMUTE_PROBS: tl.constexpr,
    FUSION_UNPAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    npu_token_count,
):
    data_type = input_ptr.dtype.element_ty
    compute_type = tl.float32

    hidden_tiles = tl.cdiv(hidden_size, BLOCK_SIZE)
    task_count = npu_token_count * hidden_tiles
    for task in range(tl.program_id(0), task_count, tl.num_programs(0)):
        pid_t = task % npu_token_count
        pid_h = task // npu_token_count
        expert_idx = 0
        current_offset = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = current_offset < hidden_size
        if PERMUTE_PROBS:
            # write 0.0 to probs_grad that are not routed
            if pid_h == 0:
                map_load_off = tl.arange(0, PROBS_LOAD_WIDTH)
                unpermuted_prob_off = (
                    pid_t * stride_unpermuted_probs_token
                    + stride_unpermuted_probs_expert * map_load_off
                )
                tl.store(
                    unpermuted_probs_ptr + unpermuted_prob_off, 0.0, mask=map_load_off < num_experts
                )
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=compute_type)
        n_routed = tl.load(
            row_id_map_ptr
            + pid_t * stride_row_id_map_token
            + num_experts * 2 * stride_row_id_map_expert
        )
        for idx in tl.range(n_routed):
            src_row = tl.load(
                row_id_map_ptr + pid_t * stride_row_id_map_token + idx * stride_row_id_map_expert
            ).to(tl.int64)
            if FUSION_UNPAD or WITH_MERGING_PROBS:
                expert_idx = tl.load(
                    row_id_map_ptr
                    + pid_t * stride_row_id_map_token
                    + (num_experts + idx) * stride_row_id_map_expert
                )
            if FUSION_UNPAD:
                pad_off = tl.load(pad_offsets_ptr + expert_idx)
                src_row = src_row + pad_off
            input_off = src_row * stride_input_token + current_offset * stride_input_hidden
            inp = tl.load(input_ptr + input_off, mask=mask)
            inp = inp.to(compute_type)
            if WITH_MERGING_PROBS:
                merging_prob_off = (
                    pid_t * stride_merging_probs_token + expert_idx * stride_merging_probs_expert
                )
                merging_prob = tl.load(merging_probs_ptr + merging_prob_off).to(compute_type)
                inp *= merging_prob
            accumulator += inp
            if PERMUTE_PROBS:
                if pid_h == 0:
                    expert_idx = tl.load(
                        row_id_map_ptr
                        + pid_t * stride_row_id_map_token
                        + (num_experts + idx) * stride_row_id_map_expert
                    )
                    unpermuted_prob_off = (
                        pid_t * stride_unpermuted_probs_token
                        + expert_idx * stride_unpermuted_probs_expert
                    )
                    permuted_prob_off = src_row * stride_permuted_probs_token
                    prob = tl.load(permuted_probs_ptr + permuted_prob_off)
                    tl.store(unpermuted_probs_ptr + unpermuted_prob_off, prob)
        accumulator = accumulator.to(data_type)
        dst_row = pid_t.to(tl.int64)
        output_off = dst_row * stride_output_token + current_offset * stride_output_hidden
        tl.store(output_ptr + output_off, accumulator, mask=mask)


@triton.jit
def _chunk_sort_offsets(
    split_sizes, sorted_indices, ends, output_starts, NUM_SPLITS: tl.constexpr, WIDTH: tl.constexpr
):
    slots = tl.arange(0, WIDTH)
    valid = slots < NUM_SPLITS
    sizes = tl.load(split_sizes + slots, mask=valid, other=0).to(tl.int32)
    ends_in_input_order = tl.cumsum(sizes)
    chunks = tl.load(sorted_indices + slots, mask=valid, other=0).to(tl.int32)
    reordered_sizes = tl.load(split_sizes + chunks, mask=valid, other=0).to(tl.int32)
    starts_in_output_order = tl.cumsum(reordered_sizes) - reordered_sizes
    tl.store(ends + slots, ends_in_input_order, mask=valid)
    tl.store(output_starts + chunks, starts_in_output_order, mask=valid)


@triton.jit(do_not_specialize=["num_tokens"])
def _chunk_sort_expand(
    ends,
    output_starts,
    mapping,
    num_tokens,
    NUM_SPLITS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    total = tl.load(ends + NUM_SPLITS - 1)
    offsets = tl.arange(0, BLOCK)
    for tile in range(tl.program_id(0), tl.cdiv(num_tokens, BLOCK), tl.num_programs(0)):
        rows = tile * BLOCK + offsets
        valid = (rows < total) & (rows < num_tokens)
        lo = tl.full((BLOCK,), 0, tl.int32)
        hi = tl.full((BLOCK,), NUM_SPLITS, tl.int32)
        # Upper bound on the inclusive ends skips all empty chunks.
        for _ in tl.range(0, SEARCH_STEPS):
            mid = (lo + hi) // 2
            advance = tl.load(ends + mid, mask=valid & (mid < NUM_SPLITS), other=2147483647) <= rows
            lo = tl.where(advance, mid + 1, lo)
            hi = tl.where(advance, hi, mid)
        found = valid & (lo < NUM_SPLITS)
        input_start = tl.load(ends + lo - 1, mask=found & (lo > 0), other=0)
        output_start = tl.load(output_starts + lo, mask=found, other=0)
        dst_rows = rows - input_start + output_start
        tl.store(mapping + rows, dst_rows, mask=rows < num_tokens)


@triton.autotune(configs=_permutation_autotune_configs(), key=["hidden_size"])
@triton.jit(do_not_specialize=["npu_token_count"])
def _sort_chunks_by_map_kernel_npu(
    # input pointers
    input_ptr,
    row_id_map_ptr,
    probs_ptr,
    # Pre-allocated output buffer for JAX input_output_aliases.
    # Aliased to output_ptr in JAX so they point to the same memory.
    # In PyTorch, pass the same tensor as output_ptr.
    output_buf_ptr,  # pylint: disable=unused-argument
    # strides
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_probs_token,
    stride_permuted_probs_token,
    # output pointers
    output_ptr,
    permuted_probs_ptr,
    # metas
    hidden_size: tl.constexpr,
    PERMUTE_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FORWARD: tl.constexpr,
    npu_token_count,
):
    hidden_tiles = tl.cdiv(hidden_size, BLOCK_SIZE)
    task_count = npu_token_count * hidden_tiles
    for task in range(tl.program_id(0), task_count, tl.num_programs(0)):
        pid_t = task % npu_token_count
        pid_h = task // npu_token_count
        if FORWARD:
            src_row = pid_t.to(tl.int64)
            dst_row = tl.load(row_id_map_ptr + pid_t).to(tl.int64)
        else:
            src_row = tl.load(row_id_map_ptr + pid_t).to(tl.int64)
            dst_row = pid_t.to(tl.int64)
        current_offset = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = current_offset < hidden_size
        input_offsets = src_row * stride_input_token + current_offset * stride_input_hidden
        output_offsets = dst_row * stride_output_token + current_offset * stride_output_hidden
        inp = tl.load(input_ptr + input_offsets, mask=mask)
        tl.store(output_ptr + output_offsets, inp, mask=mask)
        if PERMUTE_PROBS:
            if pid_h == 0:
                prob_off = src_row * stride_probs_token
                prob = tl.load(probs_ptr + prob_off)
                permuted_prob_off = dst_row * stride_permuted_probs_token
                tl.store(permuted_probs_ptr + permuted_prob_off, prob)


@triton.jit(do_not_specialize=["num_tokens", "num_out_tokens"])
def _pack_mask_map_routes(
    row_map,
    source_rows,
    expert_rows,
    num_tokens,
    num_out_tokens,
    stride_token,
    stride_expert,
    EXPERTS: tl.constexpr,
    WITH_PROBS: tl.constexpr,
    TOKENS: tl.constexpr,
):
    lanes = tl.arange(0, TOKENS)
    for tile in range(tl.program_id(0), tl.cdiv(num_tokens, TOKENS), tl.num_programs(0)):
        rows = tile * TOKENS + lanes
        row_base = row_map + rows.to(tl.int64) * stride_token
        counts = tl.load(row_base + 2 * EXPERTS * stride_expert, rows < num_tokens, other=0)
        for route in tl.range(tl.max(counts, 0)):
            valid = (rows < num_tokens) & (route < counts)
            dst = tl.load(row_base + route * stride_expert, valid, other=-1)
            valid = valid & (dst >= 0) & (dst < num_out_tokens)
            tl.store(source_rows + dst, rows, valid)
            if WITH_PROBS:
                expert = tl.load(row_base + (EXPERTS + route) * stride_expert, valid, other=0)
                tl.store(expert_rows + dst, expert, valid)


@triton.jit(do_not_specialize=["num_rows"])
def _invert_chunk_row_map(row_map, inverse, num_rows, ROWS: tl.constexpr):
    lanes = tl.arange(0, ROWS)
    for tile in range(tl.program_id(0), tl.cdiv(num_rows, ROWS), tl.num_programs(0)):
        rows = tile * ROWS + lanes
        valid = rows < num_rows
        dst = tl.load(row_map + rows, valid, other=0)
        tl.store(inverse + dst, rows, valid)


@triton.jit(do_not_specialize=["num_rows"])
def _gather_permutation_rows(
    inp,
    source_rows,
    expert_rows,
    probs,
    output,
    output_probs,
    num_rows,
    stride_input_row,
    stride_input_hidden,
    stride_output_row,
    stride_output_hidden,
    stride_probs_row,
    stride_probs_expert,
    stride_output_probs,
    HIDDEN: tl.constexpr,
    WITH_PROBS: tl.constexpr,
    MASK_MAP: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    hidden_tiles: tl.constexpr = tl.cdiv(HIDDEN, COLS)
    tasks = tl.cdiv(num_rows, ROWS) * hidden_tiles
    for task in range(tl.program_id(0), tasks, tl.num_programs(0)):
        rows = (task // hidden_tiles) * ROWS + tl.arange(0, ROWS)
        hidden = (task % hidden_tiles) * COLS + tl.arange(0, COLS)
        valid = rows < num_rows
        src = tl.load(source_rows + rows, valid, other=0).to(tl.int64)
        if MASK_MAP:
            valid = valid & (src >= 0)
            # Keep masked addresses in bounds for NPU vectorized loads.
            src = tl.where(valid, src, 0)
        mask = valid[:, None] & (hidden[None, :] < HIDDEN)
        values = tl.load(
            inp + src[:, None] * stride_input_row + hidden[None, :] * stride_input_hidden,
            mask,
            other=0,
        )
        if WITH_PROBS:
            if MASK_MAP:
                expert = tl.load(expert_rows + rows, valid, other=0)
                prob = tl.load(
                    probs + src * stride_probs_row + expert * stride_probs_expert,
                    valid,
                    other=0,
                )
                zero = tl.full((ROWS, COLS), 0, values.dtype)
                values = tl.where((prob == 0)[:, None], zero, values)
            if task % hidden_tiles == 0:
                if not MASK_MAP:
                    prob = tl.load(probs + src * stride_probs_row, valid, other=0)
                tl.store(output_probs + rows * stride_output_probs, prob, valid)
        tl.store(
            output
            + rows.to(tl.int64)[:, None] * stride_output_row
            + hidden[None, :] * stride_output_hidden,
            values,
            mask,
        )


@triton.jit(do_not_specialize=["num_tokens"])
def _combine_mask_map_rows(
    inp,
    row_map,
    weights,
    output,
    num_tokens,
    stride_map_token,
    stride_map_expert,
    stride_input_row,
    stride_input_hidden,
    stride_output_row,
    stride_output_hidden,
    stride_weight_token,
    stride_weight_expert,
    EXPERTS: tl.constexpr,
    HIDDEN: tl.constexpr,
    WITH_WEIGHTS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    hidden_tiles: tl.constexpr = tl.cdiv(HIDDEN, COLS)
    tasks = tl.cdiv(num_tokens, ROWS) * hidden_tiles
    for task in range(tl.program_id(0), tasks, tl.num_programs(0)):
        rows = (task // hidden_tiles) * ROWS + tl.arange(0, ROWS)
        hidden = (task % hidden_tiles) * COLS + tl.arange(0, COLS)
        valid = rows < num_tokens
        row_base = row_map + rows.to(tl.int64) * stride_map_token
        counts = tl.load(row_base + 2 * EXPERTS * stride_map_expert, valid, other=0)
        hidden_mask = hidden[None, :] < HIDDEN
        acc = tl.full((ROWS, COLS), 0, tl.float32)
        for route in tl.range(tl.max(counts, 0)):
            active = valid & (route < counts)
            src = tl.load(row_base + route * stride_map_expert, active, other=0).to(tl.int64)
            values = tl.load(
                inp + src[:, None] * stride_input_row + hidden[None, :] * stride_input_hidden,
                active[:, None] & hidden_mask,
                other=0,
            ).to(tl.float32)
            if WITH_WEIGHTS:
                expert = tl.load(row_base + (EXPERTS + route) * stride_map_expert, active, other=0)
                weight = tl.load(
                    weights
                    + rows.to(tl.int64) * stride_weight_token
                    + expert * stride_weight_expert,
                    active,
                    other=0,
                ).to(tl.float32)
                values = values * weight[:, None]
            # Preserve route order and leave lanes with fewer routes unchanged.
            acc = tl.where(active[:, None], acc + values, acc)
        tl.store(
            output
            + rows.to(tl.int64)[:, None] * stride_output_row
            + hidden[None, :] * stride_output_hidden,
            acc.to(inp.dtype.element_ty),
            valid[:, None] & hidden_mask,
        )


@triton.jit(do_not_specialize=["num_tokens"])
def _row_map_scan_blocks(
    routing_map,
    local_prefix,
    block_counts,
    num_tokens,
    stride_token,
    stride_expert,
    EXPERTS: tl.constexpr,
    TOKENS: tl.constexpr,
    COLUMNS: tl.constexpr,
):
    """Scan disjoint token blocks, retaining their per-expert totals."""
    token_block = tl.program_id(0)
    expert = tl.program_id(1) * COLUMNS + tl.arange(0, COLUMNS)
    token = token_block * TOKENS + tl.arange(0, TOKENS)
    valid = (token[:, None] < num_tokens) & (expert[None, :] < EXPERTS)
    routes = tl.load(
        routing_map + token.to(tl.int64)[:, None] * stride_token + expert[None, :] * stride_expert,
        valid,
        other=0,
    ).to(tl.int32)
    prefix = tl.cumsum(routes, axis=0)
    tl.store(
        local_prefix + token.to(tl.int64)[:, None] * EXPERTS + expert[None, :],
        prefix,
        valid,
    )
    tl.store(
        block_counts + token_block * EXPERTS + expert,
        tl.sum(routes, axis=0),
        expert < EXPERTS,
    )


@triton.jit
def _row_map_block_offsets(
    block_counts,
    block_offsets,
    expert_counts,
    BLOCKS: tl.constexpr,
    EXPERTS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """Scan block totals independently for each expert."""
    expert = tl.program_id(0)
    block = tl.arange(0, WIDTH)
    counts = tl.load(block_counts + block * EXPERTS + expert, block < BLOCKS, other=0)
    prefix = tl.cumsum(counts, axis=0)
    tl.store(block_offsets + block * EXPERTS + expert, prefix - counts, block < BLOCKS)
    tl.store(expert_counts + expert, tl.sum(counts, axis=0))


@triton.jit(do_not_specialize=["num_tokens"])
def _row_map_write(
    routing_map,
    local_prefix,
    block_offsets,
    expert_counts,
    row_map,
    num_tokens,
    stride_token,
    stride_expert,
    EXPERTS: tl.constexpr,
    WIDTH: tl.constexpr,
    TOKENS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    """Combine prefixes and compact valid routes in descending expert order."""
    expert = tl.arange(0, WIDTH)
    expert_valid = expert < EXPERTS
    counts = tl.load(expert_counts + expert, expert_valid, other=0)
    expert_start = tl.cumsum(counts, axis=0) - counts
    for tile in range(tl.program_id(0), tl.cdiv(num_tokens, TOKENS), tl.num_programs(0)):
        token = tile * TOKENS + tl.arange(0, TOKENS)
        valid = (token[:, None] < num_tokens) & expert_valid[None, :]
        routes = tl.load(
            routing_map
            + token.to(tl.int64)[:, None] * stride_token
            + expert[None, :] * stride_expert,
            valid,
            other=0,
        ).to(tl.int32)
        route_counts = tl.sum(routes, axis=1)
        slot = route_counts[:, None] - tl.cumsum(routes, axis=1)
        local = tl.load(
            local_prefix + token.to(tl.int64)[:, None] * EXPERTS + expert[None, :],
            valid,
            other=0,
        )
        offset = tl.load(
            block_offsets + (token // BLOCK_TOKENS)[:, None] * EXPERTS + expert[None, :],
            valid,
            other=0,
        )
        position = expert_start[None, :] + offset + local - 1
        row_base = row_map + token.to(tl.int64)[:, None] * (2 * EXPERTS + 1)
        active = valid & (routes != 0)
        tl.store(row_base + slot, position, active)
        tl.store(
            row_base + EXPERTS + slot, tl.broadcast_to(expert[None, :], (TOKENS, WIDTH)), active
        )
        tl.store(
            row_map + token.to(tl.int64) * (2 * EXPERTS + 1) + 2 * EXPERTS,
            route_counts,
            token < num_tokens,
        )
