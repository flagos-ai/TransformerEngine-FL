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


@triton.jit(do_not_specialize=["npu_token_count"])
def _make_chunk_sort_map_kernel_npu(
    # pointers
    split_sizes_ptr,
    sorted_indices_ptr,
    dst_rows_ptr,
    # sizes
    num_splits: tl.constexpr,
    # metas
    IDX_LOAD_WIDTH: tl.constexpr,
    npu_token_count,
):
    task_count = npu_token_count
    for pid in range(tl.program_id(0), task_count, tl.num_programs(0)):

        load_split_offset = tl.arange(0, IDX_LOAD_WIDTH)
        sorted_indices = tl.load(
            sorted_indices_ptr + load_split_offset, mask=load_split_offset < num_splits
        )

        # get chunk idx of the current token in the input tensor
        input_split_sizes = tl.load(
            split_sizes_ptr + load_split_offset, mask=load_split_offset < num_splits, other=0
        ).to(tl.int32)
        input_split_sizes_cumsum = tl.cumsum(input_split_sizes)

        # Compute total valid tokens and skip phantom/padding tokens.
        # When the input buffer is larger than sum(split_sizes), tokens beyond
        # the valid range should map to themselves (identity mapping) to avoid
        # corrupting valid output positions.
        total_valid_tokens = tl.sum(input_split_sizes)

        input_split_sizes_mask = tl.where(input_split_sizes_cumsum <= pid, 1, 0)
        input_chunk_idx = tl.sum(input_split_sizes_mask)
        input_split_sizes_presum = tl.sum(input_split_sizes * input_split_sizes_mask)
        in_chunk_offset = pid - input_split_sizes_presum

        # get chunk idx of the current token in the output tensor
        output_chunk_mask = tl.where(sorted_indices == input_chunk_idx, 1, 0)
        output_chunk_idx = tl.argmax(output_chunk_mask, axis=-1)

        # make row_id_map
        output_split_sizes = tl.load(
            split_sizes_ptr + sorted_indices, mask=load_split_offset < num_splits
        ).to(tl.int32)
        output_pre_split_sizes = tl.where(
            load_split_offset < output_chunk_idx, output_split_sizes, 0
        )
        dst_row = tl.sum(output_pre_split_sizes) + in_chunk_offset

        # For tokens beyond the valid range (pid >= total_valid_tokens),
        # use identity mapping to avoid corrupting valid data
        dst_row = tl.where(pid < total_valid_tokens, dst_row, pid)

        tl.store(dst_rows_ptr + pid, dst_row)


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
