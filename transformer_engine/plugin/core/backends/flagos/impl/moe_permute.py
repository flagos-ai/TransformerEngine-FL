import torch
from typing import Tuple, List, Optional
import triton
import triton.language as tl


@triton.jit
def moe_unpermute_kernel_triton(
    input_ptr,  # [total_expert_tokens, num_cols]
    output_ptr,  # [num_tokens, num_cols]
    row_id_map_ptr,  # [topK, num_rows] or [topK * num_rows]
    prob_ptr,  # [num_rows, topK]
    num_rows: tl.constexpr,  # num_tokens
    num_cols: tl.constexpr,
    topK: tl.constexpr,
    HAS_PROB: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    #  blockIdx.x
    source_token = tl.program_id(0)

    # Traverse along the hidden dimention
    for col_offset in range(0, num_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        col_mask = cols < num_cols

        # frag_sum
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for k in range(topK):
            #  k * num_rows + source_token
            map_idx = k * num_rows + source_token

            source_row = tl.load(row_id_map_ptr + map_idx)

            # source_row == -1
            if source_row != -1:
                in_ptrs = input_ptr + source_row * num_cols + cols
                val = tl.load(in_ptrs, mask=col_mask, other=0.0)

                if HAS_PROB:
                    #  source_token * topK + k
                    prob_val = tl.load(prob_ptr + source_token * topK + k)
                    val = val * prob_val

                acc += val

        #   unpermuted_output
        out_ptrs = output_ptr + source_token * num_cols + cols

        # store
        tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=col_mask)


@triton.jit
def _kernel_moe_permute(
    input_bwd_ptr,
    input_fwd_ptr,
    act_grad_ptr,
    prob_ptr,
    prob_grad_ptr,
    row_id_map_ptr,
    num_rows,
    num_cols,
    topk,
    BLOCK_SIZE: tl.constexpr,
    HAS_PROB: tl.constexpr,
    TOPK_P2: tl.constexpr,  # TopK padded to power of 2
):
    pid = tl.program_id(0)

    source_row_start_ptr = input_bwd_ptr + pid * num_cols

    # Accumulator for prob_grad: Shape [TOPK_P2]
    # Initialize with zeros
    prob_grad_acc = tl.zeros((TOPK_P2,), dtype=tl.float32)

    for col_offset in range(0, num_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < num_cols

        # Load source data (Reused across K)
        # [BLOCK_SIZE]
        source_vec = tl.load(source_row_start_ptr + cols, mask=mask, other=0.0)

        # Loop over experts
        for k in range(topk):
            # Map index calculation: row_id_map is [TopK, Num_Rows]
            # Stride is num_rows
            map_idx = k * num_rows + pid
            dest_row = tl.load(row_id_map_ptr + map_idx)

            if dest_row != -1:
                dest_ptr_base = act_grad_ptr + dest_row * num_cols
                dest_ptr = dest_ptr_base + cols

                if HAS_PROB:
                    # Load prob: [N, TopK] -> ptr + pid*topk + k
                    p = tl.load(prob_ptr + pid * topk + k)

                    # 1. Compute act_grad
                    val = source_vec * p
                    tl.store(dest_ptr, val, mask=mask)

                    # 2. Compute prob_grad (accumulate dot product)
                    fwd_vec = tl.load(
                        input_fwd_ptr + dest_row * num_cols + cols, mask=mask, other=0.0
                    )
                    partial_dot = tl.sum(source_vec * fwd_vec)

                    # Update accumulator at index k
                    # We use a mask to update only the k-th element of the vector
                    k_mask = tl.arange(0, TOPK_P2) == k
                    prob_grad_acc = tl.where(k_mask, prob_grad_acc + partial_dot, prob_grad_acc)
                else:
                    tl.store(dest_ptr, source_vec, mask=mask)

    if HAS_PROB:
        # Store prob_grad
        # prob_grad_ptr is [N, TopK]
        out_ptr = prob_grad_ptr + pid * topk + tl.arange(0, TOPK_P2)
        mask_store = tl.arange(0, TOPK_P2) < topk
        tl.store(out_ptr, prob_grad_acc, mask=mask_store)


@triton.jit
def moe_permute_row_map_kernel(
    sorted_row_id_ptr,  # const int *sorted_row_id
    row_id_map_ptr,  # int *row_id_map
    num_rows,  # const int num_rows
    topk,  # const int topK
    num_out_tokens,  # const int num_out_tokens
    n_elements,  # (num_rows * topk)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    source_row = tl.load(sorted_row_id_ptr + offsets, mask=mask, other=0)

    source_token_id = source_row // topk
    source_topK_id = source_row % topk

    dest_offset = source_topK_id * num_rows + source_token_id
    dest_ptr = row_id_map_ptr + dest_offset

    value_to_write = tl.where(offsets < num_out_tokens, offsets, -1)

    value_to_write = value_to_write.to(tl.int32)

    tl.store(dest_ptr, value_to_write, mask=mask)


def moe_permute_row_map(sorted_row_id, num_rows, topk, num_out_tokens):
    """ """
    sorted_row_id = sorted_row_id.contiguous()

    row_id_map = torch.empty(topk * num_rows, device=sorted_row_id.device, dtype=torch.int32)

    n_elements = num_rows * topk

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    moe_permute_row_map_kernel[grid](
        sorted_row_id, row_id_map, num_rows, topk, num_out_tokens, n_elements, BLOCK_SIZE=1024
    )

    return row_id_map


def moe_permute_fwd_fl(
    inp: torch.Tensor,
    dtype: torch.dtype,
    indices: torch.Tensor,
    num_out_tokens: int,
    workspace: Optional[List[torch.Tensor]] = None,
    max_token_num: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    num_tokens, num_cols = inp.shape
    topk = indices.shape[1]
    device = inp.device

    num_out_tokens = num_out_tokens if num_out_tokens > 0 else num_tokens * topk

    assert inp.is_cuda, "compute needs CUDA."

    # nvte_device_radix_sort_pairs
    source_row_ids = torch.arange(num_tokens * topk, dtype=torch.int32, device=device)
    keys_flat = indices.view(-1).to(torch.int32)
    sorted_expert_indices, permutation_idx = torch.sort(keys_flat, stable=True)
    sorted_row_id = source_row_ids[permutation_idx]

    # row_id_map
    row_id_map = moe_permute_row_map(sorted_row_id, num_tokens, topk, num_out_tokens)

    permuted_tokens = torch.empty((num_out_tokens, num_cols), dtype=inp.dtype, device=device)

    grid = (num_tokens,)

    _kernel_moe_permute[grid](
        input_bwd_ptr=inp,
        input_fwd_ptr=None,  #
        act_grad_ptr=permuted_tokens,
        prob_ptr=None,  #
        prob_grad_ptr=None,  #
        row_id_map_ptr=row_id_map,
        num_rows=num_tokens,
        num_cols=num_cols,
        topk=topk,
        BLOCK_SIZE=1024,
        HAS_PROB=False,
        TOPK_P2=triton.next_power_of_2(topk),
    )

    return permuted_tokens, row_id_map, (workspace if workspace is not None else [])


def moe_unpermute_bwd_fl(
    input_bwd: torch.Tensor,
    input_fwd: torch.Tensor,
    dtype: torch.dtype,
    row_id_map: torch.Tensor,
    prob: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if prob.numel() > 0:
        topk = prob.size(1)
        num_tokens = prob.size(0)
    else:
        topk = 1
        num_tokens = row_id_map.size(0)
    num_cols = input_bwd.size(1)
    device = input_bwd.device

    act_grad = torch.empty((input_fwd.size(0), num_cols), dtype=input_bwd.dtype, device=device)

    prob_grad = torch.zeros((num_tokens, topk), dtype=torch.float32, device=device)

    grid = (num_tokens,)

    if prob.numel() == 0:
        _kernel_moe_permute[grid](
            input_bwd_ptr=input_bwd,
            input_fwd_ptr=input_fwd,
            act_grad_ptr=act_grad,
            prob_ptr=None,
            prob_grad_ptr=None,
            row_id_map_ptr=row_id_map,
            num_rows=num_tokens,
            num_cols=num_cols,
            topk=topk,
            BLOCK_SIZE=1024,
            HAS_PROB=False,
            TOPK_P2=triton.next_power_of_2(topk),
        )
    else:
        _kernel_moe_permute[grid](
            input_bwd_ptr=input_bwd,
            input_fwd_ptr=input_fwd,
            act_grad_ptr=act_grad,
            prob_ptr=prob,
            prob_grad_ptr=prob_grad,
            row_id_map_ptr=row_id_map,
            num_rows=num_tokens,
            num_cols=num_cols,
            topk=topk,
            BLOCK_SIZE=1024,
            HAS_PROB=True,
            TOPK_P2=triton.next_power_of_2(topk),
        )

    return act_grad, prob_grad


def moe_permute_bwd_fl(
    input_bwd: torch.Tensor,
    dtype: torch.dtype,
    row_id_map: torch.Tensor,
    prob: torch.Tensor = None,
    num_tokens: int = None,
    topK: int = None,
) -> torch.Tensor:
    return moe_unpermute_fwd_fl(input_bwd, dtype, row_id_map, prob, num_tokens, topK)


def moe_unpermute_fwd_fl(
    input_fwd: torch.Tensor,
    dtype: torch.dtype,
    row_id_map: torch.Tensor,
    prob: torch.Tensor = None,
    num_tokens: int = None,
    topK: int = None,
) -> torch.Tensor:
    num_cols = input_fwd.size(1)
    device = input_fwd.device
    unpermuted_output = torch.empty((num_tokens, num_cols), dtype=input_fwd.dtype, device=device)

    BLOCK_SIZE = triton.next_power_of_2(num_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    grid = (num_tokens,)

    if prob.numel() == 0:
        moe_unpermute_kernel_triton[grid](
            input_fwd,
            unpermuted_output,
            row_id_map,
            None,
            num_rows=num_tokens,
            num_cols=num_cols,
            topK=topK,
            HAS_PROB=False,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        moe_unpermute_kernel_triton[grid](
            input_fwd,
            unpermuted_output,
            row_id_map,
            prob,
            num_rows=num_tokens,
            num_cols=num_cols,
            topK=topK,
            HAS_PROB=True,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return unpermuted_output
