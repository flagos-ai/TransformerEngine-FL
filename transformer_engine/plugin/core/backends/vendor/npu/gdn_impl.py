# Copyright (c) 2025, BAAI. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""GDN implementation with complete forward and backward passes using fla_npu operators.

This module implements the complete Gated Delta Net algorithm using fine-grained
AscendC and Triton operators from fla_npu.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch

from fla_npu.ops.ascendc import (
    chunk_bwd_dqkwg as ascendc_chunk_bwd_dqkwg,
    chunk_bwd_dv_local as ascendc_chunk_bwd_dv_local,
    chunk_fwd_o as ascendc_chunk_fwd_o,
    chunk_gated_delta_rule_bwd_dhu as ascendc_chunk_gated_delta_rule_bwd_dhu,
    chunk_gated_delta_rule_fwd_h as ascendc_chunk_gated_delta_rule_fwd_h,
    prepare_wy_repr_bwd_da as ascendc_prepare_wy_repr_bwd_da,
    prepare_wy_repr_bwd_full as ascendc_prepare_wy_repr_bwd_full,
    recompute_w_u_fwd as ascendc_recompute_w_u_fwd,
    solve_tri as ascendc_solve_tri,
)
from fla_npu.ops.triton import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    chunk_local_cumsum,
    chunk_scaled_dot_kkt_fwd,
    input_guard,
    l2norm_bwd,
    l2norm_fwd,
    solve_tril_npu,
)


def _as_int_list(value: Optional[torch.Tensor]) -> Optional[list]:
    """Convert tensor to list of integers."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().flatten().tolist()]
    return [int(x) for x in value]


def _chunk_list(chunk_indices_list: Optional[Dict], chunk_size: int) -> Optional[list]:
    """Get chunk indices list for specific chunk size."""
    if chunk_indices_list is None:
        return None
    return chunk_indices_list.get(str(chunk_size))


def _chunk_tensor(chunk_indices: Optional[Dict], chunk_size: int) -> Optional[torch.Tensor]:
    """Get chunk indices tensor for specific chunk size."""
    if chunk_indices is None:
        return None
    return chunk_indices.get(str(chunk_size))


def solve_tri_ascendc(
    A: torch.Tensor,
    cu_seqlens: Optional[list] = None,
    chunk_indices: Optional[list] = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """Solve triangular system using AscendC."""
    A_in = A.to(output_dtype).contiguous()

    if cu_seqlens is None:
        # Standard batch mode
        return ascendc_solve_tri(A_in, layout="bsnd")

    # Varlen mode
    if A_in.ndim != 4 or A_in.shape[0] != 1:
        raise ValueError(
            f"solve_tri varlen expects A with shape [1, T, H, BT], got {tuple(A_in.shape)}"
        )
    if chunk_indices is None:
        raise ValueError("solve_tri varlen requires chunk_indices")

    out = ascendc_solve_tri(
        A_in.squeeze(0),
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        layout="tnd",
    )
    return out.unsqueeze(0)


def solve_tri(
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    chunk_indices: Optional[Dict],
    cu_seqlens_list: Optional[list],
    chunk_indices_list: Optional[list],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Solve triangular system with backend selection."""
    backend = os.getenv("FLA_NPU_GDN_SOLVE_TRI_BACKEND", "ascendc").strip().lower()

    if backend == "ascendc":
        return solve_tri_ascendc(
            A,
            cu_seqlens=cu_seqlens_list,
            chunk_indices=chunk_indices_list,
            output_dtype=output_dtype,
        )
    elif backend == "triton":
        return solve_tril_npu(
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices_out=chunk_indices,
            output_dtype=output_dtype,
        )
    else:
        raise ValueError(
            f"FLA_NPU_GDN_SOLVE_TRI_BACKEND must be 'ascendc' or 'triton', got {backend!r}"
        )


def recompute_w_u(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
    cu_seqlens: Optional[list],
    chunk_indices: Optional[list],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recompute W and U from K, V, beta, A."""
    w, u = ascendc_recompute_w_u_fwd(
        k, v, beta, A, chunk_size,
        g=g,
        gk=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u


def flash_chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_list: Optional[list] = None,
    chunk_indices: Optional[Dict] = None,
    chunk_indices_list: Optional[Dict] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Forward pass: 6 steps."""

    # Step 1: Compute cumulative sum of g
    g = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
    )

    # Step 2: Compute A = K^T @ K (scaled with decay)
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )

    # Step 3: Solve triangular system
    A = solve_tri(
        A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices_list=_chunk_list(chunk_indices_list, chunk_size),
        output_dtype=k.dtype,
    )

    # Transpose for AscendC operators
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()
    A = A.transpose(1, 2).contiguous()

    # Step 4: Recompute W and U
    w, u = recompute_w_u(
        k, v, beta, A, g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    # Step 5: Compute recurrent state H
    h, v_new, final_state = ascendc_chunk_gated_delta_rule_fwd_h(
        k, w, u,
        g=g,
        gk=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    if not output_final_state:
        final_state = None

    # Step 6: Compute output O
    o = ascendc_chunk_fwd_o(
        q, k, v_new, h, scale,
        g=g,
        g_gamma=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )

    # Transpose back
    g = g.transpose(1, 2).contiguous()
    o = o.transpose(1, 2).contiguous()

    return g, o, A, final_state


def flash_chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    do: torch.Tensor,
    dht: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_list: Optional[list] = None,
    chunk_indices: Optional[Dict] = None,
    chunk_indices_list: Optional[Dict] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Backward pass: 6 steps."""

    # Transpose for AscendC operators
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()

    # Recompute W and U
    w, u = recompute_w_u(
        k, v, beta, A, g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    do = do.transpose(1, 2).contiguous()

    # Recompute forward intermediate values
    h, v_new, _ = ascendc_chunk_gated_delta_rule_fwd_h(
        k, w, u,
        g=g,
        gk=None,
        initial_state=initial_state,
        output_final_state=False,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    # Step 1: Compute dV (local)
    dv = ascendc_chunk_bwd_dv_local(
        q, k, do, g, scale, chunk_size,
        g_gamma=None,
        A=A,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    # Step 2: Compute dH and update dV
    dh, dh0, dv = ascendc_chunk_gated_delta_rule_bwd_dhu(
        q, k, w, do, dv, scale, chunk_size,
        g=g,
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )
    dh0 = None

    # Step 3: Compute dQ, dK, dW, dG
    dq, dk, dw, dg = ascendc_chunk_bwd_dqkwg(
        q, k, v_new, g, h, do, dh, dv, chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        w=None,
        g_gamma=None,
        scale=scale,
        use_exp2=False,
        transpose_state_layout=False,
    )

    # Step 4: Compute dA
    dA = ascendc_prepare_wy_repr_bwd_da(
        k, v, beta.float(), A, dw, dv, g.float(),
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    # Step 5: Full WY representation backward
    dk2, dv, db, dg2 = ascendc_prepare_wy_repr_bwd_full(
        k, v, beta, A, dA, dw, dv, g, chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    # Transpose gradients back
    db = db.transpose(1, 2).contiguous()
    dg2 = dg2.transpose(1, 2).contiguous()
    dg = dg.transpose(1, 2).contiguous()

    # Accumulate gradients
    dk.add_(dk2)
    dg.add_(dg2)

    # Step 6: Reverse cumsum for dG
    dg = chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
    )

    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """Autograd function for GDN with complete forward/backward."""

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_list: Optional[list] = None,
        chunk_indices: Optional[Dict] = None,
        chunk_indices_list: Optional[Dict] = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
    ):
        # Apply L2 normalization if requested
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        # Forward pass
        g, o, A, final_state = flash_chunk_gated_delta_rule_fwd(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
            chunk_size=chunk_size,
        )

        # Save for backward
        ctx.save_for_backward(q, k, v, g, beta, A)
        ctx.q_rstd = q_rstd
        ctx.k_rstd = k_rstd
        ctx.initial_state = initial_state
        ctx.cu_seqlens = cu_seqlens
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens_list = cu_seqlens_list
        ctx.chunk_indices = chunk_indices
        ctx.chunk_indices_list = chunk_indices_list

        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: Optional[torch.Tensor]):
        q, k, v, g, beta, A = ctx.saved_tensors

        # Backward pass
        dq, dk, dv, db, dg, dh0 = flash_chunk_gated_delta_rule_bwd(
            q=q, k=k, v=v, g=g, beta=beta, A=A,
            scale=ctx.scale,
            initial_state=ctx.initial_state,
            do=do,
            dht=dht,
            cu_seqlens=ctx.cu_seqlens,
            cu_seqlens_list=ctx.cu_seqlens_list,
            chunk_indices=ctx.chunk_indices,
            chunk_indices_list=ctx.chunk_indices_list,
            chunk_size=ctx.chunk_size,
        )

        # Apply L2 norm backward if used in forward
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, ctx.q_rstd, dq)
            dk = l2norm_bwd(k, ctx.k_rstd, dk)

        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            dg.to(g.dtype),
            db.to(beta.dtype),
            None,  # scale
            dh0,   # initial_state
            None,  # output_final_state
            None,  # cu_seqlens
            None,  # cu_seqlens_list
            None,  # chunk_indices
            None,  # chunk_indices_list
            None,  # use_qk_l2norm_in_kernel
            None,  # chunk_size
        )


def flash_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_list: Optional[list] = None,
    chunk_indices: Optional[Dict] = None,
    chunk_indices_list: Optional[Dict] = None,
    chunk_size: int = 64,
    head_first: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Main entry point for GDN.

    Expects BHSD layout for q, k, v and BSH for g, beta.
    Returns output in BSHD layout.
    """
    # Input validation
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError("q, k, v must have the same dtype")
    if q.dtype == torch.float32:
        raise ValueError("GDN does not support float32, use float16/bfloat16")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be rank-4 [B, H, T, D] tensors")
    if g.ndim != 3 or beta.ndim != 3:
        raise ValueError("g, beta must be rank-3 [B, T, H] tensors")

    if scale is None:
        scale = k.shape[-1] ** -0.5

    # Call autograd function
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q, k, v, g, beta,
        float(scale),
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_list,
        chunk_indices,
        chunk_indices_list,
        use_qk_l2norm_in_kernel,
        chunk_size,
    )

    return o, final_state
