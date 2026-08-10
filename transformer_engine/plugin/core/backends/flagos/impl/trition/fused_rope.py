# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - exercised only on systems without Triton.
    triton = None
    tl = None


NVTE_SBHD = 0
NVTE_BSHD = 1
NVTE_THD = 2

__all__ = [
    "fused_rope_forward_fl",
    "fused_rope_backward_fl",
    "fused_qkv_rope_forward_fl",
    "fused_qkv_rope_backward_fl",
]


def _require_triton() -> None:
    if triton is None:
        raise RuntimeError(
            "FlagOS fused RoPE requires the Triton Python package, but it is not installed."
        )


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _choose_block_d(d: int) -> int:
    return min(max(16, _next_power_of_2(min(d, 128))), 128)


def _choose_rope_block_h(h: int) -> int:
    return 4 if h < 16 else 8


def _choose_qkv_block_h(h: int) -> int:
    return min(8, _next_power_of_2(max(1, h)))


def _num_warps(block_h: int) -> int:
    return max(1, min(8, block_h))


def _check_freqs(freqs: torch.Tensor, name: str) -> None:
    if freqs.dim() != 4:
        raise ValueError(f"{name} must be a 4D tensor")
    if freqs.size(1) != 1 or freqs.size(2) != 1:
        raise ValueError(f"{name} must have shape (s, 1, 1, d)")
    if freqs.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")


def _check_qkv_splits(qkv_split_arg_list: List[int]) -> Tuple[int, int, int]:
    if len(qkv_split_arg_list) != 3:
        raise ValueError("qkv_split_arg_list must contain exactly three integers")
    q_split, k_split, v_split = [int(x) for x in qkv_split_arg_list]
    if q_split <= 0 or k_split <= 0 or v_split <= 0:
        raise ValueError("qkv split sizes must be positive")
    if k_split != v_split:
        raise ValueError("FlagOS fused QKV RoPE requires equal K and V head dimensions")
    if q_split % k_split != 0:
        raise ValueError("Q split size must be an integer multiple of the K/V head dimension")
    return q_split, k_split, v_split


if triton is not None:

    @triton.jit
    def _fused_rope_kernel(
        src,
        cu_seqlens,
        freqs,
        start_positions,
        dst,
        S: tl.constexpr,
        B: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        D2: tl.constexpr,
        STRIDE_S_OR_T: tl.constexpr,
        STRIDE_B: tl.constexpr,
        STRIDE_H: tl.constexpr,
        STRIDE_D: tl.constexpr,
        QKV_FORMAT: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        IS_BACKWARD: tl.constexpr,
        HAS_CU_SEQLENS: tl.constexpr,
        HAS_START_POSITIONS: tl.constexpr,
        CP_SIZE: tl.constexpr,
        CP_RANK: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        N_D_BLOCKS: tl.constexpr,
    ):
        s_id = tl.program_id(0)
        b_id = tl.program_id(1)
        hd_pid = tl.program_id(2)
        h_block = hd_pid // N_D_BLOCKS
        d_block = hd_pid - h_block * N_D_BLOCKS

        offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_h = offs_h < H
        mask_d = offs_d < D
        mask_d2 = offs_d < D2
        mask = mask_h[:, None] & mask_d[None, :]
        mask_rotary = mask_h[:, None] & mask_d2[None, :]

        if HAS_CU_SEQLENS:
            start = tl.load(cu_seqlens + b_id) // CP_SIZE
            end = tl.load(cu_seqlens + b_id + 1) // CP_SIZE
            t_id = s_id + start
            valid_token = t_id < end
            offset_block = t_id * STRIDE_S_OR_T
            offset_block_dst = t_id * H * D
            cur_seqlens = end - start
        else:
            valid_token = True
            offset_block = s_id * STRIDE_S_OR_T + b_id * STRIDE_B
            if QKV_FORMAT == 0:
                offset_block_dst = s_id * B * H * D + b_id * H * D
            else:
                offset_block_dst = b_id * S * H * D + s_id * H * D
            cur_seqlens = S

        begin_offset = 0
        if HAS_START_POSITIONS:
            begin_offset = tl.load(start_positions + b_id)
        s_id_for_freqs = s_id + begin_offset

        if CP_SIZE > 1:
            half_seq = cur_seqlens // 2
            cp_delta = tl.where(
                s_id < half_seq,
                CP_RANK * half_seq,
                cur_seqlens * CP_SIZE - (CP_RANK + 1) * half_seq - half_seq,
            )
            s_id_for_freqs += cp_delta

        src_offsets = offset_block + offs_h[:, None] * STRIDE_H + offs_d[None, :] * STRIDE_D
        dst_offsets = offset_block_dst + offs_h[:, None] * D + offs_d[None, :]

        src_values = tl.load(src + src_offsets, mask=mask & valid_token, other=0.0).to(tl.float32)
        out_values = src_values

        if INTERLEAVED:
            is_even = (offs_d % 2) == 0
            if IS_BACKWARD:
                rot_d = tl.where(is_even, offs_d + 1, offs_d - 1)
                sin_d = rot_d
                sin_sign = tl.where(is_even, 1.0, -1.0)
                rot_sign = 1.0
            else:
                rot_d = tl.where(is_even, offs_d + 1, offs_d - 1)
                sin_d = offs_d
                sin_sign = 1.0
                rot_sign = tl.where(is_even, -1.0, 1.0)
        else:
            half_d2 = D2 // 2
            first_half = (offs_d + half_d2) < D2
            rot_d = tl.where(first_half, offs_d + half_d2, offs_d + half_d2 - D2)
            if IS_BACKWARD:
                sin_d = rot_d
                sin_sign = tl.where(first_half, 1.0, -1.0)
                rot_sign = 1.0
            else:
                sin_d = offs_d
                sin_sign = 1.0
                rot_sign = tl.where(first_half, -1.0, 1.0)

        rot_offsets = offset_block + offs_h[:, None] * STRIDE_H + rot_d[None, :] * STRIDE_D
        rot_values = tl.load(src + rot_offsets, mask=mask_rotary & valid_token, other=0.0).to(
            tl.float32
        )
        freq_base = s_id_for_freqs * D2
        freq_mask = mask_d2 & valid_token
        cos_values = tl.cos(tl.load(freqs + freq_base + offs_d, mask=freq_mask, other=0.0))
        sin_values = (
            tl.sin(tl.load(freqs + freq_base + sin_d, mask=freq_mask, other=0.0)) * sin_sign
        )
        rotary_values = (
            src_values * cos_values[None, :] + rot_values * rot_sign * sin_values[None, :]
        )
        out_values = tl.where(mask_d2[None, :], rotary_values, out_values)

        tl.store(dst + dst_offsets, out_values, mask=mask & valid_token)

    @triton.jit
    def _fused_qkv_rope_kernel(
        qkv_input,
        q_freqs,
        k_freqs,
        start_positions,
        q_out,
        k_out,
        v_out,
        qkv_grad_input,
        S: tl.constexpr,
        B: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        D2: tl.constexpr,
        Q_SPLIT: tl.constexpr,
        K_SPLIT: tl.constexpr,
        V_SPLIT: tl.constexpr,
        QKV_FORMAT: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        IS_BACKWARD: tl.constexpr,
        HAS_START_POSITIONS: tl.constexpr,
        CP_SIZE: tl.constexpr,
        CP_RANK: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        N_D_BLOCKS: tl.constexpr,
    ):
        s_id = tl.program_id(0)
        b_id = tl.program_id(1)
        hd_pid = tl.program_id(2)
        h_block = hd_pid // N_D_BLOCKS
        d_block = hd_pid - h_block * N_D_BLOCKS

        offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_h = offs_h < H
        mask_d = offs_d < D
        mask_d2 = offs_d < D2

        total_d = Q_SPLIT + K_SPLIT + V_SPLIT
        if QKV_FORMAT == 0:
            input_base = s_id * B * H * total_d + b_id * H * total_d
            q_base = s_id * B * H * Q_SPLIT + b_id * H * Q_SPLIT
            k_base = s_id * B * H * K_SPLIT + b_id * H * K_SPLIT
            v_base = s_id * B * H * V_SPLIT + b_id * H * V_SPLIT
        else:
            input_base = b_id * S * H * total_d + s_id * H * total_d
            q_base = b_id * S * H * Q_SPLIT + s_id * H * Q_SPLIT
            k_base = b_id * S * H * K_SPLIT + s_id * H * K_SPLIT
            v_base = b_id * S * H * V_SPLIT + s_id * H * V_SPLIT

        if CP_SIZE > 1:
            half_seq = S // 2
            s_id_for_freqs = tl.where(
                s_id < half_seq,
                s_id + CP_RANK * half_seq,
                S * CP_SIZE - (CP_RANK + 1) * half_seq + s_id - half_seq,
            )
        else:
            if IS_BACKWARD:
                s_id_for_freqs = s_id
            else:
                begin_offset = 0
                if HAS_START_POSITIONS:
                    begin_offset = tl.load(start_positions + b_id)
                s_id_for_freqs = s_id + begin_offset

        if INTERLEAVED:
            is_even = (offs_d % 2) == 0
            if IS_BACKWARD:
                rot_d = tl.where(is_even, offs_d + 1, offs_d - 1)
                sin_d = rot_d
                sin_sign = tl.where(is_even, 1.0, -1.0)
                rot_sign = 1.0
            else:
                rot_d = tl.where(is_even, offs_d + 1, offs_d - 1)
                sin_d = offs_d
                sin_sign = 1.0
                rot_sign = tl.where(is_even, -1.0, 1.0)
        else:
            half_d2 = D2 // 2
            first_half = (offs_d + half_d2) < D2
            rot_d = tl.where(first_half, offs_d + half_d2, offs_d + half_d2 - D2)
            if IS_BACKWARD:
                sin_d = rot_d
                sin_sign = tl.where(first_half, 1.0, -1.0)
                rot_sign = 1.0
            else:
                sin_d = offs_d
                sin_sign = 1.0
                rot_sign = tl.where(first_half, -1.0, 1.0)

        q_cos = tl.cos(tl.load(q_freqs + s_id_for_freqs * D2 + offs_d, mask=mask_d2, other=0.0))
        q_sin = (
            tl.sin(tl.load(q_freqs + s_id_for_freqs * D2 + sin_d, mask=mask_d2, other=0.0))
            * sin_sign
        )
        k_cos = tl.cos(tl.load(k_freqs + s_id_for_freqs * D2 + offs_d, mask=mask_d2, other=0.0))
        k_sin = (
            tl.sin(tl.load(k_freqs + s_id_for_freqs * D2 + sin_d, mask=mask_d2, other=0.0))
            * sin_sign
        )

        for row_offset in tl.static_range(0, Q_SPLIT, D):
            component_d = row_offset + offs_d
            mask = mask_h[:, None] & (component_d[None, :] < Q_SPLIT) & mask_d[None, :]
            mask_rotary = mask_h[:, None] & (component_d[None, :] < Q_SPLIT) & mask_d2[None, :]
            if IS_BACKWARD:
                src_base = q_base
                dst_base = input_base
                src_row_length = Q_SPLIT
                dst_row_offset = row_offset
            else:
                src_base = input_base
                dst_base = q_base
                src_row_length = total_d
                dst_row_offset = row_offset
            src_offsets = src_base + offs_h[:, None] * src_row_length + component_d[None, :]
            rot_offsets = (
                src_base + offs_h[:, None] * src_row_length + (row_offset + rot_d)[None, :]
            )
            dst_offsets = dst_base + offs_h[:, None] * total_d + dst_row_offset + offs_d[None, :]
            if not IS_BACKWARD:
                dst_offsets = dst_base + offs_h[:, None] * Q_SPLIT + component_d[None, :]

            if IS_BACKWARD:
                values = tl.load(q_out + src_offsets, mask=mask, other=0.0).to(tl.float32)
                rot_values = tl.load(q_out + rot_offsets, mask=mask_rotary, other=0.0).to(
                    tl.float32
                )
            else:
                values = tl.load(qkv_input + src_offsets, mask=mask, other=0.0).to(tl.float32)
                rot_values = tl.load(qkv_input + rot_offsets, mask=mask_rotary, other=0.0).to(
                    tl.float32
                )
            rotary_values = values * q_cos[None, :] + rot_values * rot_sign * q_sin[None, :]
            out_values = tl.where(mask_d2[None, :], rotary_values, values)
            if IS_BACKWARD:
                tl.store(qkv_grad_input + dst_offsets, out_values, mask=mask)
            else:
                tl.store(q_out + dst_offsets, out_values, mask=mask)

        for row_offset in tl.static_range(0, K_SPLIT, D):
            component_d = row_offset + offs_d
            input_row_offset = Q_SPLIT + row_offset
            mask = mask_h[:, None] & (component_d[None, :] < K_SPLIT) & mask_d[None, :]
            mask_rotary = mask_h[:, None] & (component_d[None, :] < K_SPLIT) & mask_d2[None, :]
            if IS_BACKWARD:
                src_offsets = k_base + offs_h[:, None] * K_SPLIT + component_d[None, :]
                rot_offsets = k_base + offs_h[:, None] * K_SPLIT + (row_offset + rot_d)[None, :]
                dst_offsets = (
                    input_base + offs_h[:, None] * total_d + input_row_offset + offs_d[None, :]
                )
                values = tl.load(k_out + src_offsets, mask=mask, other=0.0).to(tl.float32)
                rot_values = tl.load(k_out + rot_offsets, mask=mask_rotary, other=0.0).to(
                    tl.float32
                )
                rotary_values = values * k_cos[None, :] + rot_values * rot_sign * k_sin[None, :]
                out_values = tl.where(mask_d2[None, :], rotary_values, values)
                tl.store(qkv_grad_input + dst_offsets, out_values, mask=mask)
            else:
                src_offsets = (
                    input_base + offs_h[:, None] * total_d + input_row_offset + offs_d[None, :]
                )
                rot_offsets = (
                    input_base + offs_h[:, None] * total_d + (input_row_offset + rot_d)[None, :]
                )
                dst_offsets = k_base + offs_h[:, None] * K_SPLIT + component_d[None, :]
                values = tl.load(qkv_input + src_offsets, mask=mask, other=0.0).to(tl.float32)
                rot_values = tl.load(qkv_input + rot_offsets, mask=mask_rotary, other=0.0).to(
                    tl.float32
                )
                rotary_values = values * k_cos[None, :] + rot_values * rot_sign * k_sin[None, :]
                out_values = tl.where(mask_d2[None, :], rotary_values, values)
                tl.store(k_out + dst_offsets, out_values, mask=mask)

        component_d = offs_d
        mask = mask_h[:, None] & (component_d[None, :] < V_SPLIT) & mask_d[None, :]
        if IS_BACKWARD:
            src_offsets = v_base + offs_h[:, None] * V_SPLIT + component_d[None, :]
            dst_offsets = (
                input_base + offs_h[:, None] * total_d + Q_SPLIT + K_SPLIT + offs_d[None, :]
            )
            values = tl.load(v_out + src_offsets, mask=mask, other=0.0)
            tl.store(qkv_grad_input + dst_offsets, values, mask=mask)
        else:
            src_offsets = (
                input_base + offs_h[:, None] * total_d + Q_SPLIT + K_SPLIT + offs_d[None, :]
            )
            dst_offsets = v_base + offs_h[:, None] * V_SPLIT + component_d[None, :]
            values = tl.load(qkv_input + src_offsets, mask=mask, other=0.0)
            tl.store(v_out + dst_offsets, values, mask=mask)


def fused_rope_forward_fl(
    input: torch.Tensor,
    freqs: torch.Tensor,
    start_positions: Optional[torch.Tensor],
    qkv_format,
    interleaved: bool,
    cu_seqlens: Optional[torch.Tensor],
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    _require_triton()
    _check_freqs(freqs, "freqs")
    if not freqs.is_contiguous():
        freqs = freqs.contiguous()
    qkv_format = int(qkv_format)
    output = torch.empty(input.size(), dtype=input.dtype, device=input.device)

    if qkv_format == NVTE_THD:
        if input.dim() != 3:
            raise ValueError("input must be a 3D tensor for THD format")
        if cu_seqlens is None:
            raise ValueError("cu_seqlens is required for THD format")
        s = freqs.size(0)
        b = cu_seqlens.numel() - 1
        h = input.size(1)
        d = input.size(2)
        stride_s_or_t = input.stride(0)
        stride_b = 0
        stride_h = input.stride(1)
        stride_d = input.stride(2)
        has_cu_seqlens = True
    else:
        if input.dim() != 4:
            raise ValueError("input must be a 4D tensor for SBHD/BSHD format")
        if qkv_format == NVTE_SBHD:
            s = input.size(0)
            b = input.size(1)
            stride_s_or_t = input.stride(0)
            stride_b = input.stride(1)
        else:
            s = input.size(1)
            b = input.size(0)
            stride_s_or_t = input.stride(1)
            stride_b = input.stride(0)
        h = input.size(2)
        d = input.size(3)
        stride_h = input.stride(2)
        stride_d = input.stride(3)
        has_cu_seqlens = False

    d2 = freqs.size(3)
    if d < d2:
        raise ValueError("input last dimension must be greater than or equal to freqs last dim")
    if qkv_format != NVTE_THD and s * cp_size > freqs.size(0):
        raise ValueError("freqs sequence length is too short for input and cp_size")

    block_h = _choose_rope_block_h(h)
    block_d = _choose_block_d(d)
    d_blocks = triton.cdiv(d, block_d)
    grid = (s, b, triton.cdiv(h, block_h) * d_blocks)
    dummy_cu = cu_seqlens if cu_seqlens is not None else input
    dummy_start = start_positions if start_positions is not None else input
    _fused_rope_kernel[grid](
        input,
        dummy_cu,
        freqs,
        dummy_start,
        output,
        s,
        b,
        h,
        d,
        d2,
        stride_s_or_t,
        stride_b,
        stride_h,
        stride_d,
        qkv_format,
        interleaved,
        False,
        has_cu_seqlens,
        start_positions is not None,
        cp_size,
        cp_rank,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        N_D_BLOCKS=d_blocks,
        num_warps=_num_warps(block_h),
    )
    return output


def fused_rope_backward_fl(
    output_grads: torch.Tensor,
    freqs: torch.Tensor,
    start_positions: Optional[torch.Tensor],
    qkv_format,
    interleaved: bool,
    cu_seqlens: Optional[torch.Tensor],
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    _require_triton()
    _check_freqs(freqs, "freqs")
    if not freqs.is_contiguous():
        freqs = freqs.contiguous()
    qkv_format = int(qkv_format)
    input_grads = torch.empty(
        output_grads.size(), dtype=output_grads.dtype, device=output_grads.device
    )

    if qkv_format == NVTE_THD:
        if output_grads.dim() != 3:
            raise ValueError("output_grads must be a 3D tensor for THD format")
        if cu_seqlens is None:
            raise ValueError("cu_seqlens is required for THD format")
        s = freqs.size(0)
        b = cu_seqlens.numel() - 1
        h = output_grads.size(1)
        d = output_grads.size(2)
        stride_s_or_t = output_grads.stride(0)
        stride_b = 0
        stride_h = output_grads.stride(1)
        stride_d = output_grads.stride(2)
        has_cu_seqlens = True
    else:
        if output_grads.dim() != 4:
            raise ValueError("output_grads must be a 4D tensor for SBHD/BSHD format")
        if qkv_format == NVTE_SBHD:
            s = output_grads.size(0)
            b = output_grads.size(1)
            stride_s_or_t = output_grads.stride(0)
            stride_b = output_grads.stride(1)
        else:
            s = output_grads.size(1)
            b = output_grads.size(0)
            stride_s_or_t = output_grads.stride(1)
            stride_b = output_grads.stride(0)
        h = output_grads.size(2)
        d = output_grads.size(3)
        stride_h = output_grads.stride(2)
        stride_d = output_grads.stride(3)
        has_cu_seqlens = False

    d2 = freqs.size(3)
    if d < d2:
        raise ValueError(
            "output_grads last dimension must be greater than or equal to freqs last dim"
        )
    if qkv_format != NVTE_THD and s * cp_size > freqs.size(0):
        raise ValueError("freqs sequence length is too short for output_grads and cp_size")

    block_h = _choose_rope_block_h(h)
    block_d = _choose_block_d(d)
    d_blocks = triton.cdiv(d, block_d)
    grid = (s, b, triton.cdiv(h, block_h) * d_blocks)
    dummy_cu = cu_seqlens if cu_seqlens is not None else output_grads
    dummy_start = start_positions if start_positions is not None else output_grads
    _fused_rope_kernel[grid](
        output_grads,
        dummy_cu,
        freqs,
        dummy_start,
        input_grads,
        s,
        b,
        h,
        d,
        d2,
        stride_s_or_t,
        stride_b,
        stride_h,
        stride_d,
        qkv_format,
        interleaved,
        True,
        has_cu_seqlens,
        start_positions is not None,
        cp_size,
        cp_rank,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        N_D_BLOCKS=d_blocks,
        num_warps=_num_warps(block_h),
    )
    return input_grads


def fused_qkv_rope_forward_fl(
    qkv_input: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    start_positions: Optional[torch.Tensor],
    qkv_split_arg_list: List[int],
    qkv_format,
    interleaved: bool,
    cp_size: int,
    cp_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_triton()
    _check_freqs(q_freqs, "q_freqs")
    _check_freqs(k_freqs, "k_freqs")
    if not q_freqs.is_contiguous():
        q_freqs = q_freqs.contiguous()
    if not k_freqs.is_contiguous():
        k_freqs = k_freqs.contiguous()
    if qkv_input.dim() != 4:
        raise ValueError("qkv_input must be a 4D tensor")
    if not qkv_input.is_contiguous():
        raise ValueError("qkv_input must be contiguous")

    qkv_format = int(qkv_format)
    is_sbhd = qkv_format == NVTE_SBHD
    s = qkv_input.size(0) if is_sbhd else qkv_input.size(1)
    b = qkv_input.size(1) if is_sbhd else qkv_input.size(0)
    h = qkv_input.size(2)
    q_split, k_split, v_split = _check_qkv_splits(qkv_split_arg_list)
    if qkv_input.size(3) != q_split + k_split + v_split:
        raise ValueError("qkv_input last dimension must equal the sum of qkv split sizes")
    d = v_split
    d2 = q_freqs.size(3)
    if d < d2:
        raise ValueError("qkv value split must be greater than or equal to q_freqs last dim")
    if q_freqs.size(3) != k_freqs.size(3):
        raise ValueError("q_freqs and k_freqs must have the same rotary dimension")

    q_out_size = list(qkv_input.size())
    q_out_size[2] = q_out_size[2] * q_split // k_split
    q_out_size[3] = k_split
    k_out_size = list(qkv_input.size())
    k_out_size[3] = k_split
    v_out_size = list(qkv_input.size())
    v_out_size[3] = v_split
    q_out = torch.empty(q_out_size, dtype=qkv_input.dtype, device=qkv_input.device)
    k_out = torch.empty(k_out_size, dtype=qkv_input.dtype, device=qkv_input.device)
    v_out = torch.empty(v_out_size, dtype=qkv_input.dtype, device=qkv_input.device)

    block_h = _choose_qkv_block_h(h)
    block_d = _choose_block_d(d)
    d_blocks = triton.cdiv(d, block_d)
    grid = (s, b, triton.cdiv(h, block_h) * d_blocks)
    dummy_start = start_positions if start_positions is not None else qkv_input
    _fused_qkv_rope_kernel[grid](
        qkv_input,
        q_freqs,
        k_freqs,
        dummy_start,
        q_out,
        k_out,
        v_out,
        qkv_input,
        s,
        b,
        h,
        d,
        d2,
        q_split,
        k_split,
        v_split,
        qkv_format,
        interleaved,
        False,
        start_positions is not None,
        cp_size,
        cp_rank,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        N_D_BLOCKS=d_blocks,
        num_warps=_num_warps(block_h),
    )
    return q_out, k_out, v_out


def fused_qkv_rope_backward_fl(
    q_grad_out: torch.Tensor,
    k_grad_out: torch.Tensor,
    v_grad_out: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    qkv_split_arg_list: List[int],
    qkv_format,
    interleaved: bool,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    _require_triton()
    _check_freqs(q_freqs, "q_freqs")
    _check_freqs(k_freqs, "k_freqs")
    if not q_freqs.is_contiguous():
        q_freqs = q_freqs.contiguous()
    if not k_freqs.is_contiguous():
        k_freqs = k_freqs.contiguous()
    q_grad_out = q_grad_out.contiguous()
    k_grad_out = k_grad_out.contiguous()
    v_grad_out = v_grad_out.contiguous()

    qkv_format = int(qkv_format)
    is_sbhd = qkv_format == NVTE_SBHD
    s = q_grad_out.size(0) if is_sbhd else q_grad_out.size(1)
    b = q_grad_out.size(1) if is_sbhd else q_grad_out.size(0)
    q_split, k_split, v_split = _check_qkv_splits(qkv_split_arg_list)
    if q_grad_out.size(3) != k_split or k_grad_out.size(3) != k_split:
        raise ValueError("Q and K gradient last dimensions must match the K split size")
    if v_grad_out.size(3) != v_split:
        raise ValueError("V gradient last dimension must match the V split size")
    total_hd = (q_grad_out.size(2) + k_grad_out.size(2) + v_grad_out.size(2)) * q_grad_out.size(3)
    total_d = q_split + k_split + v_split
    if total_hd % total_d != 0:
        raise ValueError("Q/K/V gradient shapes are inconsistent with qkv split sizes")
    qkv_grad_size = list(q_grad_out.size())
    qkv_grad_size[2] = total_hd // total_d
    qkv_grad_size[3] = total_d
    h = qkv_grad_size[2]
    d = v_split
    d2 = q_freqs.size(3)
    if d < d2:
        raise ValueError("qkv value split must be greater than or equal to q_freqs last dim")
    if q_freqs.size(3) != k_freqs.size(3):
        raise ValueError("q_freqs and k_freqs must have the same rotary dimension")

    qkv_grad_input = torch.empty(qkv_grad_size, dtype=q_grad_out.dtype, device=q_grad_out.device)
    block_h = _choose_qkv_block_h(h)
    block_d = _choose_block_d(d)
    d_blocks = triton.cdiv(d, block_d)
    grid = (s, b, triton.cdiv(h, block_h) * d_blocks)
    _fused_qkv_rope_kernel[grid](
        q_grad_out,
        q_freqs,
        k_freqs,
        q_grad_out,
        q_grad_out,
        k_grad_out,
        v_grad_out,
        qkv_grad_input,
        s,
        b,
        h,
        d,
        d2,
        q_split,
        k_split,
        v_split,
        qkv_format,
        interleaved,
        True,
        False,
        cp_size,
        cp_rank,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        N_D_BLOCKS=d_blocks,
        num_warps=_num_warps(block_h),
    )
    return qkv_grad_input
