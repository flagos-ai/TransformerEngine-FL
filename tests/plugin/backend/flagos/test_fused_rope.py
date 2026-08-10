# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from typing import Optional

import pytest
import torch

from transformer_engine.plugin.core.ops import NVTE_QKV_Format
from transformer_engine.plugin.test_utils import get_backend


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _make_freqs(seq_len: int, d2: int, device: str) -> torch.Tensor:
    values = torch.linspace(-0.7, 0.9, steps=seq_len * d2, dtype=torch.float32, device=device)
    return values.reshape(seq_len, 1, 1, d2).contiguous()


def _freq_position(
    s_id: int,
    b_id: int,
    cur_seqlens: int,
    start_positions: Optional[torch.Tensor],
    cp_size: int,
    cp_rank: int,
) -> int:
    pos = s_id
    if start_positions is not None:
        pos += int(start_positions[b_id].item())

    if cp_size > 1:
        half = cur_seqlens // 2
        if s_id < half:
            pos += cp_rank * half
        else:
            pos += cur_seqlens * cp_size - (cp_rank + 1) * half - half
    return pos


def _apply_rope_slice(
    src: torch.Tensor,
    freq: torch.Tensor,
    interleaved: bool,
    is_backward: bool,
) -> torch.Tensor:
    d2 = freq.numel()
    out = src.clone()
    src_rot = src[..., :d2].float()

    idx = torch.arange(d2, device=src.device)
    if interleaved:
        even = (idx % 2) == 0
        rot_idx = torch.where(even, idx + 1, idx - 1)
        if is_backward:
            sin_idx = rot_idx
            sin_sign = torch.where(even, 1.0, -1.0)
            rot_sign = torch.ones_like(freq)
        else:
            sin_idx = idx
            sin_sign = torch.ones_like(freq)
            rot_sign = torch.where(even, -1.0, 1.0)
    else:
        half = d2 // 2
        first_half = (idx + half) < d2
        rot_idx = torch.where(first_half, idx + half, idx + half - d2)
        if is_backward:
            sin_idx = rot_idx
            sin_sign = torch.where(first_half, 1.0, -1.0)
            rot_sign = torch.ones_like(freq)
        else:
            sin_idx = idx
            sin_sign = torch.ones_like(freq)
            rot_sign = torch.where(first_half, -1.0, 1.0)

    rotary = (
        src_rot * torch.cos(freq)
        + src_rot[..., rot_idx] * rot_sign * torch.sin(freq[sin_idx]) * sin_sign
    )
    out[..., :d2] = rotary.to(src.dtype)
    return out


def _reference_rope(
    tensor: torch.Tensor,
    freqs: torch.Tensor,
    qkv_format: NVTE_QKV_Format,
    interleaved: bool,
    cu_seqlens: Optional[torch.Tensor],
    start_positions: Optional[torch.Tensor],
    cp_size: int,
    cp_rank: int,
    is_backward: bool,
) -> torch.Tensor:
    freq_flat = freqs[:, 0, 0, :]
    out = torch.empty(tensor.size(), dtype=tensor.dtype, device=tensor.device)

    if qkv_format == NVTE_QKV_Format.NVTE_THD:
        cu = (cu_seqlens.cpu() // cp_size).tolist()
        for b_id in range(len(cu) - 1):
            start, end = cu[b_id], cu[b_id + 1]
            cur_seqlens = end - start
            for s_id in range(cur_seqlens):
                t_id = start + s_id
                pos = _freq_position(s_id, b_id, cur_seqlens, start_positions, cp_size, cp_rank)
                out[t_id] = _apply_rope_slice(
                    tensor[t_id], freq_flat[pos], interleaved, is_backward
                )
        return out

    if qkv_format == NVTE_QKV_Format.NVTE_SBHD:
        s, b = tensor.size(0), tensor.size(1)
        for s_id in range(s):
            for b_id in range(b):
                pos = _freq_position(s_id, b_id, s, start_positions, cp_size, cp_rank)
                out[s_id, b_id] = _apply_rope_slice(
                    tensor[s_id, b_id], freq_flat[pos], interleaved, is_backward
                )
        return out

    s, b = tensor.size(1), tensor.size(0)
    for b_id in range(b):
        for s_id in range(s):
            pos = _freq_position(s_id, b_id, s, start_positions, cp_size, cp_rank)
            out[b_id, s_id] = _apply_rope_slice(
                tensor[b_id, s_id], freq_flat[pos], interleaved, is_backward
            )
    return out


def _reference_qkv_forward(
    qkv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    start_positions: Optional[torch.Tensor],
    qkv_split_arg_list,
    qkv_format: NVTE_QKV_Format,
    interleaved: bool,
    cp_size: int,
    cp_rank: int,
):
    q_split, k_split, v_split = qkv_split_arg_list
    d = v_split
    is_sbhd = qkv_format == NVTE_QKV_Format.NVTE_SBHD
    s = qkv.size(0) if is_sbhd else qkv.size(1)
    b = qkv.size(1) if is_sbhd else qkv.size(0)
    h = qkv.size(2)

    q_out_size = list(qkv.size())
    q_out_size[2] = q_out_size[2] * q_split // k_split
    q_out_size[3] = k_split
    k_out_size = list(qkv.size())
    k_out_size[3] = k_split
    v_out_size = list(qkv.size())
    v_out_size[3] = v_split

    q_out = torch.empty(q_out_size, dtype=qkv.dtype, device=qkv.device)
    k_out = torch.empty(k_out_size, dtype=qkv.dtype, device=qkv.device)
    v_out = torch.empty(v_out_size, dtype=qkv.dtype, device=qkv.device)
    q_freq_flat = q_freqs[:, 0, 0, :]
    k_freq_flat = k_freqs[:, 0, 0, :]

    for s_id in range(s):
        for b_id in range(b):
            pos = _freq_position(s_id, b_id, s, start_positions, cp_size, cp_rank)
            src = qkv[s_id, b_id] if is_sbhd else qkv[b_id, s_id]
            q_flat = (q_out[s_id, b_id] if is_sbhd else q_out[b_id, s_id]).reshape(-1)
            k_flat = (k_out[s_id, b_id] if is_sbhd else k_out[b_id, s_id]).reshape(-1)
            v_flat = (v_out[s_id, b_id] if is_sbhd else v_out[b_id, s_id]).reshape(-1)

            for h_id in range(h):
                for row_offset in range(0, q_split, d):
                    q_slice = src[h_id, row_offset : row_offset + d]
                    q_flat[h_id * q_split + row_offset : h_id * q_split + row_offset + d] = (
                        _apply_rope_slice(q_slice, q_freq_flat[pos], interleaved, False)
                    )
                k_start = q_split
                for row_offset in range(0, k_split, d):
                    k_slice = src[h_id, k_start + row_offset : k_start + row_offset + d]
                    k_flat[h_id * k_split + row_offset : h_id * k_split + row_offset + d] = (
                        _apply_rope_slice(k_slice, k_freq_flat[pos], interleaved, False)
                    )
                v_start = q_split + k_split
                v_flat[h_id * v_split : (h_id + 1) * v_split] = src[
                    h_id, v_start : v_start + v_split
                ]

    return q_out, k_out, v_out


def _reference_qkv_backward(
    q_grad: torch.Tensor,
    k_grad: torch.Tensor,
    v_grad: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    qkv_split_arg_list,
    qkv_format: NVTE_QKV_Format,
    interleaved: bool,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    q_split, k_split, v_split = qkv_split_arg_list
    d = v_split
    total_d = q_split + k_split + v_split
    total_hd = (q_grad.size(2) + k_grad.size(2) + v_grad.size(2)) * q_grad.size(3)
    qkv_grad_size = list(q_grad.size())
    qkv_grad_size[2] = total_hd // total_d
    qkv_grad_size[3] = total_d
    out = torch.empty(qkv_grad_size, dtype=q_grad.dtype, device=q_grad.device)

    is_sbhd = qkv_format == NVTE_QKV_Format.NVTE_SBHD
    s = q_grad.size(0) if is_sbhd else q_grad.size(1)
    b = q_grad.size(1) if is_sbhd else q_grad.size(0)
    h = out.size(2)
    q_freq_flat = q_freqs[:, 0, 0, :]
    k_freq_flat = k_freqs[:, 0, 0, :]

    for s_id in range(s):
        for b_id in range(b):
            pos = _freq_position(s_id, b_id, s, None, cp_size, cp_rank)
            q_flat = (q_grad[s_id, b_id] if is_sbhd else q_grad[b_id, s_id]).reshape(-1)
            k_flat = (k_grad[s_id, b_id] if is_sbhd else k_grad[b_id, s_id]).reshape(-1)
            v_flat = (v_grad[s_id, b_id] if is_sbhd else v_grad[b_id, s_id]).reshape(-1)
            dst = out[s_id, b_id] if is_sbhd else out[b_id, s_id]

            for h_id in range(h):
                for row_offset in range(0, q_split, d):
                    q_slice = q_flat[h_id * q_split + row_offset : h_id * q_split + row_offset + d]
                    dst[h_id, row_offset : row_offset + d] = _apply_rope_slice(
                        q_slice, q_freq_flat[pos], interleaved, True
                    )
                k_start = q_split
                for row_offset in range(0, k_split, d):
                    k_slice = k_flat[h_id * k_split + row_offset : h_id * k_split + row_offset + d]
                    dst[h_id, k_start + row_offset : k_start + row_offset + d] = _apply_rope_slice(
                        k_slice, k_freq_flat[pos], interleaved, True
                    )
                v_start = q_split + k_split
                dst[h_id, v_start : v_start + v_split] = v_flat[
                    h_id * v_split : (h_id + 1) * v_split
                ]

    return out


@pytest.fixture(scope="module")
def flagos_backend():
    if not torch.cuda.is_available():
        pytest.skip("FlagOS fused RoPE requires a CUDA device")
    if not _triton_available():
        pytest.skip("FlagOS fused RoPE requires Triton")

    try:
        backend = get_backend("flagos")
        for op_name in (
            "fused_rope_forward",
            "fused_rope_backward",
            "fused_qkv_rope_forward",
            "fused_qkv_rope_backward",
        ):
            getattr(backend, op_name)
    except (NotImplementedError, RuntimeError) as exc:
        pytest.skip(f"FlagOS fused RoPE backend is not available: {exc}")

    return backend


@pytest.mark.parametrize(
    "qkv_format, shape, interleaved, cp_size, cp_rank, use_start",
    [
        (NVTE_QKV_Format.NVTE_SBHD, (5, 2, 3, 10), False, 1, 0, True),
        (NVTE_QKV_Format.NVTE_BSHD, (2, 4, 2, 10), True, 2, 1, False),
        (NVTE_QKV_Format.NVTE_SBHD, (4, 2, 2, 10), False, 2, 1, True),
    ],
)
def test_fused_rope_sbhd_bshd_forward_backward(
    flagos_backend,
    qkv_format,
    shape,
    interleaved,
    cp_size,
    cp_rank,
    use_start,
):
    torch.manual_seed(1234)
    device = "cuda"
    d2 = 6
    freq_len = shape[0] if qkv_format == NVTE_QKV_Format.NVTE_SBHD else shape[1]
    freq_len = max(freq_len * cp_size + 3, 12)
    freqs = _make_freqs(freq_len, d2, device)
    start_positions = None
    if use_start:
        batch = shape[1] if qkv_format == NVTE_QKV_Format.NVTE_SBHD else shape[0]
        start_positions = torch.arange(batch, dtype=torch.int32, device=device) + 1

    base = torch.randn(*shape[:-1], shape[-1] * 2, device=device)
    tensor = base[..., ::2]
    grad = torch.randn_like(tensor)
    ref_fwd = _reference_rope(
        tensor,
        freqs,
        qkv_format,
        interleaved,
        None,
        start_positions,
        cp_size,
        cp_rank,
        False,
    )
    ref_bwd = _reference_rope(
        grad,
        freqs,
        qkv_format,
        interleaved,
        None,
        start_positions,
        cp_size,
        cp_rank,
        True,
    )

    out = flagos_backend.fused_rope_forward(
        tensor,
        freqs,
        start_positions,
        qkv_format,
        interleaved,
        None,
        cp_size,
        cp_rank,
    )
    dx = flagos_backend.fused_rope_backward(
        grad,
        freqs,
        start_positions,
        qkv_format,
        interleaved,
        None,
        cp_size,
        cp_rank,
    )

    torch.testing.assert_close(out.float(), ref_fwd.float(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dx.float(), ref_bwd.float(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "cu_cpu, interleaved, cp_size, cp_rank, use_start",
    [
        (torch.tensor([0, 3, 8], dtype=torch.int32), True, 1, 0, True),
        (torch.tensor([0, 8, 20], dtype=torch.int32), False, 2, 0, False),
    ],
)
def test_fused_rope_thd_forward_backward(
    flagos_backend,
    cu_cpu,
    interleaved,
    cp_size,
    cp_rank,
    use_start,
):
    torch.manual_seed(2345)
    device = "cuda"
    cu_seqlens = cu_cpu.to(device)
    local_cu = cu_cpu // cp_size
    total_t = int(local_cu[-1].item())
    h, d, d2 = 3, 10, 6
    freq_len = max(int(cu_cpu[1:].sub(cu_cpu[:-1]).max().item()), 12)
    freqs = _make_freqs(freq_len, d2, device)
    start_positions = None
    if use_start:
        start_positions = torch.tensor([1, 0], dtype=torch.int32, device=device)

    tensor = torch.randn(total_t, h, d, device=device)
    grad = torch.randn_like(tensor)
    ref_fwd = _reference_rope(
        tensor,
        freqs,
        NVTE_QKV_Format.NVTE_THD,
        interleaved,
        cu_seqlens,
        start_positions,
        cp_size,
        cp_rank,
        False,
    )
    ref_bwd = _reference_rope(
        grad,
        freqs,
        NVTE_QKV_Format.NVTE_THD,
        interleaved,
        cu_seqlens,
        start_positions,
        cp_size,
        cp_rank,
        True,
    )

    out = flagos_backend.fused_rope_forward(
        tensor,
        freqs,
        start_positions,
        NVTE_QKV_Format.NVTE_THD,
        interleaved,
        cu_seqlens,
        cp_size,
        cp_rank,
    )
    dx = flagos_backend.fused_rope_backward(
        grad,
        freqs,
        start_positions,
        NVTE_QKV_Format.NVTE_THD,
        interleaved,
        cu_seqlens,
        cp_size,
        cp_rank,
    )

    torch.testing.assert_close(out.float(), ref_fwd.float(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dx.float(), ref_bwd.float(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "qkv_format, shape, interleaved, cp_size, cp_rank, use_start",
    [
        (NVTE_QKV_Format.NVTE_SBHD, (4, 2, 2, 32), False, 1, 0, True),
        (NVTE_QKV_Format.NVTE_BSHD, (2, 4, 2, 32), True, 2, 1, False),
    ],
)
def test_fused_qkv_rope_forward_backward(
    flagos_backend,
    qkv_format,
    shape,
    interleaved,
    cp_size,
    cp_rank,
    use_start,
):
    torch.manual_seed(3456)
    device = "cuda"
    d2 = 6
    qkv_split_arg_list = [16, 8, 8]
    seq_len = shape[0] if qkv_format == NVTE_QKV_Format.NVTE_SBHD else shape[1]
    freq_len = max(seq_len * cp_size + 3, 12)
    q_freqs = _make_freqs(freq_len, d2, device)
    k_freqs = _make_freqs(freq_len, d2, device) + 0.17
    start_positions = None
    if use_start:
        batch = shape[1] if qkv_format == NVTE_QKV_Format.NVTE_SBHD else shape[0]
        start_positions = torch.arange(batch, dtype=torch.int32, device=device)

    qkv = torch.randn(*shape, device=device).contiguous()
    ref_q, ref_k, ref_v = _reference_qkv_forward(
        qkv,
        q_freqs,
        k_freqs,
        start_positions,
        qkv_split_arg_list,
        qkv_format,
        interleaved,
        cp_size,
        cp_rank,
    )

    q_grad = torch.randn_like(ref_q)
    k_grad = torch.randn_like(ref_k)
    v_grad = torch.randn_like(ref_v)
    ref_bwd = _reference_qkv_backward(
        q_grad,
        k_grad,
        v_grad,
        q_freqs,
        k_freqs,
        qkv_split_arg_list,
        qkv_format,
        interleaved,
        cp_size,
        cp_rank,
    )

    q_out, k_out, v_out = flagos_backend.fused_qkv_rope_forward(
        qkv,
        q_freqs,
        k_freqs,
        start_positions,
        qkv_split_arg_list,
        qkv_format,
        interleaved,
        cp_size,
        cp_rank,
    )
    dqkv = flagos_backend.fused_qkv_rope_backward(
        q_grad,
        k_grad,
        v_grad,
        q_freqs,
        k_freqs,
        qkv_split_arg_list,
        qkv_format,
        interleaved,
        cp_size,
        cp_rank,
    )

    torch.testing.assert_close(q_out.float(), ref_q.float(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(k_out.float(), ref_k.float(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v_out.float(), ref_v.float(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dqkv.float(), ref_bwd.float(), rtol=1e-4, atol=1e-4)
