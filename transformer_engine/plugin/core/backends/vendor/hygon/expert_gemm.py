# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Opt-in BF16 expert GEMMs for the measured BW1100 Qwen3.5-35B shapes."""

from __future__ import annotations

import os
from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def _is_bw1100(device_index: int) -> bool:
    properties = torch.cuda.get_device_properties(device_index)
    return (
        properties.name.upper().startswith("BW11")
        and getattr(properties, "gcnArchName", "").split(":")[0] == "gfx938"
    )


def _supports_expert_gemm(
    A,
    transa,
    B,
    transb,
    out,
    m_splits,
    bias,
    single_output,
    pre_gelu_out,
    grad,
    accumulate,
    use_split_accumulator,
) -> bool:
    """Check the complete call before writing to any caller-owned output."""
    if (
        os.getenv("NVTE_HYGON_EXPERT_GEMM", "0") != "1"
        or not getattr(torch.version, "hip", None)
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or accumulate
    ):
        return False
    # This is an FP8 accumulator-mode flag. GroupedLinear also passes True
    # for BF16 dgrad/wgrad; accepting it preserves the validated BF16 path.
    # Quantized tensors are excluded below regardless of this flag's value.
    layout = (transa, transb)
    if layout not in ((True, False), (False, False), (False, True)):
        return False
    if grad != (layout != (True, False)) or single_output != (not transb):
        return False
    if (
        not all(isinstance(items, (list, tuple)) for items in (A, B, out, m_splits))
        or len(A) != 32
        or len(B) != 32
        or len(m_splits) != 32
        or len(out) != (1 if single_output else 32)
        or any(type(rows) is not int or rows < 0 for rows in m_splits)
    ):
        return False
    # Nonempty bias/activation buffers request fused operations, which this
    # path does not implement. Empty CPU sentinels are valid TE arguments.
    for buffers in (bias, pre_gelu_out):
        if (
            not isinstance(buffers, (list, tuple))
            or len(buffers) != 32
            or any(not isinstance(t, torch.Tensor) or t.numel() != 0 for t in buffers)
        ):
            return False
    tensors = (*A, *B, *out)
    if any(
        type(t) not in (torch.Tensor, torch.nn.Parameter)
        or not t.is_cuda
        or t.dtype != torch.bfloat16
        or t.ndim != 2
        or t.layout != torch.strided
        or not t.is_contiguous()
        for t in tensors
    ):
        return False
    device = A[0].device
    if any(t.device != device for t in tensors) or not _is_bw1100(device.index):
        return False
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return False

    weight_shape = tuple(out[0].shape if transb else A[0].shape)
    if weight_shape not in ((1024, 2048), (2048, 512)):
        return False
    n, k = weight_shape
    if transb:
        return all(
            tuple(a.shape) == (rows, k)
            and tuple(b.shape) == (rows, n)
            and tuple(c.shape) == weight_shape
            for a, b, c, rows in zip(A, B, out, m_splits)
        )
    input_width, output_width = (k, n) if transa else (n, k)
    return tuple(out[0].shape) == (sum(m_splits), output_width) and all(
        tuple(a.shape) == weight_shape and tuple(b.shape) == (rows, input_width)
        for a, b, rows in zip(A, B, m_splits)
    )


def _run_expert_gemm(A, transa, B, transb, out, m_splits) -> None:
    """Keep native expert splits and BF16 dW, including zero-token experts."""
    if transb:
        for a, b, c in zip(A, B, out):
            torch.mm(b.T, a, out=c)
    else:
        outputs = out[0].split(m_splits, dim=0)
        for a, b, c in zip(A, B, outputs):
            torch.mm(b, a.T if transa else a, out=c)


def try_expert_gemm(
    A,
    transa,
    B,
    transb,
    out,
    m_splits,
    bias,
    single_output,
    pre_gelu_out,
    grad,
    accumulate,
    use_split_accumulator,
) -> bool:
    """Run the supported GEMM, or return False without modifying outputs.

    The caller checks the TE output-dtype enum. Errors after execution starts
    propagate; they must not trigger a second implementation on partial output.
    """
    if not _supports_expert_gemm(
        A,
        transa,
        B,
        transb,
        out,
        m_splits,
        bias,
        single_output,
        pre_gelu_out,
        grad,
        accumulate,
        use_split_accumulator,
    ):
        return False
    _run_expert_gemm(A, transa, B, transb, out, m_splits)
    return True
