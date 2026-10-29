# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Hygon parallel CE with FP32 intermediate gradients and preserved logits.

The gradient-storage correction follows NVIDIA TransformerEngine PR #3193
by Przemek Tredak (ptrendx): https://github.com/NVIDIA/TransformerEngine/pull/3193
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import torch
import torch.distributed as dist

from ....decorators import override


_SUPPORTED_NUM_WARPS = (8, 16)
_MAX_FUSED_SIZE = 32768


def is_parallel_cross_entropy_available() -> bool:
    """Return whether the optional Hygon parallel CE implementation is enabled."""

    value = os.getenv("NVTE_HYGON_PARALLEL_CE", "1").strip().lower()
    return value not in ("0", "false", "off", "no")


def _get_num_warps() -> int:
    """Get a wave64-safe Triton warp count for Hygon devices."""

    raw_value = os.getenv("NVTE_HYGON_CE_NUM_WARPS", "16")
    try:
        num_warps = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"NVTE_HYGON_CE_NUM_WARPS must be one of {_SUPPORTED_NUM_WARPS}, but got {raw_value!r}"
        ) from exc

    if num_warps not in _SUPPORTED_NUM_WARPS:
        raise ValueError(
            f"NVTE_HYGON_CE_NUM_WARPS must be one of {_SUPPORTED_NUM_WARPS}, but got {num_warps}"
        )
    return num_warps


def parallel_cross_entropy_forward(
    inp: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_process_group: Optional[torch.distributed.ProcessGroup],
    ignore_idx: int,
):
    """Return FP32 loss and an independent FP32 gradient buffer.

    As in the baseline, distributed label-smoothing loss has a separate known
    cross-rank reduction issue. This storage fix is validated with TP>1 at
    label_smoothing=0, and with label smoothing at TP=1.
    """
    import triton

    from transformer_engine.common.triton.cross_entropy import online_softmax_kernel

    from .cross_entropy_kernel import cross_entropy_kernel

    num_warps = _get_num_warps()

    B, SQ, V = inp.shape
    n_rows = B * SQ

    assert target.numel() == (B * SQ), "Each token needs a target token ID."

    BLOCK_SIZE = min(_MAX_FUSED_SIZE, triton.next_power_of_2(V))

    # unreduced loss
    loss_1d = torch.zeros(n_rows, dtype=torch.float32, device=inp.device)

    # tensor to hold this rank's m/d/X_y values
    m_d_X_y = torch.zeros(n_rows * 3, dtype=torch.float32, device=inp.device)

    n_non_ignore = torch.zeros(1, dtype=torch.int64, device=inp.device)

    # The kernels flatten both leading dimensions and require dense rows.
    inp = inp.contiguous()
    target = target.contiguous()

    # Store the input gradient in FP32 so it is not quantized before backward.
    grad_input = torch.empty_like(inp, dtype=torch.float32)

    rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)

    online_softmax_kernel[(n_rows,)](
        X_ptr=inp,
        X_stride=inp.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),  # always 1
        m_d_X_y_ptr=m_d_X_y,
        m_d_X_y_stride=m_d_X_y.stride(-1),
        rank=rank,
        n_cols=V,
        ignore_idx=ignore_idx,
        n_non_ignore=n_non_ignore,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    n_non_ignore = torch.clamp(n_non_ignore, min=1)

    world_size = 1 if dist_process_group is None else dist.get_world_size(dist_process_group)

    if world_size > 1:
        m_d_X_y_gathered = torch.zeros(
            n_rows * 3 * world_size, dtype=torch.float32, device=inp.device
        )
        dist.all_gather_into_tensor(m_d_X_y_gathered, m_d_X_y, group=dist_process_group)
    else:
        m_d_X_y_gathered = m_d_X_y

    cross_entropy_kernel[(n_rows,)](
        X_ptr=inp,
        X_stride=inp.stride(-2),
        grad_input_ptr=grad_input,
        grad_input_stride=grad_input.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),
        loss_ptr=loss_1d,
        loss_stride=loss_1d.stride(-1),
        m_d_X_y_ptr=m_d_X_y_gathered,
        m_d_X_y_stride=m_d_X_y_gathered.stride(-1),
        rank=rank,
        world_size=world_size,
        ignore_idx=ignore_idx,
        n_cols=V,
        n_rows=n_rows,
        n_non_ignore=n_non_ignore,
        reduce_loss=reduce_loss,
        label_smoothing=label_smoothing,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    loss = (
        torch.reshape(loss_1d, (B, SQ)) if not reduce_loss else (torch.sum(loss_1d) / n_non_ignore)
    )

    return loss, grad_input


def parallel_cross_entropy_backward(
    grad_input: torch.Tensor, grad_output: torch.Tensor, is_cg_capturable: bool = False
):
    """Scale the saved FP32 gradient without writing to the caller's logits."""
    import triton

    from transformer_engine.common.triton.cross_entropy import element_mul_kernel

    num_warps = _get_num_warps()

    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
    # Only check torch.equal when not in CUDA graph capturable mode
    if not is_cg_capturable and torch.equal(
        grad_output, torch.tensor(1.0, device=grad_output.device)
    ):
        pass
    else:
        B, SQ, V = grad_input.shape
        n_rows = B * SQ
        BLOCK_SIZE = min(_MAX_FUSED_SIZE, triton.next_power_of_2(V))

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output.contiguous(),
            1 if grad_output.numel() > 1 else 0,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return grad_input


@lru_cache(maxsize=None)
def _is_supported_hygon_device(device_index: int) -> bool:
    properties = torch.cuda.get_device_properties(device_index)
    return (
        properties.name.upper().startswith("BW11")
        and getattr(properties, "gcnArchName", "").split(":")[0] == "gfx938"
    )


def _supports_hygon_cross_entropy(inp, target, *args, **kwargs) -> bool:
    """Admit only the measured runtime/devices and compatible input tensors."""
    if (
        not getattr(torch.version, "hip", None)
        or not isinstance(inp, torch.Tensor)
        or not inp.is_cuda
        or inp.ndim != 3
        or inp.dtype not in (torch.bfloat16, torch.float32)
        or not isinstance(target, torch.Tensor)
        or target.device != inp.device
    ):
        return False
    return _is_supported_hygon_device(inp.device.index)


@override(
    "transformer_engine.pytorch.cross_entropy.CrossEntropyFunction",
    vendor="HYGON",
    is_available=is_parallel_cross_entropy_available,
    supports=_supports_hygon_cross_entropy,
)
class CrossEntropyFunctionHygon(torch.autograd.Function):
    """Keep the FP32 intermediate gradient and its backward in one autograd node."""

    @staticmethod
    def forward(
        ctx,
        inp,
        target,
        label_smoothing=0.0,
        reduce_loss=False,
        dist_process_group=None,
        ignore_idx=-100,
        is_cg_capturable=False,
    ):
        """Preserve logits and save the FP32 intermediate gradient."""
        loss, grad_input = parallel_cross_entropy_forward(
            inp, target, label_smoothing, reduce_loss, dist_process_group, ignore_idx
        )
        ctx.save_for_backward(grad_input.detach())
        ctx.is_cg_capturable = is_cg_capturable
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """Run the backward belonging to the selected Hygon Function."""
        (grad_input,) = ctx.saved_tensors
        grad_input = parallel_cross_entropy_backward(grad_input, grad_output, ctx.is_cg_capturable)
        return grad_input, None, None, None, None, None, None
