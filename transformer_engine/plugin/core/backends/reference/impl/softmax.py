# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Optional
import torch
import torch.nn.functional as F

__all__ = [
    "scaled_softmax_forward_torch",
    "scaled_softmax_backward_torch",
    "scaled_masked_softmax_forward_torch",
    "scaled_masked_softmax_backward_torch",
    "scaled_upper_triang_masked_softmax_forward_torch",
    "scaled_upper_triang_masked_softmax_backward_torch",
    "scaled_aligned_causal_masked_softmax_forward_torch",
    "scaled_aligned_causal_masked_softmax_backward_torch",
]


def scaled_softmax_forward_torch(input: torch.Tensor, scale: float) -> torch.Tensor:
    return F.softmax(input * scale, dim=-1)


def scaled_softmax_backward_torch(
    output_grad: torch.Tensor,
    softmax_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    # Compute in float32 for numerical stability (matching CUDA behavior)
    orig_dtype = output_grad.dtype
    output_grad_f32 = output_grad.float()
    softmax_output_f32 = softmax_output.float()

    grad_softmax = softmax_output_f32 * (
        output_grad_f32 - (softmax_output_f32 * output_grad_f32).sum(dim=-1, keepdim=True)
    )

    return (grad_softmax * scale).to(orig_dtype)


def scaled_masked_softmax_forward_torch(
    input: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference forward matching TE CUDA `scaled_masked_softmax_warp_forward`.

    Integer/bool mask (same as uint8 kernel contract):
      - **Exactly** ``mask == 1`` means **masked** (logit set to ``-10000``, not ``input*scale`` offset).
      - Any other value (typically 0) means **unmasked** (logit is ``input * scale``).

    Floating mask: treated as **additive** bias in logit space (already scaled), added after
    ``input * scale``.

    Common pitfalls this avoids vs the old implementation:
    1) ``input * scale + (-10000)`` on masked positions ≠ CUDA's plain ``-10000``.
    2) Non-uint8 masks (bool, int) were used as direct addends → wrong (0/1 added to logits).
    3) ``mask.bool()`` masks any nonzero byte; CUDA only masks when ``mask == 1``.
    """
    if mask.dim() == 4 and mask.size(1) == 1 and input.dim() == 4:
        mask = mask.expand_as(input)

    scaled = input * scale

    if mask.is_floating_point():
        scaled = scaled + mask.to(dtype=scaled.dtype)
        return F.softmax(scaled, dim=-1)

    # Integer / bool: align with CUDA (masked iff value == 1)
    scaled = scaled.masked_fill(mask == 1, -10000.0)
    # CUDA zeros output row when every position in the softmax dim is masked (max == -10000)
    all_masked = (mask == 1).all(dim=-1, keepdim=True)
    out = F.softmax(scaled, dim=-1)
    return out.masked_fill(all_masked, 0.0)


def scaled_masked_softmax_backward_torch(
    output_grad: torch.Tensor,
    softmax_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    # Compute in float32 for numerical stability (matching CUDA behavior)
    orig_dtype = output_grad.dtype
    output_grad_f32 = output_grad.float()
    softmax_output_f32 = softmax_output.float()

    grad_softmax = softmax_output_f32 * (
        output_grad_f32 - (softmax_output_f32 * output_grad_f32).sum(dim=-1, keepdim=True)
    )

    return (grad_softmax * scale).to(orig_dtype)


def scaled_upper_triang_masked_softmax_forward_torch(
    input: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    seq_len = input.size(-1)

    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=input.device, dtype=input.dtype),
        diagonal=1,
    )

    scaled_input = input * scale + causal_mask

    return F.softmax(scaled_input, dim=-1)


def scaled_upper_triang_masked_softmax_backward_torch(
    output_grad: torch.Tensor,
    softmax_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    # Compute in float32 for numerical stability (matching CUDA behavior)
    orig_dtype = output_grad.dtype
    output_grad_f32 = output_grad.float()
    softmax_output_f32 = softmax_output.float()

    grad_softmax = softmax_output_f32 * (
        output_grad_f32 - (softmax_output_f32 * output_grad_f32).sum(dim=-1, keepdim=True)
    )

    return (grad_softmax * scale).to(orig_dtype)


def scaled_aligned_causal_masked_softmax_forward_torch(
    input: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return scaled_upper_triang_masked_softmax_forward_torch(input, scale)


def scaled_aligned_causal_masked_softmax_backward_torch(
    output_grad: torch.Tensor,
    softmax_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    # Compute in float32 for numerical stability (matching CUDA behavior)
    orig_dtype = output_grad.dtype
    output_grad_f32 = output_grad.float()
    softmax_output_f32 = softmax_output.float()

    grad_softmax = softmax_output_f32 * (
        output_grad_f32 - (softmax_output_f32 * output_grad_f32).sum(dim=-1, keepdim=True)
    )

    return (grad_softmax * scale).to(orig_dtype)
