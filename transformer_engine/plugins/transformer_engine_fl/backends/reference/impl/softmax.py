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
    grad_softmax = softmax_output * (
        output_grad - (softmax_output * output_grad).sum(dim=-1, keepdim=True)
    )

    return grad_softmax * scale


def scaled_masked_softmax_forward_torch(
    input: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    scaled_input = input * scale + mask

    return F.softmax(scaled_input, dim=-1)


def scaled_masked_softmax_backward_torch(
    output_grad: torch.Tensor,
    softmax_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    grad_softmax = softmax_output * (
        output_grad - (softmax_output * output_grad).sum(dim=-1, keepdim=True)
    )

    return grad_softmax * scale


def scaled_upper_triang_masked_softmax_forward_torch(
    input: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    seq_len = input.size(-1)

    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=input.device, dtype=input.dtype),
        diagonal=1
    )

    scaled_input = input * scale + causal_mask

    return F.softmax(scaled_input, dim=-1)


def scaled_upper_triang_masked_softmax_backward_torch(
    output_grad: torch.Tensor,
    softmax_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    grad_softmax = softmax_output * (
        output_grad - (softmax_output * output_grad).sum(dim=-1, keepdim=True)
    )

    return grad_softmax * scale


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
    grad_softmax = softmax_output * (
        output_grad - (softmax_output * output_grad).sum(dim=-1, keepdim=True)
    )

    return grad_softmax * scale
