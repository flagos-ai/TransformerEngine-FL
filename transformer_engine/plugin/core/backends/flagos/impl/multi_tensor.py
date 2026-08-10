# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import List, Tuple
import torch
import flag_gems


def multi_tensor_l2_norm_fl(
    _chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    per_tensor: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute L2 norm of tensors using flag_gems.

    Returns:
        Tuple of (total_norm, per_tensor_norms_or_dummy)
        - total_norm: The combined L2 norm of all tensors
        - per_tensor_norms_or_dummy: Per-tensor norms stacked if per_tensor=True, else dummy tensor
    """
    device = tensor_lists[0][0].device if tensor_lists and tensor_lists[0] else "cpu"

    if noop_flag.item() != 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    tensors = tensor_lists[0]

    # Compute per-tensor norms
    per_tensor_norms = []
    total_norm_sq = torch.tensor(0.0, device=device)

    for tensor in tensors:
        t_float = tensor.float()
        norm_sq = flag_gems.sum(flag_gems.mul(t_float, t_float))
        # Check for inf/nan (matches CUDA behavior)
        if not torch.isfinite(norm_sq):
            noop_flag.fill_(1)
        total_norm_sq = flag_gems.add(total_norm_sq, norm_sq)
        if per_tensor:
            per_tensor_norms.append(flag_gems.sqrt(norm_sq))

    total_norm = flag_gems.sqrt(total_norm_sq)

    if per_tensor:
        per_tensor_result = torch.stack(per_tensor_norms)
    else:
        per_tensor_result = torch.tensor(0.0, device=device)

    return total_norm, per_tensor_result


def multi_tensor_scale_fl(
    _chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    scale: float,
) -> None:
    if noop_flag.item() != 0:
        return

    for src, dst in zip(tensor_lists[0], tensor_lists[1]):
        # Check for inf/nan (matches CUDA behavior for AMP gradient scaling)
        if not torch.isfinite(src).all():
            noop_flag.fill_(1)
        flag_gems.copy_(dst, flag_gems.mul(src, scale))
