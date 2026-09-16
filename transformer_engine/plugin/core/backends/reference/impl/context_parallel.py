# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import torch


def thd_get_partitioned_indices_torch(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    world_size: int,
    rank: int,
) -> torch.Tensor:
    """Generate the load-balanced THD token indices for one CP rank."""
    if cu_seqlens.dtype != torch.int32:
        raise TypeError("cu_seqlens must have dtype torch.int32")
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a 1-D tensor with at least two entries")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if total_tokens <= 0 or total_tokens % (2 * world_size) != 0:
        raise ValueError("total_tokens must be positive and divisible by 2 * world_size")

    boundaries = cu_seqlens.detach().cpu().tolist()
    if boundaries[0] != 0 or boundaries[-1] != total_tokens:
        raise ValueError("cu_seqlens must start at 0 and end at total_tokens")

    mirrored_rank = 2 * world_size - rank - 1
    partitions = []
    for start, end in zip(boundaries, boundaries[1:]):
        sequence_length = end - start
        if sequence_length < 0 or sequence_length % (2 * world_size) != 0:
            raise ValueError("each packed sequence length must be divisible by 2 * world_size")

        chunk_size = sequence_length // (2 * world_size)
        for chunk_rank in (rank, mirrored_rank):
            chunk_start = start + chunk_rank * chunk_size
            partitions.append(
                torch.arange(
                    chunk_start,
                    chunk_start + chunk_size,
                    dtype=torch.int32,
                    device=cu_seqlens.device,
                )
            )

    if not partitions:
        return torch.empty(0, dtype=torch.int32, device=cu_seqlens.device)
    return torch.cat(partitions)
