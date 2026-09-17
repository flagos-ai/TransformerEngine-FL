"""Differentiable reference for dropless routing-map permutation.

Contract: expert-major, stable token order; mapping is a one-dimensional
token index. num_out_tokens must equal the number of true routing entries.
Capacity dropping, quantization padding and weighted combine are not part of
the initial public contract. Router probabilities are gathered during permute.
"""
import torch


def validate_inputs(tokens, routing_map, probs, num_out_tokens, drop_and_pad):
    if drop_and_pad:
        raise NotImplementedError("capacity dropping/padding is not supported")
    if tokens.ndim != 2 or tokens.shape[1] == 0:
        raise ValueError("tokens must have shape [T, H], H > 0")
    if tokens.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("initial support is FP32 or BF16 tokens")
    if routing_map.ndim != 2 or routing_map.shape[0] != tokens.shape[0]:
        raise ValueError("routing_map must have shape [T, E]")
    if routing_map.dtype != torch.bool or routing_map.shape[1] == 0:
        raise TypeError("routing_map must be bool with E > 0")
    if routing_map.device != tokens.device:
        raise ValueError("tokens and routing_map must share a device")
    if probs is not None:
        if probs.shape != routing_map.shape or probs.device != tokens.device:
            raise ValueError("probs must match routing_map shape and device")
        if probs.dtype != torch.float32:
            raise TypeError("initial support requires FP32 router probabilities")
    if isinstance(num_out_tokens, bool) or not isinstance(num_out_tokens, int):
        raise TypeError("num_out_tokens must be an explicit Python integer")
    if not 0 <= num_out_tokens <= routing_map.numel():
        raise ValueError("invalid num_out_tokens")


def moe_permute_with_routing_map(tokens, routing_map, probs=None,
                               num_out_tokens=None, drop_and_pad=False):
    validate_inputs(tokens, routing_map, probs, num_out_tokens, drop_and_pad)
    # This correctness oracle deliberately checks the full routing count.
    if int(routing_map.sum().item()) != num_out_tokens:
        raise ValueError("num_out_tokens must equal routing_map.sum()")
    flat = routing_map.T.contiguous().reshape(-1)
    positions = flat.argsort(descending=True, stable=True)[:num_out_tokens]
    indices = (positions % tokens.shape[0]).to(torch.int32) if tokens.shape[0] else (
        torch.empty(0, dtype=torch.int32, device=tokens.device)
    )
    output = tokens.index_select(0, indices.long())
    output_probs = None if probs is None else probs.T.contiguous().reshape(-1)[positions]
    return output, output_probs, indices


def validate_restore(tokens, indices, restore_shape, probs, drop_and_pad):
    if drop_and_pad or probs is not None:
        raise NotImplementedError("initial combine is dropless and unweighted")
    if tokens.ndim != 2 or tokens.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("permuted tokens must be FP32/BF16 [M, H]")
    if len(restore_shape) != 2 or restore_shape[0] < 0 or restore_shape[1] != tokens.shape[1]:
        raise ValueError("invalid restore_shape")
    if indices.ndim != 1 or indices.numel() != tokens.shape[0]:
        raise ValueError("mapping length must equal permuted token count")
    if indices.dtype not in (torch.int32, torch.int64) or indices.device != tokens.device:
        raise TypeError("mapping must be an integer tensor on the input device")


def moe_unpermute_with_routing_map(permuted_tokens, sorted_indices, restore_shape,
                                 probs=None, routing_map=None, drop_and_pad=False):
    validate_restore(permuted_tokens, sorted_indices, restore_shape, probs, drop_and_pad)
    output = permuted_tokens.new_zeros(tuple(restore_shape))
    return output.index_add(0, sorted_indices.long(), permuted_tokens)


def _chunk_row_maps(split_sizes, sorted_idxs, device):
    counts = split_sizes.detach().to(device="cpu", dtype=torch.long)
    order = sorted_idxs.detach().to(device="cpu", dtype=torch.long)
    n = int(counts.sum().item())
    sorted_counts = counts.index_select(0, order)
    chunk_ids = torch.repeat_interleave(order, sorted_counts, output_size=n)
    offsets = sorted_counts.cumsum(0) - sorted_counts
    within = torch.arange(n) - torch.repeat_interleave(offsets, sorted_counts, output_size=n)
    source_offsets = counts.cumsum(0) - counts
    row_map = source_offsets.index_select(0, chunk_ids) + within
    inverse = torch.empty_like(row_map)
    inverse[row_map] = torch.arange(n)
    return row_map.to(device=device), inverse.to(device=device)


def moe_sort_chunks_fwd(input, split_sizes, sorted_idxs, probs=None):
    row_map, inverse = _chunk_row_maps(split_sizes, sorted_idxs, input.device)
    output = input.index_select(0, row_map)
    output_probs = probs.index_select(0, row_map) if probs is not None else None
    return output, output_probs, inverse


def moe_sort_chunks_bwd(grad_output, grad_probs, inverse_row_map):
    grad_input = grad_output.index_select(0, inverse_row_map)
    grad_probs_input = grad_probs.index_select(0, inverse_row_map) if grad_probs is not None else None
    return grad_input, grad_probs_input
