"""NPU routing-map candidate; device parity is required before training use.

No TransformerEngineNPU high-level import or custom-op re-registration.
Token permutation uses raw torch_npu kernels. FP32 router probabilities keep
their own differentiable gather, avoiding the installed mixed-dtype wrapper.
"""
import torch
from ...reference.impl.permutation import validate_inputs, validate_restore


def _provider():
    import torch_npu
    return torch_npu


def is_permutation_available():
    try:
        provider = _provider()
        names = (
            "npu_moe_token_permute_with_routing_map",
            "npu_moe_token_permute_with_routing_map_grad",
            "_npu_moe_token_unpermute_with_routing_map",
        )
        return all(callable(getattr(provider, name, None)) for name in names)
    except (ImportError, OSError):
        return False


def _require_device(tokens):
    if tokens.device.type != "npu":
        raise ValueError("NPU backend requires NPU tokens")
    if torch.are_deterministic_algorithms_enabled():
        raise NotImplementedError("NPU deterministic accumulation has not been validated")


class _PermuteTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, routing_map, num_out_tokens):
        output, _, indices = _provider().npu_moe_token_permute_with_routing_map(
            tokens.contiguous(), routing_map.contiguous(), probs=None,
            num_out_tokens=num_out_tokens, drop_and_pad=False,
        )
        ctx.save_for_backward(indices, routing_map)
        ctx.tokens_num, ctx.experts_num = routing_map.shape
        ctx.mark_non_differentiable(indices)
        return output, indices

    @staticmethod
    def backward(ctx, grad_output, _grad_indices):
        indices, routing_map = ctx.saved_tensors
        grad_tokens, _ = _provider().npu_moe_token_permute_with_routing_map_grad(
            grad_output.contiguous(), None, indices, routing_map,
            ctx.experts_num, ctx.tokens_num, False,
        )
        return grad_tokens, None, None


class _UnpermuteTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, indices, restore_shape, routing_map):
        output, _, _, _ = _provider()._npu_moe_token_unpermute_with_routing_map(
            tokens.contiguous(), indices, list(restore_shape), probs=None,
            routing_map=routing_map, drop_and_pad=False,
        )
        # The installed no-probs forward does not initialize auxiliary maps.
        # For unweighted combine the adjoint is exactly the original permute.
        ctx.save_for_backward(routing_map)
        ctx.num_out_tokens = tokens.shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (routing_map,) = ctx.saved_tensors
        grad_tokens, _, _ = _provider().npu_moe_token_permute_with_routing_map(
            grad_output.contiguous(), routing_map.contiguous(), probs=None,
            num_out_tokens=ctx.num_out_tokens, drop_and_pad=False,
        )
        return grad_tokens, None, None, None


def moe_permute_with_routing_map(tokens, routing_map, probs=None,
                               num_out_tokens=None, drop_and_pad=False):
    validate_inputs(tokens, routing_map, probs, num_out_tokens, drop_and_pad)
    _require_device(tokens)
    # The installed operator's meta rounds M by T. Restrict the candidate to
    # dropless fixed-topk shapes; the caller must supply the true routing count.
    if tokens.shape[0] and num_out_tokens % tokens.shape[0]:
        raise NotImplementedError("native candidate requires fixed-topk output count")
    if not num_out_tokens:
        indices = torch.empty(0, dtype=torch.int32, device=tokens.device)
        output = tokens[:0]
        output_probs = None if probs is None else probs.reshape(-1)[:0]
        return output, output_probs, indices
    output, indices = _PermuteTokens.apply(tokens, routing_map, num_out_tokens)
    output_probs = None
    if probs is not None:
        output_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        if output_probs.numel() != num_out_tokens:
            # Fail, never retry a different backend after a submitted kernel.
            raise ValueError("routing count differs from num_out_tokens")
    return output, output_probs, indices


def moe_unpermute_with_routing_map(permuted_tokens, sorted_indices, restore_shape,
                                 probs=None, routing_map=None, drop_and_pad=False):
    validate_restore(permuted_tokens, sorted_indices, restore_shape, probs, drop_and_pad)
    _require_device(permuted_tokens)
    if routing_map is None:
        raise ValueError("NPU unweighted combine requires its original routing_map")
    if routing_map is not None:
        if routing_map.dtype != torch.bool or routing_map.ndim != 2:
            raise TypeError("routing_map must be bool [T, E]")
        if routing_map.shape[0] != restore_shape[0] or routing_map.device != permuted_tokens.device:
            raise ValueError("routing_map must match restored tokens/device")
    if not permuted_tokens.shape[0]:
        # Correct restore shape and a connected empty token gradient.
        return permuted_tokens.new_zeros(tuple(restore_shape)) + permuted_tokens.sum() * 0
    return _UnpermuteTokens.apply(permuted_tokens, sorted_indices, tuple(restore_shape), routing_map)


def _validate_chunk_metadata(input, split_sizes, sorted_idxs, probs):
    if split_sizes.ndim != 1 or sorted_idxs.ndim != 1:
        raise ValueError("split_sizes and sorted_idxs must be one-dimensional")
    if split_sizes.numel() != sorted_idxs.numel():
        raise ValueError("sorted_idxs must contain exactly one entry for each chunk")
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if split_sizes.dtype not in integer_dtypes or sorted_idxs.dtype not in integer_dtypes:
        raise TypeError("split_sizes and sorted_idxs must have integer dtype")
    counts = split_sizes.detach().to(device="cpu", dtype=torch.long)
    chunk_order = sorted_idxs.detach().to(device="cpu", dtype=torch.long)
    if torch.any(counts < 0):
        raise ValueError("split_sizes must be non-negative")
    if int(counts.sum().item()) != input.shape[0]:
        raise ValueError("sum(split_sizes) must equal input.shape[0]")
    if probs is not None and probs.shape[0] != input.shape[0]:
        raise ValueError("probs and input must have the same leading dimension")
    expected = torch.arange(counts.numel(), dtype=torch.long)
    if counts.numel() and not torch.equal(torch.sort(chunk_order).values, expected):
        raise ValueError("sorted_idxs must be a permutation of all chunk indices")
    return counts, chunk_order


def _build_chunk_row_maps(counts, chunk_order, device):
    num_rows = int(counts.sum().item())
    if num_rows == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    sorted_counts = counts.index_select(0, chunk_order)
    chunk_ids = torch.repeat_interleave(chunk_order, sorted_counts, output_size=num_rows)
    sorted_offsets = sorted_counts.cumsum(0) - sorted_counts
    offsets_within = torch.arange(num_rows) - torch.repeat_interleave(
        sorted_offsets, sorted_counts, output_size=num_rows
    )
    source_offsets = counts.cumsum(0) - counts
    row_permutation = source_offsets.index_select(0, chunk_ids) + offsets_within
    inverse = torch.empty_like(row_permutation)
    inverse[row_permutation] = torch.arange(num_rows)
    return row_permutation.to(device=device, non_blocking=True), inverse.to(device=device, non_blocking=True)


def _tefl_chunk_sort_fwd(input, split_sizes, sorted_idxs, probs=None):
    """NPU kernel for ``te_moe::chunk_sort_fwd`` using patch3 row indices."""
    _require_device(input)
    if split_sizes.device.type != "cpu" or sorted_idxs.device.type != "cpu":
        # Metadata is intentionally host-side: it is tiny and avoids 512-way
        # device split/list/cat operations.
        split_sizes = split_sizes.to(device="cpu")
        sorted_idxs = sorted_idxs.to(device="cpu")
    counts, chunk_order = _validate_chunk_metadata(input, split_sizes, sorted_idxs, probs)
    row_map, inverse = _build_chunk_row_maps(counts, chunk_order, input.device)
    output = input.index_select(0, row_map)
    permuted_probs = (
        probs.index_select(0, row_map)
        if probs is not None
        else torch.empty(0, device=input.device)
    )
    return output, permuted_probs, inverse


def _tefl_chunk_sort_bwd(
    grad_output, grad_probs, inverse_row_map, _num_tokens, _hidden_size
):
    """NPU kernel for ``te_moe::chunk_sort_bwd`` using inverse row indices."""
    _require_device(grad_output)
    grad_input = grad_output.index_select(0, inverse_row_map)
    grad_probs_input = (
        grad_probs.index_select(0, inverse_row_map)
        if grad_probs is not None and grad_probs.numel()
        else torch.empty(0, device=grad_output.device)
    )
    return grad_input, grad_probs_input


_CHUNK_SORT_KERNELS_REGISTERED = False


def ensure_chunk_sort_kernels_registered():
    """Register patch4 kernels after TE's custom-op schemas are available."""
    global _CHUNK_SORT_KERNELS_REGISTERED
    if _CHUNK_SORT_KERNELS_REGISTERED:
        return

    # Import lazily: importing this during Vendor registration would create a
    # circular import through transformer_engine.pytorch.
