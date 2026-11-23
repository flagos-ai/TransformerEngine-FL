"""Dropless, fixed-topk permutation with paired TorchNPU routing-map indices."""

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None


_NATIVE_APIS = (
    "npu_moe_token_permute_with_routing_map",
    "npu_moe_token_permute_with_routing_map_grad",
    "npu_moe_token_unpermute_with_routing_map",
    "npu_moe_token_unpermute_with_routing_map_grad",
)


def is_available():
    """Require both endpoints and their official gradient APIs."""
    return torch_npu is not None and all(
        callable(getattr(torch_npu, op, None)) for op in _NATIVE_APIS
    )


def supports_inputs(tokens, routing_map, probs, num_out_tokens, topk):
    """Check metadata; the caller guarantees exactly topk routes per token."""
    if torch.compiler.is_compiling() or tokens.device.type != "npu":
        return False
    if tokens.ndim != 2 or tokens.dtype not in (torch.bfloat16, torch.float32):
        return False
    if (
        routing_map.ndim != 2
        or routing_map.dtype != torch.bool
        or routing_map.device != tokens.device
        or routing_map.shape[0] != tokens.shape[0]
    ):
        return False
    num_tokens, hidden_size = tokens.shape
    num_experts = routing_map.shape[1]
    # Native non-pad indices are int32, with documented 24-bit N/E and k < 512.
    if not (0 < num_tokens < 16777215 and 0 < num_experts < 16777215 and hidden_size > 0):
        return False
    if not (type(topk) is int and 0 < topk < 512 and topk <= num_experts):
        return False
    if type(num_out_tokens) is not int or num_out_tokens != num_tokens * topk:
        return False
    if num_out_tokens > torch.iinfo(torch.int32).max:
        return False
    return probs is None or (
        probs.shape == routing_map.shape
        and probs.device == tokens.device
        and probs.dtype == torch.float32
    )


class _Permute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, routing_map, probs, num_out_tokens):
        output, permuted_probs, indices = torch_npu.npu_moe_token_permute_with_routing_map(
            tokens, routing_map, probs=probs, num_out_tokens=num_out_tokens, drop_and_pad=False
        )
        ctx.save_for_backward(indices, routing_map)
        ctx.mark_non_differentiable(indices)
        ctx.set_materialize_grads(False)
        return output, permuted_probs, indices

    @staticmethod
    def backward(ctx, grad_tokens, grad_probs, _grad_indices):
        indices, routing_map = ctx.saved_tensors
        num_tokens, num_experts = routing_map.shape
        tokens_grad = probs_grad = None
        if ctx.needs_input_grad[0] and grad_tokens is not None:
            tokens_grad, _ = torch_npu.npu_moe_token_permute_with_routing_map_grad(
                grad_tokens.contiguous(), None, indices, routing_map, num_experts, num_tokens, False
            )
        if ctx.needs_input_grad[2] and grad_probs is not None:
            # Keep FP32 router gradients independent of the BF16 token-gradient kernel.
            probs_grad = grad_probs.new_zeros((num_experts, num_tokens))
            probs_grad.masked_scatter_(routing_map.T, grad_probs.contiguous())
            probs_grad = probs_grad.T
        return tokens_grad, None, probs_grad, None


class _Unpermute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, indices, restore_shape, routing_map):
        output = torch_npu.npu_moe_token_unpermute_with_routing_map(
            tokens,
            indices,
            list(restore_shape),
            probs=None,
            routing_map=routing_map,
            drop_and_pad=False,
        )
        # The public native autograd may save tokens even without merge weights.
        ctx.save_for_backward(indices, routing_map)
        ctx.restore_shape = restore_shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices, routing_map = ctx.saved_tensors
        # TorchNPU's non-padding derivative uses sorted_indices for both index arguments.
        tokens_grad, _ = torch_npu.npu_moe_token_unpermute_with_routing_map_grad(
            grad_output.contiguous(),
            indices,
            indices,
            routing_map,
            None,
            None,
            False,
            list(ctx.restore_shape),
        )
        return tokens_grad, None, None, None


def permute(tokens, routing_map, probs, num_out_tokens):
    """Return expert-major tokens/probabilities and paired native int32 indices."""
    return _Permute.apply(
        tokens.contiguous(),
        routing_map.contiguous(),
        probs.contiguous() if probs is not None else None,
        num_out_tokens,
    )


def unpermute(tokens, indices, restore_shape, routing_map):
    """Sum routed rows without merge weights or saving the source activation."""
    return _Unpermute.apply(
        tokens.contiguous(), indices, tuple(restore_shape), routing_map.contiguous()
    )


permute._supports_inputs = supports_inputs
