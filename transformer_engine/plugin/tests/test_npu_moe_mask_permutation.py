"""NPU tests for the fused mask-map MoE permutation path."""

from typing import Optional

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")

if not torch.npu.is_available():
    pytest.skip("NPU is not available", allow_module_level=True)

from transformer_engine.pytorch.permutation import (  # noqa: E402
    moe_permute,
    moe_permute_with_probs,
    moe_sort_chunks_by_index_with_probs,
    moe_unpermute,
)


def _routing_map(num_tokens: int, num_experts: int, topk: int) -> torch.Tensor:
    scores = torch.rand((num_tokens, num_experts), device="npu")
    expert_indices = torch.topk(scores, topk, dim=1).indices
    return torch.zeros(
        (num_tokens, num_experts), dtype=torch.bool, device="npu"
    ).scatter_(1, expert_indices, True)


def _reference_permute(
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
):
    num_tokens = tokens.size(0)
    num_out_tokens = int(routing_map.sum().item())
    flat_sorted = (
        routing_map.T.contiguous()
        .reshape(-1)
        .argsort(descending=True, stable=True)[:num_out_tokens]
    )
    token_indices = flat_sorted % num_tokens
    permuted_probs = None
    if probs is not None:
        permuted_probs = probs.T.contiguous().reshape(-1)[flat_sorted]
    return tokens.index_select(0, token_indices), permuted_probs, token_indices


def _reference_unpermute(
    permuted_tokens: torch.Tensor,
    token_indices: torch.Tensor,
    restore_shape: torch.Size,
    merging_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if merging_probs is not None:
        permuted_tokens = permuted_tokens * merging_probs.unsqueeze(-1)
    output = torch.zeros(
        restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
    )
    return output.index_add(0, token_indices, permuted_tokens)


@pytest.mark.parametrize("with_probs", [False, True])
def test_npu_mask_map_forward_backward(with_probs):
    """The fused NPU path matches the unfused stable-argsort semantics and gradients."""
    num_tokens, num_experts, topk, hidden_size = 256, 64, 4, 128
    num_out_tokens = num_tokens * topk
    routing_map = _routing_map(num_tokens, num_experts, topk)

    ref_tokens = torch.randn(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device="npu", requires_grad=True
    )
    test_tokens = ref_tokens.detach().clone().requires_grad_(True)

    ref_probs = None
    test_probs = None
    if with_probs:
        ref_probs = torch.rand(
            (num_tokens, num_experts),
            dtype=torch.float32,
            device="npu",
            requires_grad=True,
        )
        test_probs = ref_probs.detach().clone().requires_grad_(True)

    ref_permuted, ref_permuted_probs, token_indices = _reference_permute(
        ref_tokens, routing_map, ref_probs
    )
    if with_probs:
        test_permuted, test_permuted_probs, row_id_map = moe_permute_with_probs(
            test_tokens, test_probs, routing_map, num_out_tokens=num_out_tokens
        )
    else:
        test_permuted, row_id_map = moe_permute(
            test_tokens,
            routing_map,
            num_out_tokens=num_out_tokens,
            map_type="mask",
        )
        test_permuted_probs = None

    assert row_id_map.dtype == torch.int32
    assert row_id_map.ndim == 1
    torch.testing.assert_close(test_permuted, ref_permuted, rtol=0, atol=0)
    if with_probs:
        assert test_permuted_probs.dtype == torch.float32
        torch.testing.assert_close(test_permuted_probs, ref_permuted_probs, rtol=0, atol=0)

    ref_output = _reference_unpermute(
        ref_permuted,
        token_indices,
        ref_tokens.shape,
        ref_permuted_probs,
    )
    test_output = moe_unpermute(
        test_permuted,
        row_id_map,
        merging_probs=test_permuted_probs,
        restore_shape=test_tokens.shape,
        map_type="mask",
    )
    torch.testing.assert_close(test_output, ref_output, rtol=1e-5, atol=1e-5)

    output_grad = torch.randn_like(ref_output)
    ref_output.backward(output_grad)
    test_output.backward(output_grad)
    torch.testing.assert_close(test_tokens.grad, ref_tokens.grad, rtol=0, atol=0)
    if with_probs:
        torch.testing.assert_close(test_probs.grad, ref_probs.grad, rtol=1e-5, atol=1e-5)


def test_npu_chunk_sort_large_forward_backward():
    """The facade selects TENPU chunk-sort for token counts above 65535."""
    num_tokens, hidden_size, num_splits = 131181, 16, 256
    base, remainder = divmod(num_tokens, num_splits)
    split_sizes_cpu = torch.full((num_splits,), base, dtype=torch.int32)
    split_sizes_cpu[:remainder] += 1
    sorted_index_cpu = torch.arange(num_splits, dtype=torch.int32).roll(
        num_splits // 7
    )
    split_sizes = split_sizes_cpu.npu()
    sorted_index = sorted_index_cpu.npu()

    inp = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device="npu",
        requires_grad=True,
    )
    probs = torch.randn(
        num_tokens, dtype=torch.float32, device="npu", requires_grad=True
    )
    ref_inp = inp.detach().clone().requires_grad_(True)
    ref_probs = probs.detach().clone().requires_grad_(True)

    sizes = split_sizes_cpu.tolist()
    order = sorted_index_cpu.tolist()
    inp_chunks = torch.split(ref_inp, sizes, dim=0)
    prob_chunks = torch.split(ref_probs, sizes, dim=0)
    expected = torch.cat([inp_chunks[index] for index in order], dim=0)
    expected_probs = torch.cat([prob_chunks[index] for index in order], dim=0)

    actual, actual_probs = moe_sort_chunks_by_index_with_probs(
        inp, probs, split_sizes, sorted_index
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=0, atol=0)

    grad_output = torch.randn_like(actual)
    grad_probs = torch.randn_like(actual_probs)
    torch.autograd.backward((actual, actual_probs), (grad_output, grad_probs))
    torch.autograd.backward((expected, expected_probs), (grad_output, grad_probs))
    torch.testing.assert_close(inp.grad, ref_inp.grad, rtol=0, atol=0)
    torch.testing.assert_close(probs.grad, ref_probs.grad, rtol=0, atol=0)
