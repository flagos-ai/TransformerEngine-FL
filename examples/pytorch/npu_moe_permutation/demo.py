# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Minimal Ascend demo for TE-FL's public MoE permutation API."""

from __future__ import annotations

import argparse

import torch
import torch_npu  # noqa: F401

from transformer_engine.pytorch.permutation import (
    moe_permute_with_probs,
    moe_sort_chunks_by_index_with_probs,
    moe_unpermute,
)


def run_mask_map_demo(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    device: str,
) -> None:
    token_ids = torch.arange(num_tokens, device=device).unsqueeze(1)
    expert_offsets = torch.arange(topk, device=device).unsqueeze(0)
    expert_ids = (token_ids + expert_offsets) % num_experts
    routing_map = torch.zeros(
        num_tokens, num_experts, dtype=torch.bool, device=device
    ).scatter_(1, expert_ids, True)

    tokens = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    probs = torch.rand(
        num_tokens,
        num_experts,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    permuted, permuted_probs, row_id_map = moe_permute_with_probs(
        tokens,
        probs,
        routing_map,
        num_out_tokens=num_tokens * topk,
    )
    restored = moe_unpermute(
        permuted,
        row_id_map,
        merging_probs=permuted_probs,
        restore_shape=tokens.shape,
        map_type="mask",
    )

    expected = tokens * (probs * routing_map).sum(dim=-1, keepdim=True)
    torch.testing.assert_close(restored, expected, rtol=1e-5, atol=1e-5)
    restored.float().sum().backward()
    assert tokens.grad is not None
    assert probs.grad is not None
    print(
        "PASS mask-map permute/unpermute "
        f"tokens={num_tokens} experts={num_experts} topk={topk}"
    )


def run_chunk_sort_demo(
    num_tokens: int,
    hidden_size: int,
    num_splits: int,
    device: str,
) -> None:
    num_splits = min(num_splits, num_tokens)
    base, remainder = divmod(num_tokens, num_splits)
    split_sizes_cpu = torch.full((num_splits,), base, dtype=torch.int32)
    split_sizes_cpu[:remainder] += 1
    sorted_index_cpu = torch.arange(num_splits, dtype=torch.int32).roll(
        max(1, num_splits // 7)
    )
    split_sizes = split_sizes_cpu.to(device)
    sorted_index = sorted_index_cpu.to(device)

    tokens = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    probs = torch.randn(
        num_tokens, dtype=torch.float32, device=device, requires_grad=True
    )

    actual, actual_probs = moe_sort_chunks_by_index_with_probs(
        tokens, probs, split_sizes, sorted_index
    )

    sizes = split_sizes_cpu.tolist()
    order = sorted_index_cpu.tolist()
    token_chunks = torch.split(tokens, sizes, dim=0)
    prob_chunks = torch.split(probs, sizes, dim=0)
    expected = torch.cat([token_chunks[index] for index in order], dim=0)
    expected_probs = torch.cat([prob_chunks[index] for index in order], dim=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=0, atol=0)

    (actual.float().sum() + actual_probs.sum()).backward()
    assert tokens.grad is not None
    assert probs.grad is not None
    print(
        "PASS chunk-sort "
        f"tokens={num_tokens} hidden={hidden_size} splits={num_splits}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--num-splits", type=int, default=256)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if args.topk > args.experts:
        parser.error("--topk must not exceed --experts")
    if args.tokens <= 0:
        parser.error("--tokens must be positive")

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    torch.manual_seed(123)

    run_mask_map_demo(
        args.tokens,
        args.hidden_size,
        args.experts,
        args.topk,
        device,
    )
    run_chunk_sort_demo(
        args.tokens * args.topk,
        args.hidden_size,
        args.num_splits,
        device,
    )
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
