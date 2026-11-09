# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Regression tests for Ascend MoE permutation launch-grid limits."""

import pytest
import torch


@pytest.fixture
def permutation():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    from transformer_engine.pytorch.triton import permutation

    from transformer_engine.plugin.core.manager import get_default_manager
    from transformer_engine.plugin.core.policy import SelectionPolicy, policy_context
    from transformer_engine.plugin.core.triton_dispatch.entry import ensure_native_registered

    manager = get_default_manager()
    ensure_native_registered(manager)
    op_names = (
        "triton_permute_with_mask_map",
        "triton_unpermute_with_mask_map",
        "triton_make_chunk_sort_map",
        "triton_sort_chunks_by_map",
    )
    policy = SelectionPolicy.from_dict(
        strict=True, per_op_order={op_name: ["impl:vendor.npu"] for op_name in op_names}
    )
    with policy_context(policy):
        for op_name in op_names:
            assert manager.get_selected_impl_id(op_name) == "vendor.npu"
        yield permutation


def test_permute_unpermute_over_core_dim(permutation):
    # A 64-column autotune candidate needs 16385 * 4 programs before capping.
    tokens, hidden = 16385, 256
    rows = torch.arange(tokens, dtype=torch.int32)
    inp = rows.to(torch.float32)[:, None].expand(tokens, hidden).contiguous()
    mapping = torch.stack((tokens - 1 - rows, torch.zeros_like(rows), torch.ones_like(rows)), 1)
    probs = torch.full((tokens, 1), 0.5)

    out, _, weights = permutation.permute_with_mask_map(
        inp.npu(), mapping.npu(), probs.npu(), None, None, tokens, 1, tokens, hidden, 0
    )
    restored, restored_probs = permutation.unpermute_with_mask_map(
        out, mapping.npu(), None, weights, None, tokens, 1, hidden
    )

    torch.testing.assert_close(out.cpu(), inp.flip(0), rtol=0, atol=0)
    torch.testing.assert_close(restored.cpu(), inp, rtol=0, atol=0)
    torch.testing.assert_close(restored_probs.cpu(), probs, rtol=0, atol=0)


def test_chunk_sort_over_core_dim(permutation):
    # Even a single hidden tile would exceed the original launch limit.
    tokens, hidden = 65537, 128
    split = 32768
    split_sizes = torch.tensor([split, tokens - split], dtype=torch.int32, device="npu")
    order = torch.tensor([1, 0], dtype=torch.int32, device="npu")
    inp = torch.arange(tokens, dtype=torch.float32)[:, None].expand(tokens, hidden).contiguous()
    probs = torch.arange(tokens, dtype=torch.float32)

    mapping = permutation.make_chunk_sort_map(split_sizes, order, tokens, 2)
    out, out_probs = permutation.sort_chunks_by_map(
        inp.npu(), mapping, probs.npu(), tokens, hidden, is_forward=True
    )
    restored, restored_probs = permutation.sort_chunks_by_map(
        out, mapping, out_probs, tokens, hidden, is_forward=False
    )

    torch.testing.assert_close(out.cpu(), torch.cat((inp[split:], inp[:split])), rtol=0, atol=0)
    torch.testing.assert_close(
        out_probs.cpu(), torch.cat((probs[split:], probs[:split])), rtol=0, atol=0
    )
    torch.testing.assert_close(restored.cpu(), inp, rtol=0, atol=0)
    torch.testing.assert_close(restored_probs.cpu(), probs, rtol=0, atol=0)
