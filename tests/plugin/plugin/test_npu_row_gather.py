# Copyright (c) 2026, BAAI. All rights reserved.
# See LICENSE for license information.

"""Check gather tiling through chunk autograd and sparse mask-map semantics."""

import numpy as np
import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="NPU is required")

from transformer_engine.pytorch.permutation import (
    moe_sort_chunks_by_index,
    moe_sort_chunks_by_index_with_probs,
)
from transformer_engine.plugin.core.backends.vendor.npu.triton_permutation import (
    _launch_row_gather,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_probs", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_chunk_gather_outputs_and_autograd(dtype, with_probs, strided):
    # Nonuniform chunks include empty chunks and a final partial row tile.
    sizes = [17, 0, 1024, 3, 2048, 0, 1005]
    order = [4, 1, 6, 0, 5, 3, 2]
    n, h = sum(sizes), 2048
    starts = np.cumsum([0] + sizes)
    source = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in order])
    source = torch.from_numpy(source).to(device="npu", dtype=torch.int64)
    values = torch.randint(-4, 5, (n, h + 8 if strided else h), device="npu")
    x = values.to(dtype)[:, :h].detach().requires_grad_()
    p = torch.rand(n * 2 if strided else n, device="npu")[:: 2 if strided else 1]
    p = p.detach().requires_grad_()
    splits = torch.tensor(sizes, device="npu", dtype=torch.int32)
    indices = torch.tensor(order, device="npu", dtype=torch.int32)
    if with_probs:
        actual, actual_p = moe_sort_chunks_by_index_with_probs(x, p, splits, indices)
        torch.testing.assert_close(actual_p, p.detach().index_select(0, source), rtol=0, atol=0)
    else:
        actual = moe_sort_chunks_by_index(x, splits, indices)
    torch.testing.assert_close(actual, x.detach().index_select(0, source), rtol=0, atol=0)
    gradient = torch.randint(-2, 3, actual.shape, device="npu").to(dtype)
    expected_dx = torch.empty_like(x)
    expected_dx.index_copy_(0, source, gradient)
    if with_probs:
        dp = torch.randn_like(actual_p)
        dx, actual_dp = torch.autograd.grad((actual, actual_p), (x, p), (gradient, dp))
        expected_dp = torch.empty_like(p)
        expected_dp.index_copy_(0, source, dp)
        torch.testing.assert_close(actual_dp, expected_dp, rtol=0, atol=0)
    else:
        dx = torch.autograd.grad(actual, x, gradient)[0]
    torch.testing.assert_close(dx, expected_dx, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_probs", [False, True])
def test_mask_gather_preserves_unmapped_rows_and_zero_probabilities(dtype, with_probs):
    t, e, n, h = 1025, 19, 8197, 2048
    x = torch.randn((t, h + 8), device="npu", dtype=dtype)[:, :h]
    source = (torch.arange(n, device="npu", dtype=torch.int32) * 37) % t
    source[::113] = -1
    expert = torch.arange(n, device="npu", dtype=torch.int32) % e
    probs = torch.rand((t, e * 2), device="npu")[:, ::2] if with_probs else None
    if with_probs:
        probs[:, ::3] = 0
    output = torch.full((n, h + 8), -37, device="npu", dtype=dtype)[:, :h]
    output_p = (
        torch.full((n * 2,), -37, device="npu", dtype=torch.float32)[::2] if with_probs else None
    )
    _launch_row_gather(x, source, expert, probs, output, output_p, n, h, mask_map=True)
    safe = source.clamp(min=0).long()
    expected = x.index_select(0, safe)
    valid = source >= 0
    if with_probs:
        expected_p = probs[safe, expert.long()]
        expected = torch.where((expected_p == 0)[:, None], torch.zeros_like(expected), expected)
        expected_p = torch.where(valid, expected_p, torch.full_like(expected_p, -37))
        torch.testing.assert_close(output_p, expected_p, rtol=0, atol=0)
    expected = torch.where(valid[:, None], expected, torch.full_like(expected, -37))
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "n,h,dtype",
    [
        (257, 2048, torch.bfloat16),
        (4097, 65, torch.float16),
        (4097, 2048, torch.float32),
    ],
)
def test_other_gather_shapes_keep_working(n, h, dtype):
    x = torch.randn((n, h), device="npu", dtype=dtype)
    source = torch.arange(n - 1, -1, -1, device="npu", dtype=torch.int32)
    actual = torch.empty_like(x)
    _launch_row_gather(x, source, None, None, actual, None, n, h, mask_map=False)
    torch.testing.assert_close(actual, x.index_select(0, source.long()), rtol=0, atol=0)
