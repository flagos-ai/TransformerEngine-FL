import pytest
import torch

from transformer_engine.plugin.core.backends.reference.impl.optimizer import (
    multi_tensor_compute_scale_and_scale_inv_torch,
)


@pytest.mark.parametrize(
    "amax,epsilon,pow2,expected_scale",
    [
        (8.0, 0.0, False, 56.0),
        (0.5, 1.0, False, 448.0),
        (0.0, 0.0, False, 1.0),
        (float("inf"), 0.0, False, 1.0),
        (float("nan"), 0.0, False, 1.0),
        (4.0, 0.0, True, 64.0),
    ],
)
def test_compute_scale_and_scale_inv_matches_cuda_edges(amax, epsilon, pow2, expected_scale):
    amax_tensor = torch.tensor([amax], dtype=torch.float32)
    scale = torch.full((1,), -1.0)
    scale_inv = torch.full((1,), -1.0)
    multi_tensor_compute_scale_and_scale_inv_torch(
        2048,
        torch.ones(1, dtype=torch.int32),
        [[amax_tensor], [scale], [scale_inv]],
        448.0,
        pow2,
        epsilon,
    )
    expected = torch.tensor([expected_scale], dtype=torch.float32)
    assert torch.equal(scale, expected)
    assert torch.equal(scale_inv, expected.reciprocal())


def test_compute_scale_saturates_fp32_overflow_like_cuda():
    amax = torch.tensor([torch.finfo(torch.float32).tiny])
    scale = torch.empty_like(amax)
    scale_inv = torch.empty_like(amax)
    multi_tensor_compute_scale_and_scale_inv_torch(
        2048,
        torch.zeros(1, dtype=torch.int32),
        [[amax], [scale], [scale_inv]],
        448.0,
        False,
        0.0,
    )
    assert scale.item() == torch.finfo(torch.float32).max
    assert scale_inv.item() == 0.0
