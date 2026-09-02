import itertools

import pytest
import torch

from transformer_engine import te_device_type

from .utils import available_implementations


OP_NAME = "multi_tensor_compute_scale_and_scale_inv"


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
@pytest.mark.parametrize("noop", [0, 1])
def test_compute_scale_matches_cuda_semantics(amax, epsilon, pow2, expected_scale, noop):
    implementations = available_implementations(OP_NAME)
    if len(implementations) < 2:
        pytest.skip(f"{OP_NAME} has fewer than two available implementations")

    device = torch.device(te_device_type())
    expected = torch.tensor([expected_scale], device=device, dtype=torch.float32)
    expected_inv = expected.reciprocal()
    results = {}

    for impl in implementations:
        amax_tensor = torch.tensor([amax], device=device, dtype=torch.float32)
        scale = torch.full_like(amax_tensor, -1.0)
        scale_inv = torch.full_like(amax_tensor, -1.0)
        impl.fn(
            2048,
            torch.tensor([noop], device=device, dtype=torch.int32),
            [[amax_tensor], [scale], [scale_inv]],
            448.0,
            pow2,
            epsilon,
        )
        assert torch.equal(scale, expected), f"{impl.impl_id}: unexpected scale"
        assert torch.equal(scale_inv, expected_inv), f"{impl.impl_id}: unexpected scale_inv"
        results[impl.impl_id] = (scale, scale_inv)

    for (left_id, left), (right_id, right) in itertools.combinations(results.items(), 2):
        for left_tensor, right_tensor in zip(left, right):
            assert torch.equal(left_tensor, right_tensor), f"{left_id} != {right_id}"


def test_compute_scale_saturates_fp32_overflow():
    implementations = available_implementations(OP_NAME)
    if len(implementations) < 2:
        pytest.skip(f"{OP_NAME} has fewer than two available implementations")

    device = torch.device(te_device_type())
    expected_scale = torch.finfo(torch.float32).max
    for impl in implementations:
        amax = torch.tensor([torch.finfo(torch.float32).tiny], device=device)
        scale = torch.empty_like(amax)
        scale_inv = torch.empty_like(amax)
        impl.fn(
            2048,
            torch.zeros(1, device=device, dtype=torch.int32),
            [[amax], [scale], [scale_inv]],
            448.0,
            False,
            0.0,
        )
        assert scale.item() == expected_scale, impl.impl_id
        assert scale_inv.item() == 0.0, impl.impl_id
