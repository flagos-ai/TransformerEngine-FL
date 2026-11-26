"""Numerical checks for the reference LayerNorm backend."""

import pytest
import torch
import torch.nn.functional as F

from transformer_engine.plugin.core.backends.reference.impl.normalization import (
    layernorm_bwd_torch,
    layernorm_fwd_torch,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_layernorm_matches_torch_autograd(dtype, zero_centered_gamma):
    torch.manual_seed(1234)
    x = torch.randn(8, 128, dtype=dtype, requires_grad=True)
    weight = torch.randn(128, dtype=dtype, requires_grad=True)
    bias = torch.randn(128, dtype=dtype, requires_grad=True)
    dy = torch.randn_like(x)

    gamma = (weight + 1).float() if zero_centered_gamma else weight.float()
    expected = F.layer_norm(x.float(), (128,), gamma, bias.float(), 1e-5).to(dtype)
    expected.backward(dy)

    output, mean, rsigma = layernorm_fwd_torch(
        x.detach(),
        weight.detach(),
        bias.detach(),
        1e-5,
        None,
        None,
        dtype,
        0,
        zero_centered_gamma,
    )
    dx, dgamma, dbeta = layernorm_bwd_torch(
        dy, x.detach(), mean, rsigma, weight.detach(), 0, zero_centered_gamma
    )

    torch.testing.assert_close(output, expected.detach(), rtol=0, atol=0)
    assert mean.dtype == rsigma.dtype == torch.float32
    for actual, reference in ((dx, x.grad), (dgamma, weight.grad), (dbeta, bias.grad)):
        torch.testing.assert_close(actual, reference)
