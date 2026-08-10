import sys
from unittest.mock import MagicMock

import pytest
import torch

# ==============================================================================
# Part 0: Fine-Grained Dependency Isolation (Strategic Mocking)
# ==============================================================================
mock_flag_gems = MagicMock()
sys.modules["flag_gems"] = mock_flag_gems

# Mock flag_gems.add operator
mock_flag_gems.add = lambda x, y, *args, **kwargs: x + y


# Mock forward and backward core rms_norm operators to ensure returned Tensors match expected shapes
def mock_rms_norm_forward(input_tensor, normalized_shape, weight, eps):
    # Forward returns (y, rstdevs). Intentionally add an extra dimension to rstdevs
    # to trigger the shape != view adjustment branch in the source code.
    y = input_tensor * weight
    # Construct a mismatched rstdevs shape (e.g., adding an extra dimension at the end)
    # to force triggering .view(input.shape[:-1])
    rstdevs_shape = list(input_tensor.shape[:-1]) + [1]
    rstdevs = torch.ones(rstdevs_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    return y, rstdevs


def mock_rms_norm_backward(dy, x, rsigma, normalized_shape, gamma, eps):
    # Backward returns (dx, dw)
    dx = dy * gamma
    dw = torch.ones_like(gamma)
    return dx, dw


mock_flag_gems.rms_norm_forward = mock_rms_norm_forward
mock_flag_gems.rms_norm_backward = mock_rms_norm_backward

# Directly import the implementation functions under test to bypass OpManager's dynamic routing interception
from transformer_engine.plugin.core.backends.flagos.impl.rmsnorm import (
    rmsnorm_bwd_fl,
    rmsnorm_fwd_fl,
)

# ==============================================================================
# Part 1: rmsnorm_fwd_fl Forward Path Tests
# ==============================================================================


@pytest.mark.parametrize("zero_centered_gamma", [True, False])
@pytest.mark.parametrize("input_shape", [(4, 8), (2, 3, 4)])
def test_rmsnorm_fwd_lifecycle(zero_centered_gamma, input_shape):
    """Verify forward RMSNorm lifecycle, handling gamma centering and shape reshaping."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inp = torch.randn(input_shape, device=device)
    weight = torch.ones(input_shape[-1], device=device)

    y, _, rstdevs = rmsnorm_fwd_fl(
        input=inp,
        weight=weight,
        eps=1e-5,
        ln_out=None,
        quantizer=None,
        odtype=None,
        sm_margin=0,
        zero_centered_gamma=zero_centered_gamma,
    )

    # Verify output types and correctness
    assert isinstance(y, torch.Tensor)
    assert isinstance(rstdevs, torch.Tensor)

    # Core coverage check: the shape of rstdevs must perfectly match input.shape[:-1]
    assert rstdevs.shape == inp.shape[:-1]


# ==============================================================================
# Part 2: rmsnorm_bwd_fl Backward Path Tests
# ==============================================================================


@pytest.mark.parametrize("zero_centered_gamma", [True, False])
def test_rmsnorm_bwd_lifecycle(zero_centered_gamma):
    """Verify backward RMSNorm execution and scaling adjustments."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dy = torch.randn(4, 8, device=device)
    x = torch.randn(4, 8, device=device)
    rsigma = torch.ones(4, device=device)
    gamma = torch.ones(8, device=device)

    dx, dw = rmsnorm_bwd_fl(
        dy=dy,
        x=x,
        rsigma=rsigma,
        gamma=gamma,
        sm_margin=0,
        zero_centered_gamma=zero_centered_gamma,
        eps=1e-5,
    )

    assert isinstance(dx, torch.Tensor)
    assert isinstance(dw, torch.Tensor)
    assert dx.shape == x.shape
    assert dw.shape == gamma.shape
