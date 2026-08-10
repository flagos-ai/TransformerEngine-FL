import sys
from unittest.mock import MagicMock

import pytest
import torch

# ==============================================================================
# Part 0: Fine-Grained Dependency Isolation
# Inject a fake flag_gems module before doing anything else
# ==============================================================================
mock_flag_gems = MagicMock()
sys.modules["flag_gems"] = mock_flag_gems

# Mock typical element-wise operations for flag_gems to return expected torch types
mock_flag_gems.sum = lambda x, *args, **kwargs: (
    torch.sum(x) if isinstance(x, torch.Tensor) else torch.tensor(1.0)
)
mock_flag_gems.mul = lambda x, y, *args, **kwargs: x * y
mock_flag_gems.add = lambda x, y, *args, **kwargs: x + y
mock_flag_gems.sqrt = lambda x, *args, **kwargs: (
    torch.sqrt(x) if isinstance(x, torch.Tensor) else torch.tensor(1.0)
)
mock_flag_gems.copy_ = lambda dst, src, *args, **kwargs: dst.copy_(src)

# DIRECT IMPORT: Bypass OpManager routing completely by importing the source code functions directly
from transformer_engine.plugin.core.backends.flagos.impl.multi_tensor import (
    multi_tensor_l2_norm_fl,
    multi_tensor_scale_fl,
)

# ==============================================================================
# Part 1: multi_tensor_l2_norm_fl Functional Tests
# ==============================================================================


@pytest.mark.parametrize("per_tensor", [True, False])
def test_l2_norm_standard_lifecycle(per_tensor):
    """Verify L2 norm baseline operations and shape handling logic."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    noop_flag = torch.tensor(0, dtype=torch.int32, device=device)

    tensors = [
        torch.tensor([1.0, 2.0], device=device),
        torch.tensor([3.0, 4.0], device=device),
    ]
    tensor_lists = [tensors]

    total_norm, per_tensor_res = multi_tensor_l2_norm_fl(
        _chunk_size=1024,
        noop_flag=noop_flag,
        tensor_lists=tensor_lists,
        per_tensor=per_tensor,
    )

    assert isinstance(total_norm, torch.Tensor)
    assert noop_flag.item() == 0
    if per_tensor:
        assert len(per_tensor_res) == 2
    else:
        assert per_tensor_res.item() == 0.0


def test_l2_norm_noop_shortcircuit():
    """Verify that execution drops out instantly when noop_flag is active."""
    noop_flag = torch.tensor(1, dtype=torch.int32)
    total_norm, per_tensor_res = multi_tensor_l2_norm_fl(1024, noop_flag, [], per_tensor=False)
    assert total_norm.item() == 0.0


@pytest.mark.parametrize("non_finite_val", [float("inf"), float("nan")])
def test_l2_norm_non_finite_tracking(non_finite_val):
    """Ensure that non-finite numbers set the noop_flag state to 1."""
    noop_flag = torch.tensor(0, dtype=torch.int32)
    tensor_lists = [[torch.tensor([1.0, non_finite_val])]]

    multi_tensor_l2_norm_fl(1024, noop_flag, tensor_lists, per_tensor=False)
    assert noop_flag.item() == 1


# ==============================================================================
# Part 2: multi_tensor_scale_fl Functional Tests
# ==============================================================================


def test_scale_standard_lifecycle():
    """Verify scale multiplication distributions across tensors."""
    noop_flag = torch.tensor(0, dtype=torch.int32)
    src = [torch.tensor([1.0, 2.0])]
    dst = [torch.zeros(2)]

    multi_tensor_scale_fl(1024, noop_flag, [src, dst], scale=2.0)
    assert torch.allclose(dst[0], torch.tensor([2.0, 4.0]))


def test_scale_noop_shortcircuit():
    """Verify scale operation returns immediately when noop_flag is active."""
    noop_flag = torch.tensor(1, dtype=torch.int32)
    src = [torch.tensor([1.0, 2.0])]
    dst = [torch.zeros(2)]

    multi_tensor_scale_fl(1024, noop_flag, [src, dst], scale=2.0)
    assert torch.allclose(dst[0], torch.zeros(2))


@pytest.mark.parametrize("non_finite_val", [float("inf"), float("nan")])
def test_scale_non_finite_tracking(non_finite_val):
    """Verify scale tracking captures non-finite elements and trips the noop_flag."""
    noop_flag = torch.tensor(0, dtype=torch.int32)
    src = [torch.tensor([1.0, non_finite_val])]
    dst = [torch.zeros(2)]

    multi_tensor_scale_fl(1024, noop_flag, [src, dst], scale=2.0)
    assert noop_flag.item() == 1
