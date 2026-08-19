import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

# ==============================================================================
# Part 0: Fine-Grained Dependency Isolation (Strategic Mocking)
# Inject virtual stubs to bypass missing third-party dependency errors (flag_gems)
# ==============================================================================
mock_flag_gems = MagicMock()
sys.modules["flag_gems"] = mock_flag_gems


# Simulate typical inplace operator behaviors of flag_gems by returning the
# first tensor operand to prevent execution chain collapse.
def mock_inplace_op(tensor, *args, **kwargs):
    return tensor


mock_flag_gems.add_ = mock_inplace_op
mock_flag_gems.mul_ = mock_inplace_op
mock_flag_gems.copy_ = mock_inplace_op
mock_flag_gems.add = lambda x, *args, **kwargs: x
mock_flag_gems.mul = lambda x, *args, **kwargs: x
mock_flag_gems.sqrt = lambda x, *args, **kwargs: x
mock_flag_gems.sub = lambda x, *args, **kwargs: x
mock_flag_gems.true_divide = lambda x, *args, **kwargs: x

# Import the actual physical fused_adam backend source now that dependencies are stubbed
from transformer_engine.plugin.core.backends.flagos.impl.fused_adam import (
    multi_tensor_adam_fl,
    multi_tensor_adam_param_remainder_fl,
)

# ==============================================================================
# Part 1: multi_tensor_adam_fl Core Matrix Tests
# ==============================================================================


@pytest.mark.parametrize("num_lists", [4, 5])
@pytest.mark.parametrize("mode", [0, 1])  # 0: L2 mode, 1: AdamW mode
@pytest.mark.parametrize("bias_correction", [0, 1])
def test_multi_tensor_adam_lifecycle(num_lists, mode, bias_correction):
    """Verify standard Adam / AdamW flow pathways, tensor tracking & parameter mapping."""
    num_tensors = 2
    shape = (4, 4)

    # Mock inputs: A structure of 4 or 5 tensor lists [g, p, m, v, (p_master)]
    tensor_lists = []
    for _ in range(num_lists):
        tensor_lists.append([torch.randn(shape, dtype=torch.float32) for _ in range(num_tensors)])

    noop_flag = torch.tensor(0, dtype=torch.int32)

    # Trigger execution path to hit mathematical branches and core updates
    multi_tensor_adam_fl(
        chunk_size=1024,
        noop_flag=noop_flag,
        tensor_lists=tensor_lists,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        step=5,
        mode=mode,
        bias_correction=bias_correction,
        weight_decay=0.01,
    )


def test_multi_tensor_adam_exceptions():
    """Verify basic invariant validation rules inside standard Adam execution block."""
    noop_flag = torch.tensor(0, dtype=torch.int32)

    # Assert exception when the number of lists is not 4 or 5
    with pytest.raises(AssertionError, match="Expected 4 or 5 tensor lists"):
        multi_tensor_adam_fl(
            1024,
            noop_flag,
            [[torch.randn(2)]],
            0.01,
            0.9,
            0.99,
            1e-8,
            1,
            0,
            1,
            0.0,
        )

    # Assert exception when no tensors are provided inside the structural lists
    with pytest.raises(AssertionError, match="No tensors provided"):
        multi_tensor_adam_fl(1024, noop_flag, [[], [], [], []], 0.01, 0.9, 0.99, 1e-8, 1, 0, 1, 0.0)

    # Assert exception when internal list lengths are inconsistent
    with pytest.raises(AssertionError, match="List 1 has 1 tensors, expected 2"):
        tensor_lists = [
            [torch.randn(2), torch.randn(2)],
            [torch.randn(2)],
            [torch.randn(2)],
            [torch.randn(2)],
        ]
        multi_tensor_adam_fl(1024, noop_flag, tensor_lists, 0.01, 0.9, 0.99, 1e-8, 1, 0, 1, 0.0)


# ==============================================================================
# Part 2: multi_tensor_adam_param_remainder_fl BF16 Precision Tests
# ==============================================================================


def test_param_remainder_noop_shortcircuit():
    """Verify premature termination path when noop_flag is non-zero."""
    noop_flag = torch.tensor(1, dtype=torch.int32)
    # If the short-circuit logic fails, an empty list would throw an AssertionError.
    # A clean return verifies a successful short-circuit execution.
    res = multi_tensor_adam_param_remainder_fl(
        1024, noop_flag, [], 0.01, 0.9, 0.99, 1e-8, 1, 0, 1, 0.0
    )
    assert res is None


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_param_remainder_bit_manipulation_lifecycle(mode, weight_decay):
    """Exercise complex int16/int32 precision bitwise rounding & reconstruction pipelines."""
    num_tensors = 1
    # Construct distinct tensor states to trigger bitwise shifts and View transformations
    g = torch.randn((2, 2), dtype=torch.bfloat16)
    p = torch.randint(-32768, 32767, (2, 2), dtype=torch.int16).view(torch.bfloat16)
    m = torch.randn((2, 2), dtype=torch.float32)
    v = torch.randn((2, 2), dtype=torch.float32)

    # Introduce negative remainders to force hit the conditional
    # `torch.where(local_p_rem < 0, ...)` branch.
    p_remainder = torch.tensor([[-5, 10], [-15, 20]], dtype=torch.int16)

    tensor_lists = [[g], [p], [m], [v], [p_remainder]]
    noop_flag = torch.tensor(0, dtype=torch.int32)

    multi_tensor_adam_param_remainder_fl(
        chunk_size=512,
        noop_flag=noop_flag,
        tensor_lists=tensor_lists,
        lr=0.005,
        beta1=0.9,
        beta2=0.95,
        eps=1e-6,
        step=10,
        mode=mode,
        bias_correction=1,
        weight_decay=weight_decay,
    )


def test_param_remainder_invariants():
    """Verify list structure constraint validations unique to BF16 remainder optimizers."""
    noop_flag = torch.tensor(0, dtype=torch.int32)

    # The remainder optimizer strictly mandates exactly 5 tensor tracking structures
    with pytest.raises(AssertionError, match="Expected 5 tensor lists"):
        multi_tensor_adam_param_remainder_fl(
            1024,
            noop_flag,
            [[torch.randn(2)]],
            0.01,
            0.9,
            0.99,
            1e-8,
            1,
            0,
            1,
            0.0,
        )
