# tests/pytorch/test_reference_dropout.py
import pytest
import torch

from transformer_engine.plugin.core.backends.reference.impl.dropout import (
    dropout_fwd_torch,
    dropout_bwd_torch,
)

# ==============================================================================
# Part 1: Forward Dropout Tests (Checking Probabilities and Out In-place Buffers)
# ==============================================================================


def test_dropout_fwd_zero_probability():
    """Verify forward pass logic when dropout probability is exactly 0.0."""
    inp = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    # Case A: out buffer is None
    out, mask = dropout_fwd_torch(inp, dropout_probability=0.0)
    assert torch.equal(out, inp)
    assert torch.all(mask == 1)
    assert mask.dtype == torch.uint8

    # Case B: out buffer is provided
    # NOTE: The reference implementation skips in-place out.copy_() when prob is 0.0,
    # returning a new cloned tensor instead. Thus, out_buffer remains unchanged.
    out_buffer = torch.zeros_like(inp, dtype=torch.float32)
    out, mask = dropout_fwd_torch(inp, dropout_probability=0.0, out=out_buffer)
    assert torch.equal(out, inp)
    assert torch.equal(
        out_buffer, torch.zeros_like(inp)
    )  # Remains zeros due to operator implementation detail


def test_dropout_fwd_standard_probability():
    """Verify bernoulli masking, global scale, and out-buffer copy under active dropout."""
    inp = torch.ones(
        (10, 10), dtype=torch.float32
    )  # Larger tensor to ensure statistical robustness
    p = 0.2
    expected_scale = 1.0 / (1.0 - p)

    # Case A: Basic routing
    out, mask = dropout_fwd_torch(inp, dropout_probability=p)
    assert mask.dtype == torch.uint8

    # Mathematical confirmation: Active outputs must be scaled up correctly
    for i in range(10):
        for j in range(10):
            if mask[i, j] == 1:
                assert torch.allclose(out[i, j], torch.tensor(expected_scale))
            else:
                assert out[i, j].item() == 0.0

    # Case B: Standard probability combined with designated out-buffer destination
    out_buffer = torch.empty_like(inp)
    out, mask = dropout_fwd_torch(inp, dropout_probability=p, out=out_buffer)
    assert torch.equal(out, out_buffer)


# ==============================================================================
# Part 2: Backward Dropout Tests (Verifying Gradients and In-place Buffers)
# ==============================================================================


def test_dropout_bwd_zero_probability():
    """Verify backward gradient scaling rules when dropout probability is 0.0."""
    grad_out = torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32)

    # Case A: grad_input buffer is None
    grad_in = dropout_bwd_torch(grad_out, mask=None, dropout_probability=0.0)
    assert torch.equal(grad_in, grad_out)

    # Case B: grad_input buffer is provided
    # NOTE: Similar to forward pass, grad_input.copy_() is skipped when prob is 0.0.
    # The returned tensor matches grad_out, while the provided buffer remains unchanged.
    grad_input_buffer = torch.zeros_like(grad_out)
    grad_in = dropout_bwd_torch(
        grad_out, mask=None, dropout_probability=0.0, grad_input=grad_input_buffer
    )
    assert torch.equal(grad_in, grad_out)
    assert torch.equal(
        grad_input_buffer, torch.zeros_like(grad_out)
    )  # Remains zeros due to operator implementation detail


def test_dropout_bwd_standard_probability():
    """Verify backward gradient routes scale factors based on forward masks."""
    grad_out = torch.tensor([[2.0, 4.0], [6.0, 8.0]], dtype=torch.float32)
    mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.uint8)
    p = 0.5
    expected_scale = 1.0 / (1.0 - p)  # scale = 2.0

    # Case A: Standalone computation
    grad_in = dropout_bwd_torch(grad_out, mask, dropout_probability=p)

    # Row 0 Col 0: Mask=1 -> 2.0 * 1 * 2.0 = 4.0
    # Row 0 Col 1: Mask=0 -> 4.0 * 0 * 2.0 = 0.0
    expected_grad = torch.tensor([[4.0, 0.0], [0.0, 16.0]], dtype=torch.float32)
    assert torch.allclose(grad_in, expected_grad)

    # Case B: Computation directly assigned into preallocated grad_input targets
    grad_input_buffer = torch.empty_like(grad_out)
    grad_in = dropout_bwd_torch(grad_out, mask, dropout_probability=p, grad_input=grad_input_buffer)
    assert torch.equal(grad_in, grad_input_buffer)
    assert torch.allclose(grad_input_buffer, expected_grad)
