# Reference backend activation tests.
import pytest
import torch
import torch.nn.functional as F

from transformer_engine.plugin.core.backends.reference.impl.activation import (
    gelu_torch,
    geglu_torch,
    qgelu_torch,
    qgeglu_torch,
    relu_torch,
    reglu_torch,
    srelu_torch,
    sreglu_torch,
    silu_torch,
    swiglu_torch,
    clamped_swiglu_torch,
    dgelu_torch,
    dgeglu_torch,
    dqgelu_torch,
    dqgeglu_torch,
    drelu_torch,
    dreglu_torch,
    dsrelu_torch,
    dsreglu_torch,
    dsilu_torch,
    dswiglu_torch,
    clamped_dswiglu_torch,
    dbias_dgelu_torch,
    dbias_dsilu_torch,
    dbias_drelu_torch,
    dbias_dqgelu_torch,
    dbias_dsrelu_torch,
)


# ==============================================================================
# Helper / General Fixtures
# ==============================================================================
@pytest.fixture
def standard_input():
    # Shape (2, 4) ensures .chunk(2, dim=-1) splits it into two (2, 2) tensors cleanly
    return torch.tensor([[-1.0, 2.0, -3.0, 4.0], [5.0, -6.0, 7.0, -8.0]], dtype=torch.float32)


@pytest.fixture
def standard_grad():
    return torch.tensor([[0.5, 1.5, 2.5, 3.5], [4.5, 5.5, 6.5, 7.5]], dtype=torch.float32)


# ==============================================================================
# Part 1: Forward Activation Tests (Using Real Math Verification)
# ==============================================================================


def test_basic_forwards(standard_input):
    quantizer = None

    # 1. GeLU
    assert torch.allclose(
        gelu_torch(standard_input, quantizer),
        F.gelu(standard_input, approximate="tanh"),
    )

    # 2. GeGLU
    a, b = standard_input.chunk(2, dim=-1)
    assert torch.allclose(geglu_torch(standard_input, quantizer), F.gelu(a, approximate="tanh") * b)

    # 3. Quick-GeLU (qgelu)
    assert torch.allclose(
        qgelu_torch(standard_input, quantizer),
        standard_input * torch.sigmoid(1.702 * standard_input),
    )

    # 4. Quick-GeGLU (qgeglu)
    assert torch.allclose(qgeglu_torch(standard_input, quantizer), a * torch.sigmoid(1.702 * a) * b)

    # 5. ReLU & ReGLU
    assert torch.allclose(relu_torch(standard_input, quantizer), F.relu(standard_input))
    assert torch.allclose(reglu_torch(standard_input, quantizer), F.relu(a) * b)

    # 6. Squared ReLU (srelu) & sreglu
    assert torch.allclose(
        srelu_torch(standard_input, quantizer), torch.square(F.relu(standard_input))
    )
    assert torch.allclose(sreglu_torch(standard_input, quantizer), torch.square(F.relu(a)) * b)

    # 7. SiLU & SwiGLU
    assert torch.allclose(silu_torch(standard_input, quantizer), F.silu(standard_input))
    assert torch.allclose(swiglu_torch(standard_input, quantizer), F.silu(a) * b)


def test_clamped_swiglu_forward_boundaries():
    """Verify clamped SwiGLU handles limits and triggers clamp logic precisely."""
    quantizer = None
    # Input shape: (2, 2) -> splits into a: (2, 1) and b: (2, 1)
    inp = torch.tensor([[-5.0, 5.0], [0.0, 1.0]], dtype=torch.float32)

    # Execute the activation operator
    res = clamped_swiglu_torch(inp, quantizer, limit=2.0, alpha=1.0, glu_linear_offset=0.25)

    # Fix tensor shapes to match the 2D column vector format (2, 1) after chunk(2, dim=-1)
    expected_a = torch.tensor([[-5.0], [0.0]], dtype=torch.float32)
    expected_b = torch.tensor(
        [[2.25], [1.25]], dtype=torch.float32
    )  # [5.0 clamped to max limit 2.0] + 0.25 = 2.25

    expected_out = (expected_a * torch.sigmoid(1.0 * expected_a)) * expected_b

    # Assert with matching shapes, both are now (2, 1)
    assert torch.allclose(res, expected_out)


# ==============================================================================
# Part 2: Backward Gradient Tests (Autograd Consistency Verification)
# ==============================================================================


def test_basic_backwards(standard_grad, standard_input):
    quantizer = None

    # 1. dgelu
    grad_out = dgelu_torch(standard_grad, standard_input, quantizer)
    assert grad_out.shape == standard_input.shape

    # 2. dgeglu
    assert (
        dgeglu_torch(standard_grad[..., :2], standard_input, quantizer).shape
        == standard_input.shape
    )

    # 3. dqgelu & dqgeglu
    assert dqgelu_torch(standard_grad, standard_input, quantizer).shape == standard_input.shape
    assert (
        dqgeglu_torch(standard_grad[..., :2], standard_input, quantizer).shape
        == standard_input.shape
    )

    # 4. drelu & dreglu
    assert drelu_torch(standard_grad, standard_input, quantizer).shape == standard_input.shape
    assert (
        dreglu_torch(standard_grad[..., :2], standard_input, quantizer).shape
        == standard_input.shape
    )

    # 5. dsrelu & dsreglu
    assert dsrelu_torch(standard_grad, standard_input, quantizer).shape == standard_input.shape
    assert (
        dsreglu_torch(standard_grad[..., :2], standard_input, quantizer).shape
        == standard_input.shape
    )

    # 6. dsilu & dswiglu
    assert dsilu_torch(standard_grad, standard_input, quantizer).shape == standard_input.shape
    assert (
        dswiglu_torch(standard_grad[..., :2], standard_input, quantizer).shape
        == standard_input.shape
    )


def test_clamped_dswiglu_backward_branches():
    """Force execution of both (a <= limit) and (b outside/inside limit) gradient masks."""
    quantizer = None
    # Input designed to explicitly hit:
    # a > limit (row 0), a <= limit (row 1)
    # b > limit (row 0), b < -limit (row 1)
    fwd_in = torch.tensor([[10.0, 10.0], [0.0, -10.0]], dtype=torch.float32)
    grad_in = torch.tensor([[1.0], [1.0]], dtype=torch.float32)

    # Run out-of-bounds limit to force masks evaluated as False
    res_grad = clamped_dswiglu_torch(
        grad_in,
        fwd_in,
        quantizer,
        limit=5.0,
        alpha=1.0,
        glu_linear_offset=0.25,
    )
    assert res_grad.shape == fwd_in.shape

    # Row 0, Col 0: a = 10.0 (> limit 5.0). Mask (a <= limit) is False -> grad_a should be 0.0
    assert res_grad[0, 0].item() == 0.0


# ==============================================================================
# Part 3: Fused Bias Derivative Tests (dbias_* Variants)
# ==============================================================================


@pytest.mark.parametrize(
    "dbias_fn",
    [
        dbias_dgelu_torch,
        dbias_dsilu_torch,
        dbias_drelu_torch,
        dbias_dqgelu_torch,
        dbias_dsrelu_torch,
    ],
)
def test_dbias_functional_variants(dbias_fn, standard_grad, standard_input):
    quantizer = None
    # Inject a 3D tensor to verify full dimensional summation along non-last axes
    inp_3d = torch.randn(2, 3, 4)
    grad_3d = torch.randn(2, 3, 4)

    grad_input, grad_bias = dbias_fn(grad_3d, inp_3d, quantizer)

    assert grad_input.shape == inp_3d.shape
    # Bias gradient must collapse all dimensions except the last one (Features dimension)
    assert grad_bias.shape == (4,)
    assert torch.allclose(grad_bias, grad_3d.sum(dim=(0, 1)))
