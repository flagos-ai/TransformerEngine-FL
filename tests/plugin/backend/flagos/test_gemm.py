import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

# ==============================================================================
# Part 0: Fine-Grained Dependency Isolation (Strategic Mocking)
# ==============================================================================
mock_flag_gems = MagicMock()
sys.modules["flag_gems"] = mock_flag_gems


# Ensure mock methods return a usable tensor matching standard shape conventions
def mock_mm_op(a, b, *args, **kwargs):
    # Deduces output dimensions dynamically based on matrix dimensions
    dim0 = a.shape[1] if hasattr(a, "shape") and len(a.shape) > 1 else 2
    dim1 = b.shape[1] if hasattr(b, "shape") and len(b.shape) > 1 else 2
    return torch.zeros((dim0, dim1), dtype=torch.float32)


def mock_inplace_op(tensor, *args, **kwargs):
    return tensor


mock_flag_gems.mm = mock_mm_op
mock_flag_gems.addmm = mock_mm_op
mock_flag_gems.add_ = mock_inplace_op
mock_flag_gems.copy_ = mock_inplace_op
mock_flag_gems.sum_dim = lambda x, dim, *args, **kwargs: torch.zeros((x.shape[1],), dtype=x.dtype)
mock_flag_gems.zeros = lambda shape, *args, **kwargs: torch.zeros(shape)
mock_flag_gems.gelu = lambda x, *args, **kwargs: x
mock_flag_gems.gelu_backward = lambda x, y, *args, **kwargs: x
mock_flag_gems.cat = lambda tensors, dim, *args, **kwargs: torch.cat(tensors, dim=dim)

# Import the actual physical backend functions under test
from transformer_engine.plugin.core.backends.flagos.impl.gemm import (
    _convert_dtype,
    generic_gemm_fl,
    te_general_grouped_gemm_fl,
    validate_gemm_scale,
)


@pytest.fixture(autouse=True)
def mock_gemm_module_flag_gems(monkeypatch):
    """Use the mock even when the GEMM module was imported before this test file."""
    monkeypatch.setattr(sys.modules[generic_gemm_fl.__module__], "flag_gems", mock_flag_gems)


# ==============================================================================
# Part 1: Helper and Utility Function Tests
# ==============================================================================


@pytest.mark.parametrize(
    "scale, required, expected",
    [
        (2.5, True, 2.5),
        (None, True, 1.0),
        (0.0, False, 0.0),
        (None, False, 0.0),
    ],
)
def test_validate_gemm_scale_success(scale, required, expected):
    """Verify input normalization values for various required configuration modes."""
    assert validate_gemm_scale(scale, required) == expected


def test_validate_gemm_scale_exceptions():
    """Verify ValueError is raised if scale validation is violated."""
    with pytest.raises(ValueError, match="scale must be zero"):
        validate_gemm_scale(5.0, required=False)


@pytest.mark.parametrize(
    "dtype, expected_torch_type",
    [
        (None, None),
        (torch.float32, torch.float32),
        (4, torch.float32),
        (6, torch.bfloat16),
        (99, None),
    ],
)
def test_convert_dtype_variations(dtype, expected_torch_type):
    """Exercise explicit data-type casting combinations using the internal registry map."""
    assert _convert_dtype(dtype) == expected_torch_type


def test_convert_dtype_enum_with_value_attribute():
    """Verify standard enum-like objects featuring an explicit '.value' attribute."""

    class DummyEnum:
        def __init__(self, val):
            self.value = val

    assert _convert_dtype(DummyEnum(5)) == torch.float16


# ==============================================================================
# Part 2: generic_gemm_fl Processing Matrix Tests
# ==============================================================================


@pytest.mark.parametrize("a_ndim", [2, 3])
@pytest.mark.parametrize("b_ndim", [2, 3])
@pytest.mark.parametrize("transA", [True, False])
@pytest.mark.parametrize("transB", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("grad", [True, False])
@pytest.mark.parametrize("has_D", [True, False])
@pytest.mark.parametrize("accumulate", [True, False])
def test_generic_gemm_lifecycle_matrix(
    a_ndim, b_ndim, transA, transB, has_bias, grad, has_D, accumulate
):
    """Walk through all architectural permutations within generic_gemm_fl."""
    A = torch.randn((2, 4, 4) if a_ndim == 3 else (4, 4))
    B = torch.randn((2, 4, 4) if b_ndim == 3 else (4, 4))

    D = torch.zeros((4, 4)) if has_D else None
    bias = torch.zeros((4,)) if has_bias else None
    workspace = torch.zeros((1,))

    res = generic_gemm_fl(
        A=A,
        transA=transA,
        B=B,
        transB=transB,
        D=D,
        quantizer=None,
        output_dtype=4,
        bias=bias,
        bias_type=None,
        gelu=False,
        gelu_in=None,
        grad=grad,
        workspace=workspace,
        workspace_size=0,
        accumulate=accumulate,
        use_split_accumulator=False,
    )

    assert len(res) == 4
    if has_D:
        assert res[0] is D


def test_generic_gemm_unsupported_features():
    """Verify that unsupported features raise appropriate assertion errors."""
    dummy_tensor = torch.zeros((2, 2))

    with pytest.raises(AssertionError, match="do not support gelu now"):
        generic_gemm_fl(
            dummy_tensor,
            False,
            dummy_tensor,
            False,
            None,
            None,
            None,
            None,
            None,
            gelu=True,
            gelu_in=dummy_tensor,
            grad=False,
            workspace=dummy_tensor,
            workspace_size=0,
            accumulate=False,
            use_split_accumulator=False,
        )

    with pytest.raises(AssertionError, match="do not support quantization now"):
        generic_gemm_fl(
            dummy_tensor,
            False,
            dummy_tensor,
            False,
            None,
            quantizer="mock",
            output_dtype=None,
            bias=None,
            bias_type=None,
            gelu=False,
            gelu_in=None,
            grad=False,
            workspace=dummy_tensor,
            workspace_size=0,
            accumulate=False,
            use_split_accumulator=False,
        )


def test_generic_gemm_3d_backward_output_shapes():
    """Weight gradients stay 2D while input gradients recover the token dimensions."""
    x = torch.randn(32, 2, 64)
    dy = torch.randn(32, 2, 64)
    weight = torch.randn(64, 64)
    common = dict(
        D=None,
        quantizer=None,
        output_dtype=torch.float32,
        bias=None,
        bias_type=None,
        gelu=False,
        gelu_in=None,
        grad=True,
        workspace=torch.empty(0),
        workspace_size=0,
        accumulate=False,
        use_split_accumulator=False,
    )

    with patch.object(mock_flag_gems, "mm", side_effect=torch.mm):
        dx, *_ = generic_gemm_fl(weight, False, dy, False, **common)
        dw, *_ = generic_gemm_fl(x, False, dy, True, **common)

    torch.testing.assert_close(dx, (dy.reshape(-1, 64) @ weight).reshape(32, 2, 64))
    torch.testing.assert_close(dw, dy.reshape(-1, 64).T @ x.reshape(-1, 64))
    assert dx.shape == x.shape
    assert dw.shape == weight.shape


# ==============================================================================
# Part 3: te_general_grouped_gemm_fl Execution Path Tests
# ==============================================================================


def test_grouped_gemm_single_output_validation():
    """Verify assertion trigger when single_output is enabled without D allocated."""
    with pytest.raises(ValueError, match="D should be allocated for single output case."):
        te_general_grouped_gemm_fl(
            B=[],
            transb=False,
            A=[],
            transa=False,
            D=None,
            D_type=None,
            m_splits=[],
            bias=[],
            bias_type=None,
            single_output=True,
            pre_gelu_out=[],
            grad=False,
            workspace=[],
            workspaceSize=0,
            accumulate=False,
            use_split_accumulator=False,
            math_sm_count=0,
        )


@pytest.mark.parametrize("grad", [True, False])
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("single_output", [True, False])
def test_grouped_gemm_standard_lifecycle(grad, accumulate, single_output):
    """Exercise standard forward/backward grouped GEMM list transformation operations."""
    A = [torch.randn(4, 4), torch.randn(4, 4)]
    B = [torch.randn(4, 4), torch.randn(4, 4)]
    D = [torch.zeros(4, 4), torch.zeros(4, 4)]
    bias = [torch.zeros(4), torch.zeros(4)]
    pre_gelu_out = [torch.zeros(4, 4), torch.zeros(4, 4)]

    returned_bias = te_general_grouped_gemm_fl(
        B=B,
        transb=False,
        A=A,
        transa=False,
        D=D,
        D_type=None,
        m_splits=[4, 4],
        bias=bias,
        bias_type=None,
        single_output=single_output,
        pre_gelu_out=pre_gelu_out,
        grad=grad,
        workspace=[],
        workspaceSize=0,
        accumulate=accumulate,
        use_split_accumulator=False,
        math_sm_count=80,
    )
    assert returned_bias == bias


@pytest.mark.parametrize("accumulate", [False, True])
def test_grouped_gemm_wgrad_bias_uses_output_gradient(accumulate):
    """Bias gradients sum over tokens, not over the weight-gradient matrix."""
    inputs = [torch.randn(2, 3), torch.randn(4, 3)]
    output_grads = [torch.randn(2, 5), torch.randn(4, 5)]
    weight_grads = [torch.full((5, 3), 0.25) for _ in inputs]
    bias_grads = [torch.full((5,), 0.5) for _ in inputs]

    with (
        patch.object(mock_flag_gems, "mm", side_effect=torch.mm),
        patch.object(mock_flag_gems, "sum_dim", side_effect=lambda x, dim: x.sum(dim=tuple(dim))),
        patch.object(mock_flag_gems, "copy_", side_effect=lambda dst, src: dst.copy_(src)),
        patch.object(mock_flag_gems, "add_", side_effect=lambda dst, src: dst.add_(src)),
    ):
        te_general_grouped_gemm_fl(
            inputs,
            False,
            output_grads,
            True,
            weight_grads,
            None,
            [2, 4],
            bias_grads,
            None,
            False,
            [],
            True,
            [],
            0,
            accumulate,
            False,
            0,
        )

    for x, dy, dw, db in zip(inputs, output_grads, weight_grads, bias_grads):
        torch.testing.assert_close(dw, dy.T @ x + (0.25 if accumulate else 0))
        torch.testing.assert_close(db, dy.sum(dim=0) + (0.5 if accumulate else 0))


@pytest.mark.parametrize("single_output", [True, False])
@pytest.mark.parametrize("grad", [True, False])
@pytest.mark.parametrize("accumulate", [True, False])
def test_grouped_gemm_zero_element_inputs(single_output, grad, accumulate):
    """Verify robustness and correctness when processing empty zero-element tensors."""
    A = [torch.empty((0, 4))]
    B = [torch.empty((4, 0))]
    D = [torch.empty((0, 0))]
    bias = [torch.zeros(4)]
    pre_gelu_out = [torch.zeros(0, 0)]

    returned_bias = te_general_grouped_gemm_fl(
        B=B,
        transb=False,
        A=A,
        transa=False,
        D=D,
        D_type=None,
        m_splits=[0],
        bias=bias,
        bias_type=None,
        single_output=single_output,
        pre_gelu_out=pre_gelu_out,
        grad=grad,
        workspace=[],
        workspaceSize=0,
        accumulate=accumulate,
        use_split_accumulator=False,
        math_sm_count=80,
    )
    assert returned_bias == bias
