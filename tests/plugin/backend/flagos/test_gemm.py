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


# ==============================================================================
# Part 4: Numerical Agreement With The Reference Backend
#
# The matrix above walks every structural permutation but asserts only the
# arity of the result, and the module-level mock returns zeros, so it cannot
# observe what the op computes. These cases route the mock through real torch
# ops instead and compare against general_gemm_torch, which defines the
# semantics of this op and which the NPU vendor backend states it follows.
# float64 keeps the two paths within rounding of each other, so the tolerance
# can stay tight enough to catch a dropped scale factor.
# ==============================================================================


@pytest.fixture
def real_flag_gems():
    """Swap the zeros-returning mock for real torch ops, then restore it."""
    saved = {
        name: getattr(mock_flag_gems, name)
        for name in ("mm", "addmm", "add_", "copy_", "sum_dim")
    }
    mock_flag_gems.mm = torch.mm
    mock_flag_gems.addmm = lambda bias, a, b, beta=1, alpha=1: torch.addmm(
        bias, a, b, beta=beta, alpha=alpha
    )
    mock_flag_gems.add_ = lambda tensor, other: tensor.add_(other)
    mock_flag_gems.copy_ = lambda tensor, other: tensor.copy_(other)
    mock_flag_gems.sum_dim = lambda x, dim, *a, **k: x.sum(
        dim=tuple(dim) if isinstance(dim, (list, tuple)) else dim
    )
    yield
    for name, fn in saved.items():
        setattr(mock_flag_gems, name, fn)


def _call(fn, A, transA, B, transB, D, bias, grad, accumulate, alpha, beta):
    return fn(
        A=A,
        transA=transA,
        B=B,
        transB=transB,
        D=D,
        quantizer=None,
        output_dtype=None,
        bias=bias,
        bias_type=None,
        gelu=False,
        gelu_in=None,
        grad=grad,
        workspace=torch.zeros((1,)),
        workspace_size=0,
        accumulate=accumulate,
        use_split_accumulator=False,
        alpha=alpha,
        beta=beta,
    )


# out = B_comp(m,k) @ A_comp(k,n); deliberately non-square so a wrong
# reshape cannot hide behind equal dimensions.
_M, _N, _K = 8, 6, 4


def _operands(transA, transB, b_ndim, bias, has_D, dtype=torch.float64):
    torch.manual_seed(0)
    A = torch.randn((_N, _K) if transA else (_K, _N), dtype=dtype)
    b_shape = (_K, _M) if transB else (_M, _K)
    if b_ndim == 3:
        B = torch.randn((b_shape[0] // 2, 2, b_shape[1]), dtype=dtype)
    else:
        B = torch.randn(b_shape, dtype=dtype)
    return (
        A,
        B,
        torch.randn((_N,), dtype=dtype) if bias else None,
        torch.randn((_M, _N), dtype=dtype) if has_D else None,
    )


@pytest.mark.parametrize(
    "label, transA, transB, b_ndim, bias, has_D, grad, accumulate, alpha, beta",
    [
        # alpha must scale every branch, not just the fused-bias one.
        ("alpha, no bias", False, False, 2, False, False, False, False, 2.0, None),
        ("alpha, bias", False, False, 2, True, False, False, False, 2.0, None),
        ("alpha, grad", False, False, 2, False, False, True, False, 2.0, None),
        ("alpha, grad, bias", False, False, 2, True, False, True, False, 2.0, None),
        # beta scales the accumulator D, not the GEMM result.
        ("beta accumulate", False, False, 2, False, True, False, True, 1.0, 0.5),
        ("alpha and beta", False, False, 2, False, True, False, True, 2.0, 0.5),
        # A transposed B contributes no outer dimensions to the output.
        ("3D B, transB", False, True, 3, False, False, False, False, 1.0, None),
        ("3D B, transB, bias", False, True, 3, True, False, False, False, 1.0, None),
        ("3D B, no transB", False, False, 3, False, False, False, False, 1.0, None),
        # Baselines that already agreed, kept so a fix cannot regress them.
        ("plain", False, False, 2, False, False, False, False, 1.0, None),
        ("transA", True, False, 2, False, False, False, False, 1.0, None),
        ("D overwrite", False, False, 2, False, True, False, False, 1.0, None),
    ],
)
def test_generic_gemm_matches_reference(
    real_flag_gems, label, transA, transB, b_ndim, bias, has_D, grad, accumulate, alpha, beta
):
    from transformer_engine.plugin.core.backends.reference.impl.gemm import (
        general_gemm_torch,
    )

    A, B, bias_t, D = _operands(transA, transB, b_ndim, bias, has_D)

    # D is written in place, so each implementation needs its own copy.
    got = _call(
        generic_gemm_fl, A, transA, B, transB,
        None if D is None else D.clone(), bias_t, grad, accumulate, alpha, beta,
    )
    want = _call(
        general_gemm_torch, A, transA, B, transB,
        None if D is None else D.clone(), bias_t, grad, accumulate, alpha, beta,
    )

    assert tuple(got[0].shape) == tuple(want[0].shape), (
        f"{label}: shape {tuple(got[0].shape)} != reference {tuple(want[0].shape)}"
    )
    torch.testing.assert_close(got[0], want[0], rtol=0, atol=1e-12)

    if want[1] is None:
        assert got[1] is None, f"{label}: bias gradient returned where reference returns none"
    else:
        assert got[1] is not None, f"{label}: bias gradient missing"
        torch.testing.assert_close(got[1], want[1], rtol=0, atol=1e-12)
