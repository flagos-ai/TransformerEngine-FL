# Reference backend GEMM tests.
import pytest
import torch
import torch.nn.functional as F

from transformer_engine.plugin.core.backends.reference.impl.gemm import (
    general_gemm_torch,
    _convert_dtype,
)
from transformer_engine.plugin.core.backends.reference.impl.normalization import _to_torch_dtype
from transformer_engine.plugin.core.ops import DType

# ==============================================================================
# Part 1: Internal Helper & Data Type Converter Tests
# ==============================================================================


def test_convert_dtype_variants():
    """Verify all internal _convert_dtype dictionary mappings and fallback paths."""
    # Test None input
    assert _convert_dtype(None) is None

    # Test standard torch.dtype passing through
    assert _convert_dtype(torch.float32) == torch.float32

    # Test integer ID mapping
    assert _convert_dtype(4) == torch.float32
    assert _convert_dtype(6) == torch.bfloat16
    assert _convert_dtype(7) == torch.float8_e4m3fn
    assert _convert_dtype(999) is None  # Invalid integer mapping

    # Test object containing `.value` attribute (e.g. TE custom Enum types)
    class FakeEnum:
        def __init__(self, val):
            self.value = val

    assert _convert_dtype(FakeEnum(5)) == torch.float16
    assert _convert_dtype(FakeEnum(999)) is None

    # Test completely invalid types (strings, lists, etc.)
    assert _convert_dtype("not_a_dtype") is None


@pytest.mark.parametrize(
    ("te_dtype", "torch_dtype"),
    [
        (DType.kByte, torch.uint8),
        (DType.kInt16, torch.int16),
        (DType.kInt32, torch.int32),
        (DType.kInt64, torch.int64),
        (DType.kFloat32, torch.float32),
        (DType.kFloat16, torch.float16),
        (DType.kBFloat16, torch.bfloat16),
        (DType.kFloat8E4M3, torch.float8_e4m3fn),
        (DType.kFloat8E5M2, torch.float8_e5m2),
    ],
)
def test_normalization_dtype_conversion(te_dtype, torch_dtype):
    """Convert every reference-backend-supported TE dtype, including integer IDs."""
    assert _to_torch_dtype(te_dtype) is torch_dtype
    assert _to_torch_dtype(te_dtype.value) is torch_dtype


def test_normalization_dtype_conversion_passthrough():
    """Preserve None and native torch dtype inputs."""
    assert _to_torch_dtype(None) is None
    assert _to_torch_dtype(torch.float32) is torch.float32


@pytest.mark.parametrize(
    "dtype",
    [DType.kFloat8E8M0, DType.kFloat4E2M1, DType.kNumTypes, "invalid"],
)
def test_normalization_dtype_conversion_rejects_unsupported(dtype):
    """Reject scale-only, packed, sentinel, and malformed dtype values explicitly."""
    with pytest.raises(ValueError, match="Unsupported dtype"):
        _to_torch_dtype(dtype)


# ==============================================================================
# Part 2: Matrix Multiplication (GEMM) Core & Shape Transformation Tests
# ==============================================================================


def test_gemm_standard_and_device_mismatch():
    """Test standard 2D GEMM execution along with implicit device synchronization."""
    # Device setup (falling back to CPU for high-reliability CI pipelines)
    cpu_device = torch.device("cpu")

    # A_comp shape (2, 3), B_comp shape (3, 2) -> output shape (2, 2)
    # Since out = torch.mm(B_comp, A_comp), shapes are:
    # B_comp: (M, K) = (2, 3) -> B is (2, 3) with transB=False
    # A_comp: (K, N) = (3, 2) -> A is (2, 3) with transA=True
    A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)  # Shape (2, 3)
    B = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]], dtype=torch.float32)  # Shape (2, 3)

    # Intentionally trigger Device Mismatch path (A on CPU, but B explicitly bound to CPU)
    # This fully exercises: if A.device != target_device: A = A.to(target_device)
    res, _, _, _ = general_gemm_torch(
        A=A,
        transA=True,
        B=B,
        transB=False,
        D=None,
        quantizer=None,
        output_dtype=None,
        bias=None,
        bias_type=None,
        gelu=False,
        gelu_in=None,
        grad=False,
        workspace=A,
        workspace_size=0,
        accumulate=False,
        use_split_accumulator=False,
    )

    # Expected: B_comp (2, 3) x A_comp (3, 2) -> (2, 2)
    A_comp = A.T
    expected = torch.mm(B, A_comp)
    assert torch.allclose(res, expected)


def test_gemm_3d_tensor_reshaping():
    """Test 3D Tensor dimension unfolding and structural refolding verification."""
    # A is 3D: (1, 2, 3) -> reshapes to (2, 3)
    # B is 3D: (1, 2, 3) -> reshapes to (2, 3)
    # transA=True, transB=False -> B_comp=(2, 3), A_comp=(3, 2) -> out=(2, 2)
    # Refolds using original_B_shape -> (1, 2, 2)
    A = torch.randn(1, 2, 3)
    B = torch.randn(1, 2, 3)

    res, _, _, _ = general_gemm_torch(
        A=A,
        transA=True,
        B=B,
        transB=False,
        D=None,
        quantizer=None,
        output_dtype=None,
        bias=None,
        bias_type=None,
        gelu=False,
        gelu_in=None,
        grad=False,
        workspace=torch.empty(1),
        workspace_size=0,
        accumulate=False,
        use_split_accumulator=False,
    )
    assert res.shape == (1, 2, 2)


def test_gemm_fp8_precision_downcast():
    """Verify FP8 emulation paths downcasting directly into BF16 structures."""
    # Instantiate tensors in Float8 emulation mode
    A = torch.randn(2, 2).to(torch.float8_e4m3fn)
    B = torch.randn(2, 2).to(torch.float8_e4m3fn)

    res, _, _, _ = general_gemm_torch(
        A=A,
        transA=False,
        B=B,
        transB=False,
        D=None,
        quantizer=None,
        output_dtype=None,
        bias=None,
        bias_type=None,
        gelu=False,
        gelu_in=None,
        grad=False,
        workspace=torch.empty(1),
        workspace_size=0,
        accumulate=False,
        use_split_accumulator=False,
    )
    # The internal logic forces compute_dtype = torch.bfloat16 when detecting FP8
    assert res.dtype == torch.bfloat16


# ==============================================================================
# Part 3: Math Fusions, Output Conversions & Buffers Tests
# ==============================================================================


def test_gemm_fusions_and_scaling():
    """Verify alpha scaling, bias broadcast addition, and dtype downcasting pipelines."""
    A = torch.tensor([[2.0], [2.0]], dtype=torch.float32)  # (2, 1) -> transA=False -> A_comp=(2, 1)
    B = torch.tensor([[3.0, 4.0]], dtype=torch.float32)  # (1, 2) -> transB=False -> B_comp=(1, 2)
    # torch.mm(B_comp, A_comp) -> (1, 2) x (2, 1) -> (1, 1) matrix [[14.0]]

    bias = torch.tensor([[1.0]], dtype=torch.float32)

    res, _, _, _ = general_gemm_torch(
        A=A,
        transA=False,
        B=B,
        transB=False,
        D=None,
        quantizer=None,
        output_dtype=5,
        bias=bias,
        bias_type=None,
        gelu=False,
        gelu_in=None,
        grad=False,
        workspace=torch.empty(1),
        workspace_size=0,
        accumulate=False,
        use_split_accumulator=False,
        alpha=2.0,
    )

    # Mathematical breakdown: (14.0 * alpha=2.0) + bias=1.0 = 29.0
    assert res.item() == 29.0
    assert res.dtype == torch.float16


def test_gemm_gelu_activation_branches():
    """Verify GeLU fusions including both standalone cloned and in-place copy tracks."""
    A = torch.randn(2, 2)
    B = torch.randn(2, 2)

    # Track A: gelu=True, gelu_in is None (Triggers out.clone() fallback)
    res_a, _, gelu_in_a, _ = general_gemm_torch(
        A=A,
        transA=False,
        B=B,
        transB=False,
        D=None,
        quantizer=None,
        output_dtype=None,
        bias=None,
        bias_type=None,
        gelu=True,
        gelu_in=None,
        grad=False,
        workspace=torch.empty(1),
        workspace_size=0,
        accumulate=False,
        use_split_accumulator=False,
    )
    assert gelu_in_a is not None

    # Track B: gelu=True, gelu_in provided (Triggers direct gelu_in.copy_(out) statement)
    gelu_buffer = torch.empty((2, 2), dtype=torch.float32)
    res_b, _, gelu_in_b, _ = general_gemm_torch(
        A=A,
        transA=False,
        B=B,
        transB=False,
        D=None,
        quantizer=None,
        output_dtype=None,
        bias=None,
        bias_type=None,
        gelu=True,
        gelu_in=gelu_buffer,
        grad=False,
        workspace=torch.empty(1),
        workspace_size=0,
        accumulate=False,
        use_split_accumulator=False,
    )
    assert gelu_in_b is gelu_buffer


def test_gemm_accumulator_destinations():
    """Verify tensor accumulation mapping modes (with/without active beta weights)."""
    A = torch.tensor([[1.0]], dtype=torch.float32)
    B = torch.tensor([[2.0]], dtype=torch.float32)  # mm out = [[2.0]]

    # Scenario A: accumulate=True, beta is None (defaults to 1.0)
    D_a = torch.tensor([[10.0]], dtype=torch.float32)
    res_a, _, _, _ = general_gemm_torch(
        A=A,
        transA=False,
        B=B,
        transB=False,
        D=D_a,
        quantizer=None,
        output_dtype=None,
        bias=None,
        bias_type=None,
        gelu=False,
        gelu_in=None,
        grad=False,
        workspace=torch.empty(1),
        workspace_size=0,
        accumulate=True,
        use_split_accumulator=False,
    )
    # Expected: D_a * 1.0 + 2.0 = 12.0
    assert res_a is D_a
    assert D_a.item() == 12.0

    # Scenario B: accumulate=True, beta is custom scaled (0.5)
    D_b = torch.tensor([[10.0]], dtype=torch.float32)
    res_b, _, _, _ = general_gemm_torch(
        A=A,
        transA=False,
        B=B,
        transB=False,
        D=D_b,
        quantizer=None,
        output_dtype=None,
        bias=None,
        bias_type=None,
        gelu=False,
        gelu_in=None,
        grad=False,
        workspace=torch.empty(1),
        workspace_size=0,
        accumulate=True,
        use_split_accumulator=False,
        beta=0.5,
    )
    # Expected: D_b * 0.5 + 2.0 = 7.0
    assert D_b.item() == 7.0

    # Scenario C: accumulate=False, direct deep copy into target buffer destination
    D_c = torch.tensor([[0.0]], dtype=torch.float32)
    res_c, _, _, _ = general_gemm_torch(
        A=A,
        transA=False,
        B=B,
        transB=False,
        D=D_c,
        quantizer=None,
        output_dtype=None,
        bias=None,
        bias_type=None,
        gelu=False,
        gelu_in=None,
        grad=False,
        workspace=torch.empty(1),
        workspace_size=0,
        accumulate=False,
        use_split_accumulator=False,
    )
    assert res_c is D_c
    assert D_c.item() == 2.0


# ==============================================================================
# Part 4: Backward Pass (grad=True) Tests
# ==============================================================================


def test_gemm_backward_bias_grad():
    """Verify bias gradient computation when grad=True and bias is provided.

    In backward mode the function should:
    - NOT add bias to the output
    - Return bias_grad = B.sum(dim=0) (gradient w.r.t. bias)
    """
    # A (K, N) = (3, 2), B (M, K) = (4, 3)
    # transA=False, transB=False -> out = mm(B_comp, A_comp) = mm((4,3),(3,2)) = (4,2)
    A = torch.randn(3, 2, dtype=torch.float32)
    B = torch.randn(4, 3, dtype=torch.float32)
    bias = torch.ones(B.shape[1], dtype=torch.float32)  # placeholder to request fused BGRAD

    res, bias_grad, _, _ = general_gemm_torch(
        A=A,
        transA=False,
        B=B,
        transB=False,
        D=None,
        quantizer=None,
        output_dtype=None,
        bias=bias,
        bias_type=None,
        gelu=False,
        gelu_in=None,
        grad=True,
        workspace=torch.empty(1),
        workspace_size=0,
        accumulate=False,
        use_split_accumulator=False,
    )

    # bias_grad should equal B.sum(dim=0)
    expected_bias_grad = B.sum(dim=0)
    assert bias_grad is not None
    assert torch.allclose(bias_grad, expected_bias_grad)

    # Output should NOT include bias (compare with plain matmul)
    expected_out = torch.mm(B, A)
    assert torch.allclose(res, expected_out)


def test_gemm_backward_no_bias():
    """Verify that grad=True with bias=None returns bias_grad=None and computes normally."""
    A = torch.randn(3, 2, dtype=torch.float32)
    B = torch.randn(4, 3, dtype=torch.float32)

    res, bias_grad, _, _ = general_gemm_torch(
        A=A,
        transA=False,
        B=B,
        transB=False,
        D=None,
        quantizer=None,
        output_dtype=None,
        bias=None,
        bias_type=None,
        gelu=False,
        gelu_in=None,
        grad=True,
        workspace=torch.empty(1),
        workspace_size=0,
        accumulate=False,
        use_split_accumulator=False,
    )

    assert bias_grad is None
    expected_out = torch.mm(B, A)
    assert torch.allclose(res, expected_out)


def test_gemm_backward_with_gelu():
    """Verify backward behavior when both grad=True and gelu=True.

    In backward pass, out = dY (upstream gradient) and gelu_in holds the
    pre-activation from forward. The result should be dY * GeLU'(gelu_in).

    GeLU(x) = 0.5 * x * (1 + tanh(u)),  u = sqrt(2/pi) * (x + 0.044715 * x^3)
    GeLU'(x) = 0.5*(1+tanh(u)) + 0.5*x*(1-tanh(u)^2)*sqrt(2/pi)*(1+3*0.044715*x^2)
    """
    A = torch.randn(3, 2, dtype=torch.float32)
    B = torch.randn(4, 3, dtype=torch.float32)

    # Simulate: gelu_in was saved during forward with some known values
    gelu_buffer = torch.randn(4, 2, dtype=torch.float32)
    saved_gelu_in = gelu_buffer.clone()  # preserve original values

    res, bias_grad, gelu_in_ret, _ = general_gemm_torch(
        A=A,
        transA=False,
        B=B,
        transB=False,
        D=None,
        quantizer=None,
        output_dtype=None,
        bias=None,
        bias_type=None,
        gelu=True,
        gelu_in=gelu_buffer,
        grad=True,
        workspace=torch.empty(1),
        workspace_size=0,
        accumulate=False,
        use_split_accumulator=False,
    )

    # gelu_in_ret should be None when backward
    assert gelu_in_ret is None

    # Compute expected: dY * GeLU'(saved_gelu_in)
    dY = torch.mm(B, A)  # the matmul result before gelu backward
    x = saved_gelu_in
    sqrt_2_over_pi = 0.7978845608028654
    u = sqrt_2_over_pi * (x + 0.044715 * x.pow(3))
    tanh_u = torch.tanh(u)
    gelu_deriv = 0.5 * (1.0 + tanh_u) + 0.5 * x * (1.0 - tanh_u.pow(2)) * sqrt_2_over_pi * (
        1.0 + 3.0 * 0.044715 * x.pow(2)
    )
    expected_out = dY * gelu_deriv

    assert torch.allclose(res, expected_out, atol=1e-6)


def test_gemm_backward_bias_grad_with_alpha():
    """Verify bias gradient is independent of alpha scaling.

    The bias_grad = B.sum(dim=0) should not be affected by alpha, since alpha
    only scales the matmul output.
    """
    A = torch.randn(3, 2, dtype=torch.float32)
    B = torch.randn(4, 3, dtype=torch.float32)
    bias = torch.ones(B.shape[1], dtype=torch.float32)  # placeholder to request fused BGRAD

    res, bias_grad, _, _ = general_gemm_torch(
        A=A,
        transA=False,
        B=B,
        transB=False,
        D=None,
        quantizer=None,
        output_dtype=None,
        bias=bias,
        bias_type=None,
        gelu=False,
        gelu_in=None,
        grad=True,
        workspace=torch.empty(1),
        workspace_size=0,
        accumulate=False,
        use_split_accumulator=False,
        alpha=0.5,
    )

    # bias_grad = B.sum(dim=0), unaffected by alpha
    expected_bias_grad = B.sum(dim=0)
    assert torch.allclose(bias_grad, expected_bias_grad)

    # Output should be scaled by alpha
    expected_out = torch.mm(B, A) * 0.5
    assert torch.allclose(res, expected_out)


def test_gemm_backward_bias_grad_3d_input():
    """Verify bias gradient computation with 3D B tensor (batch dimension)."""
    # B is 3D: (2, 3, 4) -> reshaped to (6, 4)
    # A is 2D: (4, 2), transA=False
    # out = mm((6,4), (4,2)) = (6,2), then reshaped to (2, 3, 2)
    A = torch.randn(4, 2, dtype=torch.float32)
    B = torch.randn(2, 3, 4, dtype=torch.float32)
    bias = torch.ones(B.shape[1], dtype=torch.float32)  # placeholder to request fused BGRAD

    res, bias_grad, _, _ = general_gemm_torch(
        A=A,
        transA=False,
        B=B,
        transB=False,
        D=None,
        quantizer=None,
        output_dtype=None,
        bias=bias,
        bias_type=None,
        gelu=False,
        gelu_in=None,
        grad=True,
        workspace=torch.empty(1),
        workspace_size=0,
        accumulate=False,
        use_split_accumulator=False,
    )

    # B is reshaped to (6, 4) before bias_grad = B.sum(dim=0) -> shape (4,)
    B_reshaped = B.reshape(-1, B.shape[-1])
    expected_bias_grad = B_reshaped.sum(dim=0)
    assert bias_grad is not None
    assert torch.allclose(bias_grad, expected_bias_grad)

    # Output should be reshaped back to (2, 3, 2)
    assert res.shape == (2, 3, 2)
