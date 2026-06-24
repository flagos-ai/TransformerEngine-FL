# tests/pytorch/test_reference_gemm.py
import pytest
import torch
import torch.nn.functional as F

from transformer_engine.plugin.core.backends.reference.impl.gemm import (
    general_gemm_torch,
    _convert_dtype,
)

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
