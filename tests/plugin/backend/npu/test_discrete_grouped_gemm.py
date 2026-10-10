"""NPU tests for dense discrete-input/output GroupedTensor GEMM adapters."""

from __future__ import annotations

import pytest
import torch


try:
    import torch_npu  # noqa: F401

    _HAS_NPU = torch.npu.is_available()
except (ImportError, AttributeError):
    _HAS_NPU = False


pytestmark = pytest.mark.skipif(not _HAS_NPU, reason="A real Ascend NPU is required")


from transformer_engine.plugin.core.backends.vendor.npu.npu import NPUBackend
from transformer_engine.pytorch.tensor.storage.grouped_tensor_storage import GroupedTensorStorage


def _make_grouped(parts: list[torch.Tensor]) -> GroupedTensorStorage:
    """Pack 2D tensors into dense GroupedTensor storage without changing values."""

    rows = [int(part.shape[0]) for part in parts]
    cols = int(parts[0].shape[1])
    assert all(part.ndim == 2 and int(part.shape[1]) == cols for part in parts)
    first_dims = None
    if any(rows[0] != row for row in rows):
        first_dims = torch.tensor(rows, dtype=torch.int64, device=parts[0].device)
    return GroupedTensorStorage(
        shape=(sum(rows), cols),
        dtype=parts[0].dtype,
        num_tensors=len(parts),
        shapes=[tuple(part.shape) for part in parts],
        quantizer=None,
        data=torch.cat([part.reshape(-1) for part in parts]).contiguous(),
        first_dims=first_dims,
    )


def _workspace(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.uint8, device=device)


def _coefficients(
    alpha: list[float], beta: list[float], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(alpha, dtype=torch.float32, device=device),
        torch.tensor(beta, dtype=torch.float32, device=device),
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("layout", ["TN", "NN"])
def test_discrete_in_forward_and_dgrad(layout: str) -> None:
    """A discrete weight list maps to one packed NPU grouped-matmul output."""

    torch.manual_seed(11)
    device = torch.device("npu", torch.npu.current_device())
    dtype = torch.bfloat16
    weights = [
        torch.randn(3, 4, dtype=dtype, device=device),
        torch.randn(3, 4, dtype=dtype, device=device),
    ]
    if layout == "TN":
        b_parts = [
            torch.randn(2, 4, dtype=dtype, device=device),
            torch.randn(3, 4, dtype=dtype, device=device),
        ]
        d_parts = [
            torch.randn(2, 3, dtype=dtype, device=device),
            torch.randn(3, 3, dtype=dtype, device=device),
        ]
        products = [b_parts[i] @ weights[i].t() for i in range(2)]
        transa = True
    else:
        b_parts = [
            torch.randn(2, 3, dtype=dtype, device=device),
            torch.randn(3, 3, dtype=dtype, device=device),
        ]
        d_parts = [
            torch.randn(2, 4, dtype=dtype, device=device),
            torch.randn(3, 4, dtype=dtype, device=device),
        ]
        products = [b_parts[i] @ weights[i] for i in range(2)]
        transa = False

    grouped_b = _make_grouped(b_parts)
    grouped_d = _make_grouped(d_parts)
    original_d = [part.clone() for part in d_parts]
    alpha, beta = _coefficients([1.25, 0.75], [0.5, -0.25], device)
    expected = torch.cat(
        [
            (products[i].to(dtype).float() * alpha[i] + original_d[i].float() * beta[i]).to(dtype)
            for i in range(2)
        ],
        dim=0,
    )

    returned = NPUBackend().te_general_grouped_gemm_for_discrete_in(
        weights,
        transa,
        grouped_b,
        False,
        grouped_d,
        None,
        None,
        alpha,
        beta,
        _workspace(device),
        _workspace(device),
        False,
        0,
    )

    assert returned is grouped_d
    _assert_close(grouped_d.rowwise_data.view_as(expected), expected)


def test_discrete_in_tn_bias_and_bias_scale() -> None:
    """Discrete-input output update preserves TE bias and row-scale semantics."""

    torch.manual_seed(17)
    device = torch.device("npu", torch.npu.current_device())
    dtype = torch.bfloat16
    weights = [
        torch.randn(3, 4, dtype=dtype, device=device),
        torch.randn(3, 4, dtype=dtype, device=device),
    ]
    x_parts = [
        torch.randn(2, 4, dtype=dtype, device=device),
        torch.randn(3, 4, dtype=dtype, device=device),
    ]
    d_parts = [
        torch.zeros(2, 3, dtype=dtype, device=device),
        torch.zeros(3, 3, dtype=dtype, device=device),
    ]
    bias_parts = [
        torch.randn(1, 3, dtype=dtype, device=device),
        torch.randn(1, 3, dtype=dtype, device=device),
    ]
    bias_scale = torch.randn(5, dtype=torch.float32, device=device)
    grouped_b = _make_grouped(x_parts)
    grouped_d = _make_grouped(d_parts)
    grouped_bias = _make_grouped(bias_parts)
    alpha, beta = _coefficients([1.0], [0.0], device)

    product = torch.cat([x_parts[i] @ weights[i].t() for i in range(2)]).to(dtype)
    expanded_bias = torch.cat([bias_parts[0].expand(2, -1), bias_parts[1].expand(3, -1)], dim=0)
    expected = (product.float() + expanded_bias.float() * bias_scale.view(-1, 1)).to(dtype)

    NPUBackend().te_general_grouped_gemm_for_discrete_in(
        weights,
        True,
        grouped_b,
        False,
        grouped_d,
        grouped_bias,
        bias_scale,
        alpha,
        beta,
        _workspace(device),
        _workspace(device),
        False,
        0,
    )

    _assert_close(grouped_d.rowwise_data.view_as(expected), expected)


def test_discrete_in_nt_packs_a_list() -> None:
    """NT discrete input packs A once and computes grouped weight gradients."""

    torch.manual_seed(23)
    device = torch.device("npu", torch.npu.current_device())
    dtype = torch.bfloat16
    a_parts = [
        torch.randn(2, 4, dtype=dtype, device=device),
        torch.randn(3, 4, dtype=dtype, device=device),
    ]
    b_parts = [
        torch.randn(2, 3, dtype=dtype, device=device),
        torch.randn(3, 3, dtype=dtype, device=device),
    ]
    d_parts = [
        torch.randn(3, 4, dtype=dtype, device=device),
        torch.randn(3, 4, dtype=dtype, device=device),
    ]
    original_d = [part.clone() for part in d_parts]
    grouped_b = _make_grouped(b_parts)
    grouped_d = _make_grouped(d_parts)
    alpha, beta = _coefficients([1.0, 0.5], [0.0, 0.25], device)
    expected = torch.cat(
        [
            (
                b_parts[i].float().t() @ a_parts[i].float() * alpha[i]
                + original_d[i].float() * beta[i]
            ).to(dtype)
            for i in range(2)
        ]
    )

    NPUBackend().te_general_grouped_gemm_for_discrete_in(
        a_parts,
        False,
        grouped_b,
        True,
        grouped_d,
        None,
        None,
        alpha,
        beta,
        _workspace(device),
        _workspace(device),
        False,
        0,
    )

    _assert_close(grouped_d.rowwise_data.view_as(expected), expected)


@pytest.mark.parametrize("layout", ["TN", "NN"])
def test_discrete_out_forward_shapes(layout: str) -> None:
    """Packed A/B operands can update a discrete output list for TN and NN."""

    torch.manual_seed(29)
    device = torch.device("npu", torch.npu.current_device())
    dtype = torch.bfloat16
    weights = [
        torch.randn(3, 4, dtype=dtype, device=device),
        torch.randn(3, 4, dtype=dtype, device=device),
    ]
    if layout == "TN":
        b_parts = [
            torch.randn(2, 4, dtype=dtype, device=device),
            torch.randn(3, 4, dtype=dtype, device=device),
        ]
        d_list = [
            torch.randn(2, 3, dtype=dtype, device=device),
            torch.randn(3, 3, dtype=dtype, device=device),
        ]
        products = [b_parts[i] @ weights[i].t() for i in range(2)]
        transa = True
    else:
        b_parts = [
            torch.randn(2, 3, dtype=dtype, device=device),
            torch.randn(3, 3, dtype=dtype, device=device),
        ]
        d_list = [
            torch.randn(2, 4, dtype=dtype, device=device),
            torch.randn(3, 4, dtype=dtype, device=device),
        ]
        products = [b_parts[i] @ weights[i] for i in range(2)]
        transa = False

    grouped_a = _make_grouped(weights)
    grouped_b = _make_grouped(b_parts)
    original_d = [tensor.clone() for tensor in d_list]
    data_ptrs = [tensor.data_ptr() for tensor in d_list]
    alpha, beta = _coefficients([1.0], [0.5], device)
    expected = [
        (products[i].to(dtype).float() + original_d[i].float() * 0.5).to(dtype) for i in range(2)
    ]

    returned = NPUBackend().te_general_grouped_gemm_for_discrete_out(
        grouped_a,
        transa,
        grouped_b,
        False,
        d_list,
        None,
        None,
        alpha,
        beta,
        _workspace(device),
        _workspace(device),
        False,
        0,
    )

    assert returned is d_list
    assert [tensor.data_ptr() for tensor in d_list] == data_ptrs
    for actual, reference in zip(d_list, expected):
        _assert_close(actual, reference)


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_discrete_out_nt_wgrad(output_dtype: torch.dtype) -> None:
    """NT uses packed grouped-matmul-add and writes into the provided D list."""

    torch.manual_seed(31)
    device = torch.device("npu", torch.npu.current_device())
    input_dtype = torch.bfloat16
    a_parts = [
        torch.randn(2, 4, dtype=input_dtype, device=device),
        torch.randn(3, 4, dtype=input_dtype, device=device),
    ]
    b_parts = [
        torch.randn(2, 3, dtype=input_dtype, device=device),
        torch.randn(3, 3, dtype=input_dtype, device=device),
    ]
    d_list = [
        torch.randn(3, 4, dtype=output_dtype, device=device),
        torch.randn(3, 4, dtype=output_dtype, device=device),
    ]
    grouped_a = _make_grouped(a_parts)
    grouped_b = _make_grouped(b_parts)
    original_d = [tensor.clone() for tensor in d_list]
    data_ptrs = [tensor.data_ptr() for tensor in d_list]
    alpha, beta = _coefficients([1.25, 0.75], [0.0, 0.5], device)
    expected = [
        (
            (b_parts[i].float().t() @ a_parts[i].float()) * alpha[i]
            + original_d[i].float() * beta[i]
        ).to(output_dtype)
        for i in range(2)
    ]

    returned = NPUBackend().te_general_grouped_gemm_for_discrete_out(
        grouped_a,
        False,
        grouped_b,
        True,
        d_list,
        None,
        None,
        alpha,
        beta,
        _workspace(device),
        _workspace(device),
        False,
        0,
    )

    assert returned is d_list
    assert [tensor.data_ptr() for tensor in d_list] == data_ptrs
    for actual, reference in zip(d_list, expected):
        _assert_close(actual, reference)


def test_qwen_expert_routing_with_empty_groups() -> None:
    """Exercise Qwen's 64-local-expert TN, NN, and NT paths with empty groups."""

    torch.manual_seed(37)
    device = torch.device("npu", torch.npu.current_device())
    dtype = torch.bfloat16
    num_experts = 64
    rows = [0 if index % 7 == 0 else index % 3 + 1 for index in range(num_experts)]
    weights = [torch.randn(3, 4, dtype=dtype, device=device) for _ in range(num_experts)]
    x_parts = [torch.randn(row, 4, dtype=dtype, device=device) for row in rows]
    dy_parts = [torch.randn(row, 3, dtype=dtype, device=device) for row in rows]
    alpha, beta = _coefficients([1.0], [0.0], device)
    backend = NPUBackend()

    forward = _make_grouped(
        [torch.full((row, 3), torch.nan, dtype=dtype, device=device) for row in rows]
    )
    backend.te_general_grouped_gemm_for_discrete_in(
        weights,
        True,
        _make_grouped(x_parts),
        False,
        forward,
        None,
        None,
        alpha,
        beta,
        _workspace(device),
        _workspace(device),
        False,
        0,
    )
    forward_reference = torch.cat(
        [x_parts[index] @ weights[index].t() for index in range(num_experts)], dim=0
    )
    _assert_close(forward.rowwise_data.view_as(forward_reference), forward_reference)

    dgrad = _make_grouped(
        [torch.full((row, 4), torch.nan, dtype=dtype, device=device) for row in rows]
    )
    backend.te_general_grouped_gemm_for_discrete_in(
        weights,
        False,
        _make_grouped(dy_parts),
        False,
        dgrad,
        None,
        None,
        alpha,
        beta,
        _workspace(device),
        _workspace(device),
        False,
        0,
    )
    dgrad_reference = torch.cat(
        [dy_parts[index] @ weights[index] for index in range(num_experts)], dim=0
    )
    _assert_close(dgrad.rowwise_data.view_as(dgrad_reference), dgrad_reference)

    wgrad = [
        torch.full((3, 4), torch.nan, dtype=torch.float32, device=device)
        for _ in range(num_experts)
    ]
    backend.te_general_grouped_gemm_for_discrete_out(
        _make_grouped(x_parts),
        False,
        _make_grouped(dy_parts),
        True,
        wgrad,
        None,
        None,
        alpha,
        beta,
        _workspace(device),
        _workspace(device),
        False,
        0,
    )
    for index in range(num_experts):
        reference = dy_parts[index].float().t() @ x_parts[index].float()
        _assert_close(wgrad[index], reference)


def test_discrete_ops_are_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """TE proxy resolves both discrete grouped-GEMM names to the NPU backend."""

    import transformer_engine_torch as tex
    from transformer_engine.plugin.core.policy import reset_global_policy

    monkeypatch.setenv(
        "TE_FL_PER_OP",
        ";".join(
            (
                "te_general_grouped_gemm_for_discrete_in=impl:vendor.npu",
                "te_general_grouped_gemm_for_discrete_out=impl:vendor.npu",
            )
        ),
    )
    reset_global_policy()
    try:
        assert callable(tex.te_general_grouped_gemm_for_discrete_in)
        assert callable(tex.te_general_grouped_gemm_for_discrete_out)

        device = torch.device("npu", torch.npu.current_device())
        dtype = torch.bfloat16
        x = torch.randn(1, 3, dtype=dtype, device=device)
        weight = torch.randn(2, 3, dtype=dtype, device=device)
        dy = torch.randn(1, 2, dtype=dtype, device=device)
        alpha, beta = _coefficients([1.0], [0.0], device)

        grouped_forward = _make_grouped([torch.full((1, 2), torch.nan, dtype=dtype, device=device)])
        tex.te_general_grouped_gemm_for_discrete_in(
            [weight],
            True,
            _make_grouped([x]),
            False,
            grouped_forward,
            None,
            None,
            alpha,
            beta,
            _workspace(device),
            _workspace(device),
            False,
            0,
        )
        _assert_close(grouped_forward.rowwise_data.view(1, 2), x @ weight.t())

        wgrad = [torch.full((2, 3), torch.nan, dtype=torch.float32, device=device)]
        tex.te_general_grouped_gemm_for_discrete_out(
            _make_grouped([x]),
            False,
            _make_grouped([dy]),
            True,
            wgrad,
            None,
            None,
            alpha,
            beta,
            _workspace(device),
            _workspace(device),
            False,
            0,
        )
        _assert_close(wgrad[0], dy.float().t() @ x.float())
    finally:
        reset_global_policy()
