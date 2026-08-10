import os
import sys
from unittest.mock import MagicMock

import pytest
import torch

_MISSING = object()
_MOCKED_MODULE_NAMES = (
    "transformer_engine.plugin.core.ops",
    "transformer_engine.plugin.core.backends.reference.impl",
    "transformer_engine.plugin.core.backends.reference.reference",
    "transformer_engine.plugin.core.backends.reference",
)


def _get_parent_attr(module_name):
    parent_name, _, attr_name = module_name.rpartition(".")
    parent_module = sys.modules.get(parent_name)
    if parent_module is None:
        return None
    return parent_module, attr_name, getattr(parent_module, attr_name, _MISSING)


_SAVED_MODULES = {
    module_name: sys.modules.get(module_name, _MISSING) for module_name in _MOCKED_MODULE_NAMES
}
_SAVED_PARENT_ATTRS = {
    module_name: _get_parent_attr(module_name) for module_name in _MOCKED_MODULE_NAMES
}

for module_name in _MOCKED_MODULE_NAMES:
    sys.modules.pop(module_name, None)


def _restore_import_state():
    for module_name, module in _SAVED_MODULES.items():
        if module is _MISSING:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = module

    for saved_attr in _SAVED_PARENT_ATTRS.values():
        if saved_attr is None:
            continue
        parent_module, attr_name, attr_value = saved_attr
        if attr_value is _MISSING:
            if hasattr(parent_module, attr_name):
                delattr(parent_module, attr_name)
        else:
            setattr(parent_module, attr_name, attr_value)


# ==============================================================================
# Part 0: High-Reliability Environment Isolation & Explicit Function Mocking
# ==============================================================================
# 1. Isolate C++ / CUDA ops dependencies safely
mock_ops = MagicMock()
sys.modules["transformer_engine.plugin.core.ops"] = mock_ops


class MockBase:
    pass


mock_ops.TEFLBackendBase = MockBase
mock_ops.DType = MagicMock()
mock_ops.FP8TensorMeta = MagicMock()
mock_ops.CommOverlapType = MagicMock()
mock_ops.NVTE_QKV_Layout = MagicMock()
mock_ops.NVTE_Bias_Type = MagicMock()
mock_ops.NVTE_Mask_Type = MagicMock()
mock_ops.NVTE_Softmax_Type = MagicMock()
mock_ops.NVTE_QKV_Format = MagicMock()
mock_ops.CommOverlap = MagicMock()


class MockFusedBackend:
    NVTE_No_Backend = 0


mock_ops.NVTE_Fused_Attn_Backend = MockFusedBackend

# 2. SEVER IMPL LINKAGE: Intercept the entire impl module to completely eliminate
# any possibility of compiler neighbor circular imports (reference <-> softmax).
mock_impl = MagicMock()
sys.modules["transformer_engine.plugin.core.backends.reference.impl"] = mock_impl

# 3. EXPLICIT SPECIFIC ASSIGNMENT: Explicitly populate only the exact required
# framework stubs to avoid dir() traversal MagicMock recursion overflows.
torch_stensors = [
    "general_gemm_torch",
    "gelu_torch",
    "geglu_torch",
    "qgelu_torch",
    "qgeglu_torch",
    "relu_torch",
    "reglu_torch",
    "srelu_torch",
    "sreglu_torch",
    "silu_torch",
    "swiglu_torch",
    "clamped_swiglu_torch",
    "dgelu_torch",
    "dgeglu_torch",
    "dqgelu_torch",
    "dqgeglu_torch",
    "drelu_torch",
    "dreglu_torch",
    "dsrelu_torch",
    "dsreglu_torch",
    "dsilu_torch",
    "dswiglu_torch",
    "clamped_dswiglu_torch",
    "dbias_dgelu_torch",
    "dbias_dsilu_torch",
    "dbias_drelu_torch",
    "dbias_dqgelu_torch",
    "dbias_dsrelu_torch",
    "scaled_softmax_forward_torch",
    "scaled_softmax_backward_torch",
    "scaled_masked_softmax_forward_torch",
    "scaled_masked_softmax_backward_torch",
    "scaled_upper_triang_masked_softmax_forward_torch",
    "scaled_upper_triang_masked_softmax_backward_torch",
    "scaled_aligned_causal_masked_softmax_forward_torch",
    "scaled_aligned_causal_masked_softmax_backward_torch",
    "dropout_bwd_torch",
]

for func in torch_stensors:
    setattr(mock_impl, func, MagicMock(return_value=torch.tensor([1.0])))

# Complex layout / structured output explicit assignments
mock_impl.layernorm_fwd_torch = MagicMock(return_value=[torch.tensor(1.0)] * 3)
mock_impl.layernorm_bwd_torch = MagicMock(return_value=[torch.tensor(1.0)] * 2)
mock_impl.rmsnorm_fwd_torch = MagicMock(return_value=[torch.tensor(1.0)] * 3)
mock_impl.rmsnorm_bwd_torch = MagicMock(return_value=[torch.tensor(1.0)] * 2)
mock_impl.dropout_fwd_torch = MagicMock(return_value=(torch.tensor(1.0), torch.tensor(1.0)))
mock_impl.multi_tensor_l2norm_torch = MagicMock(return_value=(torch.tensor(1.0), torch.tensor(1.0)))

# Non-returning tracking multi-tensor stubs
mock_impl.multi_tensor_scale_torch = MagicMock()
mock_impl.multi_tensor_adam_torch = MagicMock()
mock_impl.multi_tensor_adam_fp8_torch = MagicMock()
mock_impl.multi_tensor_adam_capturable_torch = MagicMock()
mock_impl.multi_tensor_adam_capturable_master_torch = MagicMock()
mock_impl.multi_tensor_adam_param_remainder_torch = MagicMock()
mock_impl.multi_tensor_sgd_torch = MagicMock()
mock_impl.multi_tensor_compute_scale_and_scale_inv_torch = MagicMock()
mock_impl.multi_tensor_compute_scale_inv_e8m0_torch = MagicMock()

# Safely import the real backend file now that the ecosystem is fully locked down
try:
    from transformer_engine.plugin.core.backends.reference.reference import ReferenceBackend
finally:
    _restore_import_state()

# ==============================================================================
# Part 1: Availability and Attention Routing Tests
# ==============================================================================


def test_backend_availability():
    """Verify standard static and lifecycle availability flags."""
    assert ReferenceBackend.check_available() is True
    backend = ReferenceBackend()
    assert backend.is_available() is True


@pytest.mark.parametrize(
    "env_vars, expected_backends",
    [
        ({"NVTE_FLASH_ATTN": "1", "NVTE_FUSED_ATTN": "1", "NVTE_UNFUSED_ATTN": "1"}, [1, 1, 1]),
        ({"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "0", "NVTE_UNFUSED_ATTN": "0"}, [0, 0, 0]),
    ],
)
def test_get_attention_backend(env_vars, expected_backends, monkeypatch):
    """Test dynamic environment variable evaluation for attention backends."""
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)

    backend = ReferenceBackend()
    res = backend.get_attention_backend()

    assert int(res[0]) == expected_backends[0]
    assert int(res[2]) == expected_backends[1]
    assert int(res[4]) == expected_backends[2]
    assert res[5] == expected_backends


# ==============================================================================
# Part 2: Activation and Linear Core Math Tests (Zero-Patch, Direct Assertion)
# ==============================================================================


@pytest.mark.parametrize(
    "act_fwd, act_bwd, mock_attr_fwd, mock_attr_bwd",
    [
        ("gelu", "dgelu", "gelu_torch", "dgelu_torch"),
        ("geglu", "dgeglu", "geglu_torch", "dgeglu_torch"),
        ("qgelu", "dqgelu", "qgelu_torch", "dqgelu_torch"),
        ("qgeglu", "dqgeglu", "qgeglu_torch", "dqgeglu_torch"),
        ("relu", "drelu", "relu_torch", "drelu_torch"),
        ("reglu", "dreglu", "reglu_torch", "dreglu_torch"),
        ("srelu", "dsrelu", "srelu_torch", "dsrelu_torch"),
        ("sreglu", "dsreglu", "sreglu_torch", "dsreglu_torch"),
        ("silu", "dsilu", "silu_torch", "dsilu_torch"),
        ("swiglu", "dswiglu", "swiglu_torch", "dswiglu_torch"),
    ],
)
def test_activation_forward_backward_pass_through(act_fwd, act_bwd, mock_attr_fwd, mock_attr_bwd):
    """Verify standard activations dispatch safely to their explicit mock targets."""
    backend = ReferenceBackend()
    inp = torch.randn(2, 2)

    m_fwd = getattr(mock_impl, mock_attr_fwd)
    m_bwd = getattr(mock_impl, mock_attr_bwd)
    m_fwd.reset_mock()
    m_bwd.reset_mock()

    fwd_fn = getattr(backend, act_fwd)
    bwd_fn = getattr(backend, act_bwd)

    assert fwd_fn(inp, quantizer=None) is not None
    assert bwd_fn(inp, inp, quantizer=None) is not None

    m_fwd.assert_called_once()
    m_bwd.assert_called_once()


def test_clamped_swiglu_variants():
    """Verify clamped activation branches execute without patch tracking overrides."""
    backend = ReferenceBackend()
    inp = torch.randn(2, 2)

    mock_impl.clamped_swiglu_torch.reset_mock()
    mock_impl.clamped_dswiglu_torch.reset_mock()

    assert backend.clamped_swiglu(inp, quantizer=None, limit=5.0, alpha=1.5) is not None
    assert backend.clamped_dswiglu(inp, inp, quantizer=None, limit=5.0, alpha=1.5) is not None

    mock_impl.clamped_swiglu_torch.assert_called_once()
    mock_impl.clamped_dswiglu_torch.assert_called_once()


@pytest.mark.parametrize(
    "dbias_act, mock_attr",
    [
        ("dbias_dgelu", "dbias_dgelu_torch"),
        ("dbias_dsilu", "dbias_dsilu_torch"),
        ("dbias_drelu", "dbias_drelu_torch"),
        ("dbias_dqgelu", "dbias_dqgelu_torch"),
        ("dbias_dsrelu", "dbias_dsrelu_torch"),
    ],
)
def test_dbias_fusions(dbias_act, mock_attr):
    """Verify fused bias derivative operations hit designated explicit stub locations."""
    backend = ReferenceBackend()
    inp = torch.randn(2, 2)

    m_act = getattr(mock_impl, mock_attr)
    m_act.reset_mock()

    fn = getattr(backend, dbias_act)
    assert fn(inp, inp, quantizer=None) is not None
    m_act.assert_called_once()


def test_generic_gemm_passthrough():
    """Verify general matrix multiplication arguments route cleanly to implicit core modules."""
    backend = ReferenceBackend()
    inp = torch.randn(2, 2)

    mock_impl.general_gemm_torch.reset_mock()
    res = backend.generic_gemm(
        A=inp,
        transA=False,
        B=inp,
        transB=False,
        D=None,
        quantizer=None,
        output_dtype=None,
        bias=None,
        bias_type=None,
        gelu=False,
        gelu_in=None,
        grad=False,
        workspace=inp,
        workspace_size=0,
        accumulate=False,
        use_split_accumulator=False,
    )
    assert res is not None
    mock_impl.general_gemm_torch.assert_called_once()


# ==============================================================================
# Part 3: Normalization and Softmax Functional Tests
# ==============================================================================


def test_normalization_fwd_bwd():
    """Verify LayerNorm and RMSNorm operations forward full parameter signatures."""
    backend = ReferenceBackend()
    inp = torch.randn(4, 4)
    w = torch.ones(4)

    for m in [
        mock_impl.layernorm_fwd_torch,
        mock_impl.layernorm_bwd_torch,
        mock_impl.rmsnorm_fwd_torch,
        mock_impl.rmsnorm_bwd_torch,
    ]:
        m.reset_mock()

    assert backend.layernorm_fwd(inp, w, None, 1e-5, None, None, None, 0, False) is not None
    assert backend.layernorm_bwd(inp, inp, inp, inp, w, 0, False) is not None
    assert backend.rmsnorm_fwd(inp, w, 1e-5, None, None, None, 0, False) is not None
    assert backend.rmsnorm_bwd(inp, inp, inp, w, 0, False) is not None

    mock_impl.layernorm_fwd_torch.assert_called_once()
    mock_impl.layernorm_bwd_torch.assert_called_once()
    mock_impl.rmsnorm_fwd_torch.assert_called_once()
    mock_impl.rmsnorm_bwd_torch.assert_called_once()


@pytest.mark.parametrize(
    "softmax_fwd, softmax_bwd, mock_attr_fwd, mock_attr_bwd, has_mask",
    [
        (
            "scaled_softmax_forward",
            "scaled_softmax_backward",
            "scaled_softmax_forward_torch",
            "scaled_softmax_backward_torch",
            False,
        ),
        (
            "scaled_masked_softmax_forward",
            "scaled_masked_softmax_backward",
            "scaled_masked_softmax_forward_torch",
            "scaled_masked_softmax_backward_torch",
            True,
        ),
        (
            "scaled_upper_triang_masked_softmax_forward",
            "scaled_upper_triang_masked_softmax_backward",
            "scaled_upper_triang_masked_softmax_forward_torch",
            "scaled_upper_triang_masked_softmax_backward_torch",
            False,
        ),
        (
            "scaled_aligned_causal_masked_softmax_forward",
            "scaled_aligned_causal_masked_softmax_backward",
            "scaled_aligned_causal_masked_softmax_forward_torch",
            "scaled_aligned_causal_masked_softmax_backward_torch",
            False,
        ),
    ],
)
def test_softmax_variants(softmax_fwd, softmax_bwd, mock_attr_fwd, mock_attr_bwd, has_mask):
    """Verify standard, masked, triangular, and causal masked softmax variations."""
    backend = ReferenceBackend()
    inp = torch.randn(4, 4)

    m_fwd = getattr(mock_impl, mock_attr_fwd)
    m_bwd = getattr(mock_impl, mock_attr_bwd)
    m_fwd.reset_mock()
    m_bwd.reset_mock()

    fwd_fn = getattr(backend, softmax_fwd)
    bwd_fn = getattr(backend, softmax_bwd)

    if has_mask:
        assert fwd_fn(inp, inp, 1.0) is not None
        assert bwd_fn(inp, inp, 1.0) is not None
    else:
        assert fwd_fn(inp, 1.0) is not None
        assert bwd_fn(inp, inp, 1.0) is not None

    m_fwd.assert_called_once()
    m_bwd.assert_called_once()


def test_dropout_and_version_stubs():
    """Verify dropout lifecycle execution along with framework component stubs."""
    backend = ReferenceBackend()
    inp = torch.randn(4, 4)

    mock_impl.dropout_fwd_torch.reset_mock()
    mock_impl.dropout_bwd_torch.reset_mock()

    assert backend.dropout_fwd(inp, 0.5) is not None
    assert backend.dropout_bwd(inp, inp, 0.5) is not None

    mock_impl.dropout_fwd_torch.assert_called_once()
    mock_impl.dropout_bwd_torch.assert_called_once()

    assert backend.get_cublasLt_version() == 0
    assert backend.get_cudnn_version() == 0
    assert backend.get_num_cublas_streams() == 4
    assert (
        backend.get_fused_attn_backend(
            None, None, None, None, None, None, None, 0.0, 1, 1, 1, 1, 1, 1, 0, 0, False
        )
        == 0
    )


# ==============================================================================
# Part 4: Multi-Tensor & Optimizer Pipeline Tests
# ==============================================================================


def test_multi_tensor_scale_variants():
    """Verify tensor collection scaling, including tensor to scalar unpacked conversions."""
    backend = ReferenceBackend()
    flag = torch.tensor(0)
    t_list = [[torch.tensor([1.0])]]

    mock_impl.multi_tensor_scale_torch.reset_mock()
    backend.multi_tensor_scale(1024, flag, t_list, 2.0)
    backend.multi_tensor_scale_tensor(1024, flag, t_list, torch.tensor(2.0))
    assert mock_impl.multi_tensor_scale_torch.call_count == 2


@pytest.mark.parametrize("noop_val", [0, 1])
def test_multi_tensor_unscale_l2norm(noop_val):
    """Verify unscaling behaviors drop out immediately if noop_flag trips."""
    backend = ReferenceBackend()
    flag = torch.tensor(noop_val)
    t_list = [[torch.tensor([2.0])]]
    inv_scale = torch.tensor(0.5)

    mock_impl.multi_tensor_l2norm_torch.reset_mock()
    res = backend.multi_tensor_unscale_l2norm(1024, flag, t_list, inv_scale, per_tensor=False)
    assert isinstance(res, tuple)
    if noop_val == 0:
        mock_impl.multi_tensor_l2norm_torch.assert_called_once()


def test_multi_tensor_optimizers_and_scales():
    """Verify parameter list distributions for execution pipelines like Adam, SGD, and scale calculations."""
    backend = ReferenceBackend()
    flag = torch.tensor(0)
    t_list = [[torch.tensor([1.0])]]

    opt_mocks = [
        mock_impl.multi_tensor_adam_torch,
        mock_impl.multi_tensor_adam_fp8_torch,
        mock_impl.multi_tensor_adam_param_remainder_torch,
        mock_impl.multi_tensor_adam_capturable_torch,
        mock_impl.multi_tensor_adam_capturable_master_torch,
        mock_impl.multi_tensor_sgd_torch,
        mock_impl.multi_tensor_compute_scale_and_scale_inv_torch,
        mock_impl.multi_tensor_compute_scale_inv_e8m0_torch,
    ]
    for m in opt_mocks:
        m.reset_mock()

    backend.multi_tensor_adam(1024, flag, t_list, 1e-3, 0.9, 0.99, 1e-8, 1, 0, 1, 0.01)
    backend.multi_tensor_adam_fp8(1024, flag, t_list, 1e-3, 0.9, 0.99, 1e-8, 1, 0, 1, 0.01, None)
    backend.multi_tensor_adam_param_remainder(
        1024, flag, t_list, 1e-3, 0.9, 0.99, 1e-8, 1, 0, 1, 0.01
    )

    backend.multi_tensor_adam_capturable(
        1024,
        flag,
        t_list,
        torch.tensor(1e-3),
        0.9,
        0.99,
        1e-8,
        torch.tensor(1),
        0,
        1,
        0.01,
        torch.tensor(1.0),
    )
    backend.multi_tensor_adam_capturable_master(
        1024,
        flag,
        t_list,
        torch.tensor(1e-3),
        0.9,
        0.99,
        1e-8,
        torch.tensor(1),
        0,
        1,
        0.01,
        torch.tensor(1.0),
    )

    backend.multi_tensor_sgd(1024, flag, t_list, 0.01, 0.9, 0.0, 1e-2, False, True, False, 1.0)
    backend.multi_tensor_compute_scale_and_scale_inv(1024, flag, t_list, 448.0, True, 1e-8)
    backend.multi_tensor_compute_scale_inv_e8m0(1024, flag, t_list, 16)

    for m in opt_mocks:
        m.assert_called_once()
