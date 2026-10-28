# Copyright (c) 2026, BAAI. All rights reserved.
# See LICENSE for license information.
"""NPU vendor cuDNN capability query and dispatch tests."""

import pytest

from transformer_engine.plugin.core.backends.vendor.npu import npu as npu_module
from transformer_engine.plugin.core.backends.vendor.npu.register_ops import register_builtins
from transformer_engine.plugin.core.manager import OpManager, get_default_manager
from transformer_engine.plugin.core.ops import NVTE_Fused_Attn_Backend
from transformer_engine.plugin.core.registry import OpRegistry
from transformer_engine.plugin.core.types import BackendImplKind


def test_cudnn_version_is_encoded_zero():
    """The backend contract is an integer, not a version tuple."""
    version = npu_module.NPUBackend().get_cudnn_version()
    assert isinstance(version, int) and not isinstance(version, bool)
    assert version == 0


def test_cudnn_vendor_registration(monkeypatch):
    """Use the standard NPU vendor registration and availability guard."""
    monkeypatch.setattr(npu_module, "_check_npu_available", lambda: True)
    registry = OpRegistry()
    register_builtins(registry)
    implementations = registry.snapshot().impls_by_op["get_cudnn_version"]
    assert len(implementations) == 1
    vendor = implementations[0]
    assert vendor.impl_id == "vendor.npu"
    assert vendor.kind == BackendImplKind.VENDOR
    assert vendor.vendor == "NPU" and vendor.priority == 100
    assert vendor.is_available()
    assert vendor.fn() == 0
    monkeypatch.setattr(npu_module, "_check_npu_available", lambda: False)
    assert not vendor.is_available()


def test_cudnn_not_registered_without_npu(monkeypatch):
    """Do not advertise an available NPU implementation on other platforms."""
    monkeypatch.setattr(npu_module.NPUBackend, "is_available", lambda self: False)
    registry = OpRegistry()
    register_builtins(registry)
    assert not registry.snapshot().impls_by_op.get("get_cudnn_version")


@pytest.fixture(name="real_npu_backend")
def make_real_npu_backend():
    """Require the real installed NPU vendor for integration checks."""
    pytest.importorskip("torch_npu")
    if not npu_module._check_npu_available():
        pytest.skip("Ascend NPU and TransformerEngineNPU are required")
    return npu_module.NPUBackend()


def test_real_cudnn_vendor_dispatch(real_npu_backend, monkeypatch):
    """Dispatch without runtime fallback and record the selected vendor."""
    assert real_npu_backend.is_available()
    monkeypatch.setenv("TE_FL_PREFER", "vendor")
    monkeypatch.setenv("TE_FL_STRICT", "0")
    manager = OpManager()
    assert manager.call("get_cudnn_version") == 0
    assert manager._get_last_impl_id("get_cudnn_version") == "vendor.npu"


def test_public_cudnn_version_matches_native(real_npu_backend, monkeypatch):
    """Decode zero in the public TE-FL utility and match the official NPU utility."""
    from transformer_engine.pytorch.utils import get_cudnn_version
    from transformer_engine_npu.pytorch.utils import get_cudnn_version as native_version

    assert real_npu_backend.is_available()
    monkeypatch.setenv("TE_FL_PREFER", "vendor")
    monkeypatch.setenv("TE_FL_STRICT", "0")
    get_cudnn_version.cache_clear()
    try:
        assert get_cudnn_version() == native_version() == (0, 0, 0)
        assert get_default_manager()._get_last_impl_id("get_cudnn_version") == "vendor.npu"
    finally:
        get_cudnn_version.cache_clear()


def test_npu_attention_selection_with_no_cudnn(real_npu_backend):
    """Keep NPU FlashAttention enabled and cuDNN fused attention disabled."""
    assert real_npu_backend.get_cudnn_version() == 0
    flash, _, fused, fused_backend, unfused, available = real_npu_backend.get_attention_backend()
    assert flash == 1
    assert fused == 0 and unfused == 0
    assert fused_backend == NVTE_Fused_Attn_Backend.NVTE_No_Backend
    assert available == [1, 0, 0]
