"""Check shared frontend options against legacy vendor constructor contracts."""

from enum import IntEnum
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


class CommType(IntEnum):
    RS = 0
    AG = 1


@pytest.fixture(
    params=[
        ("musa", "MUSABackend"),
        ("metax", "MetaxBackend"),
        ("hygon", "HygonBackend"),
        ("iluvatar", "IluvatarBackend"),
        ("enflame", "EnflameBackend"),
        ("biren", "BirenBackend"),
    ]
)
def backend(request):
    vendor, class_name = request.param
    module = import_module(f"transformer_engine.plugin.core.backends.vendor.{vendor}.{vendor}")
    instance = object.__new__(getattr(module, class_name))
    instance._get_tex = Mock(
        return_value=SimpleNamespace(
            CommOverlap=Mock(), CommOverlapP2P=Mock(), CommOverlapType=CommType
        )
    )
    instance._vendor = vendor
    return instance


def _is_biren(backend):
    return getattr(backend, "_vendor", None) == "biren"


@pytest.mark.parametrize("comm_type", [None, 0, 1])
def test_overlap_preserves_legacy_constructor(backend, comm_type):
    shape, helper = [128, 64], object()
    result = backend.create_comm_overlap(
        shape,
        torch.bfloat16,
        helper,
        2,
        7,
        use_cublasmp=False,
        comm_type=comm_type,
    )
    constructor = backend._get_tex.return_value.CommOverlap
    if _is_biren(backend):
        # v2.17-style: keyword args + native CommOverlapType normalization
        # (FL passes a Python IntEnum; pybind module_local enums reject it).
        assert constructor.call_count == 1
        kwargs = constructor.call_args.kwargs
        assert isinstance(kwargs["comm_type"], CommType)
        assert kwargs["comm_type"] == (
            CommType(comm_type) if comm_type is not None else CommType.RS
        )
        assert kwargs["use_cublasmp"] is False
    else:
        constructor.assert_called_once_with(
            shape, torch.bfloat16, helper, 2, 7, 3, 2, 0, 0, 16, True, False, False
        )
    assert result is constructor.return_value


def test_p2p_preserves_legacy_constructor(backend):
    shape, helper = [128, 64], object()
    result = backend.create_comm_overlap_p2p(
        shape, torch.bfloat16, helper, 2, 1, use_cublasmp=False
    )
    constructor = backend._get_tex.return_value.CommOverlapP2P
    if _is_biren(backend):
        # v2.17-style: positional comm_type normalized to native enum.
        assert constructor.call_count == 1
        args = constructor.call_args.args
        assert len(args) >= 5
        assert isinstance(args[4], CommType)
        assert args[4] == CommType.AG
        assert constructor.call_args.kwargs["use_cublasmp"] is False
    else:
        constructor.assert_called_once_with(
            shape, torch.bfloat16, helper, 2, 1, 3, 1, 0, 0, 1, False, False, True, False
        )
    assert result is constructor.return_value


@pytest.mark.parametrize("p2p", [False, True])
def test_cublasmp_rejected_before_loading_vendor_extension(backend, p2p):
    if _is_biren(backend):
        # Biren forwards use_cublasmp to the native extension (SUPA redirects
        # the cuBLASMp overlap path to suBLASMp); no early rejection.
        pytest.skip("biren forwards use_cublasmp to native")
    args = ([128, 64], torch.bfloat16, object(), 2)
    factory = backend.create_comm_overlap
    if p2p:
        args += (1,)
        factory = backend.create_comm_overlap_p2p
    with pytest.raises(NotImplementedError, match="does not support cuBLASMp"):
        factory(*args, use_cublasmp=True)
    backend._get_tex.assert_not_called()


# --- Biren-specific v2.17-style contract: keyword args + enum normalization ---


@pytest.mark.parametrize("comm_type", [None, 0, 1])
def test_biren_overlap_normalizes_comm_type(backend, comm_type):
    """create_comm_overlap must pass a native CommOverlapType instance.

    FL base.py passes the plugin-side Python IntEnum; pybind11 registers
    CommOverlapType with module_local() and cannot accept int/IntEnum/None.
    """
    if not _is_biren(backend):
        pytest.skip("biren-specific")
    shape, helper = [128, 64], object()
    result = backend.create_comm_overlap(
        shape, torch.bfloat16, helper, 2, 7,
        use_cublasmp=False, comm_type=comm_type,
    )
    tex = backend._get_tex.return_value
    constructor = tex.CommOverlap
    assert constructor.call_count == 1
    kwargs = constructor.call_args.kwargs
    assert "comm_type" in kwargs
    assert isinstance(kwargs["comm_type"], CommType)
    # None -> RS default
    expected = CommType(comm_type) if comm_type is not None else CommType.RS
    assert kwargs["comm_type"] == expected
    assert kwargs["comm_type"] is tex.CommOverlapType(expected.value)
    assert result is constructor.return_value


@pytest.mark.parametrize("comm_type", [0, 1])
def test_biren_p2p_normalizes_comm_type(backend, comm_type):
    """create_comm_overlap_p2p must pass a native CommOverlapType instance."""
    if not _is_biren(backend):
        pytest.skip("biren-specific")
    shape, helper = [128, 64], object()
    result = backend.create_comm_overlap_p2p(
        shape, torch.bfloat16, helper, 2, comm_type, use_cublasmp=False
    )
    tex = backend._get_tex.return_value
    constructor = tex.CommOverlapP2P
    assert constructor.call_count == 1
    args = constructor.call_args.args
    assert len(args) >= 5
    assert isinstance(args[4], CommType)
    assert args[4] == CommType(comm_type)
    assert result is constructor.return_value


def test_biren_cublasmp_forwarded_to_native(backend):
    """Biren forwards use_cublasmp to the native extension (SUPA translation)."""
    if not _is_biren(backend):
        pytest.skip("biren-specific")
    shape, helper = [128, 64], object()
    backend.create_comm_overlap(
        shape, torch.bfloat16, helper, 2, use_cublasmp=True
    )
    kwargs = backend._get_tex.return_value.CommOverlap.call_args.kwargs
    assert kwargs["use_cublasmp"] is True
