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
    return instance


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
    constructor.assert_called_once_with(
        shape, torch.bfloat16, helper, 2, 1, 3, 1, 0, 0, 1, False, False, True, False
    )
    assert result is constructor.return_value


@pytest.mark.parametrize("p2p", [False, True])
def test_cublasmp_rejected_before_loading_vendor_extension(backend, p2p):
    args = ([128, 64], torch.bfloat16, object(), 2)
    factory = backend.create_comm_overlap
    if p2p:
        args += (1,)
        factory = backend.create_comm_overlap_p2p
    with pytest.raises(NotImplementedError, match="does not support cuBLASMp"):
        factory(*args, use_cublasmp=True)
    backend._get_tex.assert_not_called()
