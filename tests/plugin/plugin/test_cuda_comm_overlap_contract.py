"""Check CUDA userbuffer forwarding against the native constructor contract."""

from enum import IntEnum
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from transformer_engine.plugin.core.backends.vendor.cuda.cuda import CUDABackend


class CommType(IntEnum):
    RS = 0
    AG = 1


@pytest.mark.parametrize("use_cublasmp", [False, True])
@pytest.mark.parametrize("comm_type", [None, CommType.RS, CommType.AG])
def test_overlap_forwards_options_without_positional_shift(use_cublasmp, comm_type):
    constructor = Mock(return_value=object())
    backend = object.__new__(CUDABackend)
    backend._get_tex = lambda: SimpleNamespace(CommOverlap=constructor, CommOverlapType=CommType)
    shape, helper = [128, 64], object()
    result = backend.create_comm_overlap(
        shape,
        torch.bfloat16,
        helper,
        2,
        7,
        use_cublasmp=use_cublasmp,
        comm_type=comm_type,
    )
    assert result is constructor.return_value
    assert constructor.call_args.args == (shape, torch.bfloat16, helper, 2)
    assert constructor.call_args.kwargs["num_splits"] == 7
    assert constructor.call_args.kwargs["use_cublasmp"] is use_cublasmp
    assert constructor.call_args.kwargs["comm_type"] == (
        CommType.RS if comm_type is None else comm_type
    )


@pytest.mark.parametrize("use_cublasmp", [False, True])
def test_p2p_accepts_frontend_cublasmp_option(use_cublasmp):
    constructor = Mock(return_value=object())
    backend = object.__new__(CUDABackend)
    backend._get_tex = lambda: SimpleNamespace(CommOverlapP2P=constructor, CommOverlapType=CommType)
    shape, helper = [128, 64], object()
    backend.create_comm_overlap_p2p(
        shape, torch.bfloat16, helper, 2, int(CommType.AG), use_cublasmp=use_cublasmp
    )
    assert constructor.call_args.args[:5] == (shape, torch.bfloat16, helper, 2, CommType.AG)
    assert constructor.call_args.kwargs == {"use_cublasmp": use_cublasmp}
