# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Check normalization quantizer forwarding at the CUDA extension boundary."""

from enum import IntEnum
from types import SimpleNamespace

import pytest

from transformer_engine.plugin.core.backends.vendor.cuda.cuda import CUDABackend
from transformer_engine.plugin.core.ops import DType


class NativeDType(IntEnum):
    kBFloat16 = 6
    kFloat8E4M3 = 7
    kFloat8E5M2 = 8


@pytest.mark.parametrize("operation", ["layernorm_fwd", "rmsnorm_fwd"])
@pytest.mark.parametrize("dtype", [DType.kFloat8E4M3, DType.kFloat8E5M2, NativeDType.kFloat8E4M3])
def test_norm_normalizes_quantizer_enum_and_preserves_forwarded_arguments(operation, dtype):
    calls = []
    expected = object()
    quantizer = SimpleNamespace(dtype=dtype)

    def native_norm(*args):
        qindex = 5 if operation == "layernorm_fwd" else 4
        assert args[qindex] is quantizer
        assert type(quantizer.dtype) is NativeDType
        assert type(args[qindex + 1]) is NativeDType
        calls.append(args)
        return expected

    backend = object.__new__(CUDABackend)
    backend._get_tex = lambda: SimpleNamespace(DType=NativeDType, **{operation: native_norm})
    inputs = [object(), object()]
    if operation == "layernorm_fwd":
        inputs.append(object())
    arguments = (*inputs, 1e-5, object(), quantizer, DType.kBFloat16, 3, True)
    result = getattr(backend, operation)(*arguments)
    assert result is expected
    assert len(calls) == 1
    assert calls[0][:-3] == arguments[:-3]
    assert calls[0][-2:] == arguments[-2:]
    assert int(quantizer.dtype) == int(dtype)


@pytest.mark.parametrize("operation", ["layernorm_fwd", "rmsnorm_fwd"])
@pytest.mark.parametrize(
    "quantizer", [None, SimpleNamespace(), SimpleNamespace(dtype=None), SimpleNamespace(dtype=999)]
)
def test_norm_preserves_unquantized_and_opaque_quantizers(operation, quantizer):
    calls = []
    backend = object.__new__(CUDABackend)
    backend._get_tex = lambda: SimpleNamespace(
        DType=NativeDType, **{operation: lambda *args: calls.append(args)}
    )
    inputs = [object(), object()]
    if operation == "layernorm_fwd":
        inputs.append(None)
    getattr(backend, operation)(*inputs, 1e-5, None, quantizer, None, 0, False)
    qindex = 5 if operation == "layernorm_fwd" else 4
    assert len(calls) == 1
    assert calls[0][qindex] is quantizer
    assert calls[0][qindex + 1] is None
