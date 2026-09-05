# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""TENPU-parity tests for the NPU LayerNorm forward and backward backend."""

from __future__ import annotations

import itertools
import os
import unittest

import torch
import torch.nn.functional as F


os.environ.setdefault("PLATFORM", "ascend")
os.environ.setdefault("TE_FL_SKIP_CUDA", "1")
os.environ.setdefault("NVTE_FRAMEWORK", "pytorch")
os.environ.setdefault("NVTE_WITH_NCCL_EP", "0")

try:
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = torch.npu.is_available()
except (ImportError, AttributeError):
    _NPU_AVAILABLE = False


_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_SHAPES = ((4, 16), (2, 3, 16))


def _tenpu_forward(x, weight, bias, eps, zero_centered_gamma):
    """Mirror TENPU LayerNormLinear forward semantics."""

    gamma = weight + 1 if zero_centered_gamma else weight
    output = F.layer_norm(x, [x.shape[-1]], weight=gamma, bias=bias, eps=eps)
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, unbiased=False, keepdim=True)
    rsigma = torch.rsqrt(variance + eps)
    return output, mean.squeeze(-1), rsigma.squeeze(-1)


def _tenpu_backward(grad, x, weight, mean, rsigma, zero_centered_gamma):
    """Mirror TENPU's explicit LayerNorm gradient composition."""

    if mean.ndim < x.ndim:
        mean = mean.unsqueeze(-1)
    if rsigma.ndim < x.ndim:
        rsigma = rsigma.unsqueeze(-1)
    hidden_size = x.shape[-1]
    x_hat = (x - mean) * rsigma
    gamma = weight + 1 if zero_centered_gamma else weight
    dx_hat = grad * gamma
    dvar = (dx_hat * (x - mean) * (-0.5) * rsigma.pow(3)).sum(
        dim=-1, keepdim=True
    )
    dmean = (-dx_hat * rsigma).sum(dim=-1, keepdim=True) + dvar * (
        -2.0 / hidden_size
    ) * (x - mean).sum(dim=-1, keepdim=True)
    dx = dx_hat * rsigma + dvar * 2.0 / hidden_size * (x - mean) + dmean / hidden_size
    reduce_dims = tuple(range(grad.ndim - 1))
    dweight = (grad * x_hat).sum(dim=reduce_dims)
    dbias = grad.sum(dim=reduce_dims)
    return dx, dweight, dbias


@unittest.skipUnless(_NPU_AVAILABLE, "Ascend NPU is required")
class TestLayerNormTENPUParity(unittest.TestCase):
    """Require exact parity with TENPU's forward and manual backward."""

    @classmethod
    def setUpClass(cls):
        import transformer_engine  # noqa: F401  # Install transformer_engine_torch shim.
        import transformer_engine_torch as tex
        from transformer_engine.plugin.core.ops import DType

        cls.tex = tex
        cls.DType = DType

    @staticmethod
    def _te_dtype(dtype):
        from transformer_engine.plugin.core.ops import DType

        return {
            torch.float32: DType.kFloat32,
            torch.float16: DType.kFloat16,
            torch.bfloat16: DType.kBFloat16,
        }[dtype]

    def test_dense_forward_backward_matrix(self):
        """24 cases: dtype, rank, optional bias, and zero-centered gamma."""

        for dtype, shape, with_bias, zero_centered_gamma in itertools.product(
            _DTYPES, _SHAPES, (False, True), (False, True)
        ):
            with self.subTest(
                dtype=dtype,
                shape=shape,
                with_bias=with_bias,
                zero_centered_gamma=zero_centered_gamma,
            ):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(20260901)
                x = torch.randn(shape, generator=generator, dtype=torch.float32).to(
                    device="npu", dtype=dtype
                )
                weight = torch.randn(
                    (shape[-1],), generator=generator, dtype=torch.float32
                ).to(device="npu", dtype=dtype)
                bias = (
                    torch.randn(
                        (shape[-1],), generator=generator, dtype=torch.float32
                    ).to(device="npu", dtype=dtype)
                    if with_bias
                    else None
                )
                eps = 1.0e-5

                tefl = self.tex.layernorm_fwd(
                    x,
                    weight,
                    bias,
                    eps,
                    None,
                    None,
                    self._te_dtype(dtype),
                    0,
                    zero_centered_gamma,
                )
                tenpu = _tenpu_forward(x, weight, bias, eps, zero_centered_gamma)
                for actual, expected in zip(tefl, tenpu):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

                grad = torch.randn(
                    tuple(tefl[0].shape), generator=generator, dtype=torch.float32
                ).to(device="npu", dtype=dtype)
                tefl_grad = self.tex.layernorm_bwd(
                    grad,
                    x,
                    tefl[1],
                    tefl[2],
                    weight,
                    0,
                    zero_centered_gamma,
                )
                tenpu_grad = _tenpu_backward(
                    grad, x, weight, tenpu[1], tenpu[2], zero_centered_gamma
                )
                for actual, expected in zip(tefl_grad, tenpu_grad):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dense_output_reuse_and_otype(self):
        """Honor TE-FL's preallocated dense output and output dtype ABI."""

        x = torch.randn((4, 16), device="npu", dtype=torch.float32)
        weight = torch.randn((16,), device="npu", dtype=torch.float32)
        bias = torch.randn((16,), device="npu", dtype=torch.float32)
        output_buffer = torch.empty_like(x, dtype=torch.bfloat16)
        output, mean, rsigma = self.tex.layernorm_fwd(
            x,
            weight,
            bias,
            1.0e-5,
            output_buffer,
            None,
            self.DType.kBFloat16,
            0,
            False,
        )
        expected, expected_mean, expected_rsigma = _tenpu_forward(
            x, weight, bias, 1.0e-5, False
        )
        self.assertIs(output, output_buffer)
        torch.testing.assert_close(output, expected.bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(mean, expected_mean, rtol=0, atol=0)
        torch.testing.assert_close(rsigma, expected_rsigma, rtol=0, atol=0)

    def test_z_current_scaling_quantizer_matches_separate_tenpu_quantize(self):
        """TENPU quantizes dense LayerNorm output as a separate operation."""

        device_name = torch.npu.get_device_name(0)
        if "950" not in device_name:
            self.skipTest(
                "NPU dynamic FP8 quantization requires Ascend950; "
                f"current device is {device_name}"
            )

        from transformer_engine.pytorch.constants import DType as PyDType
        from transformer_engine.pytorch.tensor.float8_tensor import (
            Float8CurrentScalingQuantizer,
        )

        x = torch.randn((4, 32), device="npu", dtype=torch.bfloat16)
        weight = torch.randn((32,), device="npu", dtype=torch.bfloat16)
        bias = torch.randn((32,), device="npu", dtype=torch.bfloat16)
        tefl_quantizer = Float8CurrentScalingQuantizer(
            PyDType.kFloat8E4M3,
            x.device,
            rowwise=True,
            columnwise=False,
        )
        expected_quantizer = Float8CurrentScalingQuantizer(
            PyDType.kFloat8E4M3,
            x.device,
            rowwise=True,
            columnwise=False,
        )
        output, mean, rsigma = self.tex.layernorm_fwd(
            x,
            weight,
            bias,
            1.0e-5,
            None,
            tefl_quantizer,
            self.DType.kBFloat16,
            0,
            False,
        )
        dense, expected_mean, expected_rsigma = _tenpu_forward(
            x, weight, bias, 1.0e-5, False
        )
        expected = expected_quantizer.quantize(dense)
        torch.testing.assert_close(output._data, expected._data, rtol=0, atol=0)
        torch.testing.assert_close(output._scale_inv, expected._scale_inv, rtol=0, atol=0)
        torch.testing.assert_close(mean, expected_mean, rtol=0, atol=0)
        torch.testing.assert_close(rsigma, expected_rsigma, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
