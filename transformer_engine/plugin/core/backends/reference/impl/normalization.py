# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Any, Optional, Tuple
import torch
from ....ops import DType

__all__ = [
    "layernorm_fwd_torch",
    "layernorm_bwd_torch",
]

# Mapping from DType enum to torch.dtype
_DTYPE_TO_TORCH_DTYPE = {
    DType.kByte: torch.uint8,
    DType.kInt16: torch.int16,
    DType.kInt32: torch.int32,
    DType.kInt64: torch.int64,
    DType.kFloat32: torch.float32,
    DType.kFloat16: torch.float16,
    DType.kBFloat16: torch.bfloat16,
    DType.kFloat8E4M3: torch.float8_e4m3fn,
    DType.kFloat8E5M2: torch.float8_e5m2,
}


def _to_torch_dtype(dtype):
    """Convert DType enum to torch.dtype."""
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, (int, DType)):
        dtype_enum = DType(dtype)
        if dtype_enum in _DTYPE_TO_TORCH_DTYPE:
            return _DTYPE_TO_TORCH_DTYPE[dtype_enum]
    raise ValueError(f"Unsupported dtype: {dtype}")


def layernorm_fwd_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    ln_out: Optional[torch.Tensor],
    quantizer: Any,
    odtype: DType,
    sm_margin: int,
    zero_centered_gamma: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    odtype = _to_torch_dtype(odtype)
    compute_dtype = (
        torch.float32 if input.dtype in (torch.float16, torch.bfloat16) else input.dtype
    )
    gamma = (1.0 + weight if zero_centered_gamma else weight).to(compute_dtype)
    output, mean, rsigma = torch.native_layer_norm(
        input.to(compute_dtype),
        (input.shape[-1],),
        gamma,
        bias.to(compute_dtype) if bias is not None else None,
        eps,
    )
    return output.to(odtype or input.dtype), mean.squeeze(-1), rsigma.squeeze(-1)


def layernorm_bwd_torch(
    dy: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    rsigma: torch.Tensor,
    gamma: torch.Tensor,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_dtype = x.dtype
    compute_dtype = (
        torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    )
    if mu.ndim < x.ndim:
        mu = mu.unsqueeze(-1)
    if rsigma.ndim < x.ndim:
        rsigma = rsigma.unsqueeze(-1)
    gamma_adj = (1.0 + gamma if zero_centered_gamma else gamma).to(compute_dtype)
    # Pass a tensor for bias: some PyTorch builds cannot handle None here.
    dummy_bias = torch.zeros_like(gamma_adj)
    dx, dgamma, dbeta = torch.ops.aten.native_layer_norm_backward.default(
        dy.to(compute_dtype).contiguous(),
        x.to(compute_dtype).contiguous(),
        list(gamma.shape),
        mu.to(compute_dtype).contiguous(),
        rsigma.to(compute_dtype).contiguous(),
        gamma_adj.contiguous(),
        dummy_bias,
        [True, True, True],
    )
    return dx.to(input_dtype), dgamma.to(gamma.dtype), dbeta.to(gamma.dtype)
