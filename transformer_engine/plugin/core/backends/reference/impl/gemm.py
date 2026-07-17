# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Any, Optional, Tuple, Union
import torch

from .activation import dgelu_torch

__all__ = [
    "general_gemm_torch",
]

_DTYPE_TO_TORCH = {
    0: torch.uint8,
    2: torch.int32,
    4: torch.float32,
    5: torch.float16,
    6: torch.bfloat16,
    7: torch.float8_e4m3fn,
    8: torch.float8_e5m2,
}


def _convert_dtype(dtype: Union[int, torch.dtype, None]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, int):
        return _DTYPE_TO_TORCH.get(dtype, None)
    if hasattr(dtype, "value"):
        return _DTYPE_TO_TORCH.get(dtype.value, None)
    return None


def general_gemm_torch(
    A: torch.Tensor,
    transA: bool,
    B: torch.Tensor,
    transB: bool,
    D: Optional[torch.Tensor],
    quantizer: Any,
    output_dtype: Any,
    bias: Optional[torch.Tensor],
    bias_type: Any,
    gelu: bool,
    gelu_in: Optional[torch.Tensor],
    grad: bool,
    workspace: torch.Tensor,
    workspace_size: int,
    accumulate: bool,
    use_split_accumulator: bool,
    comm_overlap: Optional[Any] = None,
    comm_type: Optional[Any] = None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
    alpha: float = 1.0,
    beta: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    import torch.nn.functional as F

    target_device = B.device

    if A.device != target_device:
        A = A.to(target_device)

    original_B_shape = None
    if B.ndim == 3:
        original_B_shape = B.shape
        B = B.reshape(-1, B.shape[-1])

    if A.ndim == 3:
        A = A.reshape(-1, A.shape[-1])

    A_comp = A.T if transA else A
    B_comp = B.T if transB else B

    if A_comp.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        compute_dtype = torch.bfloat16
        A_comp = A_comp.to(compute_dtype)
        B_comp = B_comp.to(compute_dtype)

    out = torch.mm(B_comp, A_comp)

    if alpha != 1.0:
        out = out * alpha

    # A non-transposed B contributes its outer dimensions to the output. A
    # transposed B does not, so its flattened shape must not be restored (the
    # latter is the layout normally used by weight-gradient GEMMs).
    if original_B_shape is not None and not transB:
        out = out.view(original_B_shape[0], original_B_shape[1], -1)

    gelu_input_ret = None

    # In a backward GEMM, `bias` only requests the fused BGRAD epilogue. Its
    # value is not added to the GEMM result.
    if bias is not None and not grad:
        if bias.device != target_device:
            bias = bias.to(target_device)
        out = out + bias

    if gelu:
        if grad:
            if gelu_in is None:
                raise ValueError("gelu_in must be provided for a backward GELU GEMM")
            out = dgelu_torch(out, gelu_in, quantizer=None)
        else:
            if gelu_in is not None:
                gelu_in.copy_(out)
                gelu_input_ret = gelu_in
            else:
                gelu_input_ret = out.clone()
            out = F.gelu(out, approximate="tanh")

    torch_out_dtype = _convert_dtype(output_dtype)
    if torch_out_dtype is not None and out.dtype != torch_out_dtype:
        out = out.to(torch_out_dtype)

    if D is not None:
        if D.device != target_device:
            D = D.to(target_device)
        if accumulate:
            beta_val = beta if beta is not None else 1.0
            D.mul_(beta_val).add_(out)
            out = D
        else:
            D.copy_(out)
            out = D

    bias_grad = None
    if grad and bias is not None:
        # cuBLASLt's BGRADB epilogue always reduces GEMM input B. Flattening
        # all leading dimensions also handles sequence-shaped gradient input.
        bias_grad = B.sum(dim=0).to(dtype=out.dtype)

    extra_output_ret = None

    return out, bias_grad, gelu_input_ret, extra_output_ret
