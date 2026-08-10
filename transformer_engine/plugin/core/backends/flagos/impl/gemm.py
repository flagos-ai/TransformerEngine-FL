# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Any, Dict, List, Optional, Tuple, Union
import torch

import flag_gems


__all__ = [
    "generic_gemm_fl",
    "te_general_grouped_gemm_fl",
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


def validate_gemm_scale(scale: Optional[float], required: bool) -> float:
    if required:
        return scale if scale is not None else 1.0
    if scale not in (0.0, None):
        raise ValueError("scale must be zero")
    return 0.0


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


def generic_gemm_fl(
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

    assert not gelu and gelu_in is None, "Triton-Based General Gemm do not support gelu now"
    assert quantizer is None, "Triton-Based General Gemm do not support quantization now"

    alpha = validate_gemm_scale(alpha, True)
    beta = validate_gemm_scale(beta, accumulate)

    s = -1
    b = -1
    orig_A_shape = A.shape
    orig_B_shape = B.shape
    shape_a_changed = False
    shape_b_changed = False

    if A.ndim == 3:
        A = A.view(-1, A.shape[-1])
        shape_a_changed = True

    if B.ndim == 3:
        s, b, _ = B.shape
        B = B.view(-1, B.shape[-1])
        shape_b_changed = True

    A_comp = A.T if transA else A
    B_comp = B.T if transB else B

    bias_grad = None
    if grad:
        out1 = flag_gems.mm(B_comp, A_comp)
        if bias is not None:
            bias_grad = flag_gems.sum_dim(B, dim=[0])
    else:
        # NOTE(wqq) flag_gems.addmm uses beta for bias scaling (Y = alpha * WX + beta * bias),
        # unlike the beta here (for scaling D). Always set to 1.
        if bias is not None:
            out1 = flag_gems.addmm(bias, B_comp, A_comp, beta=1, alpha=alpha)
        else:
            out1 = flag_gems.mm(B_comp, A_comp)

    if shape_b_changed:
        out1 = out1.view(s, b, -1)

    torch_out_dtype = _convert_dtype(output_dtype)
    if torch_out_dtype is not None and out1.dtype != torch_out_dtype:
        out1 = out1.to(torch_out_dtype)

    gelu_input = None
    extra_output_ret = None

    if D is not None:
        if accumulate:
            flag_gems.add_(D, out1)
        else:
            flag_gems.copy_(D, out1)
        return D, bias_grad, gelu_input, extra_output_ret
    else:
        return out1, bias_grad, gelu_input, extra_output_ret


# This function can represent both forward and backward computations.
# When grad is False (forward computation), the 'bias' is bias;
# When grad is True (backward computation/gradient calculation), the 'bias' is grad_bias;
def te_general_grouped_gemm_fl(
    B: List[torch.Tensor],
    transb: bool,
    A: List[torch.Tensor],
    transa: bool,
    D: Optional[List[torch.Tensor]],
    D_type: Any,
    m_splits: List[int],
    bias: List[torch.Tensor],  # bias or grad_bias
    bias_type: Any,
    single_output: bool,
    pre_gelu_out: List[torch.Tensor],
    grad: bool,
    workspace: List[torch.Tensor],
    workspaceSize: int,
    accumulate: bool,
    use_split_accumulator: bool,
    math_sm_count: int,
) -> Optional[List[torch.Tensor]]:
    if single_output and D is None:
        raise ValueError("not implemented, D should be allocated for single output case.")

    num_gemms = len(A)
    if D is None:
        D = []
        for i in range(num_gemms):
            m = A[i].shape[1] if transa else A[i].shape[0]
            n = B[i].shape[0] if transb else B[i].shape[1]
            D.append(torch.empty((m, n), dtype=D[i].dtype, device=A[0].device))

    temp_D = []
    for i in range(num_gemms):
        # Handle the special case of zero-element inputs
        if A[i].numel() == 0 or B[i].numel() == 0:
            if not single_output:
                if D[i].numel() != 0 and not accumulate:
                    flag_gems.copy_(D[i], flag_gems.zeros(D[i].shape))
            else:
                out = flag_gems.zeros((A[i].shape[0], B[i].shape[1]))
            if grad and len(bias) > i and bias[i] is not None and bias[i].numel() != 0:
                flag_gems.copy_(bias[i], flag_gems.zeros(bias[i].shape))
            if (
                len(pre_gelu_out) > i
                and pre_gelu_out[i] is not None
                and pre_gelu_out[i].numel() != 0
            ):
                flag_gems.copy_(pre_gelu_out[i], flag_gems.zeros(pre_gelu_out[i].shape))
            continue

        a = A[i].t() if transa else A[i]
        b = B[i].t() if transb else B[i]
        # Determine presence of epilogue tensors
        has_bias = len(bias) > i and bias[i] is not None and bias[i].numel() > 0
        has_pre_gelu = (
            len(pre_gelu_out) > i and pre_gelu_out[i] is not None and pre_gelu_out[i].numel() > 0
        )

        # Forward Pass calculation
        if not grad:
            if has_bias:
                # Fused matrix multiplication and bias addition
                out = flag_gems.addmm(bias[i], a, b)
            else:
                out = flag_gems.mm(a, b)

            # Apply GELU epilogue if pre_gelu_out is provided
            if has_pre_gelu:
                flag_gems.copy_(pre_gelu_out[i], out)
                out = flag_gems.gelu(out)
        else:
            out = flag_gems.mm(a, b)

            # Apply dGELU epilogue if requested
            if has_pre_gelu:
                out = flag_gems.gelu_backward(out, pre_gelu_out[i])

            # Compute bias gradients if requested
            if has_bias:
                bias_grad = flag_gems.sum_dim(out, dim=[0])
                if accumulate:
                    flag_gems.add_(bias[i], bias_grad)
                else:
                    flag_gems.copy_(bias[i], bias_grad)

        if not single_output:
            # Store output
            if accumulate:
                flag_gems.add_(D[i], out.to(D[i].dtype))
            else:
                flag_gems.copy_(D[i], out.to(D[i].dtype))
        else:
            temp_D.append(out.to(D[0].dtype))

    if single_output:
        if temp_D:
            temp = flag_gems.cat(temp_D, dim=0)
            flag_gems.copy_(D[0], temp)

    return bias
