# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import List, Tuple, Union
import torch

__all__ = [
    "multi_tensor_scale_torch",
    "multi_tensor_l2norm_torch",
    "multi_tensor_adam_torch",
    "multi_tensor_adam_fp8_torch",
    "multi_tensor_adam_capturable_torch",
    "multi_tensor_adam_capturable_master_torch",
    "multi_tensor_adam_param_remainder_torch",
    "multi_tensor_sgd_torch",
    "multi_tensor_compute_scale_and_scale_inv_torch",
    "multi_tensor_compute_scale_inv_e8m0_torch",
]


def multi_tensor_scale_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    scale: float,
) -> None:
    if noop_flag.item() != 0:
        return

    if len(tensor_lists) != 2:
        raise ValueError("tensor_lists should contain [input_tensors, output_tensors]")

    input_tensors, output_tensors = tensor_lists

    if len(output_tensors) != len(input_tensors):
        raise ValueError("Output and input tensor lists must have the same length")

    for in_tensor, out_tensor in zip(input_tensors, output_tensors):
        # Check for inf/nan (matches CUDA behavior for AMP gradient scaling)
        if not torch.isfinite(in_tensor).all():
            noop_flag.fill_(1)
        out_tensor.copy_(in_tensor * scale)


def multi_tensor_l2norm_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    per_tensor: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute L2 norm of tensors.

    Returns:
        Tuple of (total_norm, per_tensor_norms_or_dummy)
        - total_norm: The combined L2 norm of all tensors
        - per_tensor_norms_or_dummy: Per-tensor norms stacked if per_tensor=True, else dummy tensor
    """
    device = tensor_lists[0][0].device if tensor_lists and tensor_lists[0] else "cpu"

    if noop_flag.item() != 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    tensors = tensor_lists[0]

    # Compute per-tensor norms
    per_tensor_norms = []
    total_norm_sq = torch.tensor(0.0, device=device)

    for tensor in tensors:
        norm_sq = torch.sum(tensor.float() ** 2)
        # Check for inf/nan (matches CUDA behavior)
        if not torch.isfinite(norm_sq):
            noop_flag.fill_(1)
        total_norm_sq = total_norm_sq + norm_sq
        if per_tensor:
            per_tensor_norms.append(torch.sqrt(norm_sq))

    total_norm = torch.sqrt(total_norm_sq)

    if per_tensor:
        per_tensor_result = torch.stack(per_tensor_norms)
    else:
        per_tensor_result = torch.tensor(0.0, device=device)

    return total_norm, per_tensor_result


def multi_tensor_adam_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    step: int,
    mode: int,
    bias_correction: int,
    weight_decay: float,
) -> None:
    """
    Adam optimizer implementation matching CUDA exactly.

    mode == 0: L2 regularization (add weight_decay * param to gradient before moment update)
    mode == 1: AdamW (add weight_decay * param to update after moment computation)
    """
    if noop_flag.item() != 0:
        return

    if len(tensor_lists) != 4:
        raise ValueError("tensor_lists should contain [grads, params, exp_avgs, exp_avg_sqs]")

    grads, params, exp_avgs, exp_avg_sqs = tensor_lists

    if not (len(params) == len(grads) == len(exp_avgs) == len(exp_avg_sqs)):
        raise ValueError("All tensor lists must have the same length")

    if bias_correction:
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
    else:
        bias_correction1 = 1.0
        bias_correction2 = 1.0

    for grad, param, exp_avg, exp_avg_sq in zip(grads, params, exp_avgs, exp_avg_sqs):
        if grad is None:
            continue

        # Convert to float for computation (matches CUDA's MATH_T = float)
        g = grad.float()
        p = param.float()

        if mode == 0:  # L2 regularization
            # Add weight decay to gradient before moment update
            g = g + weight_decay * p

            # Update moments with modified gradient
            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

            # Bias correction
            m_corr = exp_avg / bias_correction1
            v_corr = exp_avg_sq / bias_correction2

            # Compute update
            denom = v_corr.sqrt().add_(epsilon)
            update = m_corr / denom

            # Update parameter
            param.add_(update, alpha=-lr)
        else:  # mode == 1, AdamW (decoupled weight decay)
            # Update moments with original gradient
            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

            # Bias correction
            m_corr = exp_avg / bias_correction1
            v_corr = exp_avg_sq / bias_correction2

            # Compute update with weight decay added (matches CUDA exactly)
            denom = v_corr.sqrt().add_(epsilon)
            update = (m_corr / denom) + (weight_decay * p)

            # Update parameter
            param.add_(update, alpha=-lr)


def multi_tensor_adam_param_remainder_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    step: int,
    mode: int,
    bias_correction: int,
    weight_decay: float,
) -> None:
    """
    Adam optimizer with parameter remainders for BF16 precision.

    This variant stores BF16 parameters + int16 remainders to reconstruct FP32 master weights
    using bit manipulation, matching the CUDA implementation exactly.

    The CUDA implementation stores:
    - p: int16 representing the high 16 bits of FP32 (viewed as BF16)
    - p_remainder: int16 representing the low 16 bits of FP32

    To reconstruct FP32:
    - If p_remainder < 0, decrement p (undo rounding)
    - Combine: fp32.int16[1] = p, fp32.int16[0] = p_remainder

    To split FP32 back:
    - p = fp32.int16[1] (high 16 bits)
    - p_remainder = fp32.int16[0] (low 16 bits)
    - If p_remainder < 0, increment p (round up)

    Args:
        chunk_size: Chunk size for processing (unused in PyTorch implementation)
        noop_flag: If non-zero, skip computation
        tensor_lists: [grads, params (int16/bf16), exp_avgs (fp32), exp_avg_sqs (fp32), param_remainders (int16)]
        lr: Learning rate
        beta1: First moment decay rate
        beta2: Second moment decay rate
        epsilon: Epsilon for numerical stability
        step: Current optimization step
        mode: 0 = L2 regularization, 1 = AdamW (decoupled weight decay)
        bias_correction: Whether to apply bias correction (1 = yes, 0 = no)
        weight_decay: Weight decay coefficient
    """
    if noop_flag.item() != 0:
        return

    if len(tensor_lists) != 5:
        raise ValueError(
            "tensor_lists should contain [grads, params, exp_avgs, exp_avg_sqs, param_remainders]"
        )

    grads, params, exp_avgs, exp_avg_sqs, param_remainders = tensor_lists

    if not (
        len(params) == len(grads) == len(exp_avgs) == len(exp_avg_sqs) == len(param_remainders)
    ):
        raise ValueError("All tensor lists must have the same length")

    if bias_correction:
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
    else:
        bias_correction1 = 1.0
        bias_correction2 = 1.0

    is_adamw = mode == 1

    for grad, param, exp_avg, exp_avg_sq, param_remainder in zip(
        grads, params, exp_avgs, exp_avg_sqs, param_remainders
    ):
        # Convert gradient to float
        g_float = grad.float()

        # Reconstruct FP32 master weight from int16 param + int16 remainder using bit manipulation
        # This matches the CUDA implementation exactly:
        # 1. If p_remainder < 0, decrement p (undo rounding)
        # 2. Combine high 16 bits (p) and low 16 bits (p_remainder) into FP32

        local_p = param.view(torch.int16).clone()
        local_p_rem = param_remainder.clone()

        # Undo rounding: if remainder < 0, decrement p
        local_p = torch.where(local_p_rem < 0, local_p - 1, local_p)

        # Combine into FP32 using bit shift operations
        # local_p is high 16 bits, local_p_rem is low 16 bits
        high_bits = local_p.to(torch.int32) << 16
        low_bits = local_p_rem.to(torch.int32) & 0xFFFF  # Mask off sign extension
        param_int32 = high_bits | low_bits
        param_master = param_int32.view(torch.float32)

        # L2 mode: add weight decay to gradient before updating moments
        if not is_adamw and weight_decay != 0:
            g_float = g_float + weight_decay * param_master

        # Update first moment: m = beta1 * m + (1 - beta1) * g
        exp_avg.mul_(beta1).add_(g_float, alpha=1 - beta1)

        # Update second moment: v = beta2 * v + (1 - beta2) * g^2
        exp_avg_sq.mul_(beta2).addcmul_(g_float, g_float, value=1 - beta2)

        # Apply bias correction
        m_corr = exp_avg / bias_correction1
        v_corr = exp_avg_sq / bias_correction2

        # Compute denominator: sqrt(v_corr) + epsilon
        denom = torch.sqrt(v_corr) + epsilon

        # Compute update
        update = m_corr / denom

        # AdamW mode: add decoupled weight decay to update
        if is_adamw and weight_decay != 0:
            update = update + weight_decay * param_master

        # Update master weight: p = p - lr * update
        param_master = param_master - lr * update

        # Split FP32 back into int16 param + int16 remainder using bit manipulation
        # This matches the CUDA implementation exactly:
        # 1. Extract high 16 bits as p
        # 2. Extract low 16 bits as p_remainder
        # 3. If p_remainder < 0, increment p (round up)

        param_int32 = param_master.view(torch.int32)
        # Extract low 16 bits (remainder) and high 16 bits (param)
        new_p_rem = (param_int32 & 0xFFFF).to(torch.int16)
        new_p = ((param_int32 >> 16) & 0xFFFF).to(torch.int16)

        # Round up: if remainder < 0, increment p
        new_p = torch.where(new_p_rem < 0, new_p + 1, new_p)

        # Write back
        param.view(torch.int16).copy_(new_p)
        param_remainder.copy_(new_p_rem)


def multi_tensor_sgd_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    lr: float,
    momentum: float,
    dampening: float,
    weight_decay: float,
    nesterov: bool,
) -> None:
    if noop_flag.item() != 0:
        return

    if len(tensor_lists) != 3:
        raise ValueError("tensor_lists should contain [params, grads, momentum_buffers]")

    params, grads, momentum_buffers = tensor_lists

    if not (len(params) == len(grads) == len(momentum_buffers)):
        raise ValueError("All tensor lists must have the same length")

    for param, grad, buf in zip(params, grads, momentum_buffers):
        if grad is None:
            continue

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if momentum != 0:
            if buf is None or buf.numel() == 0:
                buf = grad.clone().detach()
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf

        param.add_(grad, alpha=-lr)


def multi_tensor_compute_scale_and_scale_inv_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    max_fp8: float,
    force_pow_2_scales: bool = False,
    amax_epsilon: float = 0.0,
) -> None:
    """
    Compute scale and scale_inv from amax values for FP8 quantization.

    Args:
        chunk_size: Chunk size (unused in PyTorch implementation)
        noop_flag: If non-zero, skip computation
        tensor_lists: [amaxes, scales, scale_invs]
        max_fp8: Maximum representable value in FP8 format (e.g., 448.0 for E4M3)
        force_pow_2_scales: If True, force scales to be powers of 2
        amax_epsilon: Small epsilon to add to amax to avoid division by zero
    """
    if noop_flag.item() != 0:
        return

    if len(tensor_lists) != 3:
        raise ValueError("tensor_lists should contain [amaxes, scales, scale_invs]")

    amaxes, scales, scale_invs = tensor_lists

    if not (len(amaxes) == len(scales) == len(scale_invs)):
        raise ValueError("All tensor lists must have the same length")

    for amax, scale, scale_inv in zip(amaxes, scales, scale_invs):
        # Add epsilon to avoid division by zero
        amax_val = amax + amax_epsilon

        # Compute scale: max_fp8 / amax
        # Clamp amax to avoid very small values
        amax_val = torch.clamp(amax_val, min=1e-12)
        computed_scale = max_fp8 / amax_val

        if force_pow_2_scales:
            # Round scale to nearest power of 2
            log2_scale = torch.log2(computed_scale)
            log2_scale = torch.round(log2_scale)
            computed_scale = torch.pow(2.0, log2_scale)

        # Update scale and scale_inv
        scale.copy_(computed_scale)
        scale_inv.copy_(1.0 / computed_scale)


def multi_tensor_compute_scale_inv_e8m0_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    block_len: int,
) -> None:
    """
    Compute scale_inv in E8M0 format from amax values for MXFP8 quantization.

    Args:
        chunk_size: Chunk size (unused in PyTorch implementation)
        noop_flag: If non-zero, skip computation
        tensor_lists: [amaxes, scale_invs]
        block_len: Block length for block-wise scaling
    """
    if noop_flag is not None and noop_flag.item() != 0:
        return

    if len(tensor_lists) != 2:
        raise ValueError("tensor_lists should contain [amaxes, scale_invs]")

    amaxes, scale_invs = tensor_lists

    if len(amaxes) != len(scale_invs):
        raise ValueError("All tensor lists must have the same length")

    for amax, scale_inv in zip(amaxes, scale_invs):
        amax_val = torch.clamp(amax, min=2**-127)
        # E8M0: biased exponent = floor(log2(amax)) + 127
        log2_amax = torch.floor(torch.log2(amax_val))
        biased_exp = (log2_amax + 127).to(torch.uint8)
        scale_inv.copy_(biased_exp)


def multi_tensor_adam_fp8_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    step: int,
    mode: int,
    bias_correction: int,
    weight_decay: float,
    fp8_dtype,
) -> None:
    """
    FP8 adam optimizer - reference backend fallback.

    Note: This is a fallback implementation that uses FP32 computation instead of FP8.
    FP8 training is a GPU-specific feature and not supported in the reference backend.
    """
    if fp8_dtype is not None:
        raise NotImplementedError(
            "FP8 adam is not supported in the reference backend. "
            "FP8 training requires GPU acceleration. "
            "Please use a CUDA-enabled build or disable FP8 optimization."
        )

    # Fallback to regular adam with FP32 computation
    multi_tensor_adam_torch(
        chunk_size,
        noop_flag,
        tensor_lists,
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        mode,
        bias_correction,
        weight_decay,
    )


def multi_tensor_adam_capturable_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    lr: torch.Tensor,
    beta1: float,
    beta2: float,
    epsilon: float,
    step: torch.Tensor,
    mode: int,
    bias_correction: int,
    weight_decay: float,
    inv_scale: torch.Tensor,
) -> None:
    """
    Capturable adam optimizer - reference backend fallback.

    Note: This is a fallback implementation that does not support CUDA graph capture.
    CUDA graph capture is a GPU-specific feature and not supported in the reference backend.
    """
    if isinstance(lr, torch.Tensor) and lr.requires_grad:
        raise NotImplementedError(
            "Capturable adam with tensor lr is not supported in the reference backend. "
            "CUDA graph capture requires GPU acceleration. "
            "Please use a CUDA-enabled build or use scalar lr."
        )

    if isinstance(step, torch.Tensor) and step.requires_grad:
        raise NotImplementedError(
            "Capturable adam with tensor step is not supported in the reference backend. "
            "CUDA graph capture requires GPU acceleration. "
            "Please use a CUDA-enabled build or use scalar step."
        )

    # Fallback to regular adam with scalar parameters
    multi_tensor_adam_torch(
        chunk_size,
        noop_flag,
        tensor_lists,
        lr.item() if isinstance(lr, torch.Tensor) else lr,
        beta1,
        beta2,
        epsilon,
        step.item() if isinstance(step, torch.Tensor) else step,
        mode,
        bias_correction,
        weight_decay,
    )


def multi_tensor_adam_capturable_master_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    lr: torch.Tensor,
    beta1: float,
    beta2: float,
    epsilon: float,
    step: torch.Tensor,
    mode: int,
    bias_correction: int,
    weight_decay: float,
    inv_scale: torch.Tensor,
) -> None:
    """
    Capturable master adam optimizer - reference backend fallback.

    Note: This is a fallback implementation that does not support CUDA graph capture
    or master weight management. These are GPU-specific features.
    """
    if isinstance(lr, torch.Tensor) and lr.requires_grad:
        raise NotImplementedError(
            "Capturable master adam with tensor lr is not supported in the reference backend. "
            "CUDA graph capture requires GPU acceleration. "
            "Please use a CUDA-enabled build or use scalar lr."
        )

    if isinstance(step, torch.Tensor) and step.requires_grad:
        raise NotImplementedError(
            "Capturable master adam with tensor step is not supported in the reference backend. "
            "CUDA graph capture requires GPU acceleration. "
            "Please use a CUDA-enabled build or use scalar step."
        )

    # Fallback to regular adam with scalar parameters
    multi_tensor_adam_torch(
        chunk_size,
        noop_flag,
        tensor_lists,
        lr.item() if isinstance(lr, torch.Tensor) else lr,
        beta1,
        beta2,
        epsilon,
        step.item() if isinstance(step, torch.Tensor) else step,
        mode,
        bias_correction,
        weight_decay,
    )
