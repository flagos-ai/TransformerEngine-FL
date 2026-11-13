# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import List
import torch
import flag_gems


def _validate_tensor_lists(tensor_lists, expected_lists, name):
    assert (
        len(tensor_lists) == expected_lists
    ), f"{name} expected {expected_lists} tensor lists, got {len(tensor_lists)}"
    num_tensors = len(tensor_lists[0])
    assert num_tensors > 0, "No tensors provided"
    for i, tensor_list in enumerate(tensor_lists):
        assert (
            len(tensor_list) == num_tensors
        ), f"List {i} has {len(tensor_list)} tensors, expected {num_tensors}"
    return num_tensors


def _capturable_adam_fl(
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    eps,
    step,
    mode,
    bias_correction,
    weight_decay,
    inv_scale,
    master_weights,
):
    expected_lists = 5 if master_weights else 4
    name = "capturable master Adam" if master_weights else "capturable Adam"
    num_tensors = _validate_tensor_lists(tensor_lists, expected_lists, name)
    active = noop_flag.reshape(-1)[0] == 0
    correction1 = 1.0
    correction2 = 1.0
    if bias_correction == 1:
        correction1 = 1 - beta1**step
        correction2 = 1 - beta2**step

    for i in range(num_tensors):
        grad = tensor_lists[0][i]
        model_param = tensor_lists[1][i]
        exp_avg = tensor_lists[2][i]
        exp_avg_sq = tensor_lists[3][i]
        master_param = tensor_lists[4][i] if master_weights else None
        param = master_param if master_param is not None else model_param
        unscaled_grad = grad.float() * inv_scale
        adam_grad = unscaled_grad
        if mode == 0 and weight_decay != 0:
            adam_grad = adam_grad + weight_decay * param.float()
        next_avg = beta1 * exp_avg.float() + (1 - beta1) * adam_grad
        next_avg_sq = beta2 * exp_avg_sq.float() + (1 - beta2) * adam_grad.square()
        update = (next_avg / correction1) / ((next_avg_sq / correction2).sqrt() + eps)
        if mode == 1 and weight_decay != 0:
            update = update + weight_decay * param.float()
        next_param = param.float() - lr * update
        grad.copy_(torch.where(active, unscaled_grad.to(grad.dtype), grad))
        exp_avg.copy_(torch.where(active, next_avg.to(exp_avg.dtype), exp_avg))
        exp_avg_sq.copy_(torch.where(active, next_avg_sq.to(exp_avg_sq.dtype), exp_avg_sq))
        param.copy_(torch.where(active, next_param.to(param.dtype), param))
        if master_param is not None:
            model_param.copy_(torch.where(active, next_param.to(model_param.dtype), model_param))


def multi_tensor_adam_capturable_fl(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    eps,
    step,
    mode,
    bias_correction,
    weight_decay,
    inv_scale,
):
    _capturable_adam_fl(
        noop_flag,
        tensor_lists,
        lr,
        beta1,
        beta2,
        eps,
        step,
        mode,
        bias_correction,
        weight_decay,
        inv_scale,
        master_weights=False,
    )


def multi_tensor_adam_capturable_master_fl(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    eps,
    step,
    mode,
    bias_correction,
    weight_decay,
    inv_scale,
):
    _capturable_adam_fl(
        noop_flag,
        tensor_lists,
        lr,
        beta1,
        beta2,
        eps,
        step,
        mode,
        bias_correction,
        weight_decay,
        inv_scale,
        master_weights=True,
    )


def multi_tensor_adam_fp8_fl(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    eps,
    step,
    mode,
    bias_correction,
    weight_decay,
    fp8_dtype,
):
    if noop_flag.item() != 0:
        return
    num_tensors = _validate_tensor_lists(tensor_lists, 8, "FP8 Adam")
    dtype_value = int(fp8_dtype)
    if dtype_value == 7:
        torch_dtype = torch.float8_e4m3fn
    elif dtype_value == 8:
        torch_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 dtype: {fp8_dtype}")
    grads, params, exp_avgs, exp_avg_sqs, masters, scales, amaxes, scale_invs = tensor_lists
    multi_tensor_adam_fl(
        chunk_size,
        noop_flag,
        [grads, masters, exp_avgs, exp_avg_sqs],
        lr,
        beta1,
        beta2,
        eps,
        step,
        mode,
        bias_correction,
        weight_decay,
    )
    for i in range(num_tensors):
        quantized = (masters[i] * scales[i]).to(torch_dtype)
        params[i].copy_(
            quantized.view(torch.uint8) if params[i].dtype == torch.uint8 else quantized
        )
        amaxes[i].copy_(torch.maximum(amaxes[i], masters[i].abs().max()))
        scale_invs[i].copy_(scales[i].reciprocal())


def multi_tensor_adam_fl(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
    mode: int,
    bias_correction: int,
    weight_decay: float,
) -> None:

    if noop_flag.item() != 0:
        return

    num_lists = len(tensor_lists)
    assert num_lists in [4, 5], f"Expected 4 or 5 tensor lists, got {num_lists}"

    num_tensors = len(tensor_lists[0])
    assert num_tensors > 0, "No tensors provided"

    for i, lst in enumerate(tensor_lists):
        assert len(lst) == num_tensors, f"List {i} has {len(lst)} tensors, expected {num_tensors}"

    bias_correction1 = 1.0
    bias_correction2 = 1.0
    if bias_correction == 1:
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

    is_adamw = mode == 1

    for i in range(num_tensors):
        g = tensor_lists[0][i]
        model_param = tensor_lists[1][i]
        m = tensor_lists[2][i]
        v = tensor_lists[3][i]
        p_master = tensor_lists[4][i] if num_lists == 5 else None
        param_to_update = p_master if p_master is not None else model_param

        if not g.is_contiguous():
            g = g.contiguous()

        # Adam moments are maintained in FP32. Promote low-precision gradients
        # before both moment updates so the square is not rounded in BF16.
        adam_grad = g.to(dtype=m.dtype)
        if not is_adamw and weight_decay != 0:
            adam_grad = flag_gems.add(adam_grad, param_to_update, alpha=weight_decay)

        flag_gems.mul_(m, beta1)
        flag_gems.add_(m, adam_grad, alpha=1 - beta1)

        grad_squared = flag_gems.mul(adam_grad, adam_grad)
        flag_gems.mul_(v, beta2)
        flag_gems.add_(v, grad_squared, alpha=1 - beta2)

        m_corr = m
        v_corr = v
        if bias_correction == 1:
            m_corr = flag_gems.true_divide(m_corr, bias_correction1)
            v_corr = flag_gems.true_divide(v_corr, bias_correction2)

        update = flag_gems.true_divide(m_corr, flag_gems.add(flag_gems.sqrt(v_corr), eps))

        if is_adamw and weight_decay != 0:
            flag_gems.mul_(param_to_update, 1 - lr * weight_decay)

        flag_gems.add_(param_to_update, update, alpha=-lr)

        if p_master is not None:
            flag_gems.copy_(model_param, p_master.to(dtype=model_param.dtype))


def multi_tensor_adam_param_remainder_fl(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
    mode: int,
    bias_correction: int,
    weight_decay: float,
) -> None:
    """
    Adam optimizer with parameter remainders for BF16 precision (FlagOS implementation).
    """
    if noop_flag.item() != 0:
        return

    num_lists = len(tensor_lists)
    assert num_lists == 5, f"Expected 5 tensor lists, got {num_lists}"

    num_tensors = len(tensor_lists[0])
    assert num_tensors > 0, "No tensors provided"

    for i, lst in enumerate(tensor_lists):
        assert len(lst) == num_tensors, f"List {i} has {len(lst)} tensors, expected {num_tensors}"

    bias_correction1 = 1.0
    bias_correction2 = 1.0
    if bias_correction == 1:
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

    is_adamw = mode == 1

    for i in range(num_tensors):
        g = tensor_lists[0][i]
        p = tensor_lists[1][i]  # int16 parameter (high 16 bits of FP32)
        m = tensor_lists[2][i]  # FP32 first moment
        v = tensor_lists[3][i]  # FP32 second moment
        p_remainder = tensor_lists[4][i]  # int16 remainder (low 16 bits of FP32)

        if not g.is_contiguous():
            g = g.contiguous()

        # Convert gradient to float
        g_float = g.float()

        # Reconstruct FP32 master weight from int16 param + int16 remainder using bit manipulation
        # This matches the CUDA implementation exactly:
        # 1. If p_remainder < 0, decrement p (undo rounding)
        # 2. Combine high 16 bits (p) and low 16 bits (p_remainder) into FP32
        # Note: Use PyTorch native ops for bit manipulation (int16/int32 operations)

        local_p = p.view(torch.int16).clone()
        local_p_rem = p_remainder.clone()

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
            g_float = flag_gems.add(g_float, param_master, alpha=weight_decay)

        # Update first moment: m = beta1 * m + (1 - beta1) * g
        flag_gems.add_(flag_gems.mul_(m, beta1), g_float, alpha=1 - beta1)

        # Update second moment: v = beta2 * v + (1 - beta2) * g^2
        flag_gems.add_(flag_gems.mul_(v, beta2), flag_gems.mul(g_float, g_float), alpha=1 - beta2)

        # Apply bias correction
        m_corr = flag_gems.true_divide(m, bias_correction1)
        v_corr = flag_gems.true_divide(v, bias_correction2)

        # Compute denominator: sqrt(v_corr) + eps
        denom = flag_gems.add(flag_gems.sqrt(v_corr), eps)

        # Compute update
        update = flag_gems.true_divide(m_corr, denom)

        # AdamW mode: add decoupled weight decay to update
        if is_adamw and weight_decay != 0:
            update = flag_gems.add(update, param_master, alpha=weight_decay)

        # Update master weight: p = p - lr * update
        param_master = flag_gems.sub(param_master, flag_gems.mul(update, lr))

        # Split FP32 back into int16 param + int16 remainder using bit manipulation
        # This matches the CUDA implementation exactly:
        # 1. Extract high 16 bits as p
        # 2. Extract low 16 bits as p_remainder
        # 3. If p_remainder < 0, increment p (round up)
        # Note: Use PyTorch native ops for bit manipulation (int32 operations)

        param_int32 = param_master.view(torch.int32)
        # Extract low 16 bits (remainder) and high 16 bits (param)
        new_p_rem = (param_int32 & 0xFFFF).to(torch.int16)
        new_p = ((param_int32 >> 16) & 0xFFFF).to(torch.int16)

        # Round up: if remainder < 0, increment p
        new_p = torch.where(new_p_rem < 0, new_p + 1, new_p)

        # Write back
        flag_gems.copy_(p, new_p.view(torch.bfloat16))
        flag_gems.copy_(p_remainder, new_p_rem)
