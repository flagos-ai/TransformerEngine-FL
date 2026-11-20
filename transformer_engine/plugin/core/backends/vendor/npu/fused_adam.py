# Copyright (c) 2026, BAAI. All rights reserved.
# See LICENSE for license information.

"""Adam updates with TE's BF16 parameters and signed int16 master remainders."""

import torch
import triton
import triton.language as tl


@triton.jit
def _adam_param_remainder_kernel(
    G,
    P,
    M,
    V,
    R,
    N,
    lr,
    beta1,
    beta2,
    correction1,
    correction2,
    eps,
    decay,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Ascend limits coreDim to 65535; each program handles further tiles at its stride.
    for tile in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
        offsets = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        grad = tl.load(G + offsets, mask, other=0).to(tl.float32)
        high = tl.load(P + offsets, mask, other=0).to(tl.int16, bitcast=True).to(tl.int32)
        remainder = tl.load(R + offsets, mask, other=0).to(tl.int32)
        high = high - (remainder < 0).to(tl.int32)
        master = ((high << 16) | (remainder & 0xFFFF)).to(tl.float32, bitcast=True)
        moment = tl.load(M + offsets, mask, other=0).to(tl.float32)
        variance = tl.load(V + offsets, mask, other=0).to(tl.float32)

        if MODE == 0:
            grad = grad + decay * master
        moment = beta1 * moment + (1 - beta1) * grad
        variance = beta2 * variance + ((1 - beta2) * grad) * grad
        update = (moment / correction1) / (tl.sqrt(variance / correction2) + eps)
        if MODE == 1:
            update = update + decay * master
        master = master - lr * update

        # TE carries into the BF16 high bits when the signed low bits are negative,
        # including halfway cases. A normal FP32-to-BF16 cast has different tie handling.
        bits = master.to(tl.int32, bitcast=True)
        low = bits.to(tl.int16)
        high = ((bits >> 16) + (low < 0).to(tl.int32)).to(tl.int16)
        tl.store(P + offsets, high.to(tl.bfloat16, bitcast=True), mask)
        tl.store(R + offsets, low, mask)
        tl.store(M + offsets, moment, mask)
        tl.store(V + offsets, variance, mask)


def adam_param_remainder(
    tensor_lists, lr, beta1, beta2, epsilon, step, mode, bias_correction, weight_decay
):
    """Update TE's five tensor lists in place without full-size FP32 master buffers."""
    if len(tensor_lists) != 5 or any(len(t) != len(tensor_lists[0]) for t in tensor_lists):
        raise ValueError("NPU remainder Adam expects five tensor lists of equal length")
    if mode not in (0, 1) or step < 1:
        raise ValueError("NPU remainder Adam requires mode 0 or 1 and a positive step")

    # Validate the entire call before modifying state; a later unsupported tensor must
    # not leave earlier tensors updated when a later input is invalid.
    for grad, param, moment, variance, remainder in zip(*tensor_lists):
        tensors = (grad, param, moment, variance, remainder)
        if any(t.device != param.device or t.shape != param.shape for t in tensors):
            raise ValueError("NPU remainder Adam tensors must share a shape and device")
        if param.device.type != "npu" or any(not t.is_contiguous() for t in tensors):
            raise ValueError("NPU remainder Adam requires contiguous NPU tensors")
        if (
            param.dtype != torch.bfloat16
            or remainder.dtype != torch.int16
            or grad.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or moment.dtype not in (torch.float32, torch.bfloat16)
            or variance.dtype != moment.dtype
        ):
            raise TypeError(
                "NPU remainder Adam requires BF16 params, int16 remainders and FP32/BF16 moments"
            )

    correction1 = 1 - beta1**step if bias_correction else 1.0
    correction2 = 1 - beta2**step if bias_correction else 1.0
    for grad, param, moment, variance, remainder in zip(*tensor_lists):
        n = param.numel()
        if n == 0:
            continue
        block = 1024
        grid = (min(triton.cdiv(n, block), 65535),)
        with torch.npu.device(param.device):
            _adam_param_remainder_kernel[grid](
                grad,
                param,
                moment,
                variance,
                remainder,
                n,
                lr,
                beta1,
                beta2,
                correction1,
                correction2,
                epsilon,
                weight_decay,
                MODE=mode,
                BLOCK=block,
                enable_fp_fusion=False,
            )
