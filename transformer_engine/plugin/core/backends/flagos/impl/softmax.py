import torch
from typing import Union
import flag_gems


def scaled_masked_softmax_forward_fl(
    input: torch.Tensor,
    mask: torch.Tensor,
    scale_factor: Union[float, torch.Tensor],
) -> torch.Tensor:
    # Ensure `mask` and 'scale_factor' is on the same device as `input`.
    if mask.device != input.device:
        mask = flag_gems.to_copy(mask, device=input.device)
    if isinstance(scale_factor, torch.Tensor):
        if scale_factor.device != input.device:
            scale_factor = flag_gems.to_copy(scale_factor, device=input.device)

    # Keep semantics aligned with TE CUDA scaled_masked_softmax:
    # - integer/bool mask: masked iff mask == 1, masked logits set to -10000.0
    # - float mask: treated as additive bias in logit space
    if mask.dim() == 4 and mask.size(1) == 1 and input.dim() == 4:
        mask = mask.expand_as(input)

    scaled = flag_gems.mul(input, scale_factor)
    if mask.is_floating_point():
        mask_f = flag_gems.to_copy(mask, device=input.device, dtype=scaled.dtype)
        scaled = flag_gems.add(scaled, mask_f)
        return flag_gems.softmax(scaled, dim=-1)

    # Avoid using `mask == 1` (torch op) since on some devices it may fall back to CPU,
    # which would break Triton kernels inside flag_gems.
    cond = flag_gems.eq_scalar(mask, 1)
    scaled = flag_gems.masked_fill(scaled, cond, -10000.0)
    all_masked = flag_gems.all_dim(cond, dim=-1, keepdim=True)
    out = flag_gems.softmax(scaled, dim=-1)
    return flag_gems.masked_fill(out, all_masked, 0.0)


def scaled_masked_softmax_backward_fl(
    output_grad_: torch.Tensor,
    softmax_results_: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    orig_dtype = output_grad_.dtype
    # Compute in float32 for numerical stability.
    output_grad_f32 = flag_gems.to_copy(output_grad_, dtype=torch.float32)
    softmax_output_f32 = flag_gems.to_copy(
        softmax_results_, dtype=torch.float32, device=output_grad_.device
    )
    if isinstance(scale_factor, torch.Tensor):
        if scale_factor.device != output_grad_.device:
            scale_factor = flag_gems.to_copy(scale_factor, device=output_grad_.device)

    # term = softmax_output_f32 * output_grad_f32
    term = flag_gems.mul(softmax_output_f32, output_grad_f32)
    # sum_term = sum(term, dim=-1, keepdim=True)
    sum_term = flag_gems.sum_dim(term, dim=[-1], keepdim=True)
    # grad_softmax = softmax_output_f32 * (output_grad_f32 - sum_term)
    grad_softmax = flag_gems.mul(softmax_output_f32, flag_gems.sub(output_grad_f32, sum_term))
    grad_scaled = flag_gems.mul(grad_softmax, scale_factor)
    return flag_gems.to_copy(grad_scaled, dtype=orig_dtype)
