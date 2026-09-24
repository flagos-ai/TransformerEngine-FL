# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""NPU vendor backend for TE-FL plugin system.

Bridges Ascend NPU operations into the TE-FL unified plugin interface
by delegating to transformer_engine_npu (pip-installed from TransformerEngineNPU).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union
import os

import torch

from ....ops import TEFLBackendBase, NVTE_Fused_Attn_Backend, DType
from .flash_attention import NPUFlashAttention


_DTYPE_TO_TORCH = {
    0: torch.uint8,
    2: torch.int32,
    4: torch.float32,
    5: torch.float16,
    6: torch.bfloat16,
    7: torch.float8_e4m3fn,
    8: torch.float8_e5m2,
}


def _noop_requested(noop: Optional[torch.Tensor]) -> bool:
    """Return whether TE's optional no-op flag requests skipping an update."""
    if noop is None or noop.numel() == 0:
        return False
    return bool(noop.detach().reshape(-1)[0].item())


def _to_torch_dtype(dtype: Any) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype

    value = getattr(dtype, "value", dtype)
    try:
        return _DTYPE_TO_TORCH.get(int(value))
    except (TypeError, ValueError):
        return None


def _check_npu_available() -> bool:
    """Check if NPU hardware and torch_npu are available."""
    try:
        import torch_npu  # noqa: F401
        import transformer_engine_npu

        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False


def _get_torch_npu():
    """Ensure torch_npu is imported (activates NPU device support in PyTorch)."""
    import torch_npu  # noqa: F401

    return torch_npu


def _get_tenpu_optimizers():
    """Get optimizers subpackage directly, bypassing transformer_engine_npu/__init__.py
    which triggers circular imports via pytorch/__init__.py -> module -> ops."""
    import transformer_engine_npu

    return transformer_engine_npu.pytorch.optimizers


def _get_tenpu_gemm():
    """Get GEMM ops subpackage."""
    import transformer_engine_npu

    return transformer_engine_npu.pytorch.ops.gemm


class NPUBackend(TEFLBackendBase):
    """NPU backend delegating to transformer_engine_npu + torch_npu."""

    def is_available(self) -> bool:
        return _check_npu_available()

    # ===================== splits_to_offsets adaptation =====================

    def splits_to_offsets(
        self,
        first_dims: torch.Tensor,
        logical_last_dim: int,
    ) -> torch.Tensor:
        """Convert grouped-tensor first dimensions to flattened element offsets."""
        if not isinstance(first_dims, torch.Tensor):
            raise TypeError("first_dims must be a torch.Tensor")
        if first_dims.device.type != "npu":
            raise ValueError(f"first_dims must be on NPU, got {first_dims.device}")
        if first_dims.dtype != torch.int64:
            raise TypeError(f"first_dims must have dtype torch.int64, got {first_dims.dtype}")
        if first_dims.ndim != 1:
            raise ValueError(f"first_dims must be one-dimensional, got shape={first_dims.shape}")
        if first_dims.numel() == 0:
            raise ValueError("first_dims must contain at least one split")
        if type(logical_last_dim) is not int or logical_last_dim <= 0:
            raise ValueError(
                f"logical_last_dim must be a positive integer, got {logical_last_dim!r}"
            )

        cumulative = torch.cumsum(first_dims.contiguous(), dim=0)
        if logical_last_dim != 1:
            cumulative = cumulative * logical_last_dim
        return torch.cat((torch.zeros_like(first_dims[:1]), cumulative), dim=0)

    # ===================== Attention =====================

    def get_attention_backend(self, attention_params=None):
        """Return NPU attention backend selection as a 6-tuple.

        The caller (dot_product_attention.py) expects:
            (use_flash_attention, flash_attention_backend,
             use_fused_attention, fused_attention_backend,
             use_unfused_attention, available_backends)
        TransformerEngineNPU only supports FlashAttention backend
        """
        from packaging.version import Version as PkgVersion
        from ....logger_manager import get_logger

        logger = get_logger()

        # Read environment variables to determine which backends to enable
        use_flash_attention = 1
        use_fused_attention = 0
        use_unfused_attention = 0

        # Log disabled backends
        logger.info_once("TransformerEngineNPU only supports FlashAttentionNPU backend")

        # Ascend only supports FlashAttention backend, and the FlashAttention version cannot be specified.
        flash_attention_backend = 0
        fused_attention_backend = NVTE_Fused_Attn_Backend.NVTE_No_Backend

        available_backends = [use_flash_attention, use_fused_attention, use_unfused_attention]

        return (
            use_flash_attention,
            flash_attention_backend,
            use_fused_attention,
            fused_attention_backend,
            use_unfused_attention,
            available_backends,
        )

    def get_flash_attention_class(self):
        """Return FlashAttention adapter class for NPU.

        Returns the adapter that bridges TE-FL's calling convention
        to NPU's FlashAttention interface.
        """
        return NPUFlashAttention

    # ===================== LayerNorm =====================

    # NPU backend adaptation: return output and saved statistics from one native NPU call.
    def layernorm_fwd(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float,
        ln_out: Any,
        quantizer: Any,
        otype: DType,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        """Apply LayerNorm with the NPU native three-output operator."""

        del sm_margin
        gamma = weight + 1 if zero_centered_gamma else weight
        _get_torch_npu()  # Register the NPU implementation for aten native operators.
        output, mean, rsigma = torch.ops.aten.native_layer_norm.default(
            input.contiguous(),
            list(weight.shape),
            gamma.contiguous(),
            None if bias is None else bias.contiguous(),
            eps,
        )

        output_dtype = _to_torch_dtype(otype)
        if output_dtype is not None and output.dtype != output_dtype:
            output = output.to(output_dtype)

        if quantizer is not None:
            output = quantizer.quantize(output, out=ln_out)
        elif ln_out is not None:
            if not isinstance(ln_out, torch.Tensor):
                raise TypeError(
                    "Dense NPU LayerNorm output reuse requires a torch.Tensor, "
                    f"got {type(ln_out).__name__}"
                )
            if tuple(ln_out.shape) != tuple(output.shape):
                raise ValueError(
                    "NPU LayerNorm output buffer has incompatible shape: "
                    f"expected={tuple(output.shape)}, got={tuple(ln_out.shape)}"
                )
            if ln_out.device != output.device or ln_out.dtype != output.dtype:
                raise ValueError(
                    "NPU LayerNorm output buffer must match result device and dtype: "
                    f"result=({output.device}, {output.dtype}), "
                    f"buffer=({ln_out.device}, {ln_out.dtype})"
                )
            ln_out.copy_(output)
            output = ln_out

        # TE-FL exposes statistics without the trailing normalized dimension.
        # TENPU keeps that singleton during its local calculation, so squeeze
        # only that dimension before feeding the common backward ABI.
        return output, mean.squeeze(-1), rsigma.squeeze(-1)

    # NPU backend adaptation: consume saved statistics with the native NPU backward operator.
    def layernorm_bwd(
        self,
        dz: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiate LayerNorm with the NPU native backward operator."""

        del sm_margin
        if mu.ndim < x.ndim:
            mu = mu.unsqueeze(-1)
        if rsigma.ndim < x.ndim:
            rsigma = rsigma.unsqueeze(-1)

        gamma_adjusted = gamma + 1 if zero_centered_gamma else gamma
        _get_torch_npu()  # Register the NPU implementation for aten native operators.
        dx, dgamma, dbeta = torch.ops.aten.native_layer_norm_backward.default(
            dz.contiguous(),
            x.contiguous(),
            list(gamma.shape),
            mu.contiguous(),
            rsigma.contiguous(),
            gamma_adjusted.contiguous(),
            None,
            [True, True, True],
        )

        # CANN accumulates parameter gradients in FP32 for FP16/BF16 inputs,
        # while Transformer Engine returns gradients in the parameter dtype.
        if dgamma.dtype != gamma.dtype:
            dgamma = dgamma.to(gamma.dtype)
        if dbeta.dtype != gamma.dtype:
            dbeta = dbeta.to(gamma.dtype)
        return dx, dgamma, dbeta

    # ===================== RMSNorm =====================

    def rmsnorm_fwd(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        ln_out: Any,
        quantizer: Any,
        otype: Any,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        """RMSNorm forward using torch_npu.npu_rms_norm.

        TE-FL calls with: (input, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma)
        NPU kernel: npu_rms_norm(input, gamma, epsilon=eps) → (output, rstd)

        NPU kernel requires 2D input [outer_dim, inner_dim]. We reshape accordingly.
        We ignore ln_out (pre-allocated output buffer), otype, sm_margin.
        """

        if zero_centered_gamma:
            weight = weight + 1

        # NPU npu_rms_norm requires 2D input: [outer_dim, hidden_size]
        input_shape = input.shape
        inner_dim = weight.shape[0]
        x_2d = input.reshape(-1, inner_dim)

        out_2d, inv_rms = _get_torch_npu().npu_rms_norm(x_2d, weight, epsilon=eps)

        # Reshape output back to original input shape
        out = out_2d.reshape(input_shape)

        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)

        # TE-FL expects (ln_out, mu, rsigma); mu is None for RMSNorm
        # inv_rms shape is [outer_dim, 1] from NPU kernel
        return out, None, inv_rms

    def rmsnorm_bwd(
        self,
        dz: torch.Tensor,
        x: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """RMSNorm backward using torch_npu.npu_rms_norm_backward.

        TE-FL calls with: (dz, x, rsigma, gamma, sm_margin, zero_centered_gamma)
        NPU kernel expects: npu_rms_norm_backward(dy, x, gamma, rstd)
        where rstd must be FP32 and x/dy must be 2D [outer_dim, inner_dim].

        NPU supported combo (BF16):
          dy(BF16) x(BF16) rstd(FP32) gamma(BF16) → dx(BF16) dgamma(FP32)
        """

        if zero_centered_gamma:
            gamma = gamma + 1

        # NPU kernel requires 2D input
        input_shape = x.shape
        inner_dim = gamma.shape[0]
        x_2d = x.reshape(-1, inner_dim)
        dz_2d = dz.reshape(-1, inner_dim)

        # NPU kernel requires rstd in float32
        rsigma_fp32 = rsigma.float() if rsigma.dtype != torch.float32 else rsigma

        dx_2d, dw = _get_torch_npu().npu_rms_norm_backward(dz_2d, x_2d, gamma, rsigma_fp32)

        # Reshape dx back to original input shape
        dx = dx_2d.reshape(input_shape)

        return dx, dw

    # ===================== Multi-tensor Optimizers =====================

    def multi_tensor_scale(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: float,
    ):
        """Multi-tensor scale."""
        opt = _get_tenpu_optimizers()
        opt.multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale)

    def multi_tensor_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        per_tensor: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Multi-tensor L2 norm."""
        opt = _get_tenpu_optimizers()
        return opt.multi_tensor_l2norm(chunk_size, noop_flag, tensor_lists, per_tensor)

    @torch.no_grad()
    def multi_tensor_adam(
        self,
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
        """Update Adam state with TENPU's NPU fused-AdamW execution path."""
        del chunk_size  # CUDA launch partitioning has no NPU equivalent.
        if _noop_requested(noop_flag):
            return None
        if len(tensor_lists) not in (4, 5):
            raise ValueError(
                "NPU multi_tensor_adam expects four tensor lists, or five "
                "when FP32 master parameters are provided"
            )
        if mode not in (0, 1):
            raise ValueError(f"NPU multi_tensor_adam mode must be 0 or 1, got {mode}")
        if step < 1:
            raise ValueError(f"NPU multi_tensor_adam step must be positive, got {step}")

        grads, params, exp_avgs, exp_avg_sqs = tensor_lists[:4]
        master_params = tensor_lists[4] if len(tensor_lists) == 5 else None
        expected_length = len(grads)
        named_lists = {
            "params": params,
            "exp_avgs": exp_avgs,
            "exp_avg_sqs": exp_avg_sqs,
        }
        if master_params is not None:
            named_lists["master_params"] = master_params
        for name, tensors in named_lists.items():
            if len(tensors) != expected_length:
                raise ValueError(
                    "All NPU multi_tensor_adam tensor lists must have equal length: "
                    f"grads={expected_length}, {name}={len(tensors)}"
                )

        # TENPU's fused path always applies bias correction. Preserve TE-FL's
        # granular ABI for bias_correction=0 with explicit FP32 equations.
        if not bias_correction:
            for index, (grad, param, exp_avg, exp_avg_sq) in enumerate(
                zip(grads, params, exp_avgs, exp_avg_sqs)
            ):
                if grad is None:
                    continue
                update_param = master_params[index] if master_params is not None else param
                grad_fp32 = grad.float()
                param_fp32 = update_param.float()
                if mode == 0:
                    grad_fp32 = grad_fp32 + weight_decay * param_fp32
                next_exp_avg = beta1 * exp_avg.float() + (1.0 - beta1) * grad_fp32
                next_exp_avg_sq = beta2 * exp_avg_sq.float() + (1.0 - beta2) * grad_fp32 * grad_fp32
                update = next_exp_avg / (next_exp_avg_sq.sqrt() + epsilon)
                if mode == 1:
                    update = update + weight_decay * param_fp32
                next_param = param_fp32 - lr * update
                update_param.copy_(next_param.to(update_param.dtype))
                if master_params is not None:
                    param.copy_(update_param.to(param.dtype))
                exp_avg.copy_(next_exp_avg.to(exp_avg.dtype))
                exp_avg_sq.copy_(next_exp_avg_sq.to(exp_avg_sq.dtype))
            return None

        fused_adamw = getattr(torch, "_fused_adamw_", None)
        if fused_adamw is None:
            raise RuntimeError(
                "TENPU-compatible NPU multi_tensor_adam requires torch._fused_adamw_"
            )

        groups = {}
        copyback = []
        for index, (grad, param, exp_avg, exp_avg_sq) in enumerate(
            zip(grads, params, exp_avgs, exp_avg_sqs)
        ):
            if grad is None:
                continue
            update_param = master_params[index] if master_params is not None else param
            direct = (
                update_param.dtype in (torch.float32, torch.float16, torch.bfloat16)
                and grad.dtype == update_param.dtype
                and exp_avg.dtype == torch.float32
                and exp_avg_sq.dtype == torch.float32
            )
            if direct:
                work_param = update_param
                work_grad = grad
                work_exp_avg = exp_avg
                work_exp_avg_sq = exp_avg_sq
            else:
                work_param = (
                    update_param
                    if update_param.dtype == torch.float32
                    else update_param.detach().float().clone()
                )
                work_grad = grad.detach().float()
                work_exp_avg = (
                    exp_avg if exp_avg.dtype == torch.float32 else exp_avg.detach().float()
                )
                work_exp_avg_sq = (
                    exp_avg_sq if exp_avg_sq.dtype == torch.float32 else exp_avg_sq.detach().float()
                )

            fused_grad = (
                work_grad + weight_decay * work_param
                if mode == 0 and weight_decay != 0.0
                else work_grad
            )
            key = (
                work_param.device,
                work_param.dtype,
                fused_grad.dtype,
                work_exp_avg.dtype,
                work_exp_avg_sq.dtype,
            )
            group = groups.setdefault(
                key,
                {"params": [], "grads": [], "exp_avgs": [], "exp_avg_sqs": []},
            )
            group["params"].append(work_param)
            group["grads"].append(fused_grad)
            group["exp_avgs"].append(work_exp_avg)
            group["exp_avg_sqs"].append(work_exp_avg_sq)
            copyback.append(
                (
                    param,
                    update_param,
                    exp_avg,
                    exp_avg_sq,
                    work_param,
                    work_exp_avg,
                    work_exp_avg_sq,
                )
            )

        for group in groups.values():
            if not group["params"]:
                continue
            step_tensor = torch.tensor(
                step,
                dtype=torch.int64,
                device=group["params"][0].device,
            )
            fused_adamw(
                group["params"],
                group["grads"],
                group["exp_avgs"],
                group["exp_avg_sqs"],
                [],
                [step_tensor] * len(group["params"]),
                amsgrad=False,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                weight_decay=weight_decay if mode == 1 else 0.0,
                eps=epsilon,
                maximize=False,
            )

        for (
            param,
            update_param,
            exp_avg,
            exp_avg_sq,
            work_param,
            work_exp_avg,
            work_exp_avg_sq,
        ) in copyback:
            if work_param is not update_param:
                update_param.copy_(work_param.to(update_param.dtype))
            if master_params is not None:
                param.copy_(update_param.to(param.dtype))
            if work_exp_avg is not exp_avg:
                exp_avg.copy_(work_exp_avg.to(exp_avg.dtype))
            if work_exp_avg_sq is not exp_avg_sq:
                exp_avg_sq.copy_(work_exp_avg_sq.to(exp_avg_sq.dtype))
        return None

    def multi_tensor_unscale_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        inv_scale: torch.Tensor,
        per_tensor: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Multi-tensor unscale + L2 norm."""
        opt = _get_tenpu_optimizers()
        return opt.multi_tensor_unscale_l2norm(
            chunk_size, noop_flag, tensor_lists, inv_scale, per_tensor
        )

    def multi_tensor_compute_scale_and_scale_inv(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        max_fp8: float,
        force_pow_2_scales: bool,
        epsilon: float,
    ):
        """Compute per-tensor FP8 scale and scale_inv."""
        if _noop_requested(noop_flag):
            return

        opt = _get_tenpu_optimizers()
        opt.multi_tensor_compute_scale_and_scale_inv(
            chunk_size, noop_flag, tensor_lists, max_fp8, force_pow_2_scales, epsilon
        )

    def multi_tensor_compute_scale_inv_e8m0(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        block_len: int,
    ):
        """Compute scale_inv in e8m0 format for MXFP8."""
        opt = _get_tenpu_optimizers()
        opt.multi_tensor_compute_scale_inv_e8m0(chunk_size, noop_flag, tensor_lists)

    # ===================== GEMM =====================

    def generic_gemm(
        self,
        A: Any,
        transA: bool,
        B: Any,
        transB: bool,
        D: Any,
        quantizer: Any,
        output_dtype: Optional[Any],
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
    ) -> List[Any]:
        """General GEMM aligned with the generic_gemm interface.

        Computes out = B_comp @ A_comp (same as reference impl), where:
          B_comp = B.T if transB else B
          A_comp = A.T if transA else A

        Delegates to TransformerEngineNPU's general_gemm which computes:
          out = matmul(NPU_A, NPU_B) with usage-based transposition.

        Mapping: NPU_A=B, NPU_B=A, usage_a reflects transB, usage_b reflects transA.
        """
        import torch.nn.functional as F

        gemm_mod = _get_tenpu_gemm()

        # Map transA/transB to NPU TensorUsage strings
        # NPU general_gemm(A, B, usage_a, usage_b): transposes A if usage_a in USAGE_WITH_TRANS
        # We pass (B, A) as (NPU_A, NPU_B) so that NPU computes B_comp @ A_comp
        usage_a = "LT" if transB else "LN"  # controls transpose of NPU_A (which is our B)
        usage_b = "RT" if transA else "RN"  # controls transpose of NPU_B (which is our A)

        # Determine output dtype
        from ....ops import DType

        _DTYPE_TO_TORCH = {
            0: torch.uint8,
            2: torch.int32,
            4: torch.float32,
            5: torch.float16,
            6: torch.bfloat16,
            7: torch.float8_e4m3fn,
            8: torch.float8_e5m2,
        }
        torch_out_dtype = None
        if output_dtype is not None:
            if isinstance(output_dtype, torch.dtype):
                torch_out_dtype = output_dtype
            elif isinstance(output_dtype, int):
                torch_out_dtype = _DTYPE_TO_TORCH.get(output_dtype, None)
            elif hasattr(output_dtype, "value"):
                torch_out_dtype = _DTYPE_TO_TORCH.get(output_dtype.value, None)

        # Use the activation dtype of B as fallback for out_dtype
        if torch_out_dtype is None:
            torch_out_dtype = (
                B.dtype
                if B.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
                else torch.bfloat16
            )

        # Handle 3D tensors by flattening to 2D (matching reference semantics)
        original_B_shape = None
        if B.ndim == 3:
            original_B_shape = B.shape
            B = B.reshape(-1, B.shape[-1])
        if A.ndim == 3:
            A = A.reshape(-1, A.shape[-1])

        # Core GEMM: general_gemm(A, B, usage_a, usage_b, out_dtype, bias=None)
        # We pass bias=None here and handle bias/gelu ourselves to match reference semantics
        out = gemm_mod.general_gemm(B, A, usage_a, usage_b, torch_out_dtype, bias=None)

        # Restore 3D shape: a non-transposed B contributes its outer dimensions to the output
        if original_B_shape is not None and not transB:
            out = out.view(original_B_shape[0], original_B_shape[1], -1)

        if alpha != 1.0:
            out = out * alpha

        gelu_input_ret = None

        # Bias handling: in backward (grad=True), bias only requests fused BGRAD epilogue,
        # its value is NOT added to the GEMM result.
        if bias is not None and not grad:
            out = out + bias

        # GeLU handling
        if gelu:
            if grad:
                # Backward: compute dgelu(out, gelu_in)
                # out is the upstream gradient, gelu_in is the saved forward pre-activation
                if gelu_in is None:
                    raise ValueError("gelu_in must be provided for a backward GELU GEMM")
                x = gelu_in.detach().requires_grad_(True)
                with torch.enable_grad():
                    y = F.gelu(x, approximate="tanh")
                    y.backward(out)
                out = x.grad
            else:
                # Forward: save pre-gelu input and apply gelu
                if gelu_in is not None:
                    gelu_in.copy_(out)
                    gelu_input_ret = gelu_in
                else:
                    gelu_input_ret = out.clone()
                out = F.gelu(out, approximate="tanh")

        # Cast to output dtype if needed
        if torch_out_dtype is not None and out.dtype != torch_out_dtype:
            out = out.to(torch_out_dtype)

        # Accumulate into D if provided
        if D is not None:
            if accumulate:
                beta_val = beta if beta is not None else 1.0
                D.mul_(beta_val).add_(out)
                out = D
            else:
                D.copy_(out)
                out = D

        # Compute bias gradient in backward pass
        bias_grad = None
        if grad and bias is not None:
            # BGRADB epilogue: reduce over the batch/sequence dimension of B
            # At this point B is already 2D (flattened above), matching reference behavior
            bias_grad = B.sum(dim=0).to(dtype=out.dtype)

        extra_output_ret = None

        return out, bias_grad, gelu_input_ret, extra_output_ret

    # ===================== Discrete GroupedTensor GEMM adaptation =====================

    @staticmethod
    def _dense_grouped_tensor_parts(
        grouped_tensor: Any,
        name: str,
        expected_num_tensors: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int], torch.Tensor, int]:
        """Validate a dense GroupedTensor and expose its packed 2D representation."""

        num_tensors = getattr(grouped_tensor, "num_tensors", None)
        if not isinstance(num_tensors, int) or num_tensors <= 0:
            raise ValueError(f"{name}.num_tensors must be a positive integer")
        if expected_num_tensors is not None and num_tensors != expected_num_tensors:
            raise ValueError(
                f"{name}.num_tensors must be {expected_num_tensors}, got {num_tensors}"
            )
        if num_tensors > 128:
            raise ValueError(f"NPU grouped GEMM supports at most 128 groups, got {num_tensors}")
        if not hasattr(grouped_tensor, "quantizer"):
            raise TypeError(f"{name} must be a TE-FL GroupedTensor")
        if grouped_tensor.quantizer is not None:
            raise NotImplementedError(
                f"NPU discrete grouped GEMM supports quantizer=None only; {name}.quantizer is set"
            )
        if getattr(grouped_tensor, "last_dims", None) is not None:
            raise NotImplementedError(
                f"NPU discrete grouped GEMM requires a uniform last dimension for {name}"
            )

        data = getattr(grouped_tensor, "rowwise_data", None)
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"{name}.rowwise_data must be a torch.Tensor")
        if data.device.type != "npu":
            raise ValueError(f"{name}.rowwise_data must be on NPU, got {data.device}")
        if not data.is_contiguous():
            raise ValueError(f"{name}.rowwise_data must be contiguous")

        logical_shape = getattr(grouped_tensor, "logical_shape", None)
        if logical_shape is None or len(logical_shape) != 2:
            raise ValueError(f"{name}.logical_shape must be two-dimensional")
        logical_shape = (int(logical_shape[0]), int(logical_shape[1]))
        if logical_shape[0] < 0 or logical_shape[1] <= 0:
            raise ValueError(f"Invalid {name}.logical_shape: {logical_shape}")
        if data.numel() != logical_shape[0] * logical_shape[1]:
            raise ValueError(
                f"{name}.rowwise_data has {data.numel()} elements, but "
                f"logical_shape={logical_shape} requires "
                f"{logical_shape[0] * logical_shape[1]}"
            )

        first_dims = getattr(grouped_tensor, "first_dims", None)
        if first_dims is None:
            if logical_shape[0] % num_tensors != 0:
                raise ValueError(
                    f"{name}.logical_shape[0]={logical_shape[0]} is not divisible by "
                    f"num_tensors={num_tensors}"
                )
            group_sizes = torch.full(
                (num_tensors,),
                logical_shape[0] // num_tensors,
                dtype=torch.int64,
                device=data.device,
            )
        else:
            if not isinstance(first_dims, torch.Tensor):
                raise TypeError(f"{name}.first_dims must be a torch.Tensor or None")
            if first_dims.dtype != torch.int64 or first_dims.numel() != num_tensors:
                raise ValueError(f"{name}.first_dims must be int64 with {num_tensors} elements")
            if first_dims.device != data.device:
                raise ValueError(f"{name}.first_dims and rowwise_data must share a device")
            group_sizes = first_dims.contiguous().view(-1)

        return data.view(logical_shape), logical_shape, group_sizes, num_tensors

    @staticmethod
    def _validate_dense_grouped_coefficients(
        alpha: torch.Tensor,
        beta: torch.Tensor,
        device: torch.device,
        num_tensors: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate TE grouped-GEMM alpha/beta tensors."""

        for name, coefficient in (("alpha", alpha), ("beta", beta)):
            if not isinstance(coefficient, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if coefficient.device != device:
                raise ValueError(f"{name} must be on {device}, got {coefficient.device}")
            if coefficient.dtype != torch.float32:
                raise TypeError(f"{name} must have dtype torch.float32")
            if coefficient.numel() not in (1, num_tensors):
                raise ValueError(
                    f"{name} must contain 1 or {num_tensors} values, got {coefficient.numel()}"
                )
        if alpha.numel() != beta.numel():
            raise ValueError("alpha and beta must contain the same number of values")
        return alpha.reshape(-1), beta.reshape(-1)

    @staticmethod
    def _write_packed_grouped_output(
        destination: torch.Tensor,
        product: torch.Tensor,
        group_sizes: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        """Apply TE alpha/beta semantics and update a packed grouped destination."""

        if product.numel() != destination.numel():
            raise RuntimeError(
                "Unexpected NPU grouped GEMM output size: "
                f"expected {destination.numel()} elements, got {product.numel()}"
            )
        destination_2d = destination.view(destination.shape[0], -1)
        product_2d = product.reshape_as(destination_2d)

        def expand(values: torch.Tensor) -> torch.Tensor:
            if values.numel() == 1:
                return values.reshape(())
            return torch.repeat_interleave(
                values,
                group_sizes,
                output_size=destination_2d.shape[0],
            ).view(-1, 1)

        combined = product_2d.float() * expand(alpha)
        beta_expanded = expand(beta)
        destination_term = destination_2d.float() * beta_expanded
        destination_term.masked_fill_(beta_expanded == 0, 0.0)
        combined = combined + destination_term
        destination_2d.copy_(combined.to(destination.dtype))

    def te_general_grouped_gemm_for_discrete_in(self, *args, **kwargs):
        """Forward the base interface to the typed NPU implementation."""
        return self._te_general_grouped_gemm_for_discrete_in_impl(*args, **kwargs)

    def _te_general_grouped_gemm_for_discrete_in_impl(
        self,
        A: List[torch.Tensor],
        transa: bool,
        B: Any,
        transb: bool,
        D: Any,
        bias: Optional[Any],
        bias_scale: Optional[torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        workspace_setup: torch.Tensor,
        workspace_cublas: torch.Tensor,
        use_split_accumulator: bool,
        math_sm_count: int,
    ) -> Any:
        """Run dense grouped GEMM with a discrete A list and packed B/D tensors."""

        del workspace_setup, workspace_cublas, use_split_accumulator, math_sm_count

        layout = ("T" if transa else "N") + ("T" if transb else "N")
        if layout not in ("TN", "NN", "NT"):
            raise NotImplementedError(
                f"NPU discrete-input grouped GEMM supports TN, NN, and NT; got {layout}"
            )

        b_data, b_shape, b_group_sizes, num_tensors = self._dense_grouped_tensor_parts(B, "B")
        d_data, d_shape, d_group_sizes, _ = self._dense_grouped_tensor_parts(D, "D", num_tensors)
        if not isinstance(A, list) or len(A) != num_tensors:
            raise ValueError(f"A must be a list containing {num_tensors} tensors")

        supported_dtypes = (torch.float16, torch.bfloat16)
        input_dtype = b_data.dtype
        if input_dtype not in supported_dtypes:
            raise TypeError(
                f"NPU discrete-input grouped GEMM supports FP16 and BF16, got {input_dtype}"
            )
        for index, tensor in enumerate(A):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"A[{index}] must be a torch.Tensor")
            if tensor.ndim != 2:
                raise ValueError(f"A[{index}] must be 2D, got shape={tuple(tensor.shape)}")
            if tensor.device != b_data.device or tensor.dtype != input_dtype:
                raise ValueError(
                    f"A[{index}] must use device={b_data.device} and dtype={input_dtype}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"A[{index}] must be contiguous")
        if d_data.device != b_data.device or d_data.dtype != input_dtype:
            raise ValueError("B and D must have the same NPU device and dtype")

        alpha, beta = self._validate_dense_grouped_coefficients(
            alpha, beta, b_data.device, num_tensors
        )
        b_rows, b_cols = b_shape
        d_rows, d_cols = d_shape
        torch_npu = _get_torch_npu()

        if layout in ("TN", "NN"):
            weight_rows, weight_cols = map(int, A[0].shape)
            if any(tuple(tensor.shape) != (weight_rows, weight_cols) for tensor in A):
                raise ValueError("TN/NN discrete-input grouped GEMM requires uniform A shapes")
            expected_b_cols = weight_cols if layout == "TN" else weight_rows
            expected_d_cols = weight_rows if layout == "TN" else weight_cols
            if b_cols != expected_b_cols or d_shape != (b_rows, expected_d_cols):
                raise ValueError(
                    f"Invalid {layout} shapes: A[0]={tuple(A[0].shape)}, B={b_shape}, D={d_shape}"
                )
            if b_rows == 0:
                product = torch.zeros_like(d_data)
            elif num_tensors == 1:
                weight = A[0].transpose(0, 1) if layout == "TN" else A[0]
                product = torch.matmul(b_data, weight)
            else:
                weights = [tensor.transpose(0, 1) for tensor in A] if layout == "TN" else A
                product = torch_npu.npu_grouped_matmul(
                    [b_data],
                    weights,
                    group_list=b_group_sizes,
                    output_dtype=input_dtype,
                    group_type=0,
                    group_list_type=1,
                    split_item=3,
                )[0]
        else:
            a_cols = int(A[0].shape[1])
            if any(int(tensor.shape[1]) != a_cols for tensor in A):
                raise ValueError("NT discrete-input grouped GEMM requires a common A last dim")
            packed_a_rows = sum(int(tensor.shape[0]) for tensor in A)
            if packed_a_rows != b_rows or d_cols != a_cols or d_rows % num_tensors != 0:
                raise ValueError(
                    f"Invalid NT shapes: A rows={packed_a_rows}, A cols={a_cols}, "
                    f"B={b_shape}, D={d_shape}"
                )
            output_rows = d_rows // num_tensors
            if b_cols != output_rows:
                raise ValueError(
                    f"Invalid NT output: expected D group shape {(b_cols, a_cols)}, "
                    f"got {(output_rows, d_cols)}"
                )
            if b_rows == 0:
                product = torch.zeros(
                    (num_tensors, output_rows, a_cols),
                    dtype=torch.float32,
                    device=b_data.device,
                )
            elif num_tensors == 1:
                product = torch.matmul(b_data.transpose(0, 1), A[0]).float().unsqueeze(0)
            else:
                packed_a = torch.cat(A, dim=0)
                product = torch.zeros(
                    (num_tensors, output_rows, a_cols),
                    dtype=torch.float32,
                    device=b_data.device,
                )
                a_group_sizes = torch.tensor(
                    [int(tensor.shape[0]) for tensor in A],
                    dtype=torch.int64,
                    device=b_data.device,
                )
                torch_npu.npu_grouped_matmul_add_(
                    product,
                    b_data,
                    packed_a,
                    torch.cumsum(a_group_sizes, dim=0),
                )
            product = product.view(d_shape)

        self._write_packed_grouped_output(d_data, product, d_group_sizes, alpha, beta)

        if bias_scale is not None and bias is None:
            raise ValueError("bias_scale requires bias")
        if bias is not None:
            bias_data, bias_shape, _, _ = self._dense_grouped_tensor_parts(
                bias, "bias", num_tensors
            )
            if bias_data.device != d_data.device or bias_data.dtype != d_data.dtype:
                raise ValueError("bias must have the same device and dtype as D")
            if bias_shape != (num_tensors, d_cols):
                raise ValueError(
                    f"bias must have logical shape {(num_tensors, d_cols)}, got {bias_shape}"
                )
            expanded_bias = torch.repeat_interleave(
                bias_data.view(num_tensors, d_cols).float(),
                d_group_sizes,
                dim=0,
                output_size=d_rows,
            )
            if bias_scale is not None:
                if (
                    not isinstance(bias_scale, torch.Tensor)
                    or bias_scale.device != d_data.device
                    or bias_scale.dtype != torch.float32
                    or bias_scale.numel() != d_rows
                ):
                    raise ValueError(
                        f"bias_scale must be FP32 on {d_data.device} with {d_rows} elements"
                    )
                expanded_bias = expanded_bias * bias_scale.view(d_rows, 1)
            d_data.copy_((d_data.float() + expanded_bias).to(d_data.dtype))

        return D

    def te_general_grouped_gemm_for_discrete_out(self, *args, **kwargs):
        """Forward the base interface to the typed NPU implementation."""
        return self._te_general_grouped_gemm_for_discrete_out_impl(*args, **kwargs)

    def _te_general_grouped_gemm_for_discrete_out_impl(
        self,
        A: Any,
        transa: bool,
        B: Any,
        transb: bool,
        D: List[torch.Tensor],
        bias: Optional[Any],
        bias_scale: Optional[torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        workspace_setup: torch.Tensor,
        workspace_cublas: torch.Tensor,
        use_split_accumulator: bool,
        math_sm_count: int,
    ) -> List[torch.Tensor]:
        """Run dense grouped GEMM with packed A/B tensors and a discrete D list."""

        del workspace_setup, workspace_cublas, use_split_accumulator, math_sm_count
        if bias is not None or bias_scale is not None:
            raise ValueError("bias and bias_scale are not supported with discrete output")

        layout = ("T" if transa else "N") + ("T" if transb else "N")
        if layout not in ("TN", "NN", "NT"):
            raise NotImplementedError(
                f"NPU discrete-output grouped GEMM supports TN, NN, and NT; got {layout}"
            )

        a_data, a_shape, a_group_sizes, num_tensors = self._dense_grouped_tensor_parts(A, "A")
        b_data, b_shape, b_group_sizes, _ = self._dense_grouped_tensor_parts(B, "B", num_tensors)
        if not isinstance(D, list) or len(D) != num_tensors:
            raise ValueError(f"D must be a list containing {num_tensors} tensors")

        supported_dtypes = (torch.float16, torch.bfloat16)
        input_dtype = a_data.dtype
        if input_dtype not in supported_dtypes or b_data.dtype != input_dtype:
            raise TypeError("A and B must both use FP16 or BF16 with one common dtype")
        if b_data.device != a_data.device:
            raise ValueError("A and B must be on the same NPU device")

        output_dtype = D[0].dtype if D and isinstance(D[0], torch.Tensor) else None
        for index, tensor in enumerate(D):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"D[{index}] must be a torch.Tensor")
            if tensor.ndim != 2:
                raise ValueError(f"D[{index}] must be 2D, got shape={tuple(tensor.shape)}")
            if tensor.device != a_data.device or tensor.dtype != output_dtype:
                raise ValueError("All D tensors must have one NPU device and dtype")
            if not tensor.is_contiguous():
                raise ValueError(f"D[{index}] must be contiguous")
        if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"Unsupported discrete output dtype: {output_dtype}")

        alpha, beta = self._validate_dense_grouped_coefficients(
            alpha, beta, a_data.device, num_tensors
        )
        a_rows, a_cols = a_shape
        b_rows, b_cols = b_shape
        torch_npu = _get_torch_npu()

        if layout in ("TN", "NN"):
            if getattr(A, "first_dims", None) is not None or a_rows % num_tensors != 0:
                raise ValueError(f"A must have a uniform first dimension for layout {layout}")
            weight_rows = a_rows // num_tensors
            expected_b_cols = a_cols if layout == "TN" else weight_rows
            expected_d_cols = weight_rows if layout == "TN" else a_cols
            d_rows = [int(tensor.shape[0]) for tensor in D]
            if (
                b_cols != expected_b_cols
                or sum(d_rows) != b_rows
                or any(
                    tuple(tensor.shape) != (d_rows[i], expected_d_cols)
                    for i, tensor in enumerate(D)
                )
            ):
                raise ValueError(
                    f"Invalid {layout} shapes: A={a_shape}, B={b_shape}, "
                    f"D={[tuple(tensor.shape) for tensor in D]}"
                )
            if output_dtype != input_dtype:
                raise NotImplementedError(
                    f"NPU {layout} discrete-output GEMM requires output dtype {input_dtype}"
                )
            if b_rows == 0:
                packed_product = torch.zeros(
                    (b_rows, expected_d_cols), dtype=output_dtype, device=a_data.device
                )
            else:
                weight = a_data.view(num_tensors, weight_rows, a_cols)
                if layout == "TN":
                    weight = weight.transpose(1, 2).contiguous()
                if num_tensors == 1:
                    packed_product = torch.matmul(b_data, weight[0])
                else:
                    packed_product = torch_npu.npu_grouped_matmul(
                        [b_data],
                        [weight],
                        group_list=b_group_sizes,
                        output_dtype=output_dtype,
                        group_type=0,
                        group_list_type=1,
                        split_item=3,
                    )[0]
            products = torch.split(packed_product, d_rows, dim=0)
        else:
            if a_rows != b_rows:
                raise ValueError(f"NT requires equal A/B row counts, got {a_rows} and {b_rows}")
            expected_shape = (b_cols, a_cols)
            if any(tuple(tensor.shape) != expected_shape for tensor in D):
                raise ValueError(
                    f"NT requires every D tensor to have shape {expected_shape}, "
                    f"got {[tuple(tensor.shape) for tensor in D]}"
                )
            if a_rows == 0:
                packed_product = torch.zeros(
                    (num_tensors, b_cols, a_cols),
                    dtype=torch.float32,
                    device=a_data.device,
                )
            elif num_tensors == 1:
                packed_product = torch.matmul(b_data.transpose(0, 1), a_data).float().unsqueeze(0)
            else:
                packed_product = torch.zeros(
                    (num_tensors, b_cols, a_cols),
                    dtype=torch.float32,
                    device=a_data.device,
                )
                torch_npu.npu_grouped_matmul_add_(
                    packed_product,
                    b_data,
                    a_data,
                    torch.cumsum(a_group_sizes, dim=0),
                )
            products = packed_product.unbind(0)

        for index, (destination, product) in enumerate(zip(D, products)):
            alpha_i = alpha[0] if alpha.numel() == 1 else alpha[index]
            beta_i = beta[0] if beta.numel() == 1 else beta[index]
            combined = product.float() * alpha_i
            destination_term = destination.float() * beta_i
            destination_term.masked_fill_(beta_i == 0, 0.0)
            combined = combined + destination_term
            destination.copy_(combined.to(destination.dtype))

        return D

    def te_general_grouped_gemm(
        self,
        A: List[Any],
        transa: bool,
        B: List[Any],
        transb: bool,
        D: Optional[List[torch.Tensor]],
        D_type: DType,
        m_splits: List[int],
        bias: List[torch.Tensor],
        bias_type: DType,
        single_output: bool,
        pre_gelu_out: List[torch.Tensor],
        grad: bool,
        workspace: List[torch.Tensor],
        workspaceSizes: int,
        accumulate: bool,
        use_split_accumulator: bool,
        math_sm_count: int,
    ) -> Optional[List[torch.Tensor]]:
        """Grouped GEMM adapter for TransformerEngineNPU.

        TE-FL semantics for every group:

            D[i] = op(B[i], transb) @ op(A[i], transa)

        Native NPU mappings:
            Forward:  layout="TN", group_type=0
            dgrad:    layout="NN", group_type=0
            wgrad:    layout="NT", group_type=2

        The group_type=2 path requires an Ascend A2/A3 device. Operations that
        require bgrad, GELU/dGELU, mixed per-group epilogues, unsupported dtypes,
        or non-standard transpose layouts fall back to per-group generic_gemm.
        """

        num_gemms = len(A)
        if len(B) != num_gemms:
            raise ValueError(f"A/B group count mismatch: len(A)={len(A)}, len(B)={len(B)}")
        if num_gemms == 0:
            return bias

        def op_shape(tensor: Any, transpose: bool) -> Tuple[int, int]:
            if tensor.ndim != 2:
                raise ValueError(f"Grouped GEMM requires 2D tensors, got {tuple(tensor.shape)}")
            rows, cols = map(int, tensor.shape)
            return (cols, rows) if transpose else (rows, cols)

        def has_tensor(tensors, index: int) -> bool:
            return (
                tensors is not None
                and index < len(tensors)
                and tensors[index] is not None
                and tensors[index].numel() > 0
            )

        # 1. Validate GEMMs and prepare destinations.
        output_shapes: List[Tuple[int, int]] = []
        for index, (a_tensor, b_tensor) in enumerate(zip(A, B)):
            a_rows, a_cols = op_shape(a_tensor, transa)
            b_rows, b_cols = op_shape(b_tensor, transb)
            if b_cols != a_rows:
                raise ValueError(
                    f"Incompatible shapes for group {index}: "
                    f"op(B)=({b_rows}, {b_cols}), op(A)=({a_rows}, {a_cols})"
                )
            output_shapes.append((b_rows, a_cols))

        out_dtype = _to_torch_dtype(D_type)
        if out_dtype is None:
            out_dtype = D[0].dtype if D else B[0].dtype
            if out_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                out_dtype = torch.bfloat16

        if single_output:
            if D is None or len(D) != 1:
                raise ValueError("single_output=True requires exactly one D tensor")
            if len({shape[1] for shape in output_shapes}) != 1:
                raise ValueError("single_output=True requires a common output width")
            expected_shape = (
                sum(shape[0] for shape in output_shapes),
                output_shapes[0][1],
            )
            if tuple(D[0].shape) != expected_shape:
                raise ValueError(
                    f"Invalid D shape: expected {expected_shape}, got {tuple(D[0].shape)}"
                )
        else:
            if D is None:
                D = [
                    torch.empty(
                        shape,
                        dtype=out_dtype,
                        device=B[index].device,
                    )
                    for index, shape in enumerate(output_shapes)
                ]
            if len(D) != num_gemms:
                raise ValueError(f"Expected {num_gemms} output tensors, got {len(D)}")
            for index, (destination, expected_shape) in enumerate(zip(D, output_shapes)):
                if tuple(destination.shape) != expected_shape:
                    raise ValueError(
                        f"Invalid D[{index}] shape: expected {expected_shape}, "
                        f"got {tuple(destination.shape)}"
                    )

        bias_flags = [has_tensor(bias, i) for i in range(num_gemms)]
        gelu_flags = [has_tensor(pre_gelu_out, i) for i in range(num_gemms)]

        # 2. Decide whether the official native wrapper can represent this call.
        if not transb:
            native_mode = "m_split"
        elif not transa:
            native_mode = "k_split"
        else:
            native_mode = None

        dense_tensors = all(isinstance(tensor, torch.Tensor) for tensor in (*A, *B))
        dtype_ok = False
        device_ok = False
        shape_ok = False

        if dense_tensors:
            input_dtypes = {tensor.dtype for tensor in (*A, *B)}
            input_dtype = next(iter(input_dtypes)) if len(input_dtypes) == 1 else None
            dtype_ok = (
                input_dtype
                in {
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                }
                and out_dtype == input_dtype
            )
            device_ok = len({tensor.device for tensor in (*A, *B)}) == 1

            if native_mode == "m_split":
                shape_ok = (
                    len({int(tensor.shape[1]) for tensor in B}) == 1
                    and len({shape[1] for shape in output_shapes}) == 1
                )
            elif native_mode == "k_split":
                # K-split packs both operands, so every group must produce the
                # same [M, N] shape.
                shape_ok = len(set(output_shapes)) == 1

        has_bias = any(bias_flags)
        epilogue_ok = not any(gelu_flags) and (
            not has_bias or (native_mode == "m_split" and not grad and all(bias_flags))
        )

        use_native = (
            1 < num_gemms <= 128
            and native_mode is not None
            and dense_tensors
            and dtype_ok
            and device_ok
            and shape_ok
            and epilogue_ok
        )

        # 3. Native M-split/K-split path.
        if use_native:
            expected_splits = [int(tensor.shape[0]) for tensor in B]
            split_sizes = (
                [int(size) for size in m_splits]
                if m_splits is not None and len(m_splits) > 0
                else expected_splits
            )
            if split_sizes != expected_splits:
                raise ValueError(
                    "m_splits must equal the original B row counts: "
                    f"expected {expected_splits}, got {split_sizes}"
                )

            # No kernel work is needed for an entirely empty token batch.
            if sum(split_sizes) == 0:
                if native_mode == "k_split" and not accumulate:
                    for destination in D:
                        destination.zero_()
                return bias

            group_split = torch.tensor(
                split_sizes,
                dtype=torch.int64,
                device=B[0].device,
            )
            packed_b = torch.cat(B, dim=0)

            if native_mode == "m_split":
                # Final NPU operands: x=[cat(B)], weight=A.
                npu_weight = A
                group_type = 0
            else:
                # layout="NT" turns cat(B) into the left operand:
                #
                #   x      = [cat(B).T]  -> [M, sum(K_i)]
                #   weight = [cat(A)]    -> [sum(K_i), N]
                #
                # Both lists therefore have length 1, as required by K-split.
                npu_weight = torch.cat(A, dim=0)
                group_type = 2

            layout = ("T" if transa else "N") + ("T" if transb else "N")
            use_forward_bias = native_mode == "m_split" and not grad and all(bias_flags)

            packed_output = _get_tenpu_gemm().general_grouped_gemm(
                npu_weight,
                packed_b,
                group_split,
                layout=layout,
                use_bias=use_forward_bias,
                biases=bias if use_forward_bias else None,
                group_type=group_type,
                group_list_type=1,
                split_item=3,
                out_dtype=out_dtype,
            )

            if not isinstance(packed_output, torch.Tensor):
                raise TypeError(
                    "general_grouped_gemm must return one Tensor "
                    f"for split_item=3, got {type(packed_output)}"
                )

            packed_shape = (
                sum(shape[0] for shape in output_shapes),
                output_shapes[0][1],
            )
            packed_numel = packed_shape[0] * packed_shape[1]
            if packed_output.numel() != packed_numel:
                raise RuntimeError(
                    "Unexpected grouped GEMM output: "
                    f"expected {packed_numel} elements, "
                    f"got shape={tuple(packed_output.shape)}"
                )

            # M-split is already 2D. K-split [G, M, N] is flattened to TE's
            # packed [G*M, N] representation.
            packed_output = packed_output.reshape(packed_shape)

            if single_output:
                outputs = [packed_output]
            else:
                outputs = torch.split(
                    packed_output,
                    [shape[0] for shape in output_shapes],
                    dim=0,
                )

            for destination, source in zip(D, outputs):
                source = source.to(destination.dtype)
                if accumulate:
                    destination.add_(source)
                else:
                    destination.copy_(source)

            return bias

        # 4. Correctness fallback.
        output_offset = 0
        for index in range(num_gemms):
            if single_output:
                rows = output_shapes[index][0]
                destination = D[0][output_offset : output_offset + rows]
                output_offset += rows
            else:
                destination = D[index]

            if workspace:
                gemm_workspace = workspace[min(index, len(workspace) - 1)]
            else:
                gemm_workspace = torch.empty(
                    0,
                    dtype=torch.uint8,
                    device=B[index].device,
                )

            _, bias_grad, _, _ = self.generic_gemm(
                A=A[index],
                transA=transa,
                B=B[index],
                transB=transb,
                D=destination,
                quantizer=None,
                output_dtype=D_type,
                bias=bias[index] if bias_flags[index] else None,
                bias_type=bias_type,
                gelu=gelu_flags[index],
                gelu_in=(pre_gelu_out[index] if gelu_flags[index] else None),
                grad=grad,
                workspace=gemm_workspace,
                workspace_size=workspaceSizes,
                accumulate=accumulate,
                use_split_accumulator=use_split_accumulator,
            )

            if grad and bias_flags[index] and bias_grad is not None:
                bias_grad = bias_grad.to(bias[index].dtype)
                if accumulate:
                    bias[index].add_(bias_grad)
                else:
                    bias[index].copy_(bias_grad)

        _ = math_sm_count  # CUDA-only tuning knob.
        return bias

    # ===================== Dense GroupedTensor GEMM adaptation =====================

    def get_grouped_gemm_setup_workspace_size(self, num_tensors: int) -> int:
        """Return the NPU setup-workspace size for grouped-tensor GEMM.

        CUDA uses this buffer to materialize cuBLASLt pointer arrays. Ascend's
        grouped-matmul operators consume packed tensors and a group-list tensor,
        so no corresponding setup workspace is required.
        """

        if num_tensors < 0:
            raise ValueError(f"num_tensors must be non-negative, got {num_tensors}")
        return 0

    def te_general_grouped_gemm_for_grouped_tensor(self, *args, **kwargs):
        """Forward the base interface to the typed NPU implementation."""
        return self._te_general_grouped_gemm_for_grouped_tensor_impl(*args, **kwargs)

    def _te_general_grouped_gemm_for_grouped_tensor_impl(
        self,
        A: Any,
        transa: bool,
        B: Any,
        transb: bool,
        D: Any,
        bias: Optional[Any],
        bias_scale: Optional[torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        workspace_setup: torch.Tensor,
        workspace_cublas: torch.Tensor,
        use_split_accumulator: bool,
        math_sm_count: int,
    ) -> Any:
        """Run dense TE-FL GroupedTensor GEMM with native Ascend operators.

        For each group, this implements the TE-FL contract

            D[i] = alpha[i] * op(B[i]) @ op(A[i]) + beta[i] * D[i]

        followed by the optional grouped bias addition. TN and NN use
        ``torch_npu.npu_grouped_matmul``. NT (weight gradient) uses
        ``torch_npu.npu_grouped_matmul_add_`` with a temporary destination so
        arbitrary alpha/beta values retain the TE-FL meaning.

        Only dense GroupedTensors (``quantizer is None``) are accepted. The
        quantizer fields are validated but never changed by this backend.
        """

        del workspace_setup, workspace_cublas, use_split_accumulator, math_sm_count

        layout = ("T" if transa else "N") + ("T" if transb else "N")
        if layout not in ("TN", "NN", "NT"):
            raise NotImplementedError(
                f"NPU grouped-tensor GEMM supports layouts TN, NN, and NT; got {layout}"
            )

        grouped_tensors = {"A": A, "B": B, "D": D}
        if bias is not None:
            grouped_tensors["bias"] = bias

        num_tensors = getattr(A, "num_tensors", None)
        if not isinstance(num_tensors, int) or num_tensors <= 0:
            raise ValueError("NPU grouped-tensor GEMM requires a positive integer A.num_tensors")
        if num_tensors > 128:
            raise ValueError(
                f"torch_npu.npu_grouped_matmul supports at most 128 groups, got {num_tensors}"
            )

        rowwise_data: dict[str, torch.Tensor] = {}
        logical_shapes: dict[str, Tuple[int, int]] = {}
        for name, grouped_tensor in grouped_tensors.items():
            if getattr(grouped_tensor, "num_tensors", None) != num_tensors:
                raise ValueError(
                    "A, B, D, and bias must contain the same number of groups: "
                    f"A has {num_tensors}, {name} has "
                    f"{getattr(grouped_tensor, 'num_tensors', None)}"
                )
            if not hasattr(grouped_tensor, "quantizer"):
                raise TypeError(f"{name} must be a TE-FL GroupedTensor")
            if grouped_tensor.quantizer is not None:
                raise NotImplementedError(
                    "NPU te_general_grouped_gemm_for_grouped_tensor currently "
                    f"supports quantizer=None only; {name}.quantizer is set"
                )

            data = getattr(grouped_tensor, "rowwise_data", None)
            if not isinstance(data, torch.Tensor):
                raise TypeError(f"{name}.rowwise_data must be a torch.Tensor")
            if data.device.type != "npu":
                raise ValueError(f"{name}.rowwise_data must be on NPU, got {data.device}")
            if not data.is_contiguous():
                raise ValueError(f"{name}.rowwise_data must be contiguous")

            logical_shape = getattr(grouped_tensor, "logical_shape", None)
            if logical_shape is None or len(logical_shape) != 2:
                raise ValueError(f"{name}.logical_shape must be two-dimensional")
            logical_shape = (int(logical_shape[0]), int(logical_shape[1]))
            if logical_shape[0] < 0 or logical_shape[1] <= 0:
                raise ValueError(f"Invalid {name}.logical_shape: {logical_shape}")
            if data.numel() != logical_shape[0] * logical_shape[1]:
                raise ValueError(
                    f"{name}.rowwise_data has {data.numel()} elements, but "
                    f"logical_shape={logical_shape} requires "
                    f"{logical_shape[0] * logical_shape[1]}"
                )

            first_dims = getattr(grouped_tensor, "first_dims", None)
            if first_dims is not None:
                if not isinstance(first_dims, torch.Tensor):
                    raise TypeError(f"{name}.first_dims must be a torch.Tensor or None")
                if first_dims.dtype != torch.int64 or first_dims.numel() != num_tensors:
                    raise ValueError(
                        f"{name}.first_dims must be an int64 tensor with {num_tensors} elements"
                    )
                if first_dims.device != data.device:
                    raise ValueError(
                        f"{name}.first_dims and rowwise_data must be on the same device"
                    )

            rowwise_data[name] = data
            logical_shapes[name] = logical_shape

        data_device = rowwise_data["A"].device
        input_dtype = rowwise_data["A"].dtype
        supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
        if input_dtype not in supported_dtypes:
            raise TypeError(
                f"NPU dense grouped-tensor GEMM supports FP16, BF16, and FP32, got {input_dtype}"
            )
        for name in ("B", "D"):
            if rowwise_data[name].device != data_device:
                raise ValueError(f"A, B, and D must be on one NPU device; {name} differs")
            if rowwise_data[name].dtype != input_dtype:
                raise TypeError(
                    f"A, B, and D must have one dtype; A is {input_dtype}, "
                    f"{name} is {rowwise_data[name].dtype}"
                )

        for name, coefficient in (("alpha", alpha), ("beta", beta)):
            if not isinstance(coefficient, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if coefficient.device != data_device:
                raise ValueError(f"{name} must be on {data_device}, got {coefficient.device}")
            if coefficient.dtype != torch.float32:
                raise TypeError(f"{name} must have dtype torch.float32")
            if coefficient.numel() not in (1, num_tensors):
                raise ValueError(
                    f"{name} must contain 1 or {num_tensors} values, got {coefficient.numel()}"
                )
        if alpha.numel() != beta.numel():
            raise ValueError("alpha and beta must contain the same number of values")

        def require_uniform_first_dim(name: str) -> int:
            grouped_tensor = grouped_tensors[name]
            if getattr(grouped_tensor, "first_dims", None) is not None:
                raise ValueError(f"{name} must have a uniform first dimension for layout {layout}")
            total_rows = logical_shapes[name][0]
            if total_rows % num_tensors != 0:
                raise ValueError(
                    f"{name}.logical_shape[0]={total_rows} is not divisible by "
                    f"num_tensors={num_tensors}"
                )
            return total_rows // num_tensors

        def group_sizes(name: str) -> torch.Tensor:
            grouped_tensor = grouped_tensors[name]
            first_dims = getattr(grouped_tensor, "first_dims", None)
            if first_dims is not None:
                return first_dims
            common_first_dim = require_uniform_first_dim(name)
            return torch.full(
                (num_tensors,),
                common_first_dim,
                dtype=torch.int64,
                device=data_device,
            )

        a_rows, a_cols = logical_shapes["A"]
        b_rows, b_cols = logical_shapes["B"]
        d_rows, d_cols = logical_shapes["D"]

        if layout == "TN":
            weight_rows = require_uniform_first_dim("A")
            if b_cols != a_cols or d_rows != b_rows or d_cols != weight_rows:
                raise ValueError(
                    "Invalid TN grouped GEMM shapes: expected B[:, K] @ "
                    "A[G, N, K].T -> D[:, N], got "
                    f"A={logical_shapes['A']}, B={logical_shapes['B']}, "
                    f"D={logical_shapes['D']}"
                )
            split_sizes = group_sizes("B")
            x = rowwise_data["B"].view(b_rows, b_cols)
            weight = (
                rowwise_data["A"]
                .view(num_tensors, weight_rows, a_cols)
                .transpose(1, 2)
                .contiguous()
            )
        elif layout == "NN":
            weight_rows = require_uniform_first_dim("A")
            if b_cols != weight_rows or d_rows != b_rows or d_cols != a_cols:
                raise ValueError(
                    "Invalid NN grouped GEMM shapes: expected B[:, N] @ "
                    "A[G, N, K] -> D[:, K], got "
                    f"A={logical_shapes['A']}, B={logical_shapes['B']}, "
                    f"D={logical_shapes['D']}"
                )
            split_sizes = group_sizes("B")
            x = rowwise_data["B"].view(b_rows, b_cols)
            weight = rowwise_data["A"].view(num_tensors, weight_rows, a_cols)
        else:
            output_rows = require_uniform_first_dim("D")
            if a_rows != b_rows or d_rows != num_tensors * output_rows:
                raise ValueError(
                    "Invalid NT grouped GEMM row dimensions: expected "
                    "B_i.T @ A_i -> D_i, got "
                    f"A={logical_shapes['A']}, B={logical_shapes['B']}, "
                    f"D={logical_shapes['D']}"
                )
            if output_rows != b_cols or d_cols != a_cols:
                raise ValueError(
                    "Invalid NT grouped GEMM inner/output dimensions: expected "
                    "B[:, N].T @ A[:, K] -> D[G, N, K], got "
                    f"A={logical_shapes['A']}, B={logical_shapes['B']}, "
                    f"D={logical_shapes['D']}"
                )
            split_sizes = group_sizes("A")
            x = rowwise_data["B"].view(b_rows, b_cols)
            weight = rowwise_data["A"].view(a_rows, a_cols)

        output_group_sizes = group_sizes("D")
        if bias_scale is not None and bias is None:
            raise ValueError("bias_scale requires bias")
        if bias is not None:
            bias_data = rowwise_data["bias"]
            if bias_data.device != data_device or bias_data.dtype != input_dtype:
                raise ValueError("bias must have the same device and dtype as A, B, and D")
            if logical_shapes["bias"] != (num_tensors, d_cols):
                raise ValueError(
                    "Grouped bias must contain one row per group with D's last "
                    f"dimension; expected {(num_tensors, d_cols)}, "
                    f"got {logical_shapes['bias']}"
                )
            if bias_scale is not None:
                if not isinstance(bias_scale, torch.Tensor):
                    raise TypeError("bias_scale must be a torch.Tensor")
                if bias_scale.device != data_device:
                    raise ValueError("bias_scale must be on the same NPU device as D")
                if bias_scale.numel() != d_rows:
                    raise ValueError(
                        f"bias_scale must contain {d_rows} values, got {bias_scale.numel()}"
                    )

        # A one-group group_list_type=1 call is rejected by some torch_npu
        # versions. A regular NPU matmul is semantically identical in that case.
        if num_tensors == 1:
            if layout == "TN":
                product = torch.matmul(x, weight[0])
            elif layout == "NN":
                product = torch.matmul(x, weight[0])
            else:
                product = torch.matmul(x.transpose(0, 1), weight)
        elif d_rows == 0 or (layout == "NT" and a_rows == 0):
            product = torch.zeros_like(rowwise_data["D"]).view(d_rows, d_cols)
        else:
            torch_npu = _get_torch_npu()
            if layout in ("TN", "NN"):
                product = torch_npu.npu_grouped_matmul(
                    [x],
                    [weight],
                    group_list=split_sizes,
                    output_dtype=rowwise_data["D"].dtype,
                    group_type=0,
                    group_list_type=1,
                    split_item=3,
                )[0]
            else:
                # aclnnGroupedMatmulAdd requires an FP32 accumulation target
                # for dense BF16/FP16 inputs on the tested Ascend stack.
                product = torch.zeros(
                    (num_tensors, b_cols, a_cols),
                    dtype=torch.float32,
                    device=data_device,
                )
                torch_npu.npu_grouped_matmul_add_(
                    product,
                    x,
                    weight,
                    torch.cumsum(split_sizes, dim=0),
                )

        if product.numel() != rowwise_data["D"].numel():
            raise RuntimeError(
                "Unexpected NPU grouped GEMM output size: expected "
                f"{rowwise_data['D'].numel()} elements, got {product.numel()}"
            )

        product_2d = product.reshape(d_rows, d_cols)
        destination_2d = rowwise_data["D"].view(d_rows, d_cols)

        def expand_group_values(values: torch.Tensor) -> torch.Tensor:
            if values.numel() == 1:
                return values.reshape(())
            return torch.repeat_interleave(
                values,
                output_group_sizes,
                output_size=d_rows,
            ).view(d_rows, 1)

        # The NPU GEMM result has already been rounded to D's dtype. Perform
        # the alpha/beta combination in FP32 before the final in-place write.
        combined = product_2d.float() * expand_group_values(alpha)
        combined = combined + destination_2d.float() * expand_group_values(beta)
        destination_2d.copy_(combined.to(destination_2d.dtype))

        if bias is not None:
            bias_2d = bias_data.view(num_tensors, d_cols).float()
            expanded_bias = torch.repeat_interleave(
                bias_2d,
                output_group_sizes,
                dim=0,
                output_size=d_rows,
            )
            if bias_scale is not None:
                expanded_bias = expanded_bias * bias_scale.view(d_rows, 1).float()
            destination_2d.copy_((destination_2d.float() + expanded_bias).to(input_dtype))

        return D
