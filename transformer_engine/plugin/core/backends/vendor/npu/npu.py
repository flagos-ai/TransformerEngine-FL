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

from ....ops import TEFLBackendBase, NVTE_Fused_Attn_Backend
from .flash_attention import NPUFlashAttention


def _check_npu_available() -> bool:
    """Check if NPU hardware and torch_npu are available."""
    try:
        import torch_npu  # noqa: F401
        import transformer_engine_npu

        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False


def _ensure_npu_libs():
    """Ensure torch_npu is imported (activates NPU device support in PyTorch)."""
    import torch_npu  # noqa: F401


def _get_torch_npu():
    """Lazy import of torch_npu."""
    _ensure_npu_libs()
    import torch_npu

    return torch_npu


def _get_tenpu_optimizers():
    """Get optimizers subpackage directly, bypassing transformer_engine_npu/__init__.py
    which triggers circular imports via pytorch/__init__.py -> module -> ops."""
    import transformer_engine_npu

    return transformer_engine_npu.pytorch.optimizers


def _get_tenpu_gemm():
    """Get GEMM ops subpackage."""
    _ensure_npu_libs()
    import transformer_engine_npu

    return transformer_engine_npu.pytorch.ops.gemm


def _get_tenpu_activations():
    """Get NPU activation functions (uses NPU-optimized kernels like npu_gelu, npu_swiglu)."""
    _ensure_npu_libs()
    from transformer_engine_npu.pytorch.ops.basic import npu_activation

    return npu_activation


class NPUBackend(TEFLBackendBase):
    """NPU backend delegating to transformer_engine_npu + torch_npu."""

    def is_available(self) -> bool:
        return _check_npu_available()

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """RMSNorm forward using torch_npu.npu_rms_norm.

        TE-FL calls with: (input, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma)
        NPU kernel: npu_rms_norm(input, gamma, epsilon=eps) → (output, rstd)

        NPU kernel requires 2D input [outer_dim, inner_dim]. We reshape accordingly.
        We ignore ln_out (pre-allocated output buffer), otype, sm_margin.
        """
        torch_npu = _get_torch_npu()

        if zero_centered_gamma:
            weight = weight + 1

        # NPU npu_rms_norm requires 2D input: [outer_dim, hidden_size]
        input_shape = input.shape
        inner_dim = weight.shape[0]
        x_2d = input.reshape(-1, inner_dim)

        out_2d, inv_rms = torch_npu.npu_rms_norm(x_2d, weight, epsilon=eps)

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
        zero_centered_gamma: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """RMSNorm backward using torch_npu.npu_rms_norm_backward.

        TE-FL calls with: (dz, x, rsigma, gamma, sm_margin, zero_centered_gamma)
        NPU kernel expects: npu_rms_norm_backward(dy, x, gamma, rstd)
        where rstd must be FP32 and x/dy must be 2D [outer_dim, inner_dim].

        NPU supported combo (BF16):
          dy(BF16) x(BF16) rstd(FP32) gamma(BF16) → dx(BF16) dgamma(FP32)
        """
        torch_npu = _get_torch_npu()

        if zero_centered_gamma:
            gamma = gamma + 1

        # NPU kernel requires 2D input
        input_shape = x.shape
        inner_dim = gamma.shape[0]
        x_2d = x.reshape(-1, inner_dim)
        dz_2d = dz.reshape(-1, inner_dim)

        # NPU kernel requires rstd in float32
        rsigma_fp32 = rsigma.float() if rsigma.dtype != torch.float32 else rsigma

        dx_2d, dw = torch_npu.npu_rms_norm_backward(dz_2d, x_2d, gamma, rsigma_fp32)

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
        scale: float,
        epsilon: float,
    ):
        """Compute per-tensor FP8 scale and scale_inv."""
        opt = _get_tenpu_optimizers()
        opt.multi_tensor_compute_scale_and_scale_inv(
            chunk_size, noop_flag, tensor_lists, scale, epsilon
        )

    def multi_tensor_compute_scale_inv_e8m0(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
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

    def grouped_gemm(
        self,
        A: List[torch.Tensor],
        B: List[torch.Tensor],
        dtype: Any,
        workspaces: List[torch.Tensor],
        m_splits: List[int],
        accumulate: bool = False,
        out: Optional[torch.Tensor] = None,
        bias: Optional[List[torch.Tensor]] = None,
        ub_algo: Optional[Any] = None,
        ub: Optional[Any] = None,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Grouped GEMM."""
        gemm_mod = _get_tenpu_gemm()
        result = gemm_mod.general_grouped_gemm(A, B, "forward", "forward", dtype, biases=bias)
        return result, None, None

    # ===================== Activation Functions =====================
    # Delegates to transformer_engine_npu NPU-optimized kernels (npu_gelu, npu_swiglu, etc.)

    def gelu(self, input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.gelu_fwd(input)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def relu(self, input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.relu_fwd(input)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def silu(self, input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.silu_fwd(input)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def swiglu(self, input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.swiglu_fwd(input)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def geglu(self, input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.geglu_fwd(input)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def reglu(self, input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.reglu_fwd(input)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def srelu(self, input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.srelu_fwd(input)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def glu(self, input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.glu_fwd(input)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def qgelu(self, input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.qgelu_fwd(input)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def qgeglu(self, input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.qgeglu_fwd(input)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def sreglu(self, input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.sreglu_fwd(input)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    # bug
    # def clamped_swiglu(
    #     self, input: torch.Tensor, quantizer: Any = None,
    #     limit: float = 7.0, alpha: float = 1.702,
    # ) -> Any:
    #     act = _get_tenpu_activations()
    #     out = act.clamped_swiglu_fwd(input, clamp_val=limit)
    #     if quantizer is not None and hasattr(quantizer, "quantize"):
    #         out = quantizer.quantize(out)
    #     return out

    # ===================== Activation Backward =====================
    # Delegates to transformer_engine_npu's backward kernels

    def dgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.gelu_bwd(fwd_input, grad)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def dgeglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.geglu_bwd(fwd_input, grad)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def dqgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.qgelu_bwd(fwd_input, grad)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def dqgeglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.qgeglu_bwd(fwd_input, grad)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def drelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.relu_bwd(fwd_input, grad)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def dreglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.reglu_bwd(fwd_input, grad)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def dsrelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.srelu_bwd(fwd_input, grad)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def dsreglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.sreglu_bwd(fwd_input, grad)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def dsilu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.silu_bwd(fwd_input, grad)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def dswiglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.swiglu_bwd(fwd_input, grad)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    def dglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any = None) -> Any:
        act = _get_tenpu_activations()
        out = act.glu_bwd(fwd_input, grad)
        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)
        return out

    # bug
    # def clamped_dswiglu(
    #     self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any = None,
    #     limit: float = 7.0, alpha: float = 1.702,
    # ) -> Any:
    #     act = _get_tenpu_activations()
    #     out = act.clamped_swiglu_bwd(fwd_input, grad, clamp_val=limit)
    #     if quantizer is not None and hasattr(quantizer, "quantize"):
    #         out = quantizer.quantize(out)
    #     return out

    # ===================== Multi-tensor (additional) =====================

    def multi_tensor_scale_tensor(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: torch.Tensor,
    ) -> None:
        """Overflow check + scale with tensor scale factor."""
        opt = _get_tenpu_optimizers()
        opt.multi_tensor_scale_tensor(chunk_size, noop_flag, tensor_lists, scale)

    # ===================== Quantization =====================

    def quantize(
        self,
        tensor: torch.Tensor,
        quantizer: Any,
        output: Optional[torch.Tensor] = None,
        noop: Optional[torch.Tensor] = None,
    ) -> Any:
        """Quantize tensor using quantizer."""
        if quantizer is not None and hasattr(quantizer, "quantize"):
            return quantizer.quantize(tensor)
        return tensor

    # ===================== Utility =====================

    def get_cudnn_version(self) -> int:
        """NPU does not use cuDNN; return 0."""
        return 0
