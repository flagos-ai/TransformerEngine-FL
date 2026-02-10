# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
from typing import Any, List, Optional, Tuple
import torch
from ...ops import *

from .impl import (
    general_gemm_torch,
    rmsnorm_fwd_torch, rmsnorm_bwd_torch,
    layernorm_fwd_torch, layernorm_bwd_torch,
    gelu_torch, geglu_torch, qgelu_torch, qgeglu_torch,
    relu_torch, reglu_torch, srelu_torch, sreglu_torch,
    silu_torch, swiglu_torch, clamped_swiglu_torch,
    dgelu_torch, dgeglu_torch, dqgelu_torch, dqgeglu_torch,
    drelu_torch, dreglu_torch, dsrelu_torch, dsreglu_torch,
    dsilu_torch, dswiglu_torch, clamped_dswiglu_torch,
    dbias_dgelu_torch, dbias_dsilu_torch, dbias_drelu_torch,
    dbias_dqgelu_torch, dbias_dsrelu_torch,
    scaled_softmax_forward_torch, scaled_softmax_backward_torch,
    scaled_masked_softmax_forward_torch, scaled_masked_softmax_backward_torch,
    scaled_upper_triang_masked_softmax_forward_torch,
    scaled_upper_triang_masked_softmax_backward_torch,
    scaled_aligned_causal_masked_softmax_forward_torch,
    scaled_aligned_causal_masked_softmax_backward_torch,
    dropout_fwd_torch, dropout_bwd_torch,
    multi_tensor_scale_torch, multi_tensor_l2norm_torch,
    multi_tensor_adam_torch, multi_tensor_adam_param_remainder_torch,
    multi_tensor_sgd_torch,
)


class ReferenceBackend(TEFLBackendBase):
    @staticmethod
    def check_available() -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def get_attention_backend(self, _attention_params=None):
        from packaging.version import Version as PkgVersion
        from ...logger_manager import get_logger
        logger = get_logger()

        # Read environment variables to determine which backends to enable
        use_flash_attention = int(os.getenv("NVTE_FLASH_ATTN", "1"))
        use_fused_attention = int(os.getenv("NVTE_FUSED_ATTN", "1"))
        use_unfused_attention = int(os.getenv("NVTE_UNFUSED_ATTN", "1"))

        # Log disabled backends
        if not use_flash_attention:
            logger.info_once("Disabling FlashAttention due to NVTE_FLASH_ATTN=0")
        if not use_fused_attention:
            logger.info_once("Disabling FusedAttention due to NVTE_FUSED_ATTN=0")
        if not use_unfused_attention:
            logger.info_once("Disabling UnfusedDotProductAttention due to NVTE_UNFUSED_ATTN=0")

        flash_attention_backend = PkgVersion("2.6.0") if use_flash_attention else None
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

    def generic_gemm(
        self,
        A: Any,
        transA: bool,
        B: Any,
        transB: bool,
        D: Any,
        quantizer: Any,
        output_dtype: Optional[DType],
        bias: Optional[torch.Tensor],
        bias_type: DType,
        gelu: bool,
        gelu_in: Optional[torch.Tensor],
        grad: bool,
        workspace: torch.Tensor,
        workspace_size: int,
        accumulate: bool,
        use_split_accumulator: bool,
        comm_overlap: Optional[Any] = None,
        comm_type: Optional[CommOverlapType] = None,
        extra_output: Optional[torch.Tensor] = None,
        bulk_overlap: bool = False,
        alpha: float = 1.0,
        beta: Optional[float] = None,
    ) -> List[Any]:
        return general_gemm_torch(
            A, transA, B, transB, D, quantizer, output_dtype,
            bias, bias_type, gelu, gelu_in, grad, workspace, workspace_size,
            accumulate, use_split_accumulator, comm_overlap, comm_type,
            extra_output, bulk_overlap, alpha, beta
        )

    # GELU and variants
    def gelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return gelu_torch(input, quantizer)

    def geglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return geglu_torch(input, quantizer)

    def qgelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return qgelu_torch(input, quantizer)

    def qgeglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return qgeglu_torch(input, quantizer)

    # ReLU and variants
    def relu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return relu_torch(input, quantizer)

    def reglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return reglu_torch(input, quantizer)

    def srelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return srelu_torch(input, quantizer)

    def sreglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return sreglu_torch(input, quantizer)

    # SwiGLU and variants
    def silu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return silu_torch(input, quantizer)

    def swiglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return swiglu_torch(input, quantizer)

    def clamped_swiglu(
        self,
        input: torch.Tensor,
        quantizer: Any,
        limit: float = 7.0,
        alpha: float = 1.702,
    ) -> Any:
        return clamped_swiglu_torch(input, quantizer, limit, alpha)

    # Backward of GELU and variants
    def dgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dgelu_torch(grad, fwd_input, quantizer)

    def dgeglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dgeglu_torch(grad, fwd_input, quantizer)

    def dqgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dqgelu_torch(grad, fwd_input, quantizer)

    def dqgeglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dqgeglu_torch(grad, fwd_input, quantizer)

    # Backward of ReLU and variants
    def drelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return drelu_torch(grad, fwd_input, quantizer)

    def dreglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dreglu_torch(grad, fwd_input, quantizer)

    def dsrelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dsrelu_torch(grad, fwd_input, quantizer)

    def dsreglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dsreglu_torch(grad, fwd_input, quantizer)

    # Backward of SiLU and variants
    def dsilu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dsilu_torch(grad, fwd_input, quantizer)

    def dswiglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dswiglu_torch(grad, fwd_input, quantizer)

    def clamped_dswiglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
        limit: float = 7.0,
        alpha: float = 1.702,
    ) -> Any:
        return clamped_dswiglu_torch(grad, fwd_input, quantizer, limit, alpha)

    # DBias + DAct fusions
    def dbias_dgelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> List[Any]:
        return dbias_dgelu_torch(grad, fwd_input, quantizer)

    def dbias_dsilu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> List[Any]:
        return dbias_dsilu_torch(grad, fwd_input, quantizer)

    def dbias_drelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Tuple[torch.Tensor, Any]:
        return dbias_drelu_torch(grad, fwd_input, quantizer)

    def dbias_dqgelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> List[Any]:
        return dbias_dqgelu_torch(grad, fwd_input, quantizer)

    def dbias_dsrelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> List[Any]:
        return dbias_dsrelu_torch(grad, fwd_input, quantizer)

    # LayerNorm
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
    ) -> List[Any]:
        return layernorm_fwd_torch(
            input=input,
            weight=weight,
            bias=bias,
            eps=eps,
            ln_out=ln_out,
            quantizer=quantizer,
            odtype=otype,
            sm_margin=sm_margin,
            zero_centered_gamma=zero_centered_gamma,
        )

    def layernorm_bwd(
        self,
        dz: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> List[Any]:
        return layernorm_bwd_torch(
            dy=dz,
            x=x,
            mu=mu,
            rsigma=rsigma,
            gamma=gamma,
            sm_margin=sm_margin,
            zero_centered_gamma=zero_centered_gamma,
        )

    # RMSNorm
    def rmsnorm_fwd(
        self,
        input: Any,
        weight: Any,
        eps: float,
        ln_out: Any,
        quantizer: Any,
        otype: DType,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> List[Any]:
        return rmsnorm_fwd_torch(
            input=input,
            weight=weight,
            eps=eps,
            ln_out=ln_out,
            quantizer=quantizer,
            odtype=otype,
            sm_margin=sm_margin,
            zero_centered_gamma=zero_centered_gamma,
        )

    def rmsnorm_bwd(
        self,
        dz: torch.Tensor,
        x: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> List[Any]:
        return rmsnorm_bwd_torch(
            dy=dz,
            x=x,
            rsigma=rsigma,
            gamma=gamma,
            sm_margin=sm_margin,
            zero_centered_gamma=zero_centered_gamma,
        )

    # Softmax functions
    def scaled_softmax_forward(
        self,
        input: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        return scaled_softmax_forward_torch(input, scale)

    def scaled_softmax_backward(
        self,
        output_grad_: torch.Tensor,
        softmax_results_: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        return scaled_softmax_backward_torch(output_grad_, softmax_results_, scale_factor)

    def scaled_masked_softmax_forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        return scaled_masked_softmax_forward_torch(input, mask, scale_factor)

    def scaled_masked_softmax_backward(
        self,
        output_grad_: torch.Tensor,
        softmax_results_: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        return scaled_masked_softmax_backward_torch(output_grad_, softmax_results_, scale_factor)

    def scaled_upper_triang_masked_softmax_forward(
        self,
        input: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        return scaled_upper_triang_masked_softmax_forward_torch(input, scale_factor)

    def scaled_upper_triang_masked_softmax_backward(
        self,
        output_grads_: torch.Tensor,
        softmax_results_: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        return scaled_upper_triang_masked_softmax_backward_torch(output_grads_, softmax_results_, scale_factor)

    def scaled_aligned_causal_masked_softmax_forward(
        self,
        input: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        return scaled_aligned_causal_masked_softmax_forward_torch(input, scale_factor)

    def scaled_aligned_causal_masked_softmax_backward(
        self,
        output_grad_: torch.Tensor,
        softmax_results_: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        return scaled_aligned_causal_masked_softmax_backward_torch(output_grad_, softmax_results_, scale_factor)

    # Fused attention backend
    def get_fused_attn_backend(
        self,
        _is_training: bool,
        _q_dtype: DType,
        _kv_dtype: DType,
        _qkv_layout: NVTE_QKV_Layout,
        _bias_type: NVTE_Bias_Type,
        _attn_mask_type: NVTE_Mask_Type,
        _softmax_type: NVTE_Softmax_Type,
        _p_dropout: float,
        _num_attn_heads: int,
        _num_gqa_groups: int,
        _max_seqlen_q: int,
        _max_seqlen_kv: int,
        _head_dim_qk: int,
        _head_dim_v: int,
        _window_size_left: int,
        _window_size_right: int,
        _return_max_logit: bool,
    ) -> NVTE_Fused_Attn_Backend:
        return NVTE_Fused_Attn_Backend.NVTE_No_Backend

    # Dropout
    def dropout_fwd(
        self,
        input: torch.Tensor,
        dropout_probability: float,
        out: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return dropout_fwd_torch(input, dropout_probability, out)

    def dropout_bwd(
        self,
        grad_output: torch.Tensor,
        mask: torch.Tensor,
        dropout_probability: float,
        grad_input: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return dropout_bwd_torch(grad_output, mask, dropout_probability, grad_input)

    # Misc
    def get_cublasLt_version(self) -> int:
        return 0

    def get_cudnn_version(self) -> int:
        return 0

    def get_num_cublas_streams(self) -> int:
        return 0

    # Multi-tensor functions
    def multi_tensor_scale(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: float,
    ) -> None:
        return multi_tensor_scale_torch(chunk_size, noop_flag, tensor_lists, scale)

    def multi_tensor_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        per_tensor: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return multi_tensor_l2norm_torch(chunk_size, noop_flag, tensor_lists, per_tensor)

    def multi_tensor_unscale_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        inv_scale: torch.Tensor,
        per_tensor: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noop_flag.item() != 0:
            device = tensor_lists[0][0].device if tensor_lists and tensor_lists[0] else 'cpu'
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # Multiply by inv_scale
        unscaled_tensors = []
        for tensor in tensor_lists[0]:
            unscaled_tensors.append(tensor * inv_scale.item())

        return multi_tensor_l2norm_torch(chunk_size, noop_flag, [unscaled_tensors], per_tensor)

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
        return multi_tensor_adam_torch(
            chunk_size, noop_flag, tensor_lists,
            lr, beta1, beta2, epsilon, step, mode, bias_correction, weight_decay
        )

    def multi_tensor_adam_param_remainder(
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
        return multi_tensor_adam_param_remainder_torch(
            chunk_size, noop_flag, tensor_lists,
            lr, beta1, beta2, epsilon, step, mode, bias_correction, weight_decay
        )

    def multi_tensor_sgd(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        wd: float,
        momentum: float,
        dampening: float,
        lr: float,
        nesterov: bool,
        first_run: bool,
        wd_after_momentum: bool,
        scale: float,
    ) -> None:
        return multi_tensor_sgd_torch(
            chunk_size, noop_flag, tensor_lists,
            wd, momentum, dampening, lr, nesterov, first_run, wd_after_momentum, scale
        )

    def get_flash_attention_class(self):
        from .flash_attention import FlashAttentionTorch
        return FlashAttentionTorch
