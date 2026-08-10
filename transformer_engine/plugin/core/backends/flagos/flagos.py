# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
from typing import Any, List, Optional, Tuple, Union

import torch

from ...ops import *

from .impl import (
    layernorm_fwd_fl,
    layernorm_bwd_fl,
    rmsnorm_fwd_fl,
    rmsnorm_bwd_fl,
    multi_tensor_scale_fl,
    multi_tensor_adam_fl,
    multi_tensor_adam_param_remainder_fl,
    multi_tensor_l2_norm_fl,
    generic_gemm_fl,
    scaled_masked_softmax_forward_fl,
    scaled_masked_softmax_backward_fl,
    te_general_grouped_gemm_fl,
    fused_rope_forward_fl,
    fused_rope_backward_fl,
    fused_qkv_rope_forward_fl,
    fused_qkv_rope_backward_fl,
)


def _check_flagos_available() -> bool:
    return True


class FlagOSBackend(TEFLBackendBase):
    @staticmethod
    def check_available() -> bool:
        return _check_flagos_available()

    def is_available(self) -> bool:
        return _check_flagos_available()

    def get_attention_backend(self, attention_params=None):
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

    ##### transformer_engine/pytorch/csrc/extensions/pybind.cpp #####
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
        return generic_gemm_fl(
            A,
            transA,
            B,
            transB,
            D,
            quantizer,
            output_dtype,
            bias,
            bias_type,
            gelu,
            gelu_in,
            grad,
            workspace,
            workspace_size,
            accumulate,
            use_split_accumulator,
            comm_overlap,
            comm_type,
            extra_output,
            bulk_overlap,
            alpha,
            beta,
        )

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
        return te_general_grouped_gemm_fl(
            A,
            transa,
            B,
            transb,
            D,
            D_type,
            m_splits,
            bias,
            bias_type,
            single_output,
            pre_gelu_out,
            grad,
            workspace,
            workspaceSizes,
            accumulate,
            use_split_accumulator,
            math_sm_count,
        )

    # Other granular functions
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
        return layernorm_fwd_fl(
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
        return layernorm_bwd_fl(
            dy=dz,
            x=x,
            mu=mu,
            rsigma=rsigma,
            gamma=gamma,
            sm_margin=sm_margin,
            zero_centered_gamma=zero_centered_gamma,
        )

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
        return rmsnorm_fwd_fl(
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
        return rmsnorm_bwd_fl(
            dy=dz,
            x=x,
            rsigma=rsigma,
            gamma=gamma,
            sm_margin=sm_margin,
            zero_centered_gamma=zero_centered_gamma,
        )

    def get_fused_attn_backend(self, *args, **kwargs) -> int:
        return NVTE_Fused_Attn_Backend.NVTE_No_Backend

    # Softmax functions
    def scaled_masked_softmax_forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        scale_factor: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        return scaled_masked_softmax_forward_fl(input, mask, scale_factor)

    def scaled_masked_softmax_backward(
        self,
        output_grad_: torch.Tensor,
        softmax_results_: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        return scaled_masked_softmax_backward_fl(output_grad_, softmax_results_, scale_factor)

    # multi-tensor functions
    def multi_tensor_scale(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: float,
    ) -> None:
        return multi_tensor_scale_fl(chunk_size, noop_flag, tensor_lists, scale)

    def multi_tensor_scale_tensor(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: torch.Tensor,
    ) -> None:
        # Reuse multi_tensor_scale by converting tensor scale to float
        scale_value = scale.item()
        return multi_tensor_scale_fl(chunk_size, noop_flag, tensor_lists, scale_value)

    def multi_tensor_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        per_tensor: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return multi_tensor_l2_norm_fl(chunk_size, noop_flag, tensor_lists, per_tensor)

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
        return multi_tensor_adam_fl(
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
        return multi_tensor_adam_param_remainder_fl(
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

    # fused apply rope
    def fused_rope_forward(
        self,
        input: torch.Tensor,
        freqs: torch.Tensor,
        start_positions: Optional[torch.Tensor],
        qkv_format: NVTE_QKV_Format,
        interleaved: bool,
        cu_seqlens: Optional[torch.Tensor],
        cp_size: int,
        cp_rank: int,
    ) -> torch.Tensor:
        return fused_rope_forward_fl(
            input,
            freqs,
            start_positions,
            qkv_format,
            interleaved,
            cu_seqlens,
            cp_size,
            cp_rank,
        )

    def fused_rope_backward(
        self,
        output_grads: torch.Tensor,
        freqs: torch.Tensor,
        start_positions: Optional[torch.Tensor],
        qkv_format: NVTE_QKV_Format,
        interleaved: bool,
        cu_seqlens: Optional[torch.Tensor],
        cp_size: int,
        cp_rank: int,
    ) -> torch.Tensor:
        return fused_rope_backward_fl(
            output_grads,
            freqs,
            start_positions,
            qkv_format,
            interleaved,
            cu_seqlens,
            cp_size,
            cp_rank,
        )

    def fused_qkv_rope_forward(
        self,
        qkv_input: torch.Tensor,
        q_freqs: torch.Tensor,
        k_freqs: torch.Tensor,
        start_positions: Optional[torch.Tensor],
        qkv_split_arg_list: List[int],
        qkv_format: NVTE_QKV_Format,
        interleaved: bool,
        cp_size: int,
        cp_rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return fused_qkv_rope_forward_fl(
            qkv_input,
            q_freqs,
            k_freqs,
            start_positions,
            qkv_split_arg_list,
            qkv_format,
            interleaved,
            cp_size,
            cp_rank,
        )

    def fused_qkv_rope_backward(
        self,
        q_grad_out: torch.Tensor,
        k_grad_out: torch.Tensor,
        v_grad_out: torch.Tensor,
        q_freqs: torch.Tensor,
        k_freqs: torch.Tensor,
        qkv_split_arg_list: List[int],
        qkv_format: NVTE_QKV_Format,
        interleaved: bool,
        cp_size: int,
        cp_rank: int,
    ) -> torch.Tensor:
        return fused_qkv_rope_backward_fl(
            q_grad_out,
            k_grad_out,
            v_grad_out,
            q_freqs,
            k_freqs,
            qkv_split_arg_list,
            qkv_format,
            interleaved,
            cp_size,
            cp_rank,
        )

    # Misc
    def get_cublasLt_version(self) -> int:
        return 110000

    def get_cudnn_version(self) -> int:
        return 90000

    def get_num_cublas_streams(self) -> int:
        return 4  # keep consistent with transformer_engine/common/util/multi_stream.cpp, get_num_compute_streams()

    ############## class func #################################
    def get_flash_attention_class(self):
        from .attention.dot_product_attention.backends import FlashAttentionFL

        return FlashAttentionFL
