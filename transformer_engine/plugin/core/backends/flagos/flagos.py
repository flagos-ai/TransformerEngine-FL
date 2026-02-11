# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
from typing import Any, List, Optional, Tuple, Union

import torch

from ...ops import *

from .impl import (
    rmsnorm_fwd_fl, rmsnorm_bwd_fl,
    multi_tensor_scale_fl, multi_tensor_adam_fl,
    multi_tensor_adam_param_remainder_fl,
    multi_tensor_l2_norm_fl,
    generic_gemm_fl
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
            A, transA, B, transB, D, quantizer, output_dtype,
            bias, bias_type, gelu, gelu_in, grad, workspace, workspace_size,
            accumulate, use_split_accumulator, comm_overlap, comm_type,
            extra_output, bulk_overlap, alpha, beta
        )

    # Other granular functions
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
            input=input, weight=weight, eps=eps, ln_out=ln_out,
            quantizer=quantizer, odtype=otype,
            sm_margin=sm_margin, zero_centered_gamma=zero_centered_gamma,
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
            dy=dz, x=x, rsigma=rsigma, gamma=gamma,
            sm_margin=sm_margin, zero_centered_gamma=zero_centered_gamma
        )
    def get_fused_attn_backend(self, *args, **kwargs) -> int:
        return NVTE_Fused_Attn_Backend.NVTE_No_Backend

    # multi-tensor functions
    def multi_tensor_scale(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: float,
    ) -> None:
        return multi_tensor_scale_fl(chunk_size, noop_flag, tensor_lists, scale)
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
            chunk_size, noop_flag, tensor_lists, lr, beta1, beta2, epsilon,
            step, mode, bias_correction, weight_decay,
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
            chunk_size, noop_flag, tensor_lists,
            lr, beta1, beta2, epsilon,
            step, mode, bias_correction, weight_decay,
        )

    # Misc
    def get_cublasLt_version(self) -> int:
        return 110000
    def get_cudnn_version(self) -> int:
        return 90000
    def get_num_cublas_streams(self) -> int:
        return 0

############## class func #################################
    def get_flash_attention_class(self):
        from .attention.dot_product_attention.backends import FlashAttentionFL
        return FlashAttentionFL
