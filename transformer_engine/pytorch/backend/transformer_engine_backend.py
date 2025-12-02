# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import torch
from typing import Any, Callable, List, Optional, Tuple, Union
import logging

from .import_utils import safety_import

level = os.getenv("TEFL_LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("TEFL")
logger.setLevel(getattr(logging, level, logging.INFO))


### GEMM
native_general_gemm = safety_import('transformer_engine.pytorch.cpp_extensions', 'general_gemm')
te_fl_general_gemm = safety_import('transformer_engine.pytorch.backend.gemm', 'te_fl_general_gemm')

### RMSNORM
native_apply_normalization = safety_import('transformer_engine.pytorch.module._common', 'apply_normalization')
native_rmsnorm_bwd = safety_import('transformer_engine_torch', 'rmsnorm_bwd')
native_rmsnorm_fwd = safety_import('transformer_engine_torch', 'rmsnorm_fwd')
gems_rmsnorm_fwd = safety_import('transformer_engine.pytorch.backend.gems_rms_norm', 'rms_norm_forward')
gems_rmsnorm_bwd = safety_import('transformer_engine.pytorch.backend.gems_rms_norm', 'rms_norm_backward')

### AdamW
native_multi_tensor_adam = safety_import('transformer_engine_torch', 'multi_tensor_adam')
te_fl_multi_tensor_adam = safety_import('transformer_engine.pytorch.backend.fused_adam', 'te_fl_multi_tensor_adam')

### Flash-Attn
# Use lazy=True to avoid circular imports (flash_attn -> dot_product_attention -> transformer_engine_backend)
NativeFlashAttention = safety_import(
    'transformer_engine.pytorch.attention.dot_product_attention.backends',
    'FlashAttention',
    lazy=True
)
TEFLFlashAttention = safety_import(
    'transformer_engine.pytorch.backend.flash_attn',
    'TEFLFlashAttention',
    lazy=True
)


def te_fl_apply_normalization(
    inputmat: torch.Tensor,
    ln_out: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: Union[torch.Tensor, None],
    eps: float,
    output_quantizer,
    output_dtype,
    normalization: str,
    fwd_ln_sm_margin: int,
    zero_centered_gamma: bool,
):
    normalization_func = gems_rmsnorm_fwd
    ln_out, rsigma = normalization_func(
        inputmat,
        [inputmat.shape[-1]],
        ln_weight,
        eps,
    )
    mu = None
    return ln_out, mu, rsigma


def te_fl_rmsnorm_fwd(
    input,
    weight,
    eps,
    ln_out,
    quantizer,
    odtype,
    sm_margin,
    zero_centered_gamma,
):
    y, rstdevs = gems_rmsnorm_fwd(
        input,
        [input.shape[-1]],
        weight,
        eps,
    )
    return y, None, rstdevs


def te_fl_rmsnorm_bwd(
    dy,
    x,
    rsigma,
    gamma,
    sm_margin,
    zero_centered_gamma,
    eps,
):
    dx, dw = gems_rmsnorm_bwd(
        dy,
        x,
        rsigma,
        [x.shape[-1]],
        gamma,
        eps,
    )
    return dx, dw


class TransformerEngineBackend:

    def use_te_fl(self) -> bool:
        """Global flag to enable Transformer Engine FL backend."""
        global_flag = os.environ.get("USE_TRANSFORMER_ENGINE_FL", "0")
        return global_flag.lower() in ("1", "true", "yes", "on")
    
    def use_te_fl_gemm(self) -> bool:
        """Enable TE-FL GEMM. If global flag is True, always enabled. Otherwise check sub-flag."""
        if self.use_te_fl():
            return True
        # If global flag is False, check sub-flag
        sub_flag = os.environ.get("USE_TRANSFORMER_ENGINE_FL_GEMM", "0")
        return sub_flag.lower() in ("1", "true", "yes", "on")
    
    def use_te_fl_rmsnorm(self) -> bool:
        """Enable TE-FL RMSNorm. If global flag is True, always enabled. Otherwise check sub-flag."""
        if self.use_te_fl():
            return True
        # If global flag is False, check sub-flag
        sub_flag = os.environ.get("USE_TRANSFORMER_ENGINE_FL_RMSNORM", "0")
        return sub_flag.lower() in ("1", "true", "yes", "on")
    
    def use_te_fl_adam(self) -> bool:
        """Enable TE-FL Adam optimizer. If global flag is True, always enabled. Otherwise check sub-flag."""
        if self.use_te_fl():
            return True
        # If global flag is False, check sub-flag
        sub_flag = os.environ.get("USE_TRANSFORMER_ENGINE_FL_ADAM", "0")
        return sub_flag.lower() in ("1", "true", "yes", "on")
    
    def use_te_fl_flash_attention(self) -> bool:
        """Enable TE-FL Flash Attention. If global flag is True, always enabled. Otherwise check sub-flag."""
        if self.use_te_fl():
            return True
        # If global flag is False, check sub-flag
        sub_flag = os.environ.get("USE_TRANSFORMER_ENGINE_FL_FLASH_ATTENTION", "0")
        return sub_flag.lower() in ("1", "true", "yes", "on")

    def gemm(self, *args, **kwargs):
        if self.use_te_fl_gemm():
            logger.debug("TE-FL GEMM")
            return te_fl_general_gemm(*args, **kwargs)
        else:
            logger.debug("TE-Native GEMM")
            return native_general_gemm(*args, **kwargs)
    
    def apply_normalization(self, *args, **kwargs):
        if self.use_te_fl_rmsnorm():
            logger.debug("TE-FL Apply Normalization")
            return te_fl_apply_normalization(*args, **kwargs)
        else:
            logger.debug("TE-Native Apply Normalization")
            return native_apply_normalization(*args, **kwargs)
    
    def rmsnorm_fwd(self, *args, **kwargs):
        if self.use_te_fl_rmsnorm():
            logger.debug("TE-FL RmsNorm FWD")
            return te_fl_rmsnorm_fwd(*args, **kwargs)
        else:
            logger.debug("TE-Native RmsNorm FWD")
            return native_rmsnorm_fwd(*args, **kwargs)
    
    def rmsnorm_bwd(self, *args, **kwargs):
        if self.use_te_fl_rmsnorm():
            logger.debug("TE-FL RmsNorm BWD")
            return te_fl_rmsnorm_bwd(*args, **kwargs)
        else:
            logger.debug("TE-Native RmsNorm BWD")
            trimmed_args = args[:-1]  # cut eps
            return native_rmsnorm_bwd(*trimmed_args, **kwargs)
    
    def multi_tensor_adam(self):
        if self.use_te_fl_adam():
            logger.debug("TE-FL Fused Adam")
            return te_fl_multi_tensor_adam
        else:
            logger.debug("TE-Native Fused Adam")
            return native_multi_tensor_adam
    
    def flash_attention(self, *args, **kwargs):
        if self.use_te_fl_flash_attention():
            logger.debug("TE-FL Flash Attention")
            return TEFLFlashAttention(*args, **kwargs)
        else:
            logger.debug("TE-Native Flash Attention")
            return NativeFlashAttention(*args, **kwargs)

backend = TransformerEngineBackend()
