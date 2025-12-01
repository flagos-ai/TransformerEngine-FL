# transformer_engine_backend.py
import os
import torch
from typing import Any, Callable, List, Optional, Tuple, Union
import logging

logger = logging.getLogger("TEFL")
logger.setLevel(logging.INFO)

### GEMM
from transformer_engine.pytorch.cpp_extensions import (
    general_gemm,
)
from transformer_engine.pytorch.backend.gemm import (
    gems_general_gemm,
)

### RMSNORM
from transformer_engine.pytorch.module._common import (
    apply_normalization,
)
from transformer_engine_torch import rmsnorm_bwd, rmsnorm_fwd

from transformer_engine.pytorch.backend.gems_rms_norm import (
    rms_norm_forward,
    rms_norm_backward,
)
def gems_apply_normalization(
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
    normalization_func = rms_norm_forward
    ln_out, rsigma = normalization_func(
        inputmat,
        [inputmat.shape[-1]],
        ln_weight,
        eps,
    )
    mu = None
    return ln_out, mu, rsigma

def gems_rmsnorm_fwd(
    input,
    weight,
    eps,
    ln_out,
    quantizer,
    odtype,
    sm_margin,
    zero_centered_gamma,
):
    y, rstdevs = rms_norm_forward(
        input,
        [input.shape[-1]],
        weight,
        eps,
    )
    return y, rstdevs

def gems_rmsnorm_bwd(
    dy,
    x,
    rsigma,
    gamma,
    sm_margin,
    zero_centered_gamma,
    eps,
):
    dx, dw = rms_norm_backward(
        dy,
        x,
        rsigma,
        [x.shape[-1]],
        gamma,
        eps,
    )
    return dx, dw

### AdamW
from transformer_engine.pytorch.backend.fused_adam import (
    fl_multi_tensor_adam,
)
from transformer_engine_torch import multi_tensor_adam

### Flash-Attn

class TransformerEngineBackend:

    def use_te_fl(self) -> bool:
        return bool(os.environ.get("USE_TRANSFORMER_ENGINE_FL", False))

    def gemm(self, *args, **kwargs):
        if self.use_te_fl():
            logger.info("TE-FL GEMM")
            return gems_general_gemm(*args, **kwargs)
        else:
            logger.info("TE-Native GEMM")
            return general_gemm(*args, **kwargs)
    
    def apply_normalization(self, *args, **kwargs):
        if self.use_te_fl():
            logger.info("TE-FL Apply Normalization")
            return gems_apply_normalization(*args, **kwargs)
        else:
            logger.info("TE-Native Apply Normalization")
            return apply_normalization(*args, **kwargs)
    
    def rmsnorm_fwd(self, *args, **kwargs):
        if self.use_te_fl():
            logger.info("TE-FL RmsNorm FWD")
            y, rstdevs = gems_rmsnorm_fwd(*args, **kwargs)
            return y, None, rstdevs
        else:
            logger.info("TE-Native RmsNorm FWD")
            return rmsnorm_fwd(*args, **kwargs)
    
    def rmsnorm_bwd(self, *args, **kwargs):
        if self.use_te_fl():
            logger.info("TE-FL RmsNorm BWD")
            return gems_rmsnorm_bwd(*args, **kwargs)
        else:
            logger.info("TE-Native RmsNorm BWD")
            trimmed_args = args[:-1]  # cut eps
            return rmsnorm_bwd(*trimmed_args, **kwargs)
    
    def multi_tensor_adam(self):
        if self.use_te_fl():
            logger.info("TE-FL Fused Adam")
            return fl_multi_tensor_adam
        else:
            logger.info("TE-Native Fused Adam")
            return multi_tensor_adam
    
    def flash_attention(self, *args, **kwargs):
        if self.use_te_fl():
            logger.info("TE-FL Gems Flash Attention")
            from transformer_engine.pytorch.backend.flash_attn import (
                GemsFlashAttention,
            )
            return GemsFlashAttention(*args, **kwargs)
        else:
            logger.info("TE-Native Flash Attention")
            from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                FlashAttention,
            )
            return FlashAttention(*args, **kwargs)

backend = TransformerEngineBackend()

