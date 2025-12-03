# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .import_utils import safety_import
from .register import register_backend
from .logger import get_logger
logger = get_logger()


### GEMM
te_fl_general_gemm = safety_import('transformer_engine.plugins.modules.gemm', 'te_fl_general_gemm')
### RMSNORM
gems_rmsnorm_fwd = safety_import('transformer_engine.plugins.modules.gems_rms_norm', 'rms_norm_forward')
gems_rmsnorm_bwd = safety_import('transformer_engine.plugins.modules.gems_rms_norm', 'rms_norm_backward')
### AdamW
te_fl_multi_tensor_adam = safety_import('transformer_engine.plugins.modules.fused_adam', 'te_fl_multi_tensor_adam')
### Flash-Attn
# Use lazy=True to avoid circular imports (flash_attn -> dot_product_attention -> transformer_engine_backend)
TEFLFlashAttention = safety_import(
    'transformer_engine.plugins.modules.flash_attn',
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

def register_te_fl_backend():
    # Register TE-FL backend
    register_backend("te_fl", {
        "gemm": te_fl_general_gemm,
        "apply_normalization": te_fl_apply_normalization,
        "rmsnorm_fwd": te_fl_rmsnorm_fwd,
        "rmsnorm_bwd": te_fl_rmsnorm_bwd,
        "adam": te_fl_multi_tensor_adam,
        "flash_attention": TEFLFlashAttention,
    })