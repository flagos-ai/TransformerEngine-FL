# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .import_utils import safety_import
from .register_backend import register_backend
from .logger import get_logger
logger = get_logger()


### GEMM
native_general_gemm = safety_import('transformer_engine.pytorch.cpp_extensions', 'general_gemm')
### RMSNORM
native_apply_normalization = safety_import('transformer_engine.pytorch.module._common', 'apply_normalization')
native_rmsnorm_bwd = safety_import('transformer_engine_torch', 'rmsnorm_bwd')
native_rmsnorm_fwd = safety_import('transformer_engine_torch', 'rmsnorm_fwd')
### AdamW
native_multi_tensor_adam = safety_import('transformer_engine_torch', 'multi_tensor_adam')
### Flash-Attn
# Use lazy=True to avoid circular imports (flash_attn -> dot_product_attention -> transformer_engine_backend)
NativeFlashAttention = safety_import(
    'transformer_engine.pytorch.attention.dot_product_attention.backends',
    'FlashAttention',
    lazy=True
)

# Register native backend
def register_native_backend():
    # Note: native_rmsnorm_bwd doesn't take eps as the last argument, so we wrap it
    def native_rmsnorm_bwd_wrapper(*args, **kwargs):
        return native_rmsnorm_bwd(*args[:-1], **kwargs)
    register_backend("native", {
        "gemm": native_general_gemm,
        "apply_normalization": native_apply_normalization,
        "rmsnorm_fwd": native_rmsnorm_fwd,
        "rmsnorm_bwd": native_rmsnorm_bwd_wrapper,
        "adam": native_multi_tensor_adam,
        "flash_attention": NativeFlashAttention,
    })
