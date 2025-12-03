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
fl_general_gemm = safety_import('transformer_engine.plugins.cpp_extensions.gemm', 'fl_general_gemm')
### RMSNORM
fl_apply_normalization = safety_import('transformer_engine.plugins.module._common', 'fl_apply_normalization')
fl_rmsnorm_bwd = safety_import('transformer_engine.plugins.cpp_extensions', 'fl_rmsnorm_bwd')
fl_rmsnorm_fwd = safety_import('transformer_engine.plugins.cpp_extensions', 'fl_rmsnorm_fwd')
### AdamW
fl_multi_tensor_adam = safety_import('transformer_engine.plugins.cpp_extensions', 'fl_multi_tensor_adam')
### Flash-Attn
# Use lazy=True to avoid circular imports
FLFlashAttention = safety_import(
    'transformer_engine.plugins.attention.dot_product_attention.backends',
    'FLFlashAttention',
    lazy=True
)

def register_fl_backend():
    # Register TE-FL backend
    register_backend("te_fl", {
        "gemm": fl_general_gemm,
        "apply_normalization": fl_apply_normalization,
        "rmsnorm_fwd": fl_rmsnorm_fwd,
        "rmsnorm_bwd": fl_rmsnorm_bwd,
        "adam": fl_multi_tensor_adam,
        "flash_attention": FLFlashAttention,
    })