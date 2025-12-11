# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Stub module for transformer_engine_torch when CUDA extensions are not available.

This module provides mock implementations of the transformer_engine_torch (tex)
module interfaces to allow importing transformer_engine.pytorch modules in
FL-only mode (TE_FL_SKIP_CUDA_BUILD=1).

Note: The actual FP8 and CUDA-specific operations are NOT supported in this mode.
Only basic type definitions and fallback behaviors are provided.
"""

import warnings
from enum import IntEnum
from typing import Any, List, Optional
import torch


# =============================================================================
# Enum Types
# =============================================================================

class DType(IntEnum):
    """Mock TE DType enum matching transformer_engine_torch.DType"""
    kByte = 0
    kInt32 = 1
    kFloat32 = 2
    kFloat16 = 3
    kBFloat16 = 4
    kFloat8E4M3 = 5
    kFloat8E5M2 = 6
    kFloat4E2M1 = 7


class Float8BlockScaleTensorFormat(IntEnum):
    """Mock Float8BlockScaleTensorFormat enum"""
    COMPACT = 0
    GEMM_READY = 1


class NVTE_Bias_Type(IntEnum):
    """Mock NVTE_Bias_Type enum"""
    NVTE_NO_BIAS = 0
    NVTE_PRE_SCALE_BIAS = 1
    NVTE_POST_SCALE_BIAS = 2
    NVTE_ALIBI = 3


class NVTE_Activation_Type(IntEnum):
    """Mock NVTE_Activation_Type enum"""
    NVTE_GELU = 0
    NVTE_GEGLU = 1
    NVTE_SILU = 2
    NVTE_SWIGLU = 3
    NVTE_RELU = 4
    NVTE_REGLU = 5
    NVTE_QGELU = 6
    NVTE_QGEGLU = 7
    NVTE_SRELU = 8
    NVTE_SREGLU = 9


class NVTE_QKV_Layout(IntEnum):
    """Mock NVTE_QKV_Layout enum - comprehensive list"""
    NVTE_SB3HD = 0
    NVTE_SBH3D = 1
    NVTE_SBHD_SB2HD = 2
    NVTE_SBHD_SBH2D = 3
    NVTE_SBHD_SBHD_SBHD = 4
    NVTE_BS3HD = 5
    NVTE_BSH3D = 6
    NVTE_BSHD_BS2HD = 7
    NVTE_BSHD_BSH2D = 8
    NVTE_BSHD_BSHD_BSHD = 9
    NVTE_T3HD = 10
    NVTE_TH3D = 11
    NVTE_THD_T2HD = 12
    NVTE_THD_TH2D = 13
    NVTE_THD_THD_THD = 14
    # Cross-format layouts
    NVTE_SBHD_BSHD_BSHD = 15
    NVTE_BSHD_SBHD_SBHD = 16
    NVTE_THD_BSHD_BSHD = 17
    NVTE_THD_SBHD_SBHD = 18
    # Paged KV layouts
    NVTE_Paged_KV_BSHD_BSHD_BSHD = 19
    NVTE_Paged_KV_BSHD_SBHD_SBHD = 20
    NVTE_Paged_KV_SBHD_BSHD_BSHD = 21
    NVTE_Paged_KV_SBHD_SBHD_SBHD = 22
    NVTE_Paged_KV_THD_BSHD_BSHD = 23
    NVTE_Paged_KV_THD_SBHD_SBHD = 24


class NVTE_QKV_Format(IntEnum):
    """Mock NVTE_QKV_Format enum"""
    NVTE_SBHD = 0
    NVTE_BSHD = 1
    NVTE_THD = 2
    NVTE_SBHD_2BSHD = 3
    NVTE_BSHD_2SBHD = 4
    NVTE_THD_2BSHD = 5
    NVTE_THD_2SBHD = 6


class NVTE_Mask_Type(IntEnum):
    """Mock NVTE_Mask_Type enum"""
    NVTE_NO_MASK = 0
    NVTE_PADDING_MASK = 1
    NVTE_CAUSAL_MASK = 2
    NVTE_PADDING_CAUSAL_MASK = 3
    NVTE_CAUSAL_BOTTOM_RIGHT_MASK = 4
    NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK = 5
    NVTE_ARBITRARY_MASK = 6


class NVTE_Softmax_Type(IntEnum):
    """Mock NVTE_Softmax_Type enum"""
    NVTE_SOFTMAX = 0
    NVTE_SOFTMAX_LOG = 1
    NVTE_VANILLA_SOFTMAX = 2
    NVTE_OFF_BY_ONE_SOFTMAX = 3
    NVTE_LEARNABLE_SOFTMAX = 4


class NVTE_Fused_Attn_Backend(IntEnum):
    """Mock NVTE_Fused_Attn_Backend enum"""
    NVTE_No_Backend = 0
    NVTE_F16_max512_seqlen = 1
    NVTE_F16_arbitrary_seqlen = 2
    NVTE_FP8 = 3
    NVTE_FA3 = 4


class CommGemmOverlapRole(IntEnum):
    """Mock CommGemmOverlapRole enum"""
    INPUT = 0
    OUTPUT = 1


class CommOverlapType(IntEnum):
    """Mock CommOverlapType enum"""
    DEFAULT = 0
    SPLIT_PIPELINED_AG = 1
    SPLIT_PIPELINED_RS = 2
    SPLIT_PIPELINED_AG_P2P = 3
    SPLIT_PIPELINED_RS_P2P = 4
    ATOMIC_GEMM_RS = 5
    ATOMIC_GEMM_AG = 6
    BULK_OVERLAP_RS = 7
    BULK_OVERLAP_AG = 8


class FP8FwdTensors(IntEnum):
    """Mock FP8FwdTensors enum for FP8 forward tensors"""
    GEMM1_INPUT = 0
    GEMM1_WEIGHT = 1
    GEMM1_OUTPUT = 2
    GEMM2_INPUT = 3
    GEMM2_WEIGHT = 4
    GEMM2_OUTPUT = 5
    GEMM3_INPUT = 6
    GEMM3_WEIGHT = 7
    GEMM3_OUTPUT = 8


class FP8BwdTensors(IntEnum):
    """Mock FP8BwdTensors enum for FP8 backward tensors"""
    GRAD_OUTPUT1 = 0
    GRAD_INPUT1 = 1
    GRAD_OUTPUT2 = 2
    GRAD_INPUT2 = 3
    GRAD_OUTPUT3 = 4
    GRAD_INPUT3 = 5


# =============================================================================
# Classes
# =============================================================================

class CommGemmOverlapAlgoConfig:
    """Mock CommGemmOverlapAlgoConfig class"""
    def __init__(self, *args, **kwargs):
        pass


class FP8TensorMeta:
    """Mock FP8TensorMeta class"""
    def __init__(self, *args, **kwargs):
        self.scale = None
        self.scale_inv = None
        self.amax_history = None


class FusedAdamCUDAKernel:
    """Mock FusedAdamCUDAKernel class"""
    def __init__(self, *args, **kwargs):
        _not_supported()


class FusedSGDCUDAKernel:
    """Mock FusedSGDCUDAKernel class"""
    def __init__(self, *args, **kwargs):
        _not_supported()


class CommOverlap:
    """Mock CommOverlap class for userbuffers communication overlap"""
    def __init__(self, *args, **kwargs):
        pass


class CommOverlapP2P:
    """Mock CommOverlapP2P class for P2P communication overlap"""
    def __init__(self, *args, **kwargs):
        pass


# =============================================================================
# Error function for unsupported operations
# =============================================================================

def _not_supported(*args, **kwargs):
    """Raise error for unsupported operations."""
    raise RuntimeError(
        "This operation requires CUDA extensions (transformer_engine_torch) which are not available. "
        "Please rebuild Transformer Engine with CUDA support, or avoid using FP8/CUDA-specific features "
        "when running in FL-only mode (TE_FL_SKIP_CUDA_BUILD=1)."
    )


# =============================================================================
# Version and Info Functions
# =============================================================================

def get_cublasLt_version() -> int:
    """Return a version that indicates no cuBLAS support."""
    return 0


def get_cudnn_version() -> int:
    """Return a version that indicates no cuDNN support."""
    return 0


def get_fused_attn_backend(*args, **kwargs):
    """Return no backend available for fused attention."""
    return NVTE_Fused_Attn_Backend.NVTE_No_Backend


def get_num_cublas_streams() -> int:
    """Return number of cuBLAS streams (0 for stub)."""
    return 0


def get_stream_priority_range():
    """Return stream priority range."""
    return (0, 0)


# =============================================================================
# Stub Functions - All raise RuntimeError when called
# =============================================================================

# GEMM operations
gemm = _not_supported
fp8_gemm = _not_supported
te_grouped_gemm = _not_supported
te_grouped_gemm_single_output = _not_supported

# Cast operations
cast_to_fp8 = _not_supported
cast_from_fp8 = _not_supported
fused_cast_transpose = _not_supported
fused_cast_transpose_bgrad = _not_supported

# Normalization operations
layernorm_fwd = _not_supported
layernorm_bwd = _not_supported
rmsnorm_fwd = _not_supported
rmsnorm_bwd = _not_supported

# Attention operations
fused_attn_fwd = _not_supported
fused_attn_bwd = _not_supported

# MoE operations
moe_permute_fwd = _not_supported
moe_permute_bwd = _not_supported
moe_unpermute_fwd = _not_supported
moe_unpermute_bwd = _not_supported
fused_topk_with_score_function_fwd = _not_supported
fused_topk_with_score_function_bwd = _not_supported
fused_score_for_moe_aux_loss_fwd = _not_supported
fused_score_for_moe_aux_loss_bwd = _not_supported
fused_moe_aux_loss_fwd = _not_supported
fused_moe_aux_loss_bwd = _not_supported

# Scale and amax operations
fused_amax_and_scale_update_after_reduction = _not_supported
compute_amax = _not_supported
fp8_block_scaling_compute_partial_amax = _not_supported
fp8_block_scaling_partial_cast = _not_supported

# Multi-tensor operations
multi_tensor_scale = _not_supported
multi_tensor_compute_scale_and_scale_inv = _not_supported
multi_tensor_unscale_l2norm = _not_supported
multi_tensor_adam = _not_supported
multi_tensor_adam_fp8 = _not_supported
multi_tensor_adam_capturable = _not_supported
multi_tensor_adam_capturable_master = _not_supported
multi_tensor_sgd = _not_supported
multi_tensor_l2norm = _not_supported

# Misc operations
swap_first_dims = _not_supported
bulk_overlap_ag_with_external_gemm = _not_supported

# Activation operations
gelu = _not_supported
relu = _not_supported
geglu = _not_supported
reglu = _not_supported
swiglu = _not_supported
qgelu = _not_supported
srelu = _not_supported
dgelu = _not_supported
drelu = _not_supported
dgeglu = _not_supported
dreglu = _not_supported
dswiglu = _not_supported
dqgelu = _not_supported
dsrelu = _not_supported
gelu_fp8 = _not_supported
relu_fp8 = _not_supported
geglu_fp8 = _not_supported
reglu_fp8 = _not_supported
swiglu_fp8 = _not_supported
qgelu_fp8 = _not_supported
srelu_fp8 = _not_supported
dgelu_fp8 = _not_supported
drelu_fp8 = _not_supported
dgeglu_fp8 = _not_supported
dreglu_fp8 = _not_supported
dswiglu_fp8 = _not_supported
dqgelu_fp8 = _not_supported
dsrelu_fp8 = _not_supported

# Fused operations
fused_multi_cast_transpose = _not_supported
fused_multi_quantize = _not_supported
fused_fp8_transpose = _not_supported
fused_multi_row_reduction = _not_supported
fused_layernorm_cast_transpose = _not_supported
fused_rmsnorm_cast_transpose = _not_supported

# Userbuffers
userbuffers_init = _not_supported
userbuffers_create_communicator = _not_supported
userbuffers_destroy_communicator = _not_supported

# CUDNN operations
cudnn_fused_dgrad_dbias_gelu = _not_supported
cudnn_fused_dgrad_dbias_relu = _not_supported

# Bias + activation derivatives
dbias_dgelu = _not_supported
dbias_drelu = _not_supported
dbias_dswiglu = _not_supported
dbias_dgeglu = _not_supported
dbias_dreglu = _not_supported
dbias_dsrelu = _not_supported
dbias_dqgelu = _not_supported

# Quantization operations
quantize = _not_supported
dequantize = _not_supported

# RoPE operations
fused_rope_forward = _not_supported
fused_rope_backward = _not_supported
fused_qkv_rope_forward = _not_supported
fused_qkv_rope_backward = _not_supported

# KV cache operations
copy_to_kv_cache = _not_supported

# Multi-tensor adam remainder
multi_tensor_adam_param_remainder = _not_supported

# LayerNorm class placeholder
class LayerNorm:
    """Mock LayerNorm class"""
    def __init__(self, *args, **kwargs):
        _not_supported()

# =============================================================================
# Define __all__ to control what gets exported with "from module import *"
# =============================================================================

__all__ = [
    # Enums
    "DType",
    "Float8BlockScaleTensorFormat",
    "NVTE_Bias_Type",
    "NVTE_Activation_Type",
    "NVTE_QKV_Layout",
    "NVTE_QKV_Format",
    "NVTE_Mask_Type",
    "NVTE_Softmax_Type",
    "NVTE_Fused_Attn_Backend",
    "CommGemmOverlapRole",
    "CommOverlapType",
    "FP8FwdTensors",
    "FP8BwdTensors",
    # Classes
    "CommGemmOverlapAlgoConfig",
    "FP8TensorMeta",
    "FusedAdamCUDAKernel",
    "FusedSGDCUDAKernel",
    "CommOverlap",
    "CommOverlapP2P",
    # Functions
    "get_cublasLt_version",
    "get_cudnn_version",
    "gemm",
    "fp8_gemm",
    "cast_to_fp8",
    "cast_from_fp8",
    "layernorm_fwd",
    "layernorm_bwd",
    "rmsnorm_fwd",
    "rmsnorm_bwd",
    "fused_attn_fwd",
    "fused_attn_bwd",
    "multi_tensor_scale",
    "multi_tensor_compute_scale_and_scale_inv",
    "multi_tensor_unscale_l2norm",
    "multi_tensor_adam",
    "multi_tensor_adam_fp8",
    "multi_tensor_adam_capturable",
    "multi_tensor_adam_capturable_master",
    "multi_tensor_sgd",
    "multi_tensor_l2norm",
]
