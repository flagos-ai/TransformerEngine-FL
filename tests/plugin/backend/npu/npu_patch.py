# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Runtime patches for running Transformer Engine pytest suites on Ascend NPU."""

from __future__ import annotations

import os


def _set_ascend_env() -> None:
    os.environ.setdefault("PLATFORM", "ascend")
    os.environ.setdefault("TE_FL_SKIP_CUDA", "1")
    os.environ.setdefault("NVTE_FRAMEWORK", "pytorch")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def _unsupported(reason: str):
    return False, reason


def apply_ascend_npu_patch() -> None:
    """Configure TE and patch CUDA-only helpers for Ascend test execution."""
    _set_ascend_env()

    import torch

    try:
        import torch_npu
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"torch_npu is required for Ascend tests: {exc}") from exc

    # Translate CUDA-oriented shared tests to the equivalent Torch-NPU APIs.
    import torch_npu.contrib.transfer_to_npu  # noqa: F401

    import transformer_engine

    transformer_engine.TE_DEVICE_TYPE = "npu"
    transformer_engine.TE_PLATFORM = torch_npu.npu

    # Some TE PyTorch paths query CUDA graph state unconditionally.
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_capability = lambda device=None: (0, 0)
    torch.cuda.is_current_stream_capturing = lambda: False

    _patch_quantization_capability_checks()
    _patch_te_gemm_workspace()


def _patch_quantization_capability_checks() -> None:
    import transformer_engine.pytorch.module.layernorm_mlp as layernorm_mlp
    import transformer_engine.pytorch.quantization as quantization
    import transformer_engine.pytorch.utils as pytorch_utils

    # Torch-NPU has no CUDA compute capability. Shared gates should select
    # their non-FP8 path instead of attempting to inspect CUDA properties.
    pytorch_utils._get_device_compute_capability = lambda device: (0, 0)

    # LayerNormMLP constructs its activation table eagerly. Filter out
    # operators such as glu/dglu that are not registered by FlagOS so they do
    # not block supported GELU, ReLU, SiLU, and gated activation paths.
    def _npu_activation_table(recipe=None):
        candidates = {
            "gelu": ("gelu", "dgelu", "dbias_dgelu"),
            "geglu": ("geglu", "dgeglu", None),
            "glu": ("glu", "dglu", None),
            "qgelu": ("qgelu", "dqgelu", "dbias_dqgelu"),
            "qgeglu": ("qgeglu", "dqgeglu", None),
            "relu": ("relu", "drelu", "dbias_drelu"),
            "reglu": ("reglu", "dreglu", None),
            "srelu": ("srelu", "dsrelu", "dbias_dsrelu"),
            "sreglu": ("sreglu", "dsreglu", None),
            "silu": ("silu", "dsilu", "dbias_dsilu"),
            "swiglu": ("swiglu", "dswiglu", None),
            "clamped_swiglu": ("clamped_swiglu", "clamped_dswiglu", None),
        }
        delayed = recipe is not None and (recipe.delayed() or recipe.mxfp8())
        table = {}
        for activation, (forward_name, backward_name, dbias_name) in candidates.items():
            try:
                forward = getattr(layernorm_mlp.tex, forward_name)
                backward = getattr(layernorm_mlp.tex, backward_name)
                dbias = getattr(layernorm_mlp.tex, dbias_name) if delayed and dbias_name else None
            except AttributeError:
                continue
            table[activation] = (forward, backward, dbias)
        return table

    layernorm_mlp._get_act_func_supported_list = _npu_activation_table

    quantization.check_fp8_support = lambda: _unsupported("FP8 execution is not supported on npu.")
    quantization.check_mxfp8_support = lambda: _unsupported(
        "MXFP8 execution is not supported on npu."
    )
    quantization.check_nvfp4_support = lambda: _unsupported(
        "NVFP4 execution is not supported on npu."
    )
    quantization.check_fp8_block_scaling_support = lambda: _unsupported(
        "FP8 block scaling is not supported on npu."
    )


def _patch_te_gemm_workspace() -> None:
    import torch
    import transformer_engine.pytorch.cpp_extensions.gemm as gemm

    def _npu_workspace(device, ub, grouped_gemm):
        device_index = torch.npu.current_device() if device is None else int(device)
        npu_device = torch.device("npu", device_index)
        workspace_size = 4_194_304

        if ub:
            return torch.empty(workspace_size * 3, dtype=torch.uint8, device=npu_device)
        if grouped_gemm:
            return [torch.empty(workspace_size, dtype=torch.uint8, device=npu_device)]
        return torch.empty(workspace_size, dtype=torch.uint8, device=npu_device)

    cache_clear = getattr(gemm.get_cublas_workspace, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    gemm.get_cublas_workspace = _npu_workspace
