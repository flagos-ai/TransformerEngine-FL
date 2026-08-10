# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from ....ops import *


def _load_kunlunxin_libs():
    import ctypes
    from pathlib import Path
    import importlib
    import platform

    def get_ext():
        system = platform.system()
        return ".so" if system == "Linux" else ".dylib" if system == "Darwin" else ".dll"

    ext = get_ext()

    try:
        import transformer_engine_klx_torch

        spec = importlib.machinery.PathFinder.find_spec("transformer_engine_klx_torch")
        base_path = Path(spec.origin).parent
        for search_dir in [base_path, base_path / "transformer_engine_klx_torch"]:

            if search_dir.exists():
                matches = list(search_dir.glob(f"transformer_engine*{ext}*"))

                if matches:
                    ctypes.CDLL(str(matches[0]), mode=ctypes.RTLD_GLOBAL)
                    return True

        return False

    except Exception as e:
        return False


_kunlunxin_libs_loaded = False


def _ensure_kunlunxin_available():
    global _kunlunxin_libs_loaded
    if not _kunlunxin_libs_loaded:
        _kunlunxin_libs_loaded = _load_kunlunxin_libs()
        if _kunlunxin_libs_loaded:
            print(f"[KunLunXin] Successfully loaded KunLunXin libs")
    return _kunlunxin_libs_loaded


def _check_kunlunxin_available() -> bool:
    """Check if xpu-smi command can be executed successfully."""
    if _ensure_kunlunxin_available():
        return True
    else:
        return False


def _get_kunlunxin_tex():
    _ensure_kunlunxin_available()
    import transformer_engine_klx_torch

    return transformer_engine_klx_torch


class KunLunXinBackend(TEFLBackendBase):
    @staticmethod
    def check_available() -> bool:
        return _check_kunlunxin_available()

    def __init__(self):
        self._tex = None

    def _get_tex(self):
        if self._tex is None:
            self._tex = _get_kunlunxin_tex()
        return self._tex

    def is_available(self) -> bool:
        return _check_kunlunxin_available()

    def get_flash_attention_class(self):
        from .flash_attention import FlashAttentionTorch

        return FlashAttentionTorch

    def rmsnorm_bwd(
        self,
        dz: torch.Tensor,
        x: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> List[Any]:
        tex = self._get_tex()
        return tex.rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma)

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
        tex = self._get_tex()
        otype = tex.DType(int(otype)) if otype is not None else None
        return tex.layernorm_fwd(
            input,
            weight,
            bias,
            eps,
            ln_out,
            quantizer,
            otype,
            sm_margin,
            zero_centered_gamma,
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
        tex = self._get_tex()
        return tex.layernorm_bwd(dz, x, mu, rsigma, gamma, sm_margin, zero_centered_gamma)

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
        tex = self._get_tex()
        return tex.multi_tensor_adam(
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

    def multi_tensor_scale(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: float,
    ) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale)

    def multi_tensor_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        per_tensor: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.multi_tensor_l2norm(chunk_size, noop_flag, tensor_lists, per_tensor)

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
        tex = self._get_tex()
        otype = tex.DType(int(otype)) if otype is not None else None
        y, rstdevs = tex.rmsnorm_fwd(input, weight, eps, sm_margin, zero_centered_gamma)
        return y, None, rstdevs

    def multi_tensor_adam_fp8(
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
        fp8_dtype: DType,
    ) -> None:
        tex = self._get_tex()
        fp8_dtype = tex.DType(int(fp8_dtype)) if fp8_dtype is not None else None
        return tex.multi_tensor_adam_fp8(
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
            fp8_dtype,
        )

    def multi_tensor_adam_capturable(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        lr: torch.Tensor,
        beta1: float,
        beta2: float,
        epsilon: float,
        step: torch.Tensor,
        mode: int,
        bias_correction: int,
        weight_decay: float,
        inv_scale: torch.Tensor,
    ) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_adam_capturable(
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
            inv_scale,
        )

    def multi_tensor_adam_capturable_master(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        lr: torch.Tensor,
        beta1: float,
        beta2: float,
        epsilon: float,
        step: torch.Tensor,
        mode: int,
        bias_correction: int,
        weight_decay: float,
        inv_scale: torch.Tensor,
    ) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_adam_capturable_master(
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
            inv_scale,
        )

    def cast_to_fp8(
        self,
        input: torch.Tensor,
        scale: torch.Tensor,
        amax: torch.Tensor,
        scale_inv: torch.Tensor,
        otype: int,
        scale_offset: int,
        amax_offset: int,
        scale_inv_offset: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.cast_to_fp8(
            input,
            scale,
            amax,
            scale_inv,
            otype,
            scale_offset,
            amax_offset,
            scale_inv_offset,
        )

    def bulk_overlap_ag_with_external_gemm(
        self,
        allgather_communicator: CommOverlap,
        send_stream: Any,
        recv_stream: Any,
    ) -> Any:
        tex = self._get_tex()
        return tex.bulk_overlap_ag_with_external_gemm(
            allgather_communicator, send_stream, recv_stream
        )

    def get_cudnn_version(self) -> int:
        return 0

    def get_attention_backend(self, attention_params=None):
        tex = self._get_tex()

        (
            use_flash_attention,
            flash_attention_backend,
            use_fused_attention,
            fused_attention_backend,
            use_unfused_attention,
            available_backends,
        ) = tex.get_attention_backend(attention_params)

        return (
            use_flash_attention,
            flash_attention_backend,
            use_fused_attention,
            fused_attention_backend,
            use_unfused_attention,
            available_backends,
        )

    def scaled_masked_softmax_forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        output = torch.empty_like(input)
        torch.ops.custom_ops.softmax_with_mask(input, mask, scale_factor, output=output)
        return output

    def scaled_masked_softmax_backward(
        self,
        output_grad_: torch.Tensor,
        softmax_results_: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        d_input = torch.empty_like(softmax_results_)

        torch.ops.custom_ops.softmax_with_mask_backward(
            output_grad_,
            softmax_results_,
            scale_factor,
            d_input=d_input,
        )
        return d_input

    def multi_tensor_compute_scale_and_scale_inv(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        max_fp8: float,
        force_pow_2_scales: bool,
        epsilon: float,
    ) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_compute_scale_and_scale_inv(
            chunk_size, noop_flag, tensor_lists, max_fp8, force_pow_2_scales, epsilon
        )

    def multi_tensor_compute_scale_inv_e8m0(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        block_len: int,
    ) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_compute_scale_inv_e8m0(
            chunk_size,
            noop_flag,
            tensor_lists,
            block_len,
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
        tex = self._get_tex()
        return tex.generic_gemm(
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
        A: List[torch.Tensor],
        transa: bool,
        B: List[torch.Tensor],
        transb: bool,
        D: Optional[List[torch.Tensor]],
        D_type,
        m_splits: List[int],
        bias: List[torch.Tensor],
        bias_type,
        single_output: bool,
        pre_gelu_out: List[torch.Tensor],
        grad: bool,
        workspace: List[torch.Tensor],
        workspaceSizes: int,
        accumulate: bool,
        use_split_accumulator: bool,
        math_sm_count: int,
    ) -> Optional[List[torch.Tensor]]:
        tex = self._get_tex()
        return tex.te_general_grouped_gemm(
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
