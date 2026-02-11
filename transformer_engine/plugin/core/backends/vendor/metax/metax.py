# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Any, Dict, List, Optional, Tuple, Union

import ctypes
from pathlib import Path
import importlib.util
import platform
import os
import functools
import inspect

import torch

from ....ops import *

def _load_metax_libs():

    def get_ext():
        system = platform.system()
        return ".so" if system == "Linux" else ".dylib" if system == "Darwin" else ".dll"

    ext = get_ext()

    try:
        import transformer_engine_metax
        te_path = Path(importlib.util.find_spec("transformer_engine_metax").origin).parent.parent
        for search_dir in [te_path, te_path / "transformer_engine_metax"]:
            if search_dir.exists():
                matches = list(search_dir.glob(f"libtransformer_engine{ext}*"))
                if matches:
                    ctypes.CDLL(str(matches[0]), mode=ctypes.RTLD_GLOBAL)
                    return True
        return False
    except Exception as e:
        print(f"[Metax] Failed to load Metax libs: {e}")
        return False

_metax_libs_loaded = False

def _ensure_metax_libs():
    global _metax_libs_loaded
    if not _metax_libs_loaded:
        _metax_libs_loaded = _load_metax_libs()
    return _metax_libs_loaded

def _check_metax_available() -> bool:
    if not torch.cuda.is_available():
        return False

    try:
        from ...._build_config import SKIP_METAX_BUILD
        if SKIP_METAX_BUILD:
            print("[Metax] Disabled: Metax was skipped at build time")
            return False
    except ImportError:
        if bool(int(os.environ.get("TE_FL_SKIP_METAX", "0"))):
            print("[Metax] Disabled: TE_FL_SKIP_METAX=1")
            return False

    try:
        if not _ensure_metax_libs():
            return False
        import transformer_engine_torch_metax
        return True
    except (ImportError, OSError) as e:
        print(f"[Metax] Import failed: {e}")
        return False

def _get_tex():
    _ensure_metax_libs()
    import transformer_engine_torch_metax
    return transformer_engine_torch_metax

class MetaxBackend(TEFLBackendBase):
    @staticmethod
    def check_available() -> bool:
        return _check_metax_available()

    def __init__(self):
        self._tex = None

    def _get_tex(self):
        if self._tex is None:
            self._tex = _get_tex()
        return self._tex

    def is_available(self) -> bool:
        return _check_metax_available()

    def get_attention_backend(self, attention_params=None):
        # Import the metax get_attention_backend function
        try:
            from transformer_engine_metax.pytorch.attention.dot_product_attention import utils
            return utils.get_attention_backend(attention_params)

        except ImportError as e:
            raise RuntimeError(
                f"Failed to import metax FlashAttention: {e}. "
                "Please ensure flash-attn is installed and transformer_engine_metax is available."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get_attention_backend: {e}. "
                f"Attention_params: {self.attention_params}"
            )

##### transformer_engine/pytorch/csrc/extensions/pybind.cpp #####
    def quantize(
        self,
        tensor: torch.Tensor,
        quantizer: Any,
        output: Optional[torch.Tensor] = None,
        noop: Optional[torch.Tensor] = None,
    ) -> Any:
        tex = self._get_tex()
        return tex.quantize(tensor, quantizer, output, noop)

    def dequantize(
        self,
        input: Any,
        otype: DType,
    ) -> Any:
        tex = self._get_tex()
        otype = tex.DType(int(otype)) if otype is not None else None
        return tex.dequantize(input, otype)

    def bgrad_quantize(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> List[Any]:
        tex = self._get_tex()
        return tex.bgrad_quantize(input, quantizer)

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
        
        bias_type = tex.DType(int(bias_type)) if bias_type is not None else None
        comm_type = tex.CommOverlapType(int(comm_type)) if comm_type is not None else None
        output_dtype = tex.DType(int(output_dtype)) if output_dtype is not None else None
        return tex.generic_gemm(
            A, transA, B, transB, D, quantizer, output_dtype,
            bias, bias_type, gelu, gelu_in, grad, workspace, workspace_size,
            accumulate, use_split_accumulator, comm_overlap, comm_type,
            extra_output, bulk_overlap, alpha, beta
        )
    # GELU and variants #
    def gelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.gelu(input, quantizer)
    def geglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.geglu(input, quantizer)
    def qgelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.qgelu(input, quantizer)
    def qgeglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.qgeglu(input, quantizer)
    # ReLU and variants #
    def relu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.relu(input, quantizer)
    def reglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.reglu(input, quantizer)
    def srelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.srelu(input, quantizer)
    def sreglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.sreglu(input, quantizer)
    # SwiGLU and variants #
    def silu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.silu(input, quantizer)
    def swiglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.swiglu(input, quantizer)
    def clamped_swiglu(
        self,
        input: torch.Tensor,
        quantizer: Any,
        limit: float = 7.0,
        alpha: float = 1.702,
    ) -> Any:
        tex = self._get_tex()
        return tex.clamped_swiglu(input, quantizer, limit, alpha)
    # Backward of GELU and variants #
    def dgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dgelu(grad, fwd_input, quantizer)
    def dgeglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dgeglu(grad, fwd_input, quantizer)
    def dqgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dqgelu(grad, fwd_input, quantizer)
    def dqgeglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dqgeglu(grad, fwd_input, quantizer)
    # Backward of ReLU and variants #
    def drelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.drelu(grad, fwd_input, quantizer)
    def dreglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dreglu(grad, fwd_input, quantizer)
    def dsrelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dsrelu(grad, fwd_input, quantizer)
    def dsreglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dsreglu(grad, fwd_input, quantizer)
    # Backward of SiLU and variants #
    def dsilu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dsilu(grad, fwd_input, quantizer)
    def dswiglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dswiglu(grad, fwd_input, quantizer)
    def clamped_dswiglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
        limit: float = 7.0,
        alpha: float = 1.702,
    ) -> Any:
        tex = self._get_tex()
        return tex.clamped_dswiglu(grad, fwd_input, quantizer, limit, alpha)
    # DBias + DAct fusions #
    def dbias_dgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> List[Any]:
        tex = self._get_tex()
        return tex.dbias_dgelu(grad, fwd_input, quantizer)
    def dbias_dsilu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> List[Any]:
        tex = self._get_tex()
        return tex.dbias_dsilu(grad, fwd_input, quantizer)
    def dbias_drelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> List[Any]:
        tex = self._get_tex()
        return tex.dbias_drelu(grad, fwd_input, quantizer)
    def dbias_dqgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> List[Any]:
        tex = self._get_tex()
        return tex.dbias_dqgelu(grad, fwd_input, quantizer)
    def dbias_dsrelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> List[Any]:
        tex = self._get_tex()
        return tex.dbias_dsrelu(grad, fwd_input, quantizer)
    # Permutation functions                                                                                                                                                                                                                                                                                             
    def moe_permute_fwd(
        self,
        input: torch.Tensor,
        dtype: DType,
        indices: torch.Tensor,
        num_out_tokens: int,
        workspace: List[torch.Tensor],
        max_expanded_token_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        tex = self._get_tex()
        dtype = tex.DType(int(dtype)) if dtype is not None else None
        return tex.moe_permute_fwd(input, dtype,indices,num_out_tokens,workspace,max_expanded_token_num)
    def moe_permute_bwd(
        self,
        input: torch.Tensor,
        dtype: DType,
        row_id_map: torch.Tensor,
        prob: torch.Tensor,
        num_tokens: int,
        topK: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        dtype = tex.DType(int(dtype)) if dtype is not None else None
        return tex.moe_permute_bwd(input,dtype,row_id_map,prob,num_tokens,topK)
    def moe_unpermute_fwd(
        self,
        input: torch.Tensor,
        dtype: DType,
        row_id_map: torch.Tensor,
        prob: torch.Tensor,
        num_tokens: int,
        topK: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        dtype = tex.DType(int(dtype)) if dtype is not None else None
        return tex.moe_unpermute_fwd(input,dtype,row_id_map,prob,num_tokens,topK)
    def moe_unpermute_bwd(
        self,
        input_bwd: torch.Tensor,
        input_fwd: torch.Tensor,
        dtype: DType,
        row_id_map: torch.Tensor,
        prob: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        dtype = tex.DType(int(dtype)) if dtype is not None else None
        return tex.moe_unpermute_bwd(input_bwd,input_fwd,dtype,row_id_map,prob)
    # Softmax functions
    def scaled_softmax_forward(
        self,
        input: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_softmax_forward(input, scale)
    def scaled_softmax_backward(
        self,
        output_grad_: torch.Tensor,
        softmax_results_: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_softmax_backward(output_grad_, softmax_results_, scale_factor)
    def scaled_masked_softmax_forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_masked_softmax_forward(input, mask, scale_factor)
    def scaled_masked_softmax_backward(
        self,
        output_grad_: torch.Tensor,
        softmax_results_: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_masked_softmax_backward(output_grad_, softmax_results_, scale_factor)
    def scaled_upper_triang_masked_softmax_forward(
        self,
        input: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_upper_triang_masked_softmax_forward(input, scale_factor)
    def scaled_upper_triang_masked_softmax_backward(
        self,
        output_grads_: torch.Tensor,
        softmax_results_: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_upper_triang_masked_softmax_backward(
            output_grads_, softmax_results_, scale_factor
        )
    def scaled_aligned_causal_masked_softmax_forward(
        self,
        input: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_aligned_causal_masked_softmax_forward(input, scale_factor)
    def scaled_aligned_causal_masked_softmax_backward(
        self,
        output_grad_: torch.Tensor,
        softmax_results_: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_aligned_causal_masked_softmax_backward(
            output_grad_, softmax_results_, scale_factor
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
        tex = self._get_tex()
        otype = tex.DType(int(otype)) if otype is not None else None
        return tex.layernorm_fwd(
            input, weight, bias, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma
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
        return tex.layernorm_bwd(
            dz, x, mu, rsigma, gamma, sm_margin, zero_centered_gamma
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
        tex = self._get_tex()
        otype = tex.DType(int(otype)) if otype is not None else None
        return tex.rmsnorm_fwd(
            input, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma
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
        tex = self._get_tex()
        return tex.rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma)
    def rmsnorm_bwd_add(
        self,
        dz: torch.Tensor,
        x: torch.Tensor,
        add: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> List[Any]:
        tex = self._get_tex()
        return tex.rmsnorm_bwd_add(dz, x, add, rsigma, gamma, sm_margin, zero_centered_gamma)

    def multi_tensor_quantize(
        self,
        tensor_list: List[torch.Tensor],
        quantizer_list: List[Any],
    ) -> List[Any]:
        tex = self._get_tex()
        return tex.multi_tensor_quantize(tensor_list, quantizer_list)
    def split_quantize(
        self,
        tensor: torch.Tensor,
        split_sections: List[int],
        quantizer_list: List[Any],
    ) -> List[Any]:
        tex = self._get_tex()
        return tex.split_quantize(tensor, split_sections, quantizer_list)
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
        tex = self._get_tex()
        D_type = tex.DType(int(D_type)) if D_type is not None else None
        bias_type = tex.DType(int(bias_type)) if bias_type is not None else None
        return tex.te_general_grouped_gemm(
            A, transa, B, transb, D, D_type, m_splits, bias, bias_type,
            single_output, pre_gelu_out, grad, workspace, workspaceSizes,
            accumulate, use_split_accumulator, math_sm_count
        )
    def fp8_transpose(
        self,
        input: torch.Tensor,
        dtype: DType,
        out: Optional[torch.Tensor],
    ) -> torch.Tensor:
        tex = self._get_tex()
        dtype = tex.DType(int(dtype)) if dtype is not None else None
        return tex.fp8_transpose(input, dtype, out)
    def swap_first_dims(
        self,
        tensor: torch.Tensor,
        out: Optional[torch.Tensor],
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.swap_first_dims(tensor, out)
    def get_fused_attn_backend(
        self,
        is_training: bool,
        q_dtype: DType,
        kv_dtype: DType,
        qkv_layout: NVTE_QKV_Layout,
        bias_type: NVTE_Bias_Type,
        attn_mask_type: NVTE_Mask_Type,
        softmax_type: NVTE_Softmax_Type,
        p_dropout: float,
        num_attn_heads: int,
        num_gqa_groups: int,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        head_dim_qk: int,
        head_dim_v: int,
        window_size_left: int,
        window_size_right: int,
        return_max_logit: bool,
    ) -> NVTE_Fused_Attn_Backend:
        tex = self._get_tex()

        q_dtype = tex.DType(int(q_dtype)) if q_dtype is not None else None
        kv_dtype = tex.DType(int(kv_dtype)) if kv_dtype is not None else None
        qkv_layout = tex.NVTE_QKV_Layout(int(qkv_layout)) if qkv_layout is not None else None
        bias_type = tex.NVTE_Bias_Type(int(bias_type)) if bias_type is not None else None
        attn_mask_type = tex.NVTE_Mask_Type(int(attn_mask_type)) if attn_mask_type is not None else None
        softmax_type = tex.NVTE_Softmax_Type(int(softmax_type)) if softmax_type is not None else None

        result = tex.get_fused_attn_backend(
            is_training, q_dtype, kv_dtype, qkv_layout, bias_type,
            attn_mask_type, softmax_type, p_dropout, num_attn_heads,
            num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim_qk,
            head_dim_v, window_size_left, window_size_right, return_max_logit
        )
        return NVTE_Fused_Attn_Backend(result)

    def compute_amax(
        self,
        input: torch.Tensor,
        amax: torch.Tensor,
    ) -> None:
        tex = self._get_tex()
        return tex.compute_amax(input, amax)
    def fused_amax_and_scale_update_after_reduction(
        self,
        amax_reduction_buffer: torch.Tensor,
        amax_histories: List[torch.Tensor],
        scales: List[torch.Tensor],
        amax_compute_algo: str,
        fp8_dtype: DType,
        margin: float,
    ) -> None:
        tex = self._get_tex()
        fp8_dtype = tex.DType(int(fp8_dtype)) if fp8_dtype is not None else None
        return tex.fused_amax_and_scale_update_after_reduction(
            amax_reduction_buffer, amax_histories, scales,
            amax_compute_algo, fp8_dtype, margin
        )
    def fp8_block_scaling_compute_partial_amax(
        self,
        tensor: torch.Tensor,
        amax: torch.Tensor,
        h: int,
        w: int,
        start_offset: int,
        block_len: int,
    ) -> None:
        tex = self._get_tex()
        return tex.fp8_block_scaling_compute_partial_amax(
            tensor, amax, h, w, start_offset, block_len
        )
    def fp8_block_scaling_partial_cast(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        scale: torch.Tensor,
        h: int,
        w: int,
        start_offset: int,
        block_len: int,
        out_dtype: DType,
    ) -> None:
        tex = self._get_tex()
        out_dtype = tex.DType(int(out_dtype)) if out_dtype is not None else None
        return tex.fp8_block_scaling_partial_cast(
            inp, out, scale, h, w, start_offset, block_len, out_dtype
        )
    def fused_multi_row_padding(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        input_row_list: List[int],
        padded_input_row_list: List[int],
    ) -> None:
        tex = self._get_tex()
        return tex.fused_multi_row_padding(
            input, output, input_row_list, padded_input_row_list
        )
    def fused_multi_row_unpadding(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        input_row_list: List[int],
        unpadded_input_row_list: List[int],
    ) -> None:
        tex = self._get_tex()
        return tex.fused_multi_row_unpadding(
            input, output, input_row_list, unpadded_input_row_list
        )

    # attention kernels
    def fa_prepare_fwd(
        self,
        qkvi: torch.Tensor,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.fa_prepare_fwd(qkvi)
    def fa_prepare_bwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.fa_prepare_bwd(q, k, v)
    def fused_attn_fwd(
        self,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        is_training: bool,
        attn_scale: float,
        p_dropout: float,
        set_zero: bool,
        qkv_layout: NVTE_QKV_Layout,
        bias_type: NVTE_Bias_Type,
        attn_mask_type: NVTE_Mask_Type,
        softmax_type: NVTE_Softmax_Type,
        window_size: List[int],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        Q: Any,
        K: Any,
        V: Any,
        fake_dtype: torch.dtype,
        cu_seqlens_q_padded: Optional[torch.Tensor],
        cu_seqlens_kv_padded: Optional[torch.Tensor],
        page_table_k: Optional[torch.Tensor],
        page_table_v: Optional[torch.Tensor],
        s_quantizer: Any,
        o_quantizer: Any,
        Bias: Optional[torch.Tensor],
        SoftmaxOffset: Optional[torch.Tensor],
        rng_gen: Optional[torch.Generator],
        rng_elts_per_thread: int,
        return_max_logit: bool,
    ) -> List[Any]:
        tex = self._get_tex()

        qkv_layout = tex.NVTE_QKV_Layout(int(qkv_layout)) if qkv_layout is not None else None
        bias_type = tex.NVTE_Bias_Type(int(bias_type)) if bias_type is not None else None
        attn_mask_type = tex.NVTE_Mask_Type(int(attn_mask_type)) if attn_mask_type is not None else None
        softmax_type = tex.NVTE_Softmax_Type(int(softmax_type)) if softmax_type is not None else None

        return tex.fused_attn_fwd(
            max_seqlen_q,
            max_seqlen_kv,
            is_training,
            attn_scale,
            p_dropout,
            set_zero,
            qkv_layout,
            bias_type,
            attn_mask_type,
            softmax_type,
            window_size,
            cu_seqlens_q,
            cu_seqlens_kv,
            Q,
            K,
            V,
            fake_dtype,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            page_table_k,
            page_table_v,
            s_quantizer,
            o_quantizer,
            Bias,
            SoftmaxOffset,
            rng_gen,
            rng_elts_per_thread,
            return_max_logit
        )
    def fused_attn_bwd(
        self,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        attn_scale: float,
        p_dropout: float,
        set_zero: bool,
        qkv_layout: NVTE_QKV_Layout,
        bias_type: NVTE_Bias_Type,
        attn_mask_type: NVTE_Mask_Type,
        softmax_type: NVTE_Softmax_Type,
        window_size: List[int],
        deterministic: bool,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        Q: Any,
        K: Any,
        V: Any,
        O: Any,
        dO: Any,
        fake_dtype: torch.dtype,
        dqkv_type: DType,
        Aux_CTX_Tensors: List[torch.Tensor],
        cu_seqlens_q_padded: Optional[torch.Tensor],
        cu_seqlens_kv_padded: Optional[torch.Tensor],
        s_quantizer: Any,
        dp_quantizer: Any,
        dqkv_quantizer: Any,
    ) -> List[Any]:
        tex = self._get_tex()

        qkv_layout = tex.NVTE_QKV_Layout(int(qkv_layout)) if qkv_layout is not None else None
        bias_type = tex.NVTE_Bias_Type(int(bias_type)) if bias_type is not None else None
        attn_mask_type = tex.NVTE_Mask_Type(int(attn_mask_type)) if attn_mask_type is not None else None
        softmax_type = tex.NVTE_Softmax_Type(int(softmax_type)) if softmax_type is not None else None
        dqkv_type = tex.DType(int(dqkv_type)) if dqkv_type is not None else None

        return tex.fused_attn_bwd(
            max_seqlen_q,
            max_seqlen_kv,
            attn_scale,
            p_dropout,
            set_zero,
            qkv_layout,
            bias_type,
            attn_mask_type,
            softmax_type,
            window_size,
            deterministic,
            cu_seqlens_q,
            cu_seqlens_kv,
            Q,
            K,
            V,
            O,
            dO,
            fake_dtype,
            dqkv_type,
            Aux_CTX_Tensors,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            s_quantizer,
            dp_quantizer,
            dqkv_quantizer
        )
    def copy_to_kv_cache(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cu_new_lens: torch.Tensor,
        cu_cached_lens: torch.Tensor,
        qkv_format: NVTE_QKV_Format,
        b: int,
        max_ctx_len: int,
        max_seq_len: int,
        max_pages_per_seq: int,
        is_non_paged: bool,
    ) -> None:
        tex = self._get_tex()
        qkv_format = tex.NVTE_QKV_Format(int(qkv_format)) if qkv_format is not None else None
        return tex.copy_to_kv_cache(
            new_k,
            new_v,
            k_cache,
            v_cache,
            page_table,
            cu_new_lens,
            cu_cached_lens,
            qkv_format,
            b,
            max_ctx_len,
            max_seq_len,
            max_pages_per_seq,
            is_non_paged
        )
    def convert_thd_to_bshd(
        self,
        tensor: torch.Tensor,
        cu_seqlens: torch.Tensor,
        b: int,
        max_seq_len: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.convert_thd_to_bshd(tensor, cu_seqlens, b, max_seq_len)
    def convert_bshd_to_thd(
        self,
        tensor: torch.Tensor,
        cu_seqlens: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.convert_bshd_to_thd(tensor, cu_seqlens, t)

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
        tex = self._get_tex()
        qkv_format = tex.NVTE_QKV_Format(int(qkv_format)) if qkv_format is not None else None
        return tex.fused_rope_forward(
            input, freqs, start_positions, qkv_format,
            interleaved, cu_seqlens, cp_size, cp_rank
        )
    def fused_rope_backward(
        self,
        output_grads: torch.Tensor,
        freqs: torch.Tensor,
        qkv_format: NVTE_QKV_Format,
        interleaved: bool,
        cu_seqlens: Optional[torch.Tensor],
        cp_size: int,
        cp_rank: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        qkv_format = tex.NVTE_QKV_Format(int(qkv_format)) if qkv_format is not None else None
        return tex.fused_rope_backward(
            output_grads, freqs, qkv_format,
            interleaved, cu_seqlens, cp_size, cp_rank
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
        tex = self._get_tex()
        qkv_format = tex.NVTE_QKV_Format(int(qkv_format)) if qkv_format is not None else None
        return tex.fused_qkv_rope_forward(
            qkv_input, q_freqs, k_freqs, start_positions,
            qkv_split_arg_list, qkv_format, interleaved,
            cp_size, cp_rank
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
        tex = self._get_tex()
        qkv_format = tex.NVTE_QKV_Format(int(qkv_format)) if qkv_format is not None else None
        return tex.fused_qkv_rope_backward(
            q_grad_out, k_grad_out, v_grad_out,
            q_freqs, k_freqs, qkv_split_arg_list,
            qkv_format, interleaved, cp_size, cp_rank
        )

    # fused router
    def fused_topk_with_score_function_fwd(
        self,
        logits: torch.Tensor,
        topk: int,
        use_pre_softmax: bool,
        num_groups: Optional[int],
        group_topk: Optional[int],
        scaling_factor: Optional[float],
        score_function: str,
        expert_bias: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.fused_topk_with_score_function_fwd(
            logits,
            topk,
            use_pre_softmax,
            num_groups,
            group_topk,
            scaling_factor,
            score_function,
            expert_bias,
        )
    def fused_topk_with_score_function_bwd(
        self,
        num_tokens: int,
        num_experts: int,
        routing_map: torch.Tensor,
        intermediate_output: torch.Tensor,
        grad_probs: torch.Tensor,
        topk: int,
        use_pre_softmax: bool,
        scaling_factor: Optional[float],
        score_function: str,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.fused_topk_with_score_function_bwd(
            num_tokens,
            num_experts,
            routing_map,
            intermediate_output,
            grad_probs,
            topk,
            use_pre_softmax,
            scaling_factor,
            score_function,
        )
    def fused_score_for_moe_aux_loss_fwd(
        self,
        logits: torch.Tensor,
        topk: int,
        score_function: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.fused_score_for_moe_aux_loss_fwd(
            logits,
            topk,
            score_function,
        )
    def fused_score_for_moe_aux_loss_bwd(
        self,
        num_tokens: int,
        num_experts: int,
        intermediate_output: torch.Tensor,
        grad_scores: torch.Tensor,
        topk: int,
        score_function: str,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.fused_score_for_moe_aux_loss_bwd(
            num_tokens,
            num_experts,
            intermediate_output,
            grad_scores,
            topk,
            score_function,
        )
    def fused_moe_aux_loss_fwd(
        self,
        probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        total_num_tokens: int,
        num_experts: int,
        num_rows: int,
        num_cols: int,
        topk: int,
        coeff: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.fused_moe_aux_loss_fwd(
            probs,
            tokens_per_expert,
            total_num_tokens,
            num_experts,
            num_rows,
            num_cols,
            topk,
            coeff,
        )
    def fused_moe_aux_loss_bwd(
        self,
        Const_buf: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        num_rows: int,
        num_cols: int,
        grad_aux_loss: torch.Tensor,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.fused_moe_aux_loss_bwd(Const_buf, tokens_per_expert, num_rows, num_cols, grad_aux_loss)

    # Dropout
    def dropout_fwd(
        self,
        input: torch.Tensor,
        dropout_probability: float,
        out: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.dropout_fwd(input, dropout_probability, out)
    def dropout_bwd(
        self,
        grad_output: torch.Tensor,
        mask: torch.Tensor,
        dropout_probability: float,
        grad_input: Optional[torch.Tensor],
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.dropout_bwd(grad_output, mask, dropout_probability, grad_input)

    # Misc
    def get_cublasLt_version(self) -> int:
        tex = self._get_tex()
        return tex.get_cublasLt_version()
    def get_cudnn_version(self) -> int:
        tex = self._get_tex()
        return tex.get_cudnn_version()
    def get_num_cublas_streams(self) -> int:
        tex = self._get_tex()
        return tex.get_num_cublas_streams()

    # Support THD format for Context Parallel
    def thd_read_half_tensor(
        self,
        tensor: torch.Tensor,
        cu_seqlens: torch.Tensor,
        half_idx: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.thd_read_half_tensor(tensor, cu_seqlens, half_idx)
    def thd_second_half_lse_correction(
        self,
        lse: torch.Tensor,
        lse_per_step: torch.Tensor,
        cu_seqlens: torch.Tensor,
        lse_packed: bool,
    ) -> None:
        tex = self._get_tex()
        return tex.thd_second_half_lse_correction(
            lse, lse_per_step, cu_seqlens, lse_packed
        )
    def thd_read_second_half_lse(
        self,
        lse: torch.Tensor,
        cu_seqlens: torch.Tensor,
        lse_packed: bool,
        second_half_lse_seqlen: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.thd_read_second_half_lse(
            lse, cu_seqlens, lse_packed, second_half_lse_seqlen
        )
    def thd_out_correction(
        self,
        out: torch.Tensor,
        out_per_step: torch.Tensor,
        lse: torch.Tensor,
        lse_per_step: torch.Tensor,
        cu_seqlens: torch.Tensor,
        only_second_half: bool,
        lse_packed: bool,
    ) -> None:
        tex = self._get_tex()
        return tex.thd_out_correction(
            out, out_per_step, lse, lse_per_step,
            cu_seqlens, only_second_half, lse_packed
        )
    def thd_grad_correction(
        self,
        grad: torch.Tensor,
        grad_per_step: torch.Tensor,
        cu_seqlens: torch.Tensor,
        first_half: str,
        second_half: str,
    ) -> None:
        tex = self._get_tex()
        return tex.thd_grad_correction(
            grad, grad_per_step, cu_seqlens,
            first_half, second_half
        )
    def thd_get_partitioned_indices(
        self,
        cu_seqlens: torch.Tensor,
        total_tokens: int,
        world_size: int,
        rank: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.thd_get_partitioned_indices(
            cu_seqlens, total_tokens, world_size, rank
        )

    # nvshmem functions
    def init_nvshmem_backend(
        self,
        process_group: Any,
    ) -> None:
        tex = self._get_tex()
        return tex.init_nvshmem_backend(process_group)
    def create_nvshmem_tensor(
        self,
        shape: List[int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.create_nvshmem_tensor(shape, dtype)
    def nvshmem_send_on_current_stream(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        peer: int,
        signal: torch.Tensor,
    ) -> None:
        tex = self._get_tex()
        return tex.nvshmem_send_on_current_stream(src, dst, peer, signal)
    def nvshmem_wait_on_current_stream(
        self,
        signal: torch.Tensor,
        wait_kind: str,
    ) -> None:
        tex = self._get_tex()
        return tex.nvshmem_wait_on_current_stream(signal, wait_kind)
    def nvshmem_finalize(self) -> None:
        tex = self._get_tex()
        return tex.nvshmem_finalize()

    # multi-tensor functions
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
    def multi_tensor_unscale_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        inv_scale: torch.Tensor,
        per_tensor: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.multi_tensor_unscale_l2norm(
            chunk_size, noop_flag, tensor_lists, inv_scale, per_tensor
        )
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
            chunk_size, noop_flag, tensor_lists,
            lr, beta1, beta2, epsilon,
            step, mode, bias_correction, weight_decay
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
        tex = self._get_tex()
        return tex.multi_tensor_adam_param_remainder(
            chunk_size, noop_flag, tensor_lists,
            lr, beta1, beta2, epsilon,
            step, mode, bias_correction, weight_decay
        )
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
            chunk_size, noop_flag, tensor_lists,
            lr, beta1, beta2, epsilon,
            step, mode, bias_correction, weight_decay,
            fp8_dtype
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
            chunk_size, noop_flag, tensor_lists,
            lr, beta1, beta2, epsilon,
            step, mode, bias_correction, weight_decay,
            inv_scale
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
            chunk_size, noop_flag, tensor_lists,
            lr, beta1, beta2, epsilon,
            step, mode, bias_correction, weight_decay,
            inv_scale
        )
    def multi_tensor_sgd(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        wd: float,
        momentum: float,
        dampening: float,
        lr: float,
        nesterov: bool,
        first_run: bool,
        wd_after_momentum: bool,
        scale: float,
    ) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_sgd(
            chunk_size, noop_flag, tensor_lists,
            wd, momentum, dampening,
            lr, nesterov, first_run,
            wd_after_momentum, scale
        )
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
            chunk_size, noop_flag, tensor_lists,
            max_fp8, force_pow_2_scales, epsilon
        )

    # Comm+GEMM Overlap
    def bulk_overlap_ag_with_external_gemm(
        self,
        allgather_communicator: CommOverlap,
        send_stream: Any,
        recv_stream: Any,
    ) -> Any:
        tex = self._get_tex()
        return tex.bulk_overlap_ag_with_external_gemm(allgather_communicator, send_stream, recv_stream)

############## class func #################################
    def get_flash_attention_class(self):
        from .flash_attention import FlashAttentionMETAX
        return FlashAttentionMETAX
    def create_fp8_tensor_meta(self) -> FP8TensorMeta:
        tex = self._get_tex()
        return tex.FP8TensorMeta()
    def create_comm_overlap_helper(
        self,
        world_group: Optional[Any] = None,
        intra_node_group: Optional[Any] = None,
    ) -> "CommOverlapHelper":
        tex = self._get_tex()
        return tex.CommOverlapHelper(world_group, intra_node_group)
    def create_comm_overlap(
        self,
        buffer_shape: List[int],
        buffer_dtype: torch.dtype,
        helper: Any,
        tp_size: int,
        num_splits: int = 3,
        num_max_streams: int = 3,
        comm_cga_size: int = 2,
        gemm_priority: int = 0,
        comm_priority: int = 0,
        num_comm_sm: int = 16,
        set_sm_margin: bool = True,
        atomic_gemm: bool = False,
        rs_overlap_first_gemm: bool = False,
    ) -> "CommOverlap":
        tex = self._get_tex()
        return tex.CommOverlap(
            buffer_shape, buffer_dtype, helper, tp_size,
            num_splits, num_max_streams, comm_cga_size,
            gemm_priority, comm_priority, num_comm_sm,
            set_sm_margin, atomic_gemm, rs_overlap_first_gemm
        )
    def create_comm_overlap_p2p(
        self,
        buffer_shape: List[int],
        buffer_dtype: torch.dtype,
        helper: Any,
        tp_size: int,
        comm_type: Any,
        num_max_streams: int = 3,
        comm_cga_size: int = 1,
        gemm_priority: int = 0,
        comm_priority: int = 0,
        num_comm_sm: int = 1,
        set_sm_margin: bool = False,
        atomic_gemm: bool = False,
        use_ce: bool = True,
        aggregate: bool = False,
    ) -> "CommOverlapP2P":
        tex = self._get_tex()
        return tex.CommOverlapP2P(
            buffer_shape, buffer_dtype, helper, tp_size, comm_type,
            num_max_streams, comm_cga_size, gemm_priority, comm_priority,
            num_comm_sm, set_sm_margin, atomic_gemm, use_ce, aggregate
        )
