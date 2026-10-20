# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.
import functools
import inspect
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from ....ops import *
from ..compat import vendor_te_compat

_logger = logging.getLogger(__name__)

# Causal mask types (explicit set instead of relying on enum value ordering).
_CAUSAL_MASK_TYPES = frozenset({
    NVTE_Mask_Type.NVTE_CAUSAL_MASK,
    NVTE_Mask_Type.NVTE_PADDING_CAUSAL_MASK,
    NVTE_Mask_Type.NVTE_CAUSAL_BOTTOM_RIGHT_MASK,
    NVTE_Mask_Type.NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK,
})


def _load_biren_libs():
    import ctypes
    import os
    from pathlib import Path
    import importlib.util
    import platform
    import glob as glob_module

    def get_ext():
        system = platform.system()
        return ".so" if system == "Linux" else ".dylib" if system == "Darwin" else ".dll"

    ext = get_ext()

    try:
        te_path_override = os.environ.get("TE_LIB_PATH")
        if te_path_override:
            te_path = Path(te_path_override)
        else:
            te_path = Path(importlib.util.find_spec("transformer_engine").origin).parent.parent
        for search_dir in [te_path, te_path / "transformer_engine"]:
            if search_dir.exists():
                matches = list(search_dir.glob(f"libtransformer_engine{ext}*"))
                if matches:
                    ctypes.CDLL(str(matches[0]), mode=ctypes.RTLD_GLOBAL)
                    return True
        return False
    except Exception:
        return False


_biren_libs_loaded = False


def _ensure_biren_libs():
    global _biren_libs_loaded
    if not _biren_libs_loaded:
        _biren_libs_loaded = _load_biren_libs()
    return _biren_libs_loaded


def _check_biren_available() -> bool:
    if not bool(int(os.getenv("TE_FL_USE_SUPA", "1"))):
        return False

    try:
        import torch_supa
        if not torch.cuda.is_available():
            return False
    except ImportError:
        return False

    try:
        if not _ensure_biren_libs():
            return False
        import transformer_engine_torch_biren
        return True
    except (ImportError, OSError) as e:
        _logger.warning(
            "Biren/SUPA backend unavailable (torch_supa present but the "
            "transformer_engine_torch_biren extension could not be imported; "
            "build/load wiring may be missing): %s", e,
        )
        return False


def _get_tex():
    _ensure_biren_libs()
    import transformer_engine_torch_biren
    return vendor_te_compat(transformer_engine_torch_biren, "Biren")


# --- DType auto-patching decorator ---
# pybind11 module_local() makes DType enums non-interchangeable across modules.
# This decorator auto-converts DType params and quantizer.dtype on method entry.

_DTYPE_PARAMS = frozenset({
    'otype', 'output_dtype', 'q_dtype', 'kv_dtype', 'fp8_dtype',
    'out_dtype', 'D_type', 'dqkv_type', 'dtype',
})
_QUANTIZER_PARAMS = frozenset({'quantizer'})
_QUANTIZER_LIST_PARAMS = frozenset({'quantizer_list'})


def _auto_patch_dtype(cls):
    """Class decorator: auto-convert foreign DType enum on method entry."""
    tex_ref = [None]

    def _cvt_dtype(val, tex):
        if val is None:
            return None
        try:
            return tex.DType(int(val))
        except (TypeError, ValueError):
            return val

    def _cvt_quantizer(q, tex):
        if q is None:
            return q
        if hasattr(q, 'dtype') and q.dtype is not None:
            try:
                q.dtype = tex.DType(int(q.dtype))
            except (TypeError, ValueError):
                pass
        return q

    def _make_wrapper(method, dtype_idx, quantizer_idx, qlist_idx,
                      dtype_names, quantizer_names, qlist_names):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if tex_ref[0] is None:
                tex_ref[0] = self._get_tex()
            tex = tex_ref[0]
            args = list(args)
            for i in dtype_idx:
                if i < len(args):
                    args[i] = _cvt_dtype(args[i], tex)
            for i in quantizer_idx:
                if i < len(args):
                    args[i] = _cvt_quantizer(args[i], tex)
            for i in qlist_idx:
                if i < len(args) and args[i] is not None:
                    for q in args[i]:
                        _cvt_quantizer(q, tex)
            for k in dtype_names:
                if k in kwargs:
                    kwargs[k] = _cvt_dtype(kwargs[k], tex)
            for k in quantizer_names:
                if k in kwargs:
                    kwargs[k] = _cvt_quantizer(kwargs[k], tex)
            for k in qlist_names:
                if k in kwargs and kwargs[k] is not None:
                    for q in kwargs[k]:
                        _cvt_quantizer(q, tex)
            return method(self, *args, **kwargs)
        return wrapper

    for name in list(vars(cls)):
        if name.startswith('_'):
            continue
        method = getattr(cls, name)
        if not callable(method):
            continue
        try:
            sig = inspect.signature(method)
        except (ValueError, TypeError):
            continue
        params = list(sig.parameters.keys())[1:]  # skip self

        dtype_idx = [i for i, p in enumerate(params) if p in _DTYPE_PARAMS]
        quantizer_idx = [i for i, p in enumerate(params) if p in _QUANTIZER_PARAMS]
        qlist_idx = [i for i, p in enumerate(params) if p in _QUANTIZER_LIST_PARAMS]

        if not (dtype_idx or quantizer_idx or qlist_idx):
            continue

        dtype_names = [p for p in params if p in _DTYPE_PARAMS]
        quantizer_names = [p for p in params if p in _QUANTIZER_PARAMS]
        qlist_names = [p for p in params if p in _QUANTIZER_LIST_PARAMS]

        setattr(cls, name, _make_wrapper(method, dtype_idx, quantizer_idx, qlist_idx,
                                         dtype_names, quantizer_names, qlist_names))
    return cls


@_auto_patch_dtype
class BirenBackend(TEFLBackendBase):
    @staticmethod
    def check_available() -> bool:
        return _check_biren_available()

    def __init__(self):
        self._tex = None

    def _get_tex(self):
        if self._tex is None:
            self._tex = _get_tex()
        return self._tex

    def is_available(self) -> bool:
        return _check_biren_available()

    def get_attention_backend(self, attention_params=None):
        try:
            import flashattn_train  # noqa: F401
        except ImportError:
            return None

        import torch
        from packaging.version import Version as PkgVersion

        if attention_params is not None:
            dtype = getattr(attention_params, "qkv_dtype", None)
            fp8 = getattr(attention_params, "fp8", False)

            if fp8:
                # suDNN does not support causal/sliding_window mask with dropout > 0
                dropout = getattr(attention_params, "attention_dropout", 0.0)
                mask_type = getattr(attention_params, "attn_mask_type", "no_mask")
                is_causal = "causal" in mask_type if mask_type else False
                window_size = getattr(attention_params, "window_size", None)
                is_sliding = window_size is not None and window_size != (-1, -1)
                if (is_causal or is_sliding) and dropout > 0.0:
                    pass  # fall through to FlashAttentionBiren (dequants FP8→BF16)
                else:
                    # suDNN supports FP8 attention natively via FusedAttention (cuDNN frontend)
                    from transformer_engine.pytorch.attention.dot_product_attention import utils as dpa_utils
                    return dpa_utils._original_get_attention_backend(attention_params)
            elif dtype is not None and dtype not in (torch.float16, torch.bfloat16):
                from transformer_engine.pytorch.attention.dot_product_attention import utils as dpa_utils
                return dpa_utils._original_get_attention_backend(attention_params)

            head_dim = getattr(attention_params, "head_dim_qk", None)
            if head_dim is not None and head_dim not in (32, 64, 128):
                return (False, None, False, None, True, [False, False, True])

        # Respect env var overrides for backend selection
        use_flash = int(os.getenv("NVTE_FLASH_ATTN", "1"))
        use_fused = int(os.getenv("NVTE_FUSED_ATTN", "1"))
        use_unfused = int(os.getenv("NVTE_UNFUSED_ATTN", "1"))

        # suDNN fused attention limitations:
        # 1. Does not support padding masks (required for KV cache inference)
        # 2. Does not reliably support attention bias on cmodel
        # 3. Does not support causal/sliding_window mask with dropout > 0
        inference_params = getattr(attention_params, "inference_params", None) if attention_params else None
        fused_available = True
        if inference_params is not None:
            use_fused = 0
            fused_available = False

        if attention_params is not None:
            bias_type = getattr(attention_params, "core_attention_bias_type", "no_bias")
            if bias_type not in ("no_bias", None):
                use_fused = 0
                fused_available = False

            dropout = getattr(attention_params, "attention_dropout", 0.0)
            mask_type = getattr(attention_params, "attn_mask_type", "no_mask")
            is_causal = "causal" in mask_type if mask_type else False
            window_size = getattr(attention_params, "window_size", None)
            is_sliding = window_size is not None and window_size != (-1, -1)
            if (is_causal or is_sliding) and dropout > 0.0:
                use_fused = 0
                fused_available = False

        fused_backend = NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen if use_fused else None
        return (
            bool(use_flash), PkgVersion("3.0.0") if use_flash else None,
            bool(use_fused), fused_backend,
            bool(use_unfused),
            [True, fused_available, True],
        )


    ##### Quantization #####
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
        return tex.dequantize(input, otype)

    def bgrad_quantize(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> List[Any]:
        tex = self._get_tex()
        if quantizer is None:
            bias_size = input.shape[-1]
            grad_bias = input.reshape(-1, bias_size).sum(dim=0)
            return [grad_bias, input.contiguous()]
        # Avoid the fused bgrad+quantize kernel (IS_DBIAS=true) which requires SM >= 10.0.
        # Instead compute grad_bias separately and use the plain quantize kernel.
        bias_size = input.shape[-1]
        grad_bias = input.reshape(-1, bias_size).sum(dim=0)
        quantized = tex.quantize(input, quantizer, None, None)
        return [grad_bias, quantized]

    def group_quantize(
        self,
        tensor: torch.Tensor,
        quantizer: Any,
        num_tensors: int,
        first_dims: List[int],
        tensor_offsets: Optional[Any] = None,
    ) -> Any:
        tex = self._get_tex()
        return tex.group_quantize(tensor, quantizer, num_tensors, first_dims, tensor_offsets)

    def bgrad_group_quantize(
        self,
        tensor: torch.Tensor,
        quantizer: Any,
        num_tensors: int,
        first_dims: List[int],
        tensor_offsets: Optional[Any] = None,
    ) -> Any:
        tex = self._get_tex()
        return tex.bgrad_group_quantize(tensor, quantizer, num_tensors, first_dims, tensor_offsets)

    ##### GEMM #####
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
        return tex.generic_gemm(
            A, transA, B, transB, D, quantizer, output_dtype, bias,
            bias_type, gelu, gelu_in, grad, workspace, workspace_size,
            accumulate, use_split_accumulator, comm_overlap, comm_type,
            extra_output, bulk_overlap, alpha, beta,
        )

    ##### Activations #####
    def glu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.glu(input, quantizer)

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

    def silu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.silu(input, quantizer)

    def swiglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.swiglu(input, quantizer)

    def clamped_swiglu(self, input: torch.Tensor, quantizer: Any,
                       limit: float = 7.0, alpha: float = 1.702,
                       glu_linear_offset: float = 1.0) -> Any:
        tex = self._get_tex()
        return tex.clamped_swiglu(input, quantizer, limit, alpha, glu_linear_offset)

    ##### Activation Backward #####
    def dglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dglu(grad, fwd_input, quantizer)

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

    def dsilu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dsilu(grad, fwd_input, quantizer)

    def dswiglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dswiglu(grad, fwd_input, quantizer)

    def clamped_dswiglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any,
                        limit: float = 7.0, alpha: float = 1.702,
                        glu_linear_offset: float = 1.0) -> Any:
        tex = self._get_tex()
        return tex.clamped_dswiglu(grad, fwd_input, quantizer, limit, alpha, glu_linear_offset)

    ##### DBias + DAct #####
    # Closed-form activation gradients aligned with common/util/math.h; used
    # by the dbias_d* fallbacks below (no autograd graph construction).
    @staticmethod
    def _dgelu_closed(grad_output, x):
        """Closed-form tanh-GELU gradient, aligned with common/util/math.h dgelu()."""
        k_alpha = 0.7978845608028654  # sqrt(2/pi)
        tanh_val = torch.tanh(k_alpha * (x + 0.044715 * x * x * x))
        dtanh = 1.0 - tanh_val * tanh_val
        dx = (
            0.5 * (1.0 + tanh_val)
            + 0.5 * x * dtanh * k_alpha * (1.0 + 3.0 * 0.044715 * x * x)
        )
        return grad_output * dx

    @staticmethod
    def _dsilu_closed(grad_output, x):
        """Closed-form SiLU gradient: sigmoid(x)*(1 + x*(1 - sigmoid(x)))."""
        s = torch.sigmoid(x)
        return grad_output * s * (1.0 + x * (1.0 - s))

    @staticmethod
    def _drelu_closed(grad_output, x):
        """Closed-form ReLU gradient."""
        return grad_output * (x > 0).to(x.dtype)

    @staticmethod
    def _dqgelu_closed(grad_output, x):
        """Closed-form QuickGELU gradient: derivative of x*sigmoid(1.702*x)."""
        a = 1.702
        s = torch.sigmoid(a * x)
        return grad_output * (s + a * x * s * (1.0 - s))

    @staticmethod
    def _dsrelu_closed(grad_output, x):
        """Closed-form squared-ReLU gradient: derivative of relu(x)^2."""
        return grad_output * 2.0 * torch.nn.functional.relu(x)

    _DBIAS_DACT_GRADS = {
        "dbias_dgelu": _dgelu_closed,
        "dbias_dsilu": _dsilu_closed,
        "dbias_drelu": _drelu_closed,
        "dbias_dqgelu": _dqgelu_closed,
        "dbias_dsrelu": _dsrelu_closed,
    }

    def _dbias_dact_fallback(self, grad: torch.Tensor, fwd_input: torch.Tensor, dact_fn, quantizer=None) -> List[Any]:
        """dbias+dact via closed-form gradients (no autograd graph build).

        Returns [grad_bias, grad_input] to match the native dact_dbias
        convention and the pytorch/ops/fused caller (``db, dx = ...``).
        """
        grad_input = dact_fn(grad, fwd_input)
        grad_bias = grad_input.sum(dim=tuple(range(grad_input.ndim - 1)))
        if quantizer is not None:
            tex = self._get_tex()
            grad_input = tex.quantize(grad_input, quantizer, None, None)
        return [grad_bias, grad_input]

    def dbias_dgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> List[Any]:
        """GELU backward + bias gradient (tanh approximation).

        Returns ``[grad_bias, grad_input]``.
        """
        return self._dbias_dact_fallback(grad, fwd_input, self._dgelu_closed, quantizer)

    def dbias_dsilu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> List[Any]:
        """SiLU backward + bias gradient. Returns ``[grad_bias, grad_input]``."""
        return self._dbias_dact_fallback(grad, fwd_input, self._dsilu_closed, quantizer)

    def dbias_drelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> List[Any]:
        """ReLU backward + bias gradient. Returns ``[grad_bias, grad_input]``."""
        return self._dbias_dact_fallback(grad, fwd_input, self._drelu_closed, quantizer)

    def dbias_dqgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> List[Any]:
        """QuickGELU backward + bias gradient. Returns ``[grad_bias, grad_input]``."""
        return self._dbias_dact_fallback(grad, fwd_input, self._dqgelu_closed, quantizer)

    def dbias_dsrelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> List[Any]:
        """Squared-ReLU backward + bias gradient. Returns ``[grad_bias, grad_input]``."""
        return self._dbias_dact_fallback(grad, fwd_input, self._dsrelu_closed, quantizer)

    ##### Normalization #####
    def layernorm_fwd(self, input, weight, bias, eps, ln_out, quantizer, otype, sm_margin,
                      zero_centered_gamma) -> List[Any]:
        tex = self._get_tex()
        return tex.layernorm_fwd(input, weight, bias, eps, ln_out, quantizer, otype,
                                 sm_margin, zero_centered_gamma)

    def layernorm_bwd(self, dz, x, mu, rsigma, gamma, sm_margin,
                      zero_centered_gamma) -> List[Any]:
        tex = self._get_tex()
        return tex.layernorm_bwd(dz, x, mu, rsigma, gamma, sm_margin, zero_centered_gamma)

    def rmsnorm_fwd(self, input, weight, eps, ln_out, quantizer, otype, sm_margin,
                    zero_centered_gamma) -> List[Any]:
        tex = self._get_tex()
        return tex.rmsnorm_fwd(input, weight, eps, ln_out, quantizer, otype,
                               sm_margin, zero_centered_gamma)

    def rmsnorm_bwd(self, dz, x, rsigma, gamma, sm_margin, zero_centered_gamma) -> List[Any]:
        tex = self._get_tex()
        return tex.rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma)

    def rmsnorm_bwd_add(self, dz, x, add, rsigma, gamma, sm_margin,
                        zero_centered_gamma) -> List[Any]:
        tex = self._get_tex()
        return tex.rmsnorm_bwd_add(dz, x, add, rsigma, gamma, sm_margin, zero_centered_gamma)

    ##### Permutation #####
    def moe_permute_fwd(self, input, dtype, indices, num_out_tokens, workspace,
                        max_expanded_token_num) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        tex = self._get_tex()
        return tex.moe_permute_fwd(input, dtype, indices, num_out_tokens, workspace,
                                   max_expanded_token_num)

    def moe_permute_bwd(self, input, dtype, row_id_map, prob, num_tokens, topK) -> torch.Tensor:
        tex = self._get_tex()
        return tex.moe_permute_bwd(input, dtype, row_id_map, prob, num_tokens, topK)

    def moe_unpermute_fwd(self, input, dtype, row_id_map, prob, num_tokens, topK) -> torch.Tensor:
        tex = self._get_tex()
        return tex.moe_unpermute_fwd(input, dtype, row_id_map, prob, num_tokens, topK)

    def moe_unpermute_bwd(self, input_bwd, input_fwd, dtype, row_id_map,
                          prob) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.moe_unpermute_bwd(input_bwd, input_fwd, dtype, row_id_map, prob)

    ##### Softmax #####
    def scaled_softmax_forward(self, input, scale) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_softmax_forward(input, scale)

    def scaled_softmax_backward(self, output_grad_, softmax_results_, scale_factor) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_softmax_backward(output_grad_, softmax_results_, scale_factor)

    def scaled_masked_softmax_forward(self, input, mask, scale_factor) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_masked_softmax_forward(input, mask, scale_factor)

    def scaled_masked_softmax_backward(self, output_grad_, softmax_results_,
                                       scale_factor) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_masked_softmax_backward(output_grad_, softmax_results_, scale_factor)

    def scaled_upper_triang_masked_softmax_forward(self, input, scale_factor) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_upper_triang_masked_softmax_forward(input, scale_factor)

    def scaled_upper_triang_masked_softmax_backward(self, output_grads_, softmax_results_,
                                                    scale_factor) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_upper_triang_masked_softmax_backward(
            output_grads_, softmax_results_, scale_factor)

    def scaled_aligned_causal_masked_softmax_forward(self, input, scale_factor) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_aligned_causal_masked_softmax_forward(input, scale_factor)

    def scaled_aligned_causal_masked_softmax_backward(self, output_grad_, softmax_results_,
                                                      scale_factor) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_aligned_causal_masked_softmax_backward(
            output_grad_, softmax_results_, scale_factor)

    ##### Transpose / Swizzle #####
    def fp8_transpose(self, input, dtype, out=None) -> torch.Tensor:
        tex = self._get_tex()
        return tex.fp8_transpose(input, dtype, out=out)

    def swap_first_dims(self, tensor, out) -> torch.Tensor:
        tex = self._get_tex()
        return tex.swap_first_dims(tensor, out)

    def nvfp4_data_transpose(self, input, out=None) -> torch.Tensor:
        tex = self._get_tex()
        return tex.nvfp4_data_transpose(input, out=out)

    def swizzle_scales_for_gemm_(self, tensor) -> None:
        tex = self._get_tex()
        return tex.swizzle_scales_for_gemm_(tensor)

    def grouped_swizzle_for_gemm(self, tensor, rowwise, columnwise) -> None:
        tex = self._get_tex()
        return tex.grouped_swizzle_for_gemm(tensor, rowwise, columnwise)

    def splits_to_offsets(self, first_dims, logical_last_dim) -> torch.Tensor:
        tex = self._get_tex()
        return tex.splits_to_offsets(first_dims, logical_last_dim)

    def splits_to_offsets_multi(
        self,
        split_sizes,
        device,
        *,
        strides,
        include_leading_zero,
        dtypes,
        bulk_allocate=False,
    ):
        tex = self._get_tex()
        return tex.splits_to_offsets_multi(
            split_sizes,
            device,
            strides=strides,
            include_leading_zero=include_leading_zero,
            dtypes=dtypes,
            bulk_allocate=bulk_allocate,
        )

    ##### Attention #####
    def get_fused_attn_backend(self, is_training, q_dtype, kv_dtype, qkv_layout, bias_type,
                               attn_mask_type, softmax_type, p_dropout, num_attn_heads,
                               num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim_qk,
                               head_dim_v, window_size_left, window_size_right,
                               return_max_logit, cuda_graph=False,
                               deterministic=False) -> NVTE_Fused_Attn_Backend:
        # suDNN does not support causal/sliding_window mask with dropout > 0
        if p_dropout > 0.0:
            is_sliding = (window_size_left >= 0 or window_size_right >= 0)
            is_causal = (
                attn_mask_type is not None
                and NVTE_Mask_Type(int(attn_mask_type)) in _CAUSAL_MASK_TYPES
            )
            if is_sliding or is_causal:
                return NVTE_Fused_Attn_Backend.NVTE_No_Backend
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
            head_dim_v, window_size_left, window_size_right,
            return_max_logit, cuda_graph, deterministic,
        )
        return NVTE_Fused_Attn_Backend(result)

    def fa_prepare_fwd(self, qkvi) -> torch.Tensor:
        tex = self._get_tex()
        return tex.fa_prepare_fwd(qkvi)

    def fa_prepare_bwd(self, q, k, v) -> torch.Tensor:
        tex = self._get_tex()
        return tex.fa_prepare_bwd(q, k, v)

    def fused_attn_fwd(self, max_seqlen_q, max_seqlen_kv, is_training, attn_scale,
                       p_dropout, set_zero, qkv_layout, o_format,
                       qkv_scale_inv_format, bias_type, attn_mask_type,
                       softmax_type, window_size, bottom_right_diagonal,
                       cu_seqlens_q, cu_seqlens_kv, Q, K, V, fake_dtype,
                       cu_seqlens_q_padded, cu_seqlens_kv_padded,
                       page_table_k, page_table_v, s_quantizer, o_quantizer,
                       Bias, SoftmaxOffset, rng_gen, rng_elts_per_thread,
                       return_max_logit, cuda_graph=False) -> List[Any]:
        tex = self._get_tex()
        qkv_layout = tex.NVTE_QKV_Layout(int(qkv_layout)) if qkv_layout is not None else None
        o_format = tex.NVTE_QKV_Format(int(o_format)) if o_format is not None else None
        qkv_scale_inv_format = (
            tex.NVTE_QKV_Format(int(qkv_scale_inv_format)) if qkv_scale_inv_format is not None else None
        )
        bias_type = tex.NVTE_Bias_Type(int(bias_type)) if bias_type is not None else None
        attn_mask_type = tex.NVTE_Mask_Type(int(attn_mask_type)) if attn_mask_type is not None else None
        softmax_type = tex.NVTE_Softmax_Type(int(softmax_type)) if softmax_type is not None else None
        return tex.fused_attn_fwd(
            max_seqlen_q, max_seqlen_kv, is_training, attn_scale, p_dropout,
            set_zero, qkv_layout, o_format, qkv_scale_inv_format,
            bias_type, attn_mask_type, softmax_type,
            window_size, bottom_right_diagonal, cu_seqlens_q, cu_seqlens_kv,
            Q, K, V, fake_dtype, cu_seqlens_q_padded, cu_seqlens_kv_padded,
            page_table_k, page_table_v, s_quantizer, o_quantizer, Bias,
            SoftmaxOffset, rng_gen, rng_elts_per_thread, return_max_logit, cuda_graph,
        )

    def fused_attn_bwd(self, max_seqlen_q, max_seqlen_kv, attn_scale, p_dropout,
                       set_zero, qkv_layout, o_format, do_format, dqkv_layout,
                       qkv_scale_inv_format, do_scale_inv_format,
                       bias_type, attn_mask_type, softmax_type,
                       window_size, bottom_right_diagonal, deterministic,
                       cu_seqlens_q, cu_seqlens_kv, Q, K, V, O, dO, fake_dtype,
                       Aux_CTX_Tensors, cu_seqlens_q_padded,
                       cu_seqlens_kv_padded, s_quantizer, dp_quantizer,
                       dqkv_quantizer, cuda_graph=False) -> List[Any]:
        tex = self._get_tex()
        qkv_layout = tex.NVTE_QKV_Layout(int(qkv_layout)) if qkv_layout is not None else None
        o_format = tex.NVTE_QKV_Format(int(o_format)) if o_format is not None else None
        do_format = tex.NVTE_QKV_Format(int(do_format)) if do_format is not None else None
        dqkv_layout = tex.NVTE_QKV_Layout(int(dqkv_layout)) if dqkv_layout is not None else None
        qkv_scale_inv_format = (
            tex.NVTE_QKV_Format(int(qkv_scale_inv_format)) if qkv_scale_inv_format is not None else None
        )
        do_scale_inv_format = (
            tex.NVTE_QKV_Format(int(do_scale_inv_format)) if do_scale_inv_format is not None else None
        )
        bias_type = tex.NVTE_Bias_Type(int(bias_type)) if bias_type is not None else None
        attn_mask_type = tex.NVTE_Mask_Type(int(attn_mask_type)) if attn_mask_type is not None else None
        softmax_type = tex.NVTE_Softmax_Type(int(softmax_type)) if softmax_type is not None else None
        return tex.fused_attn_bwd(
            max_seqlen_q, max_seqlen_kv, attn_scale, p_dropout, set_zero,
            qkv_layout, o_format, do_format, dqkv_layout,
            qkv_scale_inv_format, do_scale_inv_format,
            bias_type, attn_mask_type, softmax_type, window_size,
            bottom_right_diagonal, deterministic, cu_seqlens_q, cu_seqlens_kv,
            Q, K, V, O, dO, fake_dtype, Aux_CTX_Tensors,
            cu_seqlens_q_padded, cu_seqlens_kv_padded, s_quantizer,
            dp_quantizer, dqkv_quantizer, cuda_graph,
        )

    def copy_to_kv_cache(self, new_k, new_v, k_cache, v_cache, page_table,
                         cu_new_lens, cu_cached_lens, qkv_format, b, max_ctx_len,
                         max_seq_len, max_pages_per_seq, is_non_paged) -> None:
        tex = self._get_tex()
        qkv_format = tex.NVTE_QKV_Format(int(qkv_format)) if qkv_format is not None else None
        return tex.copy_to_kv_cache(
            new_k, new_v, k_cache, v_cache, page_table, cu_new_lens,
            cu_cached_lens, qkv_format, b, max_ctx_len, max_seq_len,
            max_pages_per_seq, is_non_paged,
        )

    def convert_thd_to_bshd(self, tensor, cu_seqlens, b, max_seq_len) -> torch.Tensor:
        tex = self._get_tex()
        return tex.convert_thd_to_bshd(tensor, cu_seqlens, b, max_seq_len)

    def convert_bshd_to_thd(self, tensor, cu_seqlens, t) -> torch.Tensor:
        tex = self._get_tex()
        return tex.convert_bshd_to_thd(tensor, cu_seqlens, t)

    ##### RoPE #####
    def fused_rope_forward(self, input, freqs, start_positions, qkv_format,
                           interleaved, cu_seqlens, cp_size, cp_rank) -> torch.Tensor:
        tex = self._get_tex()
        qkv_format = tex.NVTE_QKV_Format(int(qkv_format)) if qkv_format is not None else None
        return tex.fused_rope_forward(input, freqs, start_positions, qkv_format,
                                      interleaved, cu_seqlens, cp_size, cp_rank)

    def fused_rope_backward(self, output_grads, freqs, start_positions, qkv_format,
                            interleaved, cu_seqlens, cp_size, cp_rank) -> torch.Tensor:
        tex = self._get_tex()
        qkv_format = tex.NVTE_QKV_Format(int(qkv_format)) if qkv_format is not None else None
        return tex.fused_rope_backward(output_grads, freqs, start_positions, qkv_format,
                                       interleaved, cu_seqlens, cp_size, cp_rank)

    def fused_qkv_rope_forward(self, qkv_input, q_freqs, k_freqs, start_positions,
                               qkv_split_arg_list, qkv_format, interleaved,
                               cp_size, cp_rank) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        qkv_format = tex.NVTE_QKV_Format(int(qkv_format)) if qkv_format is not None else None
        return tex.fused_qkv_rope_forward(qkv_input, q_freqs, k_freqs, start_positions,
                                          qkv_split_arg_list, qkv_format, interleaved,
                                          cp_size, cp_rank)

    def fused_qkv_rope_backward(self, q_grad_out, k_grad_out, v_grad_out, q_freqs, k_freqs,
                                qkv_split_arg_list, qkv_format, interleaved,
                                cp_size, cp_rank) -> torch.Tensor:
        tex = self._get_tex()
        qkv_format = tex.NVTE_QKV_Format(int(qkv_format)) if qkv_format is not None else None
        return tex.fused_qkv_rope_backward(q_grad_out, k_grad_out, v_grad_out, q_freqs, k_freqs,
                                           qkv_split_arg_list, qkv_format, interleaved,
                                           cp_size, cp_rank)

    ##### Router / MoE #####
    def fused_topk_with_score_function_fwd(self, logits, topk, use_pre_softmax, num_groups,
                                           group_topk, scaling_factor, score_function,
                                           expert_bias, routing_map_format=0
                                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.fused_topk_with_score_function_fwd(
            logits, topk, use_pre_softmax, num_groups, group_topk,
            scaling_factor, score_function, expert_bias, routing_map_format,
        )

    def fused_topk_with_score_function_bwd(self, routing_map, intermediate_output,
                                           grad_probs, grad_logits, topk, use_pre_softmax,
                                           scaling_factor, score_function,
                                           routing_map_format=0) -> torch.Tensor:
        tex = self._get_tex()
        return tex.fused_topk_with_score_function_bwd(
            routing_map, intermediate_output, grad_probs, grad_logits,
            topk, use_pre_softmax, scaling_factor, score_function, routing_map_format,
        )

    def fused_score_for_moe_aux_loss_fwd(self, logits, topk, score_function,
                                         routing_map_format=0
                                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.fused_score_for_moe_aux_loss_fwd(logits, topk, score_function,
                                                    routing_map_format)

    def fused_score_for_moe_aux_loss_bwd(self, intermediate_output, grad_scores,
                                         grad_logits, topk,
                                         score_function) -> torch.Tensor:
        tex = self._get_tex()
        return tex.fused_score_for_moe_aux_loss_bwd(
            intermediate_output, grad_scores, grad_logits, topk, score_function,
        )

    def fused_moe_aux_loss_fwd(self, probs, tokens_per_expert, total_num_tokens, num_experts,
                               num_rows, num_cols, topk,
                               coeff) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.fused_moe_aux_loss_fwd(
            probs, tokens_per_expert, total_num_tokens, num_experts,
            num_rows, num_cols, topk, coeff,
        )

    def fused_moe_aux_loss_bwd(self, Const_buf, tokens_per_expert, num_rows, num_cols,
                               grad_aux_loss) -> torch.Tensor:
        tex = self._get_tex()
        return tex.fused_moe_aux_loss_bwd(Const_buf, tokens_per_expert, num_rows, num_cols,
                                          grad_aux_loss)

    ##### Dropout #####
    def dropout_fwd(self, input, dropout_probability, out=None) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.dropout_fwd(input, dropout_probability, out)

    def dropout_bwd(self, grad_output, mask, dropout_probability,
                    grad_input=None) -> torch.Tensor:
        tex = self._get_tex()
        return tex.dropout_bwd(grad_output, mask, dropout_probability, grad_input)

    ##### FP8 Scaling #####
    def compute_amax(self, input, amax) -> None:
        tex = self._get_tex()
        return tex.compute_amax(input, amax)

    def fused_amax_and_scale_update_after_reduction(self, amax_reduction_buffer, amax_histories,
                                                    scales, amax_compute_algo, fp8_dtype,
                                                    margin) -> None:
        tex = self._get_tex()
        return tex.fused_amax_and_scale_update_after_reduction(
            amax_reduction_buffer, amax_histories, scales, amax_compute_algo, fp8_dtype, margin)

    def fp8_block_scaling_compute_partial_amax(self, tensor, amax, h, w, start_offset,
                                               block_len) -> None:
        tex = self._get_tex()
        return tex.fp8_block_scaling_compute_partial_amax(tensor, amax, h, w, start_offset, block_len)

    def fp8_block_scaling_partial_cast(self, inp, out, scale, h, w, start_offset,
                                       block_len, out_dtype) -> None:
        tex = self._get_tex()
        return tex.fp8_block_scaling_partial_cast(inp, out, scale, h, w, start_offset,
                                                  block_len, out_dtype)

    def mxfp8_scaling_compute_partial_amax(self, tensor, amax, h, w, start_offset,
                                           block_len) -> None:
        tex = self._get_tex()
        return tex.mxfp8_scaling_compute_partial_amax(tensor, amax, h, w, start_offset, block_len)

    def mxfp8_scaling_partial_cast(self, inp, out, scale, h, w, start_offset,
                                   block_len, out_dtype) -> None:
        tex = self._get_tex()
        return tex.mxfp8_scaling_partial_cast(inp, out, scale, h, w, start_offset,
                                              block_len, out_dtype)

    ##### NVFP4 #####
    def nvfp4_2d_compute_partial_amax(self, tensor, amax, h, w, start_offset,
                                      block_len=16) -> None:
        tex = self._get_tex()
        return tex.nvfp4_2d_compute_partial_amax(tensor, amax, h, w, start_offset, block_len)

    def nvfp4_multi_tensor_compute_partial_amax(self, master_weight_list, partial_amax_list,
                                                global_amax_list, h_list, w_list,
                                                start_offset_list, block_len=16) -> None:
        tex = self._get_tex()
        return tex.nvfp4_multi_tensor_compute_partial_amax(
            master_weight_list, partial_amax_list, global_amax_list,
            h_list, w_list, start_offset_list, block_len)

    def nvfp4_compute_global_scale(self, global_amaxes, global_scale_tensor) -> None:
        tex = self._get_tex()
        return tex.nvfp4_compute_global_scale(global_amaxes, global_scale_tensor)

    def nvfp4_compute_per_block_scale(self, *args, **kwargs):
        tex = self._get_tex()
        return tex.nvfp4_compute_per_block_scale(*args, **kwargs)

    def nvfp4_expand_scale_to_fp8(self, *args, **kwargs):
        tex = self._get_tex()
        return tex.nvfp4_expand_scale_to_fp8(*args, **kwargs)

    def nvfp4_fused_scale(self, *args, **kwargs):
        tex = self._get_tex()
        return tex.nvfp4_fused_scale(*args, **kwargs)

    def nvfp4_multi_tensor_fused_scale(
        self,
        block_amax_list: List[torch.Tensor],
        global_amax_list: List[torch.Tensor],
        per_block_scale_list: List[torch.Tensor],
        target_scale_list: List[torch.Tensor],
        target_amax_list: List[torch.Tensor],
        tile_rows_list: List[int],
        tile_cols_list: List[int],
        rows_padded_list: List[int],
        block_len: int,
    ) -> None:
        tex = self._get_tex()
        return tex.nvfp4_multi_tensor_fused_scale(
            block_amax_list, global_amax_list, per_block_scale_list,
            target_scale_list, target_amax_list, tile_rows_list, tile_cols_list,
            rows_padded_list, block_len)

    def nvfp4_2d_partial_cast(self, inp, out, scale, global_scale, h, w,
                              start_offset, block_len=16) -> None:
        tex = self._get_tex()
        return tex.nvfp4_2d_partial_cast(inp, out, scale, global_scale, h, w,
                                         start_offset, block_len)

    def nvfp4_multi_tensor_2d_partial_cast(self, inp_list, *args, **kwargs):
        tex = self._get_tex()
        return tex.nvfp4_multi_tensor_2d_partial_cast(inp_list, *args, **kwargs)

    def nvfp4_2d_multi_tensor_transpose(self, rowwise_data_list, columnwise_data_list,
                                        rowwise_scale_inv_list, columnwise_scale_inv_list,
                                        M_list, K_list) -> None:
        tex = self._get_tex()
        return tex.nvfp4_2d_multi_tensor_transpose(
            rowwise_data_list, columnwise_data_list, rowwise_scale_inv_list,
            columnwise_scale_inv_list, M_list, K_list)

    ##### Multi-tensor #####
    def multi_tensor_quantize(self, tensor_list, quantizer_list) -> List[Any]:
        tex = self._get_tex()
        return tex.multi_tensor_quantize(tensor_list, quantizer_list)

    def split_quantize(self, tensor, split_sections, quantizer_list,
                       disable_bulk_allocation=False) -> List[Any]:
        tex = self._get_tex()
        return tex.split_quantize(tensor, split_sections, quantizer_list, disable_bulk_allocation)

    def multi_tensor_scale(self, chunk_size, noop_flag, tensor_lists, scale) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale)

    def multi_tensor_scale_tensor(self, chunk_size, noop_flag, tensor_lists, scale) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_scale_tensor(chunk_size, noop_flag, tensor_lists, scale)

    def multi_tensor_l2norm(self, chunk_size, noop_flag, tensor_lists,
                            per_tensor=False) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.multi_tensor_l2norm(chunk_size, noop_flag, tensor_lists, per_tensor)

    def multi_tensor_unscale_l2norm(self, chunk_size, noop_flag, tensor_lists, inv_scale,
                                    per_tensor=False) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.multi_tensor_unscale_l2norm(chunk_size, noop_flag, tensor_lists,
                                               inv_scale, per_tensor)

    def multi_tensor_adam(self, chunk_size, noop_flag, tensor_lists, lr, beta1, beta2,
                         epsilon, step, mode, bias_correction, weight_decay) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_adam(chunk_size, noop_flag, tensor_lists, lr, beta1, beta2,
                                    epsilon, step, mode, bias_correction, weight_decay)

    def multi_tensor_adam_param_remainder(self, chunk_size, noop_flag, tensor_lists, lr, beta1,
                                         beta2, epsilon, step, mode, bias_correction,
                                         weight_decay) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_adam_param_remainder(
            chunk_size, noop_flag, tensor_lists, lr, beta1, beta2,
            epsilon, step, mode, bias_correction, weight_decay)

    def multi_tensor_adam_fp8(self, chunk_size, noop_flag, tensor_lists, lr, beta1, beta2,
                             epsilon, step, mode, bias_correction, weight_decay,
                             fp8_dtype) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_adam_fp8(chunk_size, noop_flag, tensor_lists, lr, beta1, beta2,
                                        epsilon, step, mode, bias_correction, weight_decay,
                                        fp8_dtype)

    def multi_tensor_adam_capturable(self, chunk_size, noop_flag, tensor_lists, lr, beta1, beta2,
                                    epsilon, step, mode, bias_correction, weight_decay,
                                    inv_scale) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_adam_capturable(
            chunk_size, noop_flag, tensor_lists, lr, beta1, beta2,
            epsilon, step, mode, bias_correction, weight_decay, inv_scale)

    def multi_tensor_adam_capturable_master(self, chunk_size, noop_flag, tensor_lists, lr, beta1,
                                           beta2, epsilon, step, mode, bias_correction,
                                           weight_decay, inv_scale) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_adam_capturable_master(
            chunk_size, noop_flag, tensor_lists, lr, beta1, beta2,
            epsilon, step, mode, bias_correction, weight_decay, inv_scale)

    def multi_tensor_sgd(self, chunk_size, noop_flag, tensor_lists, wd, momentum, dampening,
                         lr, nesterov, first_run, wd_after_momentum, scale) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_sgd(chunk_size, noop_flag, tensor_lists, wd, momentum,
                                    dampening, lr, nesterov, first_run, wd_after_momentum, scale)

    def multi_tensor_compute_scale_and_scale_inv(self, chunk_size, noop_flag, tensor_lists,
                                                 max_fp8, force_pow_2_scales, epsilon) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_compute_scale_and_scale_inv(
            chunk_size, noop_flag, tensor_lists, max_fp8, force_pow_2_scales, epsilon)

    def multi_tensor_compute_scale_inv_e8m0(self, chunk_size, noop_flag, tensor_lists,
                                            block_len) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_compute_scale_inv_e8m0(chunk_size, noop_flag, tensor_lists,
                                                       block_len)

    ##### Grouped GEMM #####
    def te_general_grouped_gemm(self, A, transa, B, transb, D, D_type, m_splits, bias,
                                bias_type, single_output, pre_gelu_out, grad, workspace,
                                workspaceSizes, accumulate, use_split_accumulator,
                                math_sm_count) -> Optional[List[torch.Tensor]]:
        tex = self._get_tex()
        bias_type = tex.DType(int(bias_type)) if bias_type is not None else None

        has_gelu = any(p is not None and p.numel() > 0 for p in pre_gelu_out) if pre_gelu_out else False
        if not has_gelu:
            return tex.te_general_grouped_gemm(
                A, transa, B, transb, D, D_type, m_splits, bias, bias_type,
                single_output, pre_gelu_out, grad, workspace, workspaceSizes,
                accumulate, use_split_accumulator, math_sm_count)

        # sublasLt only supports GeLU epilogue for FP8; fallback to GEMM + separate GeLU
        # GELU convention: tanh approximation, matching common/util/math.h
        # and the reference implementations.
        empty_gelu = [torch.empty(0, device=A[0].device, dtype=A[0].dtype) for _ in pre_gelu_out]
        tex.te_general_grouped_gemm(
            A, transa, B, transb, D, D_type, m_splits, bias, bias_type,
            single_output, empty_gelu, grad, workspace, workspaceSizes,
            accumulate, use_split_accumulator, math_sm_count)

        num_gemms = len(A)
        if not grad:
            # Forward: save GEMM output to pre_gelu_out, then apply GeLU in-place on D
            if single_output:
                m_each = D[0].shape[0] // num_gemms
                for i in range(num_gemms):
                    chunk = D[0][i * m_each:(i + 1) * m_each]
                    pre_gelu_out[i].copy_(chunk)
                    chunk.copy_(torch.nn.functional.gelu(chunk, approximate="tanh"))
            else:
                for i in range(num_gemms):
                    if pre_gelu_out[i] is not None and pre_gelu_out[i].numel() > 0:
                        pre_gelu_out[i].copy_(D[i])
                        D[i].copy_(torch.nn.functional.gelu(D[i], approximate="tanh"))
        else:
            # Backward: D already has GEMM result, apply dGeLU element-wise
            if single_output:
                m_each = D[0].shape[0] // num_gemms
                for i in range(num_gemms):
                    if pre_gelu_out[i] is not None and pre_gelu_out[i].numel() > 0:
                        chunk = D[0][i * m_each:(i + 1) * m_each]
                        chunk.copy_(self._dgelu(chunk, pre_gelu_out[i]))
            else:
                for i in range(num_gemms):
                    if pre_gelu_out[i] is not None and pre_gelu_out[i].numel() > 0:
                        D[i].copy_(self._dgelu(D[i], pre_gelu_out[i]))
        return bias

    @staticmethod
    def _dgelu(grad_output, gelu_input):
        """GELU backward: grad_output * gelu'(gelu_input), tanh approximation."""
        return BirenBackend._dgelu_closed(grad_output, gelu_input)

    def te_general_grouped_gemm_for_grouped_tensor(self, *args, **kwargs):
        tex = self._get_tex()
        return tex.te_general_grouped_gemm_for_grouped_tensor(*args, **kwargs)

    def te_general_grouped_gemm_for_discrete_in(self, *args, **kwargs):
        tex = self._get_tex()
        return tex.te_general_grouped_gemm_for_discrete_in(*args, **kwargs)

    def te_general_grouped_gemm_for_discrete_out(self, *args, **kwargs):
        tex = self._get_tex()
        return tex.te_general_grouped_gemm_for_discrete_out(*args, **kwargs)

    ##### Padding #####
    def fused_multi_row_padding(self, input, output, input_row_list,
                                padded_input_row_list) -> None:
        tex = self._get_tex()
        return tex.fused_multi_row_padding(input, output, input_row_list, padded_input_row_list)

    def fused_multi_row_unpadding(self, input, output, input_row_list,
                                  unpadded_input_row_list) -> None:
        tex = self._get_tex()
        return tex.fused_multi_row_unpadding(input, output, input_row_list, unpadded_input_row_list)

    ##### Context Parallel (THD) #####
    def thd_read_half_tensor(self, tensor, cu_seqlens, half_idx) -> torch.Tensor:
        tex = self._get_tex()
        return tex.thd_read_half_tensor(tensor, cu_seqlens, half_idx)

    def thd_second_half_lse_correction(self, lse, lse_per_step, cu_seqlens, lse_packed) -> None:
        tex = self._get_tex()
        return tex.thd_second_half_lse_correction(lse, lse_per_step, cu_seqlens, lse_packed)

    def thd_read_second_half_lse(self, lse, cu_seqlens, lse_packed,
                                 second_half_lse_seqlen) -> torch.Tensor:
        tex = self._get_tex()
        return tex.thd_read_second_half_lse(lse, cu_seqlens, lse_packed, second_half_lse_seqlen)

    def thd_out_correction(self, out, out_per_step, lse, lse_per_step, cu_seqlens,
                           only_second_half, lse_packed) -> None:
        tex = self._get_tex()
        return tex.thd_out_correction(out, out_per_step, lse, lse_per_step, cu_seqlens,
                                      only_second_half, lse_packed)

    def thd_grad_correction(self, grad, grad_per_step, cu_seqlens, first_half,
                            second_half) -> None:
        tex = self._get_tex()
        return tex.thd_grad_correction(grad, grad_per_step, cu_seqlens, first_half, second_half)

    def thd_get_partitioned_indices(self, cu_seqlens, total_tokens, world_size,
                                    rank) -> torch.Tensor:
        tex = self._get_tex()
        return tex.thd_get_partitioned_indices(cu_seqlens, total_tokens, world_size, rank)

    ##### NVSHMEM (stub) #####
    def init_nvshmem_backend(self, process_group) -> None:
        raise NotImplementedError("NVSHMEM is not supported on SUPA/Biren")

    def create_nvshmem_tensor(self, shape, dtype) -> torch.Tensor:
        raise NotImplementedError("NVSHMEM is not supported on SUPA/Biren")

    def nvshmem_send_on_current_stream(self, src, dst, peer, signal) -> None:
        raise NotImplementedError("NVSHMEM is not supported on SUPA/Biren")

    def nvshmem_wait_on_current_stream(self, signal, wait_kind) -> None:
        raise NotImplementedError("NVSHMEM is not supported on SUPA/Biren")

    def nvshmem_finalize(self) -> None:
        raise NotImplementedError("NVSHMEM is not supported on SUPA/Biren")

    ##### Comm+GEMM Overlap #####
    def bulk_overlap_ag_with_external_gemm(self, allgather_communicator, send_stream,
                                           recv_stream) -> Any:
        tex = self._get_tex()
        return tex.bulk_overlap_ag_with_external_gemm(allgather_communicator, send_stream,
                                                      recv_stream)

    ##### Misc #####
    def get_cublasLt_version(self) -> int:
        tex = self._get_tex()
        return tex.get_cublasLt_version()

    def get_cudnn_version(self) -> int:
        tex = self._get_tex()
        return tex.get_cudnn_version()

    def get_num_cublas_streams(self) -> int:
        tex = self._get_tex()
        return tex.get_num_cublas_streams()

    def create_empty_quantized_tensor(self, quantizer, shape, dtype, device, pin_memory):
        # Backs Quantizer.make_empty (FP8 TP gather paths in distributed.py,
        # aten.new_empty/empty_like on QuantizedTensor). The C++ side is
        # host-only allocation (quantizer.cpp create_tensor), safe on SUPA.
        tex = self._get_tex()
        return tex.create_empty_quantized_tensor(quantizer, shape, dtype, device, pin_memory)

    ##### Class factories #####
    def get_flash_attention_class(self):
        try:
            import flashattn_train  # noqa: F401
            from transformer_engine.plugin.core.backends.vendor.biren.flash_attention import (
                FlashAttentionBiren,
            )
            return FlashAttentionBiren
        except ImportError:
            return None

    def create_fp8_tensor_meta(self) -> FP8TensorMeta:
        tex = self._get_tex()
        return tex.FP8TensorMeta()

    def create_comm_overlap_helper(self, world_group=None, intra_node_group=None):
        tex = self._get_tex()
        return tex.CommOverlapHelper(world_group, intra_node_group)

    def create_comm_overlap(self, buffer_shape, buffer_dtype, helper, tp_size,
                            num_splits=3, num_max_streams=3, comm_cga_size=2,
                            gemm_priority=0, comm_priority=0, num_comm_sm=16,
                            set_sm_margin=True, atomic_gemm=False,
                            rs_overlap_first_gemm=False, *,
                            use_cublasmp=False, comm_type=None):
        tex = self._get_tex()
        # Normalize the FL-side Python IntEnum (plugin/core/ops.py) to the
        # native pybind enum: CommOverlapType is registered with
        # pybind11::module_local() and does not implicitly accept int or
        # IntEnum values. Same convention as the CUDA backend.
        comm_type = (
            tex.CommOverlapType.RS if comm_type is None else tex.CommOverlapType(int(comm_type))
        )
        return tex.CommOverlap(
            buffer_shape, buffer_dtype, helper, tp_size,
            use_cublasmp=use_cublasmp,
            comm_type=comm_type,
            num_splits=num_splits,
            num_max_streams=num_max_streams,
            comm_cga_size=comm_cga_size,
            gemm_priority=gemm_priority,
            comm_priority=comm_priority,
            num_comm_sm=num_comm_sm,
            set_sm_margin=set_sm_margin,
            atomic_gemm=atomic_gemm,
            rs_overlap_first_gemm=rs_overlap_first_gemm,
        )

    def create_comm_overlap_p2p(self, buffer_shape, buffer_dtype, helper, tp_size,
                                comm_type, num_max_streams=3, comm_cga_size=1,
                                gemm_priority=0, comm_priority=0, num_comm_sm=1,
                                set_sm_margin=False, atomic_gemm=False,
                                use_ce=True, aggregate=False, *,
                                use_cublasmp=False):
        tex = self._get_tex()
        # Normalize the FL-side Python IntEnum to the native pybind enum,
        # matching the CUDA backend (comm_type is required for P2P).
        comm_type = tex.CommOverlapType(int(comm_type))
        return tex.CommOverlapP2P(
            buffer_shape, buffer_dtype, helper, tp_size, comm_type,
            num_max_streams, comm_cga_size, gemm_priority, comm_priority,
            num_comm_sm, set_sm_margin, atomic_gemm, use_ce, aggregate,
            use_cublasmp=use_cublasmp)
