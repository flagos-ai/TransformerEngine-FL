# Copyright (c) 2026, BAAI. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""NPU Flash Attention adapter.

Bridges TE-FL's FlashAttention calling convention to NPU's npu_fusion_attention kernel.

TE-FL passes many parameters (qkv_layout, window_size, cp_group, fp8, etc.)
that NPU's FlashAttention doesn't support. This adapter:
  1. Accepts the full TE-FL parameter set
  2. Maps qkv_layout → qkv_format (sbhd/thd)
  3. Forwards only the supported parameters to NPU's FlashAttention
  4. Silently ignores unsupported features (sliding window, CP, FP8, ALiBi)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from transformer_engine.plugin.core.ops import FlashAttentionBase


_COMPRESSED_MASK_SIZE = 2048

_COMPRESSED_CAUSAL_MASK = None


def get_compressed_causal_mask(device="npu"):
    global _COMPRESSED_CAUSAL_MASK
    if _COMPRESSED_CAUSAL_MASK is None:
        _COMPRESSED_CAUSAL_MASK = torch.triu(
            torch.ones(
                (_COMPRESSED_MASK_SIZE, _COMPRESSED_MASK_SIZE),
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
    return _COMPRESSED_CAUSAL_MASK


class NPUFlashAttention(FlashAttentionBase):
    """FlashAttention adapter for NPU (Ascend) hardware.

    Wraps transformer_engine_npu's FlashAttention, which calls
    torch_npu.npu_fusion_attention under the hood.

    Supported features:
      - sbhd, bshd (via transpose), and thd formats
      - causal / padding mask types
      - Variable-length sequences (cu_seqlens)
      - Sparse mask optimization (via NPU's get_fa_config)

    Not supported (silently ignored):
      - Sliding window attention (window_size)
      - ALiBi slopes
      - Context Parallelism (cp_group, cp_stream, etc.)
      - FP8 / quantization
      - KV cache (inference_params)
      - FA v2/v3 version selection
    """

    def __init__(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = None,
        attention_type: str = "self",
        layer_number: Optional[int] = None,
        deterministic: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            softmax_scale=softmax_scale,
            attention_dropout=attention_dropout,
            attention_dropout_ctx=attention_dropout_ctx,
            attention_type=attention_type,
            layer_number=layer_number,
            deterministic=deterministic,
        )
        self.softmax_scale = softmax_scale
        self.attention_dropout = attention_dropout
        self.attention_type = attention_type
        self.layer_number = layer_number
        self._npu_flash = None

    def _ensure_backend(self):
        """Lazy-initialize NPU FlashAttention backend."""
        if self._npu_flash is not None:
            return
        from transformer_engine_npu.pytorch.attention.dot_product_attention.backends import (
            FlashAttention as _NPUFlashAttention,
        )

        self._npu_flash = _NPUFlashAttention(self.softmax_scale)

    @staticmethod
    def _layout_to_format(qkv_layout: Optional[str]) -> str:
        """Map TE-FL qkv_layout string to NPU qkv_format."""
        if qkv_layout is None:
            return "sbhd"
        layout = qkv_layout.lower()
        if "thd" in layout or layout.startswith("t"):
            return "thd"
        return "sbhd"

    @staticmethod
    def _is_bshd_layout(qkv_layout: Optional[str]) -> bool:
        """Whether the separate Q/K/V tensors use batch-major BSHD layout."""
        if qkv_layout is None:
            return False
        layout = qkv_layout.lower()
        # Covers bs3hd, bsh3d, and bshd_bshd_bshd.
        return layout.startswith("bs")

    def _forward_impl(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        qkv_layout: Optional[str] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        cp_group: Optional[Any] = None,
        cp_global_ranks: Optional[List[int]] = None,
        cp_stream: Optional[Any] = None,
        cp_comm_type: str = "p2p",
        fp8: bool = False,
        fp8_meta: Optional[Dict[str, Any]] = None,
        quantizers: Optional[Any] = None,
        inference_params: Optional[Any] = None,
        flash_attention_backend: Optional[Any] = None,
        fp8_output: bool = False,
        num_splits: Optional[int] = 1,
    ) -> torch.Tensor:
        """Forward pass — adapts TE-FL args to NPU FlashAttention interface.

        Only passes: query, key, value, attention_mask, qkv_format,
                     cu_seqlens_q, cu_seqlens_kv, attn_mask_type

        Raises:
            NotImplementedError: For features that would produce incorrect results
                if silently ignored (window_size, alibi_slopes, cp_group).

        Warns:
            For features that don't affect correctness but differ from user
            expectation (fp8, inference_params).
        """
        # --- Validate: features that would silently produce wrong results ---
        if window_size is not None and window_size not in ((-1, -1), (-1, 0)):
            raise NotImplementedError(
                "NPU FlashAttention does not support sliding window attention "
                f"(window_size={window_size}). npu_fusion_attention only computes "
                "full causal/padding attention. Either disable sliding window or "
                "use UnfusedDotProductAttention as fallback."
            )

        if alibi_slopes is not None:
            raise NotImplementedError(
                "NPU FlashAttention does not support ALiBi position encoding "
                "(alibi_slopes). npu_fusion_attention has no ALiBi parameter. "
                "Use RoPE or other position encoding supported by NPU."
            )

        if cp_group is not None:
            raise NotImplementedError(
                "NPU FlashAttention does not support Context Parallelism "
                "(cp_group). Ring attention / CP requires NPU-specific HCCL "
                "implementation which is not yet available."
            )

        # --- Warn: features that don't break correctness but differ from expectation ---
        if fp8:
            import warnings

            warnings.warn(
                "NPU FlashAttention does not support FP8 attention computation. "
                "Falling back to BF16/FP16 precision. Results are correct but "
                "without FP8 performance optimization.",
                stacklevel=2,
            )

        if inference_params is not None:
            import warnings

            warnings.warn(
                "NPU FlashAttention does not support KV cache (inference_params). "
                "Full recomputation will be used. This is correct but slower for "
                "autoregressive inference.",
                stacklevel=2,
            )

        # TransformerEngineNPU only accepts sequence-major SBHD or packed THD.
        # Convert batch-major BSHD inputs explicitly instead of only relabeling
        # their layout, which would swap the semantic batch and sequence axes.
        input_is_bshd = self._is_bshd_layout(qkv_layout)
        if input_is_bshd:
            query_layer = query_layer.transpose(0, 1).contiguous()
            key_layer = key_layer.transpose(0, 1).contiguous()
            value_layer = value_layer.transpose(0, 1).contiguous()

        qkv_format = self._layout_to_format(qkv_layout)
        if attn_mask_type in ("causal", "padding_causal", "padding,causal", "causal,padding"):
            attention_mask = get_compressed_causal_mask(query_layer.device)

        self._ensure_backend()
        output = self._npu_flash(
            query_layer,
            key_layer,
            value_layer,
            attention_mask=attention_mask,
            qkv_format=qkv_format,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            attn_mask_type=attn_mask_type,
        )

        if input_is_bshd:
            output = output.transpose(0, 1).contiguous()

        return output
